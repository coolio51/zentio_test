# streamlit_app.py
# Production Scheduler (CP-SAT) — Phased Operations (Setup/Run/Clean)
# - Machine capacity & eligibility
# - Worker skills & fractional attention (scaled by CAP)
# - Calendars for machines & workers
# - Precedence (DAG) and due dates (tardiness)
# - Validation + clear errors + Gantt-like view
#
# Acronyms:
# - CP-SAT = Constraint Programming with a SAT-based solver (in OR-Tools)
# - OR-Tools = Google’s open-source optimization toolkit
# - DAG = Directed Acyclic Graph (for operation precedence)
# - KPI = Key Performance Indicator (makespan, tardiness, etc.)

import json, traceback
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from ortools.sat.python import cp_model

st.set_page_config(page_title="Production Scheduler (CP-SAT)", layout="wide")

# -----------------------------
# Sidebar glossary
# -----------------------------
with st.sidebar.expander("Glossary (acronyms & terms)", expanded=True):
    st.markdown("""
- **CP-SAT**: Constraint Programming with a SAT-based solver. A combinatorial optimization solver in **OR-Tools** that mixes integer programming and SAT search.
- **OR-Tools**: Open-source optimization toolkit by Google (routing, linear, CP-SAT, scheduling).
- **DAG**: Directed Acyclic Graph — nodes + directed edges with no cycles (used for operation precedence).
- **KPI**: Key Performance Indicator — e.g., makespan, tardiness, setup time.
- **Gantt**: Timeline chart showing resource allocation (machines/workers) over time.
""")

# -----------------------------
# Default demo data
# -----------------------------
def default_data():
    return {
        "Machines": ["M1", "M2"],
        "Workers":  ["W1", "W2", "W3", "W4", "W5"],
        "Skills":   ["Setup", "Run", "Clean"],
        "Phases":   ["Setup", "Run", "Clean"],

        # T = time buckets (discrete units, e.g., each = 15 minutes)
        "T": list(range(1, 11)),

        # Ops = operations (each will run Setup -> Run -> Clean)
        "Ops": ["OpA", "OpB", "OpC"],

        # JobOf maps each operation to a job (two jobs here)
        "JobOf": {"OpA": "J1", "OpB": "J1", "OpC": "J2"},
        "Jobs": ["J1", "J2"],

        # Due dates (in bucket indices)
        # Adjusted due date so Job J1 can finish within horizon
        "Due": {"J1": 10, "J2": 6},

        # E = eligibility (op x machine -> 1 allowed / 0 not allowed)
        "E": {
            "OpA": {"M1": 1, "M2": 0},
            "OpB": {"M1": 1, "M2": 1},
            "OpC": {"M1": 0, "M2": 1},
        },

        # D = durations (buckets) per op/phase/machine
        "D": {
            "OpA": {"Setup": {"M1": 1}, "Run": {"M1": 3}, "Clean": {"M1": 1}},
            "OpB": {
                "Setup": {"M1": 1, "M2": 1},
                "Run":   {"M1": 3, "M2": 3},
                "Clean": {"M1": 1, "M2": 1},
            },
            "OpC": {"Setup": {"M2": 1}, "Run": {"M2": 2}, "Clean": {"M2": 1}},
        },

        # Need = operator capacity need per op/phase/skill (1.0 = one full worker)
        "Need": {
            "OpA": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}},
            "OpB": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}},
            "OpC": {"Setup": {"Setup": 1.0}, "Run": {"Run": 1.0}, "Clean": {"Clean": 1.0}},
        },

        # Q = worker skills (1 qualified, 0 not)
        # Updated worker skills so every phase has at least one qualified worker
        "Q": {
            "W1": {"Setup": 1, "Run": 1, "Clean": 1},
            "W2": {"Setup": 0, "Run": 1, "Clean": 0},
            "W3": {"Setup": 0, "Run": 0, "Clean": 1},
            "W4": {"Setup": 1, "Run": 1, "Clean": 1},
            "W5": {"Setup": 0, "Run": 1, "Clean": 0},
        },

        # A_M = machine availability per bucket (1 available, 0 down)
        "A_M": {
            "M1": {str(t): 1 for t in range(1, 11)},
            "M2": {str(t): 1 for t in range(1, 11)},
        },

        # A_W_frac = worker availability per bucket (0..1 fraction of attention)
        "A_W_frac": {
            w: {str(t): 1.0 for t in range(1, 11)} for w in ["W1","W2","W3","W4","W5"]
        },

        # Precedence edges [pred, succ]
        "PredEdges": [["OpA", "OpB"]],

        # CAP = scaling from fractional attention to integers (10 -> 0.1 steps)
        "CAP": 10
    }

# -----------------------------
# Helpers
# -----------------------------
def val_at(d, t):
    """Safely read value at time bucket t from a dict that may use int or str keys."""
    return d.get(t, d.get(str(t)))

def normalize_calendars(data):
    """Normalize A_M and A_W_frac so keys are ints and cover all T."""
    T = data["T"]
    # Machines
    A_M = {}
    for m in data["Machines"]:
        row = data["A_M"].get(m, {})
        A_M[m] = {t: int(val_at(row, t) or 0) for t in T}
    # Workers
    A_W_frac = {}
    for w in data["Workers"]:
        row = data["A_W_frac"].get(w, {})
        A_W_frac[w] = {t: float(val_at(row, t) if val_at(row, t) is not None else 0.0) for t in T}
    return A_M, A_W_frac

def validate_data(d):
    errors = []

    required_top = ["Machines","Workers","Skills","Phases","T","Ops","JobOf","Jobs","Due","E","D","Need","Q","A_M","A_W_frac","CAP"]
    for k in required_top:
        if k not in d:
            errors.append(f"Missing top-level key: '{k}'")

    if errors:
        return errors

    if not isinstance(d["T"], list) or not d["T"] or not all(isinstance(t, int) for t in d["T"]):
        errors.append("T must be a non-empty list of integers (time bucket indices).")

    if set(d["Phases"]) != {"Setup","Run","Clean"}:
        errors.append("Phases must be exactly ['Setup','Run','Clean'].")

    # Op -> Job mapping
    for p in d["Ops"]:
        if p not in d["JobOf"]:
            errors.append(f"JobOf missing op '{p}'.")
    for p, j in d["JobOf"].items():
        if p not in d["Ops"]:
            errors.append(f"JobOf references unknown op '{p}'.")
        if j not in d["Jobs"]:
            errors.append(f"JobOf[{p}] references unknown job '{j}'.")

    # E & D consistency
    for p in d["Ops"]:
        elig = [m for m in d["Machines"] if d["E"].get(p,{}).get(m,0)==1]
        if not elig:
            errors.append(f"Operation '{p}' has no eligible machines in E.")
        else:
            for m in elig:
                for k in d["Phases"]:
                    dur = d["D"].get(p,{}).get(k,{}).get(m, None)
                    if dur is None:
                        errors.append(f"Missing D[{p}][{k}][{m}] for eligible machine.")
                    elif not (isinstance(dur,int) and dur>=0):
                        errors.append(f"D[{p}][{k}][{m}] must be a non-negative integer.")

    # Need map must reference known skills and be >=0
    for p in d["Ops"]:
        for k in d["Phases"]:
            for s,val in d["Need"].get(p,{}).get(k,{}).items():
                if s not in d["Skills"]:
                    errors.append(f"Need[{p}][{k}] uses unknown skill '{s}'.")
                if not isinstance(val,(int,float)) or val<0:
                    errors.append(f"Need[{p}][{k}]['{s}'] must be a non-negative number.")

    # Worker skills Q
    for w in d["Workers"]:
        if w not in d["Q"]:
            errors.append(f"Q missing worker '{w}'.")
            continue
        for s,v in d["Q"][w].items():
            if s not in d["Skills"]:
                errors.append(f"Q[{w}] references unknown skill '{s}'.")
            if v not in (0,1):
                errors.append(f"Q[{w}]['{s}'] must be 0 or 1.")

    # Due times
    for j in d["Jobs"]:
        if j not in d["Due"] or not isinstance(d["Due"][j], int):
            errors.append(f"Due['{j}'] must be an integer bucket index.")

    # CAP
    if not isinstance(d["CAP"], int) or d["CAP"] <= 0:
        errors.append("CAP must be a positive integer.")

    return errors

# -----------------------------
# Solver
# -----------------------------
def solve_cp_sat(data, time_limit_sec=15, log_progress=False):
    Machines = data["Machines"]
    Workers  = data["Workers"]
    Skills   = data["Skills"]
    Phases   = data["Phases"]
    T        = data["T"]                                        # time buckets
    Ops      = data["Ops"]                                      # operations
    JobOf    = data["JobOf"]                                    # op -> job mapping
    Jobs     = data["Jobs"]                                     # jobs
    Due      = data["Due"]                                      # due dates
    E        = data["E"]                                        # eligibility
    D        = data["D"]                                        # durations
    Need     = data["Need"]                                     # operator needs
    Q        = data["Q"]                                        # worker skills
    A_M_raw  = data["A_M"]                                      # machine availability
    A_W_raw  = data["A_W_frac"]                                 # worker availability (fractions)
    PredEdges = [tuple(x) for x in data.get("PredEdges", [])]   # precedence edges
    CAP      = int(data.get("CAP", 10)) # capacity scaling

    # Normalize calendars (accept "1" or 1 as keys)
    A_M, A_W_frac = normalize_calendars({
        "Machines": Machines, "Workers": Workers, "T": T,
        "A_M": A_M_raw, "A_W_frac": A_W_raw
    })

    # Convert worker availability to integer capacity units
    A_W = {w: {t: int(round(A_W_frac[w][t] * CAP)) for t in T} for w in Workers}

    m = cp_model.CpModel()

    # Decision variables
    z = {(p, mach): m.NewBoolVar(f"z[{p},{mach}]") for p in Ops for mach in Machines}
    x = {(p, mach, t): m.NewBoolVar(f"x[{p},{mach},{t}]") for p in Ops for mach in Machines for t in T}
    y = {(p, k, t): m.NewBoolVar(f"y[{p},{k},{t}]") for p in Ops for k in Phases for t in T}
    u = {(p, k, s, w, t): m.NewIntVar(0, CAP, f"u[{p},{k},{s},{w},{t}]")
         for p in Ops for k in Phases for s in Skills for w in Workers for t in T}
    C_op  = {p: m.NewIntVar(0, max(T), f"C[{p}]") for p in Ops}
    C_job = {j: m.NewIntVar(0, max(T), f"C_job[{j}]") for j in Jobs}
    Tard  = {j: m.NewIntVar(0, max(T), f"Tard[{j}]") for j in Jobs}
    C_max = m.NewIntVar(0, max(T), "C_max")

    # 1) Exactly one eligible machine per operation
    for p in Ops:
        m.Add(sum(z[(p, mach)] for mach in Machines) == 1)
        for mach in Machines:
            if E.get(p, {}).get(mach, 0) == 0:
                m.Add(z[(p, mach)] == 0)

    # 2) Phase durations
    for p in Ops:
        for k in Phases:
            rhs_terms = []
            for mach in Machines:
                dur = D.get(p, {}).get(k, {}).get(mach, None)
                if E.get(p, {}).get(mach, 0) == 1 and dur is not None:
                    rhs_terms.append(z[(p, mach)] * int(dur))
            if not rhs_terms:
                raise ValueError(f"No durations in D for any eligible machine (op '{p}', phase '{k}').")
            m.Add(sum(y[(p, k, t)] for t in T) == sum(rhs_terms))

    # 3) Link phase activity to machine occupancy and availability
    for p in Ops:
        for t in T:
            m.Add(sum(y[(p, k, t)] for k in Phases) <= sum(x[(p, mach, t)] for mach in Machines))
            for mach in Machines:
                m.Add(x[(p, mach, t)] <= z[(p, mach)])
                m.Add(x[(p, mach, t)] <= A_M[mach][t])

    # 4) Machine capacity
    for mach in Machines:
        for t in T:
            m.Add(sum(x[(p, mach, t)] for p in Ops) <= 1)

    # 5) Operator skill coverage + qualification + availability
    for p in Ops:
        for k in Phases:
            need_map = Need.get(p, {}).get(k, {})
            for s in Skills:
                req = int(round(need_map.get(s, 0.0) * CAP))
                if req == 0:
                    for w in Workers:
                        for t in T:
                            m.Add(u[(p, k, s, w, t)] == 0)
                    continue
                for t in T:
                    m.Add(sum(u[(p, k, s, w, t)] for w in Workers) >= req * y[(p, k, t)])
                    for w in Workers:
                        if Q.get(w, {}).get(s, 0) == 0:
                            m.Add(u[(p, k, s, w, t)] == 0)
                        else:
                            m.Add(u[(p, k, s, w, t)] <= A_W[w][t])
                            m.Add(u[(p, k, s, w, t)] <= CAP * y[(p, k, t)])

    # 6) Worker capacity per bucket
    for w in Workers:
        for t in T:
            m.Add(sum(u[(p, k, s, w, t)] for p in Ops for k in Phases for s in Skills) <= A_W[w][t])

    # 7) Phase order: Setup before Run before Clean
    # A run bucket can occur only after some setup bucket has occurred at an earlier time
    # and a clean bucket can occur only after some run bucket.
    for p in Ops:
        for t in T:
            m.Add(y[(p, "Run", t)]
                  <= sum(y[(p, "Setup", tau)] for tau in T if tau < t))
            m.Add(y[(p, "Clean", t)]
                  <= sum(y[(p, "Run", tau)] for tau in T if tau < t))

    # 8) Precedence: successor's setup can only start after predecessor's cleanup has started at some earlier time
    for (pred_op, succ_op) in PredEdges:
        for t in T:
            m.Add(y[(succ_op, "Setup", t)] <= sum(y[(pred_op, "Clean", tau)] for tau in T if tau < t))

    # 9) Completion times
    for p in Ops:
        for k in Phases:
            for t in T:
                m.Add(C_op[p] >= t * y[(p, k, t)])

    # 10) Job completion, tardiness, makespan
    for j in Jobs:
        for p in Ops:
            if JobOf[p] == j:
                m.Add(C_job[j] >= C_op[p])
        m.Add(Tard[j] >= C_job[j] - Due[j])
    for p in Ops:
        m.Add(C_max >= C_op[p])

    # Objective: minimize total tardiness (weight 100) + makespan
    m.Minimize(sum(Tard[j] for j in Jobs) * 100 + C_max)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(15)
    solver.parameters.num_search_workers = 8
    solver.parameters.log_search_progress = bool(False)

    status = solver.Solve(m)
    status_map = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }
    out = {"status": status_map.get(status, str(status))}

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["KPIs"] = {
            "makespan": solver.Value(C_max),
            "jobs": {j: {"completion": solver.Value(C_job[j]), "due": int(Due[j]), "tardiness": solver.Value(Tard[j])}
                     for j in Jobs}
        }
        chosen = {}
        for p in Ops:
            chosen[p] = [mach for mach in Machines if solver.Value(z[(p, mach)]) == 1]
        out["chosen_machines"] = chosen

        # Machine schedule table
        ms_rows = []
        for mach in Machines:
            for t in T:
                label = "."
                for p in Ops:
                    if solver.Value(x[(p, mach, t)]) == 1:
                        phases_active = [k for k in Phases if solver.Value(y[(p, k, t)]) == 1]
                        label = f"{p}:{'+'.join(phases_active) if phases_active else '?'}"
                ms_rows.append({"machine": mach, "t": t, "slot": label})
        out["machine_schedule"] = pd.DataFrame(ms_rows)

        # Worker assignment table (non-zero u)
        wa_rows = []
        for t in T:
            for w in Workers:
                for p in Ops:
                    for k in Phases:
                        for s in Skills:
                            val = solver.Value(u[(p, k, s, w, t)])
                            if val > 0:
                                wa_rows.append({
                                    "t": t, "worker": w, "op": p, "phase": k, "skill": s, "attention": val / CAP
                                })
        out["worker_assignments"] = pd.DataFrame(wa_rows)

        # Phase activity table
        pa_rows = []
        for p in Ops:
            for k in Phases:
                for t in T:
                    if solver.Value(y[(p, k, t)]) == 1:
                        pa_rows.append({"op": p, "phase": k, "t": t})
        out["phase_activity"] = pd.DataFrame(pa_rows)

    return out

# -----------------------------
# UI
# -----------------------------
st.title("Production Scheduler (CP-SAT)")
st.caption("Discrete-time phased model with machines, worker skills/calendars, and precedence.")

left, right = st.columns([1,1])

with left:
    st.subheader("1) Model Data")
    demo = default_data()
    default_json = json.dumps(demo, indent=2)
    if "data_text" not in st.session_state:
        st.session_state["data_text"] = default_json

    if st.button("Reset to defaults"):
        st.session_state["data_text"] = default_json

    data_text = st.text_area("Edit JSON (or keep defaults)", value=st.session_state["data_text"], height=450)
    validate_btn = st.button("Validate only")
    solve_btn = st.button("Solve")

with right:
    st.subheader("2) Results")

def show_errors(errs):
    st.error("Validation errors:")
    for e in errs:
        st.write("• " + e)

if validate_btn:
    try:
        d = json.loads(data_text)
        errs = validate_data(d)
        if errs: show_errors(errs)
        else: st.success("Schema looks good ✔")
    except Exception as e:
        st.error(f"JSON parsing failed: {e}")
        st.code(traceback.format_exc())

if solve_btn:
    try:
        d = json.loads(data_text)
        errs = validate_data(d)
        if errs: show_errors(errs)
        else:
            res = solve_cp_sat(d)
            st.write(f"**Status:** {res['status']}")
            if res["status"] in ("OPTIMAL","FEASIBLE"):
                st.markdown("### KPIs")
                kpi = res["KPIs"]
                st.write({"makespan": kpi["makespan"]})
                st.table(pd.DataFrame(kpi["jobs"]).T)

                st.markdown("### Chosen machine per operation")
                st.json(res["chosen_machines"])

                st.markdown("### Machine schedule (table)")
                st.dataframe(res["machine_schedule"], use_containerWidth=True)

                st.markdown("### Machine schedule (Gantt-like)")
                df = res["machine_schedule"]
                fig, ax = plt.subplots(figsize=(10, 3 + 0.4 * len(df['machine'].unique())))
                yticks, ylabels = [], []
                y = 0
                for mach in df["machine"].unique():
                    segs = df[df["machine"] == mach]
                    for _, row in segs.iterrows():
                        if row["slot"] != ".":
                            ax.barh(y, 1.0, left=row["t"]-1, edgecolor="black")
                            ax.text(row["t"]-0.5, y, row["slot"], va="center", ha="center", fontsize=8)
                    ylabels.append(mach); yticks.append(y); y += 1
                ax.set_yticks(yticks, ylabels)
                ax.set_xlabel("time bucket"); ax.set_ylabel("machine")
                st.pyplot(fig)

                st.markdown("### Worker assignments (non-zero)")
                st.dataframe(res["worker_assignments"], use_container_width=True)

                st.markdown("### Phase activity (y=1)")
                st.dataframe(res["phase_activity"], use_container_width=True)
            else:
                st.warning("No feasible schedule found. Try relaxing due dates, checking eligibility, or extending T.")
    except Exception as e:
        st.error(f"Failed to parse/solve:\n{e}")
        st.code(traceback.format_exc())
