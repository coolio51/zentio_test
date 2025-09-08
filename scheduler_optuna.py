
# Streamlit Matrix Scheduler — GA + CP-SAT + Optuna (Snapshots)
# Run:
#   pip install streamlit numpy pandas matplotlib ortools optuna
#   streamlit run streamlit_matrix_scheduler_ga_cpsat_optuna_snap.py

import json, math, time, random, datetime
from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except Exception:
    HAS_CPSAT = False

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False

st.set_page_config(page_title="Matrix Scheduler — GA + CP-SAT + Optuna (Snapshots)", layout="wide")

# ====== persistence boot ======
if "snapshots" not in st.session_state:
    st.session_state["snapshots"] = []   # list of dicts: {"label", "genome_bytes", "csv_bytes"}
if "data" not in st.session_state:
    st.session_state["data"] = None
# Expected session_state keys
# GA best: ga_best_present (bool), ga_best_genome_json, ga_best_sched_csv, ga_best_sched_obj, ga_best_hist,
#          ga_best_total_ms, ga_best_score, ga_best_genomes_evaluated, ga_best_eval
# CP-SAT polish: cp_best_present (bool), cp_best_genome_json, cp_best_sched_csv, cp_best_sched_obj,
#                cp_best_eval, cp_best_polish_ms, cp_best_decode_ms
st.session_state.setdefault("ga_best_present", False)
st.session_state.setdefault("cp_best_present", False)
st.session_state.setdefault("ga_best_genome_json", None)
st.session_state.setdefault("ga_best_sched_csv", None)
st.session_state.setdefault("ga_best_sched_obj", None)
st.session_state.setdefault("ga_best_hist", None)
st.session_state.setdefault("ga_best_total_ms", None)
st.session_state.setdefault("ga_best_score", None)
st.session_state.setdefault("ga_best_genomes_evaluated", None)
st.session_state.setdefault("ga_best_eval", None)
st.session_state.setdefault("cp_best_genome_json", None)
st.session_state.setdefault("cp_best_sched_csv", None)
st.session_state.setdefault("cp_best_sched_obj", None)
st.session_state.setdefault("cp_best_eval", None)
st.session_state.setdefault("cp_best_polish_ms", None)
st.session_state.setdefault("cp_best_decode_ms", None)


# =============== utilities ===============
def now_ns(): return time.perf_counter_ns()
def ns_to_ms(ns): return ns / 1e6

def sliding_all_true(x_bool: np.ndarray, L: int) -> np.ndarray:
    T = x_bool.shape[0]
    if L <= 0 or L > T:
        return np.zeros(0, dtype=bool)
    x_int = x_bool.astype(np.int32)
    csum = np.empty(T + 1, dtype=np.int32); csum[0] = 0; np.cumsum(x_int, out=csum[1:])
    sums = csum[L:] - csum[:-L]
    return sums == L

# =============== configs ===============
@dataclass
class HorizonCfg: days:int=30; start_hour:int=8; end_hour:int=22; bucket_min:int=10
@dataclass
class ScaleCfg: machines:int=12; workers:int=25; jobs:int=50; ops_min:int=5; ops_max:int=20
@dataclass
class MaintCfg: blocks_per_machine:int=1; block_len_min:int=3; block_len_max:int=8
@dataclass
class NeedsCfg: cap:int=10
@dataclass
class ShiftCfg: break_bucket_offset:int=3; day_start:int=8; day_end:int=16; eve_start:int=16; eve_end:int=22; assign_evening_fraction:float=0.5

# =============== dataset gen ===============
def build_time_axis(hcfg: HorizonCfg):
    bph = 60 // hcfg.bucket_min
    bpd = (hcfg.end_hour - hcfg.start_hour) * bph
    T = np.arange(hcfg.days * bpd, dtype=np.int32)
    return {"T": T, "bph": bph, "bpd": bpd}

def generate_stress_dataset(seed: int, hcfg: HorizonCfg, scfg: ScaleCfg, mcfg: MaintCfg, ncfg: NeedsCfg, shcfg: ShiftCfg):
    rng = random.Random(seed)
    ax = build_time_axis(hcfg)
    T = ax["T"]; bph = ax["bph"]; bpd = ax["bpd"]

    Machines = [f"M{i+1}" for i in range(scfg.machines)]
    Workers  = [f"Ada_{i+1}" for i in range(scfg.workers)]
    process_skills = ["Drilling","Milling","Turning","Cutting","Holes","Injection"]
    machine_skills = [f"{role}_{m}" for m in Machines for role in ("setup","operate","clean")]
    Skills = process_skills + machine_skills

    # Machine availability with one maintenance block per machine
    A_M = np.ones((len(Machines), len(T)), dtype=bool)
    for mi, m in enumerate(Machines):
        L = rng.randint(mcfg.block_len_min, mcfg.block_len_max)
        day = rng.randint(0, hcfg.days-1)
        start_hour = rng.randint(hcfg.start_hour, hcfg.end_hour-1)
        start_bucket = day * bpd + (start_hour - hcfg.start_hour) * bph
        start_bucket = min(start_bucket, len(T) - L)
        A_M[mi, start_bucket:start_bucket+L] = False

    # Worker availability (fixed shifts + break)
    A_W = np.zeros((len(Workers), len(T)), dtype=bool)
    for wi, w in enumerate(Workers):
        eve = rng.random() < shcfg.assign_evening_fraction
        for d in range(hcfg.days):
            base = d * bpd
            if eve:
                s = base + (shcfg.eve_start - hcfg.start_hour) * bph
                e = base + (shcfg.eve_end   - hcfg.start_hour) * bph
            else:
                s = base + (shcfg.day_start - hcfg.start_hour) * bph
                e = base + (shcfg.day_end   - hcfg.start_hour) * bph
            if s < e:
                A_W[wi, s:e] = True
                brk = s + shcfg.break_bucket_offset
                if brk < e:
                    A_W[wi, brk] = False

    # Qualifications
    Q = np.zeros((len(Workers), len(Skills)), dtype=bool)
    s_idx = {s:i for i,s in enumerate(Skills)}
    for m in Machines:
        ops = rng.sample(Workers, k=min(len(Workers), rng.randint(3,6)))
        setups = rng.sample(Workers, k=min(len(Workers), rng.randint(2,4)))
        cleans = rng.sample(Workers, k=min(len(Workers), rng.randint(2,4)))
        for w in ops:    Q[Workers.index(w), s_idx[f"operate_{m}"]] = True
        for w in setups: Q[Workers.index(w), s_idx[f"setup_{m}"]]   = True
        for w in cleans: Q[Workers.index(w), s_idx[f"clean_{m}"]]   = True
    for wi, w in enumerate(Workers):
        for s in process_skills:
            if rng.random() < 0.35:
                Q[wi, s_idx[s]] = True

    # Jobs & ops
    Jobs = [f"J{j+1}" for j in range(scfg.jobs)]
    Ops = []; JobOf = {}; PredEdges = []
    base_setup = {m: rng.randint(1,4) for m in Machines}
    base_run   = {m: rng.randint(6,20) for m in Machines}
    base_clean = {m: rng.randint(1,3) for m in Machines}
    E = {}; D = {}; Need = {}
    op_counter = 1
    for j in Jobs:
        n_ops = rng.randint(scfg.ops_min, scfg.ops_max)
        job_ops = []
        for k in range(n_ops):
            op = f"O{op_counter}"; op_counter += 1
            job_ops.append(op); Ops.append(op); JobOf[op] = j
            k_m = rng.choice([1,2,2,3])
            elig = rng.sample(Machines, k=min(k_m, len(Machines)))
            E[op] = {m: 1 if m in elig else 0 for m in Machines}
            D[op] = {"Setup":{}, "Run":{}, "Clean":{}}
            Need[op] = {"Setup":{}, "Run":{}, "Clean":{}}
            run_proc_skill = random.choice(process_skills) if rng.random() < 0.30 else None
            for m in Machines:
                if E[op][m]==1:
                    su = max(0, base_setup[m] + rng.randint(-1,1))
                    rn = max(3, base_run[m]   + rng.randint(-2,2))
                    cl = max(0, base_clean[m] + rng.randint(-1,1))
                    D[op]["Setup"][m] = su; D[op]["Run"][m] = rn; D[op]["Clean"][m] = cl
            for m in Machines:
                if E[op][m]==1:
                    Need[op]["Setup"][f"setup_{m}"]  = float(random.choice((0.5,1.0,1.5)))
                    Need[op]["Run"][f"operate_{m}"]  = float(random.choice((1.0,1.5,2.0)))
                    if run_proc_skill is not None:
                        Need[op]["Run"][run_proc_skill] = float(random.choice((0.5,1.0)))
                    Need[op]["Clean"][f"clean_{m}"]  = float(random.choice((0.5,1.0)))
        # DAG layers
        L = rng.randint(3,5)
        layers = [[] for _ in range(L)]
        for op in job_ops:
            layers[rng.randint(0, L-1)].append(op)
        for target in (0, L-1):
            if not layers[target]:
                for l in range(L):
                    if layers[l]:
                        layers[target].append(layers[l].pop()); break
        for l in range(L-1):
            if not layers[l+1]: continue
            for a in layers[l]:
                succs = rng.sample(layers[l+1], k=min(len(layers[l+1]), rng.randint(1, min(3, len(layers[l+1])))))
                for b in succs: PredEdges.append([a,b])

    # JSONize calendars/quals
    A_M_json = {m: list(A_M[i].astype(int)) for i,m in enumerate(Machines)}
    A_W_frac = {w: list(A_W[i].astype(float)) for i,w in enumerate(Workers)}
    Q_json = {w: {s: int(Q[wi, s_idx[s]]) for s in Skills} for wi, w in enumerate(Workers)}

    # Due dates: rough CP + slack
    def op_min_time(op):
        mins = [D[op]["Setup"][m] + D[op]["Run"][m] + D[op]["Clean"][m] for m in Machines if E[op][m]==1]
        return min(mins) if mins else 0
    Due = {}; T_len = len(T)
    for j in Jobs:
        job_ops = [op for op in Ops if JobOf[op]==j]
        cp = sum(op_min_time(op) for op in job_ops)
        slack = int(round(0.20 * cp))
        Due[j] = min(T_len, cp + slack + random.randint(0, bpd))

    payload = {
        "Machines": Machines, "Workers": Workers, "Skills": Skills, "Phases": ["Setup","Run","Clean"],
        "T": list(T.astype(int)), "Ops": Ops, "JobOf": JobOf, "Jobs": Jobs, "Due": Due,
        "E": E, "D": D, "Need": Need, "Q": Q_json, "A_M": A_M_json, "A_W_frac": A_W_frac,
        "PredEdges": PredEdges, "CAP": ncfg.cap
    }
    return payload, ax

# =============== matrices ===============
def build_matrices(data: Dict):
    Ops=data["Ops"]; Machines=data["Machines"]; Workers=data["Workers"]; Skills=data["Skills"]; Phases=data["Phases"]
    T=np.array(data["T"],dtype=np.int32); CAP=int(data["CAP"])
    nO,nM,nW,nS,nT=len(Ops),len(Machines),len(Workers),len(Skills),len(T)
    idx={"op":{op:i for i,op in enumerate(Ops)}, "m":{m:i for i,m in enumerate(Machines)}, "w":{w:i for i,w in enumerate(Workers)},
         "s":{s:i for i,s in enumerate(Skills)}, "ph":{p:i for i,p in enumerate(Phases)}, "job":{j:i for i,j in enumerate(data["Jobs"])}}
    A_M=np.array([data["A_M"][m] for m in Machines], dtype=bool)  # (M,T)
    A_W_frac=np.array([data["A_W_frac"][w] for w in Workers], dtype=float)  # (W,T)
    C_W=np.rint(A_W_frac * CAP).astype(np.int16)  # (W,T)
    Q=np.array([[data["Q"][w][s] for s in Skills] for w in Workers], dtype=bool)  # (W,S)
    C_S=Q.T.astype(np.int16) @ C_W  # (S,T)
    E=np.array([[data["E"][op][m] for m in Machines] for op in Ops], dtype=bool)  # (O,M)
    D=np.zeros((nO, len(Phases), nM), dtype=np.int16); NeedKernels={}; total_len={}
    for oi, op in enumerate(Ops):
        for mi, m in enumerate(Machines):
            if not E[oi, mi]: continue
            offset=0; tot=0
            for pi, ph in enumerate(Phases):
                L=int(data["D"][op][ph].get(m, 0)); D[oi, pi, mi]=L
                if L>0:
                    for s, need in data["Need"][op][ph].items():
                        demand=int(math.ceil(float(need) * CAP))
                        si=idx["s"].get(s, None)
                        if si is not None and demand>0:
                            NeedKernels.setdefault((oi,mi), []).append((si, offset, L, demand))
                    offset+=L; tot+=L
            total_len[(oi,mi)]=tot
    preds=[(idx["op"][a], idx["op"][b]) for a,b in data["PredEdges"]]
    JobOf=np.array([idx["job"][data["JobOf"][op]] for op in Ops], dtype=np.int32)
    Due=np.array([data["Due"][j] for j in data["Jobs"]], dtype=np.int32)
    return {"idx":idx,"A_M":A_M,"C_W":C_W,"Q":Q,"C_S":C_S,"E":E,"D":D,"NeedKernels":NeedKernels,"total_len":total_len,
            "preds":preds,"JobOf":JobOf,"Due":Due,
            "n":{"O":nO,"M":nM,"W":nW,"S":nS,"T":nT,"CAP":CAP},
            "meta":{"Ops":Ops,"Machines":Machines,"Workers":Workers,"Skills":Skills,"Phases":Phases,"T":T}}

# =============== decoder ===============
@dataclass
class DecodeMetrics: total_ns:int=0; feas_ns:int=0; assign_ns:int=0; ops_scheduled:int=0; ops_failed:int=0; tries:int=0

def decode_schedule(mats: Dict, op_order: np.ndarray, machine_choice: np.ndarray, hard_deadlines: bool=False) -> Dict:
    t0=now_ns()
    A_M=mats["A_M"].copy(); C_S_base=mats["C_S"].copy(); E=mats["E"]; NeedK=mats["NeedKernels"]; total_len=mats["total_len"]; preds=mats["preds"]
    nO, nT, CAP = mats["n"]["O"], mats["n"]["T"], mats["n"]["CAP"]
    M_busy=np.zeros_like(A_M, dtype=bool); S_used=np.zeros_like(C_S_base, dtype=np.int16); W_busy=np.zeros((mats["n"]["W"], nT), dtype=bool)
    Q=mats["Q"]; C_W=mats["C_W"]
    pred_list=[[] for _ in range(nO)]
    for a,b in preds: pred_list[b].append(a)
    start=np.full(nO, -1, dtype=np.int32); finish=np.full(nO, -1, dtype=np.int32); chosen_m=np.full(nO, -1, dtype=np.int16)
    assigned_workers={}; dm=DecodeMetrics()

    def worker_assign(oi, mi, t_start)->bool:
        nonlocal S_used, W_busy, assigned_workers
        segments=NeedK.get((oi,mi), []);
        if not segments: return True
        t_end=t_start + total_len[(oi,mi)]
        req={}
        for (si, off, Ls, demand) in segments:
            for t in range(t_start+off, t_start+off+Ls):
                req.setdefault(si, np.zeros((t_end - t_start,), dtype=np.int16))
                req[si][t - t_start] += demand
        op_assign={}
        for local_t in range(t_end - t_start):
            abs_t=t_start + local_t
            skills_here=[si for si,arr in req.items() if arr[local_t] > 0]
            if not skills_here: continue
            counts=[]
            for si in skills_here:
                cand=np.where(np.logical_and(Q[:,si], np.logical_and(C_W[:,abs_t]>0, np.logical_not(W_busy[:,abs_t]))))[0]
                counts.append((si, cand.size))
            skills_sorted=[si for si,_ in sorted(counts, key=lambda x:x[1])]
            for si in skills_sorted:
                need_units=req[si][local_t];
                if need_units<=0: continue
                K=int(math.ceil(need_units / CAP))
                cand=np.where(np.logical_and(Q[:,si], np.logical_and(C_W[:,abs_t]>0, np.logical_not(W_busy[:,abs_t]))))[0]
                if cand.size < K: return False
                chosen=cand[:K]; W_busy[chosen,abs_t]=True; S_used[si,abs_t]+=K*CAP; op_assign.setdefault(abs_t, [])
                for w in chosen: op_assign[abs_t].append((int(w), int(si)))
        assigned_workers[oi]=op_assign; return True

    def earliest_pred_finish(oi):
        preds_here = pred_list[oi]
        if not preds_here: return 0
        valids = [finish[p] for p in preds_here if finish[p] >= 0]
        return int(max(valids)) if valids else 0

    for oi in op_order:
        dm.ops_scheduled += 1; dm.tries += 1
        mi_req=machine_choice[oi]; elig=np.where(E[oi])[0]
        cand_machines=list(elig) if mi_req<0 or mi_req not in elig else [mi_req]
        if not cand_machines: dm.ops_failed += 1; continue
        est=earliest_pred_finish(oi); best_t=None; best_mi=None
        feas_start=now_ns()
        for mi in cand_machines:
            tot=total_len.get((oi,mi), 0);
            if tot<=0: continue
            free_m=np.logical_and(A_M[mi], np.logical_not(M_busy[mi]))
            mach_ok=sliding_all_true(free_m, tot)
            if mach_ok.size==0: continue
            Rem=C_S_base - S_used; feas_vec=mach_ok.copy()
            for (si, off, Ls, demand) in NeedK.get((oi,mi), []):
                ok=sliding_all_true(Rem[si] >= demand, Ls)
                if ok.size==0: feas_vec[:]=False; break
                aligned=np.zeros_like(feas_vec, dtype=bool); max_t=min(feas_vec.size, ok.size - off)
                if max_t>0: aligned[:max_t]=ok[off:off+max_t]
                feas_vec=np.logical_and(feas_vec, aligned)
                if not feas_vec.any(): break
            if est>0 and est<feas_vec.size: feas_vec[:est]=False
            if feas_vec.any():
                t_candidate=int(np.argmax(feas_vec))
                if (best_t is None) or (t_candidate < best_t):
                    best_t=t_candidate; best_mi=mi
        dm.feas_ns += now_ns() - feas_start
        if best_t is None: dm.ops_failed += 1; continue
        assign_start=now_ns(); ok=worker_assign(oi, best_mi, best_t); dm.assign_ns += now_ns() - assign_start
        if not ok: dm.ops_failed += 1; continue
        tot=total_len[(oi, best_mi)]
        M_busy[best_mi, best_t:best_t+tot]=True; chosen_m[oi]=best_mi; start[oi]=best_t; finish[oi]=best_t+tot

    res={"feasible": bool(dm.ops_failed == 0) if hard_deadlines else True, "start":start, "finish":finish, "machine":chosen_m,
         "M_busy":M_busy, "W_busy":W_busy, "S_used":S_used, "assigned":assigned_workers, "metrics":dm}
    res["metrics"].total_ns = now_ns() - t0; return res

# =============== genome helpers & GA ===============
def default_genome(mats: Dict, strategy="min_total_len"):
    E=mats["E"]; total_len=mats["total_len"]; nO=mats["n"]["O"]
    op_order=np.arange(nO, dtype=np.int32); machine_choice=-np.ones(nO, dtype=np.int16)
    for oi in range(nO):
        elig=np.where(E[oi])[0]
        if elig.size==0: continue
        if strategy=="min_total_len":
            best=None; best_val=None
            for mi in elig:
                tot=total_len.get((oi,mi), 10**9)
                if best is None or tot<best_val: best, best_val = mi, tot
            machine_choice[oi]=best
        else:
            machine_choice[oi]=elig[0]
    return op_order, machine_choice

def random_genome(mats: Dict, rng: np.random.Generator):
    nO=mats["n"]["O"]
    op_order=np.arange(nO, dtype=np.int32); rng.shuffle(op_order)
    E=mats["E"]
    machine_choice=-np.ones(nO, dtype=np.int16)
    for oi in range(nO):
        elig=np.where(E[oi])[0]
        if elig.size>0: machine_choice[oi]=int(rng.choice(elig))
    return op_order, machine_choice

def evaluate_objective(mats: Dict, schedule: Dict, alpha: float = 0.03):
    Due=mats["Due"]; JobOf=mats["JobOf"]; finish=schedule["finish"]; nJ=Due.shape[0]
    job_finish=np.zeros(nJ, dtype=np.int32)
    for j in range(nJ):
        ops=np.where(JobOf==j)[0]
        if ops.size>0: job_finish[j]=np.max(finish[ops])
    tardiness=int(np.maximum(0, job_finish - Due).sum())
    makespan=int(finish.max()) if finish.size>0 else 0
    score=float(tardiness + alpha * makespan)
    return {"tardiness":tardiness,"makespan":makespan,"score":score,"job_finish":job_finish}

# precedence-aware repair (topological projection)
def repair_order_topo(mats, order):
    preds = mats["preds"]; nO = mats["n"]["O"]
    pos = np.empty(nO, dtype=np.int32)
    for rank, oi in enumerate(order):
        pos[int(oi)] = rank
    indeg = np.zeros(nO, dtype=np.int32); succ = {}
    for a,b in preds:
        indeg[b] += 1; succ.setdefault(a, []).append(b)
    import heapq
    heap = []
    for oi in range(nO):
        if indeg[oi] == 0:
            heapq.heappush(heap, (int(pos[oi]), oi))
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in succ.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (int(pos[v]), v))
    if len(out) != nO:
        return order.copy()
    return np.array(out, dtype=np.int32)

@dataclass
class GASettings:
    pop:int=150; gens:int=40; elite:int=10; tour_k:int=4
    swap_rate:float=0.30; flip_rate:float=0.20; alpha:float=0.03; seed:int=123; patience:int=8

def tournament_select(scores, k, rng):
    n = len(scores)
    k = max(1, min(int(k), n))
    idxs = rng.choice(n, size=k, replace=False)
    best_i = int(idxs[0]); best_s = scores[best_i]
    for i in idxs[1:]:
        i = int(i)
        if scores[i] < best_s:
            best_s = scores[i]; best_i = i
    return int(best_i)

def ox_crossover(order_a: np.ndarray, order_b: np.ndarray, rng: np.random.Generator):
    n = order_a.size
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = -np.ones(n, dtype=np.int32)
    child[i:j+1] = order_a[i:j+1]
    used = set(child[i:j+1].tolist())
    pos = (j+1) % n
    for k in range(n):
        gene = order_b[(j+1+k) % n]
        if gene not in used:
            child[pos] = gene
            used.add(gene)
            pos = (pos + 1) % n
    return child

def one_point_crossover(mc_a: np.ndarray, mc_b: np.ndarray, rng: np.random.Generator):
    n = mc_a.size
    cut = int(rng.integers(0, n)) if n>0 else 0
    child = mc_a.copy()
    if n>0: child[cut:] = mc_b[cut:]
    return child

def mutate_genome(mats, op_order, machine_choice, rng, swap_rate=0.3, flip_rate=0.2):
    n=op_order.size
    if rng.random()<swap_rate and n>=2:
        i,j=rng.choice(n, size=2, replace=False); op_order[i],op_order[j]=op_order[j],op_order[i]
    if rng.random()<flip_rate:
        oi=int(rng.integers(0,n)); elig=np.where(mats["E"][oi])[0]
        if elig.size>0:
            if rng.random()<0.25: machine_choice[oi]=-1
            else: machine_choice[oi]=int(rng.choice(elig))
    op_order = repair_order_topo(mats, op_order)
    return op_order, machine_choice

def build_initial_population(mats, pop, rng):
    pop_genomes = []
    base_oo, base_mc = default_genome(mats, strategy="min_total_len")
    base_oo = repair_order_topo(mats, base_oo)
    pop_genomes.append((base_oo.copy(), base_mc.copy()))
    if pop > 1:
        oo2, mc2 = random_genome(mats, rng)
        oo2 = repair_order_topo(mats, oo2)
        pop_genomes.append((oo2, mc2))
    if pop > 2:
        JobOf = mats["JobOf"]; Due = mats["Due"]
        order = np.argsort(Due[JobOf])
        order = repair_order_topo(mats, order.astype(np.int32))
        mc = default_genome(mats, strategy="min_total_len")[1]
        pop_genomes.append((order, mc))
    while len(pop_genomes) < pop:
        oo, mc = random_genome(mats, rng)
        oo = repair_order_topo(mats, oo)
        pop_genomes.append((oo, mc))
    return pop_genomes

def run_ga(mats, settings: GASettings, hard_deadlines=False, progress_cb=None):
    rng=np.random.default_rng(settings.seed)
    pop_list = build_initial_population(mats, settings.pop, rng)
    # evaluate initial pop
    t0=time.perf_counter_ns(); scores=[]; dec_scheds=[]; evs=[]
    for (oo, mc) in pop_list:
        sc=decode_schedule(mats, oo.copy(), mc.copy(), hard_deadlines=hard_deadlines)
        ev=evaluate_objective(mats, sc, alpha=settings.alpha)
        scores.append(ev["score"]); dec_scheds.append(sc); evs.append(ev)
    t1=time.perf_counter_ns()
    history={"best":[], "avg":[], "time_ns":[t1-t0]}
    best_i=int(np.argmin(scores)); best={"score":float(scores[best_i]), "genome":pop_list[best_i], "sched":dec_scheds[best_i], "eval":evs[best_i]}
    no_improve=0
    for g in range(settings.gens):
        gen_start=time.perf_counter_ns()
        eff_elite = max(0, min(int(settings.elite), len(scores)-1))
        elite_idx=np.argsort(scores)[:eff_elite]
        new_pop=[(pop_list[i][0].copy(), pop_list[i][1].copy()) for i in elite_idx]
        while len(new_pop)<settings.pop:
            pa=pop_list[tournament_select(scores, settings.tour_k, rng)]
            pb=pop_list[tournament_select(scores, settings.tour_k, rng)]
            child_order=ox_crossover(pa[0], pb[0], rng); child_mc=one_point_crossover(pa[1], pb[1], rng)
            child_order, child_mc=mutate_genome(mats, child_order, child_mc, rng, settings.swap_rate, settings.flip_rate)
            new_pop.append((child_order, child_mc))
        pop_list=new_pop; scores=[]; dec_scheds=[]; evs=[]
        for (oo,mc) in pop_list:
            sc=decode_schedule(mats, oo.copy(), mc.copy(), hard_deadlines=hard_deadlines); ev=evaluate_objective(mats, sc, alpha=settings.alpha)
            scores.append(ev["score"]); dec_scheds.append(sc); evs.append(ev)
        gen_end=time.perf_counter_ns(); history["time_ns"].append(gen_end - gen_start)
        avg=float(np.mean(scores)); best_idx=int(np.argmin(scores)); best_score=float(scores[best_idx]); history["avg"].append(avg); history["best"].append(best_score)
        if best_score + 1e-9 < best["score"]:
            best={"score":best_score, "genome":pop_list[best_idx], "sched":dec_scheds[best_idx], "eval":evs[best_idx]}; no_improve=0
        else: no_improve+=1
        if progress_cb: progress_cb(g, best_score, avg)
        if settings.patience and no_improve>=settings.patience: break
    total_ns=sum(history["time_ns"]) + history["time_ns"][0]
    return best, history, {"total_ns": total_ns, "gens_done": len(history["best"])}

# =============== CP-SAT polish ===============
def cpsat_polish_machine_window(mats, base_sched, machine_index: int, window_start: int, window_len: int, time_limit_s: float = 3.0):
    if not HAS_CPSAT: return {}
    start=base_sched["start"]; finish=base_sched["finish"]; chosen_m=base_sched["machine"]
    D=mats["D"]; preds=mats["preds"]; Due=mats["Due"]; JobOf=mats["JobOf"]; nO=mats["n"]["O"]
    tot_by_m={}
    for oi in range(nO):
        if chosen_m[oi]==machine_index:
            tot=int(np.sum(D[oi, :, machine_index])); tot_by_m[oi]=tot
    w_end=window_start + window_len
    cand_ops=[oi for oi in tot_by_m.keys() if not (finish[oi] <= window_start or start[oi] >= w_end)]
    if not cand_ops: return {}
    model=cp_model.CpModel(); horizon=window_start + window_len
    s_var={}; e_var={}; ivar={}
    for oi in cand_ops:
        dur=tot_by_m[oi]; s=model.NewIntVar(window_start, horizon, f"s_{oi}"); e=model.NewIntVar(window_start, horizon+dur, f"e_{oi}")
        iv=model.NewIntervalVar(s, dur, e, f"iv_{oi}"); s_var[oi]=s; e_var[oi]=e; ivar[oi]=iv
    model.AddNoOverlap([ivar[oi] for oi in cand_ops])
    pred_map={}
    for a,b in preds: pred_map.setdefault(b, []).append(a)
    for oi in cand_ops:
        for p in pred_map.get(oi, []):
            if p in cand_ops: model.Add(s_var[oi] >= e_var[p])
    # Stability + due soft penalties
    for oi in cand_ops:
        orig = int(start[oi])
        delta = model.NewIntVar(-horizon, horizon, f"delta_{oi}")
        model.Add(delta == s_var[oi] - orig)
        absd = model.NewIntVar(0, horizon, f"abs_{oi}")
        model.AddAbsEquality(absd, delta); model.Minimize(absd)
        j=JobOf[oi]; due=int(Due[j]); late=model.NewIntVar(0, horizon, f"late_{oi}"); model.Add(late >= e_var[oi] - due); model.Minimize(late)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=float(time_limit_s); solver.parameters.num_search_workers=8
    res=solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return {}
    return {oi:int(solver.Value(s_var[oi])) for oi in cand_ops}

# =============== diagnostics & visuals ===============
def machine_utilization(mats, sched):
    A_M=mats["A_M"]; M_busy=sched["M_busy"]
    util=[]
    for mi in range(mats["n"]["M"]):
        avail=A_M[mi].sum(); busy=M_busy[mi].sum()
        util.append((mi, busy, avail, (busy/avail if avail>0 else 0.0)))
    util.sort(key=lambda x:x[3], reverse=True)
    return util

def unscheduled_ops(sched):
    start=sched["start"]; finish=sched["finish"]
    idx=np.where((start<0) | (finish<0))[0]
    return idx

def plot_machine_gantt(mats, sched, machine_index:int, title:str="", max_ops:int=200, window=None):
    Ops=mats["meta"]["Ops"]; Machines=mats["meta"]["Machines"]
    start=sched["start"]; finish=sched["finish"]; chosen=sched["machine"]
    rows=[]
    for oi in range(mats["n"]["O"]):
        if chosen[oi]==machine_index and start[oi]>=0 and finish[oi]>=0:
            s=int(start[oi]); f=int(finish[oi])
            if window:
                t0,t1=window
                if f<=t0 or s>=t1: continue
                s=max(s,t0); f=min(f,t1)
            rows.append((Ops[oi], s, f-s))
    rows=sorted(rows, key=lambda r:r[1])[:max_ops]
    if not rows:
        fig, ax = plt.subplots(); ax.set_title(f"{Machines[machine_index]} — no ops in view")
        return fig
    fig, ax = plt.subplots(figsize=(10, min(6, 0.25*len(rows)+1)))
    y_ticks=[]
    for y,(op,s,dur) in enumerate(rows):
        ax.barh(y, dur, left=s)
        y_ticks.append(op)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(y_ticks, fontsize=8)
    ax.set_xlabel("Bucket"); ax.set_title(title or f"Gantt — {Machines[machine_index]}")
    return fig

# =============== UI ===============
st.title("Matrix Scheduler — GA + CP-SAT + Optuna (Snapshots)")
st.caption("Fast matrix decoder + GA for global search + CP-SAT to polish bottlenecks. With diagnostics, Gantt visuals, and robust snapshot downloads.")

with st.expander("Workflow"):
    st.markdown("""
1) Generate a dataset (sidebar).  
2) GA Optimize or Optuna Tuner.  
3) Save **Snapshot** ⇒ persists best genome & schedule across reruns.  
4) Diagnostics (bottlenecks & unscheduled ops).  
5) CP-SAT polish ⇒ then **Save Snapshot (polished)**.
""")

# Sidebar dataset controls
st.sidebar.header("Dataset")
seed=st.sidebar.number_input("Seed", value=2025, step=1)
jobs=st.sidebar.number_input("Jobs", 10, 200, 50, 5)
ops_min=st.sidebar.number_input("Ops per job (min)", 3, 30, 5, 1)
ops_max=st.sidebar.number_input("Ops per job (max)", ops_min, 40, 20, 1)
machines=st.sidebar.number_input("Machines", 4, 30, 12, 1)
workers=st.sidebar.number_input("Workers", 10, 100, 25, 1)
evening_frac=st.sidebar.slider("Evening shift fraction", 0.0, 1.0, 0.5, 0.05)
cap=st.sidebar.number_input("CAP (units per worker)", 5, 50, 10, 1)
maint_len_min=st.sidebar.number_input("Maintenance min (buckets)", 1, 50, 3, 1)
maint_len_max=st.sidebar.number_input("Maintenance max (buckets)", 1, 50, 8, 1)

if st.sidebar.button("Generate dataset"):
    hcfg=HorizonCfg(); scfg=ScaleCfg(machines=int(machines), workers=int(workers), jobs=int(jobs), ops_min=int(ops_min), ops_max=int(ops_max))
    ncfg=NeedsCfg(cap=int(cap)); mcfg=MaintCfg(blocks_per_machine=1, block_len_min=int(maint_len_min), block_len_max=int(maint_len_max))
    shcfg=ShiftCfg(assign_evening_fraction=float(evening_frac))
    data, ax = generate_stress_dataset(int(seed), hcfg, scfg, mcfg, ncfg, shcfg)
    st.session_state["data"]=data; st.session_state["ax"]=ax
    st.sidebar.success("Dataset generated.")

# Load or default
if st.session_state.get("data") is None:
    hcfg=HorizonCfg(); scfg=ScaleCfg(); ncfg=NeedsCfg(); mcfg=MaintCfg(); shcfg=ShiftCfg()
    data, ax = generate_stress_dataset(2025, hcfg, scfg, mcfg, ncfg, shcfg)
    st.session_state["data"]=data; st.session_state["ax"]=ax

data=st.session_state.get("data")

# Build matrices
t_build0=now_ns(); mats=build_matrices(data); t_build=now_ns()-t_build0
st.subheader("Scale summary")
nO,nM,nW,nS,nT=mats["n"]["O"],mats["n"]["M"],mats["n"]["W"],mats["n"]["S"],mats["n"]["T"]
c1,c2,c3,c4=st.columns(4); c1.metric("Ops", nO); c2.metric("Machines", nM); c3.metric("Workers", nW); c4.metric("Buckets", nT)
st.caption(f"Matrix build time: {ns_to_ms(t_build):.1f} ms")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single decode", "Random benchmark", "GA optimize", "Diagnostics", "CP-SAT polish"])

with tab1:
    st.caption("Decode a heuristic genome (min total duration per op).")
    op_order, machine_choice = default_genome(mats, strategy="min_total_len")
    hard_deadlines = st.checkbox("Hard deadlines (reject if any due missed)", value=False, key="sd_hd")
    t0=now_ns(); sched=decode_schedule(mats, op_order.copy(), machine_choice.copy(), hard_deadlines=hard_deadlines); t1=now_ns()
    eva=evaluate_objective(mats, sched)
    a,b,c,d=st.columns(4)
    a.metric("Decode time", f"{ns_to_ms(t1-t0):.1f} ms"); b.metric("Feasible", str(sched["feasible"]))
    c.metric("Tardiness", eva["tardiness"]); d.metric("Makespan", eva["makespan"])
    st.write(f"Feasibility: {ns_to_ms(sched['metrics'].feas_ns):.1f} ms · Identity: {ns_to_ms(sched['metrics'].assign_ns):.1f} ms")
    # Preview
    Ops=mats["meta"]["Ops"]; Machines=mats["meta"]["Machines"]
    df = pd.DataFrame({"Op": Ops[:50],
                       "Machine": [Machines[m] if m>=0 else "" for m in sched["machine"][:50]],
                       "Start": sched["start"][:50], "Finish": sched["finish"][:50]})
    st.dataframe(df)

with tab2:
    st.caption("Measure decoder throughput with random genomes.")
    bench_n=st.number_input("Random decodes", 10, 2000, 200, 50); seed_b=st.number_input("Benchmark seed", value=7, step=1)
    if st.button("Run benchmark"):
        rng=np.random.default_rng(int(seed_b)); times=[]; feas_ms=[]; assign_ms=[]
        for i in range(int(bench_n)):
            oo,mc=random_genome(mats, rng); t0=now_ns(); sc=decode_schedule(mats, oo, mc, hard_deadlines=False); t1=now_ns()
            times.append(ns_to_ms(t1-t0)); feas_ms.append(ns_to_ms(sc["metrics"].feas_ns)); assign_ms.append(ns_to_ms(sc["metrics"].assign_ns))
        times=np.array(times); feas_ms=np.array(feas_ms); assign_ms=np.array(assign_ms)
        x1,x2,x3,x4,x5=st.columns(5)
        x1.metric("Decodes", len(times)); x2.metric("Avg (ms)", f"{times.mean():.1f}"); x3.metric("Median (ms)", f"{np.median(times):.1f}")
        x4.metric("p95 (ms)", f"{np.percentile(times,95):.1f}"); x5.metric("Decodes/sec", f"{1000.0 / max(1e-6, times.mean()):.1f}")
        fig, axp=plt.subplots(); axp.hist(times, bins=30); axp.set_title("Decode time (ms)"); st.pyplot(fig)

with tab3:
    st.caption("Full GA optimization (with precedence-aware repair).")
    subtab = st.tabs(["Run GA", "Optuna tuner"])
    with subtab[0]:
        col = st.columns(2)
        with col[0]:
            pop=st.number_input("Population", 10, 1000, 150, 10)
            gens=st.number_input("Generations", 1, 500, 40, 1)
            elite=st.number_input("Elite", 0, int(pop)-1 if int(pop)>1 else 0, min(10, max(0,int(pop)-1)), 1)
            tour_k=st.number_input("Tournament k", 2, int(pop), min(4, int(pop)), 1)
            alpha=st.slider("Alpha (makespan weight)", 0.0, 0.2, 0.03, 0.01)
            patience=st.number_input("Early-stop patience", 0, 50, 8, 1)
            hard_deadlines_ga=st.checkbox("Hard deadlines", value=False, key="ga_hd")
        with col[1]:
            swap=st.slider("Mutation: swap rate", 0.0, 1.0, 0.30, 0.05)
            flip=st.slider("Mutation: machine flip rate", 0.0, 1.0, 0.20, 0.05)
            seed_ga=st.number_input("GA seed", value=123, step=1)
            run_ga_btn=st.button("Run GA")

        if run_ga_btn:
            settings=GASettings(pop=int(pop), gens=int(gens), elite=int(elite), tour_k=int(tour_k),
                                swap_rate=float(swap), flip_rate=float(flip), alpha=float(alpha), seed=int(seed_ga), patience=int(patience))
            progress=st.progress(0); status=st.empty()
            def cb(gen_idx, best_score, avg_score):
                pct=int(100*(gen_idx+1)/max(1,settings.gens)); progress.progress(min(pct,100)); status.text(f"Gen {gen_idx+1}/{settings.gens} — best {best_score:.2f} · avg {avg_score:.2f}")
            t0=time.perf_counter_ns(); best, hist, stats=run_ga(mats, settings, hard_deadlines=hard_deadlines_ga, progress_cb=cb); t1=time.perf_counter_ns()
            total_ms=(t1-t0)/1e6
            genomes_evaluated=settings.pop * max(1, len(hist["best"]) + 1)

            Ops=mats["meta"]["Ops"]; Machines=mats["meta"]["Machines"]; best_sched=best["sched"]
            df_best=pd.DataFrame({"Op":Ops,"Machine":[Machines[m] if m>=0 else "" for m in best_sched["machine"]],
                                  "Start":best_sched["start"],"Finish":best_sched["finish"]}).sort_values(by=["Start"])

            # Persist results in memory
            best_oo,best_mc=best["genome"]
            genome_json={"op_order":best_oo.astype(int).tolist(),"machine_choice":best_mc.astype(int).tolist(),"alpha":settings.alpha}
            genome_bytes = json.dumps(genome_json).encode("utf-8")
            csv_bytes = df_best.to_csv(index=False).encode("utf-8")
            st.session_state["ga_best_present"] = True
            st.session_state["ga_best_total_ms"] = total_ms
            st.session_state["ga_best_score"] = float(best["score"])
            st.session_state["ga_best_hist"] = hist
            st.session_state["ga_best_genomes_evaluated"] = genomes_evaluated
            st.session_state["ga_best_genome_json"] = genome_bytes
            st.session_state["ga_best_sched_csv"] = csv_bytes
            st.session_state["ga_best_sched_obj"] = best_sched
            st.session_state["ga_best_eval"] = best["eval"]

        has_ga = st.session_state.get("ga_best_present") and st.session_state.get("ga_best_genome_json") and st.session_state.get("ga_best_sched_csv")
        if has_ga:
            hist = st.session_state.get("ga_best_hist")
            total_ms = st.session_state.get("ga_best_total_ms")
            best_score = st.session_state.get("ga_best_score")
            genomes_evaluated = st.session_state.get("ga_best_genomes_evaluated")
            st.success(f"GA completed in {total_ms:.1f} ms • gens: {len(hist['best'])} • best score: {best_score:.2f}")
            st.write(f"≈ Genomes evaluated: {genomes_evaluated} → ~{1000.0*genomes_evaluated/max(1.0,total_ms):.1f} genomes/sec")

            fig, axp = plt.subplots(); axp.plot(hist["best"], label="best"); axp.plot(hist["avg"], label="avg")
            axp.set_xlabel("generation"); axp.set_ylabel("score"); axp.legend(); st.pyplot(fig)

            Ops=mats["meta"]["Ops"]; Machines=mats["meta"]["Machines"]; best_sched=st.session_state.get("ga_best_sched_obj")
            df_best=pd.DataFrame({"Op":Ops,"Machine":[Machines[m] if m>=0 else "" for m in best_sched["machine"]],
                                  "Start":best_sched["start"],"Finish":best_sched["finish"]}).sort_values(by=["Start"])
            st.dataframe(df_best.head(50))

            cA, cB = st.columns(2)
            with cA:
                st.download_button("Download best genome (JSON)", data=st.session_state.get("ga_best_genome_json"), file_name="best_genome.json", mime="application/json", key="dl_genome_ga")
                st.download_button("Download best schedule (CSV)", data=st.session_state.get("ga_best_sched_csv"), file_name="best_schedule.csv", mime="text/csv", key="dl_sched_ga")
            with cB:
                snap_label = st.text_input("Snapshot label", value=f"GA_best_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                save_snapshot_ga = st.button("Save Snapshot", key="save_snapshot_ga")
                if save_snapshot_ga:
                    st.session_state.get("snapshots").append({"label": snap_label, "genome_bytes": st.session_state.get("ga_best_genome_json"), "csv_bytes": st.session_state.get("ga_best_sched_csv")})
                    st.success(f"Saved snapshot: {snap_label}")
        else:
            st.caption("Run GA to generate results.")

    with subtab[1]:
        st.caption("Use Optuna to discover good GA hyperparameters for this dataset size.")
        if not HAS_OPTUNA:
            st.error("optuna is not installed. Run: pip install optuna")
        else:
            trials = st.number_input("Trials", 5, 200, 20, 1)
            seed_base = st.number_input("Base seed", 1, 999999, 2025, 1)
            run_optuna = st.button("Run Optuna Tuner")
            if run_optuna:
                def objective(trial):
                    pop   = trial.suggest_int("pop", 50, 300, step=10)
                    gens  = trial.suggest_int("gens", 10, 80, step=5)
                    elite = trial.suggest_int("elite", 0, max(0, pop//5))
                    tourk = trial.suggest_int("tour_k", 2, min(12,pop))
                    swap  = trial.suggest_float("swap_rate", 0.1, 0.6)
                    flip  = trial.suggest_float("flip_rate", 0.05, 0.5)
                    alpha = trial.suggest_float("alpha", 0.0, 0.1)
                    patience = trial.suggest_int("patience", 4, 20)
                    seed_ga = seed_base + trial.number
                    settings=GASettings(pop=pop, gens=gens, elite=elite, tour_k=tourk, swap_rate=swap, flip_rate=flip, alpha=alpha, seed=seed_ga, patience=patience)
                    best, hist, stats = run_ga(mats, settings, hard_deadlines=False, progress_cb=None)
                    return best["score"]
                study = optuna.create_study(direction="minimize")
                with st.spinner("Tuning GA hyperparameters..."):
                    study.optimize(objective, n_trials=int(trials))
                st.success("Optuna finished.")
                st.write("**Best params:**")
                st.json(study.best_params)
                st.write(f"Best score: {study.best_value:.2f}")
                st.session_state["optuna_best_params"] = study.best_params

with tab4:
    st.caption("Bottlenecks & unscheduled operations (diagnostics).")
    base_choice = st.selectbox("Analyze which schedule", ["Heuristic decode (fresh)", "GA best (if available)"])
    if base_choice == "GA best (if available)" and st.session_state.get("ga_best_present", False):
        sched = st.session_state.get("ga_best_sched_obj")
        eva = st.session_state.get("ga_best_eval")
    else:
        op_order, machine_choice = default_genome(mats, strategy="min_total_len")
        sched = decode_schedule(mats, op_order.copy(), machine_choice.copy(), hard_deadlines=False)
        eva = evaluate_objective(mats, sched)
    colA,colB,colC,colD = st.columns(4)
    colA.metric("Tardiness", eva["tardiness"]); colB.metric("Makespan", eva["makespan"])
    uns = unscheduled_ops(sched); colC.metric("Unscheduled ops", int(len(uns)))
    colD.metric("Decode time (ms)", f"{ns_to_ms(sched['metrics'].total_ns):.1f}")
    st.markdown("### Machine utilization (top 10)")
    util = machine_utilization(mats, sched)[:10]
    util_df = pd.DataFrame([{"Machine": mats["meta"]["Machines"][mi], "Busy": busy, "Avail": avail, "Util%": round(100*u,1)} for (mi,busy,avail,u) in util])
    st.dataframe(util_df)
    st.markdown("### Gantt-like view (choose machine)")
    Machines=mats["meta"]["Machines"]
    mi_name=st.selectbox("Machine", Machines, index=0, key="diag_mi")
    mi=Machines.index(mi_name)
    window_start=st.number_input("Window start (bucket)", 0, mats["n"]["T"]-1, 0, 10, key="diag_ws")
    window_len=st.number_input("Window length (buckets)", 10, mats["n"]["T"], 300, 10, key="diag_wl")
    fig = plot_machine_gantt(mats, sched, mi, title=f"Gantt — {mi_name}", window=(int(window_start), int(window_start+window_len)))
    st.pyplot(fig)
    if len(uns)>0:
        st.markdown("### Unscheduled operations")
        Ops=mats["meta"]["Ops"]
        st.dataframe(pd.DataFrame({"Op":[Ops[i] for i in uns]}))

with tab5:
    st.caption("CP-SAT local polish on a single machine and time window. Start from GA best or heuristic.")
    if not HAS_CPSAT:
        st.error("ortools is not installed. Run: pip install ortools")
    else:
        use_ga_best = st.checkbox("Use GA best as base (if available)", value=True)
        if use_ga_best and st.session_state.get("ga_best_present", False):
            base_sched = st.session_state.get("ga_best_sched_obj")
            base_eval = st.session_state.get("ga_best_eval")
        else:
            base_oo, base_mc = default_genome(mats, strategy="min_total_len")
            base_sched = decode_schedule(mats, base_oo.copy(), base_mc.copy(), hard_deadlines=False)
            base_eval = evaluate_objective(mats, base_sched)

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Base tardiness", base_eval["tardiness"]); c2.metric("Base makespan", base_eval["makespan"])
        base_uns = unscheduled_ops(base_sched); c3.metric("Base unscheduled", int(len(base_uns)))
        c4.metric("Base decode (ms)", f"{ns_to_ms(base_sched['metrics'].total_ns):.1f}")

        Machines=mats["meta"]["Machines"]
        mi_name=st.selectbox("Machine to polish", Machines, index=0, key="cp_mi")
        mi=Machines.index(mi_name)
        window_start=st.number_input("Window start (bucket)", 0, mats["n"]["T"]-1, 0, 10, key="cp_ws")
        window_len=st.number_input("Window length (buckets)", 10, mats["n"]["T"], 200, 10, key="cp_wl")
        tl=st.number_input("CP-SAT time limit (seconds)", 0.1, 30.0, 3.0, 0.1, key="cp_tl")
        run_polish=st.button("Run CP-SAT polish")

        if run_polish:
            # Reset any previous CP-SAT polish results to avoid stale data
            for k in list(st.session_state.keys()):
                if k.startswith("cp_best_"):
                    st.session_state.pop(k)

            t0=now_ns();
            new_starts=cpsat_polish_machine_window(mats, base_sched, mi, int(window_start), int(window_len), float(tl));
            t1=now_ns()
            polish_ms = ns_to_ms(t1-t0)
            if not new_starts:
                st.warning("No changes (or no ops in window).")
            else:
                # Build a nudged order: polished ops by new start first, then the rest
                nO=mats["n"]["O"]; base_order=np.arange(nO, dtype=np.int32)
                polished_ops=sorted(new_starts.keys(), key=lambda oi: new_starts[oi])
                mask=np.ones_like(base_order, dtype=bool); mask[polished_ops]=False; rest=base_order[mask]
                new_order=np.array(polished_ops + rest.tolist(), dtype=np.int32)
                base_mc = -np.ones(nO, dtype=np.int16)
                t2=now_ns(); new_sched=decode_schedule(mats, new_order, base_mc, hard_deadlines=False); t3=now_ns()
                new_eval=evaluate_objective(mats, new_sched)
                decode_ms = ns_to_ms(t3-t2)

                df_new = pd.DataFrame({"Op":mats["meta"]["Ops"],
                                       "Machine":[mats['meta']['Machines'][m] if m>=0 else "" for m in new_sched["machine"]],
                                       "Start":new_sched["start"],"Finish":new_sched["finish"]}).sort_values(by=["Start"])
                genome_json = {"op_order": new_order.astype(int).tolist(), "machine_choice": (-np.ones(nO, dtype=int)).tolist(), "alpha": 0.0}
                genome_bytes = json.dumps(genome_json).encode("utf-8")
                csv_bytes = df_new.to_csv(index=False).encode("utf-8")

                # Persist polish results in session state
                st.session_state["cp_best_present"] = True
                st.session_state["cp_best_polish_ms"] = polish_ms
                st.session_state["cp_best_decode_ms"] = decode_ms
                st.session_state["cp_best_sched_obj"] = new_sched
                st.session_state["cp_best_eval"] = new_eval
                st.session_state["cp_best_sched_csv"] = csv_bytes
                st.session_state["cp_best_genome_json"] = genome_bytes

        has_cp = st.session_state.get("cp_best_present") and st.session_state.get("cp_best_genome_json") and st.session_state.get("cp_best_sched_csv")
        if has_cp:
            st.write(f"CP-SAT polish time: {st.session_state.get('cp_best_polish_ms', 0.0):.1f} ms")
            new_sched = st.session_state.get("cp_best_sched_obj")
            new_eval = st.session_state.get("cp_best_eval")

            d1,d2,d3,d4=st.columns(4)
            d1.metric("New tardiness", new_eval["tardiness"], delta=int(new_eval["tardiness"]-base_eval["tardiness"]))
            d2.metric("New makespan", new_eval["makespan"], delta=int(new_eval["makespan"]-base_eval["makespan"]))
            new_uns = unscheduled_ops(new_sched); d3.metric("New unscheduled", int(len(new_uns)), delta=int(len(new_uns)-len(base_uns)))
            d4.metric("Re-decode (ms)", f"{st.session_state.get('cp_best_decode_ms', 0.0):.1f}")

            # Before/After Gantt for the polished machine and window
            st.markdown("#### Before (base)")
            fig_before = plot_machine_gantt(mats, base_sched, mi, title=f"Before — {mi_name}", window=(int(window_start), int(window_start+window_len)))
            st.pyplot(fig_before)
            st.markdown("#### After (polished)")
            fig_after = plot_machine_gantt(mats, new_sched, mi, title=f"After — {mi_name}", window=(int(window_start), int(window_start+window_len)))
            st.pyplot(fig_after)

            # Affected ops table
            Ops=mats["meta"]["Ops"]
            changed=np.where(new_sched["start"]!=base_sched["start"])[0]
            rows=[{"Op":Ops[oi], "Old start":int(base_sched["start"][oi]), "New start":int(new_sched["start"][oi])} for oi in sorted(changed, key=lambda oi:new_sched["start"][oi])]
            if rows:
                df_polished = pd.DataFrame(rows).sort_values(by="New start")
                st.dataframe(df_polished)

            # Download buttons and snapshot controls
            cA,cB=st.columns(2)
            with cA:
                st.download_button("Download polished genome (JSON)", data=st.session_state.get("cp_best_genome_json"), file_name="polished_genome.json", mime="application/json", key="dl_genome_cp")
                st.download_button("Download polished schedule (CSV)", data=st.session_state.get("cp_best_sched_csv"), file_name="polished_schedule.csv", mime="text/csv", key="dl_sched_cp")
            with cB:
                snap_label = st.text_input("Snapshot label (polished)", value=f"CP_polished_{mi_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                save_snapshot_cp = st.button("Save Snapshot (polished)", key="save_snapshot_cp")
                if save_snapshot_cp:
                    st.session_state.get("snapshots").append({"label": snap_label, "genome_bytes": st.session_state.get("cp_best_genome_json"), "csv_bytes": st.session_state.get("cp_best_sched_csv")})
                    st.success(f"Saved snapshot: {snap_label}")
        else:
            st.caption("Run CP-SAT polish to generate results.")

# Sidebar persistent snapshots
st.sidebar.markdown("---")
st.sidebar.subheader("Snapshots")
if st.session_state.get("ga_best_present") or st.session_state.get("cp_best_present"):
    st.sidebar.caption("Current results available in main tabs.")
if st.session_state.get("snapshots"):
    for i, snap in enumerate(st.session_state.get("snapshots", [])):
        st.sidebar.write(snap["label"])
        st.sidebar.download_button(f"Genome {i+1}", data=snap["genome_bytes"], file_name=f"{snap['label']}_genome.json", mime="application/json", key=f"snap_g_{i}")
        st.sidebar.download_button(f"Schedule {i+1}", data=snap["csv_bytes"], file_name=f"{snap['label']}_schedule.csv", mime="text/csv", key=f"snap_s_{i}")
else:
    st.sidebar.caption("No snapshots yet. Save from GA or CP-SAT tabs.")
