
# Streamlit Matrix Scheduler — GA + CP-SAT Polishing (Stress Test Edition)
# (See previous cell for the full description and features.)
import json, math, time, random
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except Exception:
    HAS_CPSAT = False

st.set_page_config(page_title="Matrix Scheduler — GA + CP-SAT", layout="wide")

def now_ns(): return time.perf_counter_ns()
def ns_to_ms(ns): return ns / 1e6

def sliding_all_true(x_bool: np.ndarray, L: int) -> np.ndarray:
    T = x_bool.shape[0]
    if L <= 0 or L > T: return np.zeros(0, dtype=bool)
    x_int = x_bool.astype(np.int32)
    csum = np.empty(T + 1, dtype=np.int32); csum[0] = 0; np.cumsum(x_int, out=csum[1:])
    sums = csum[L:] - csum[:-L]
    return sums == L

@dataclass
class HorizonCfg: days:int=30; start_hour:int=8; end_hour:int=22; bucket_min:int=10
@dataclass
class ScaleCfg: machines:int=12; workers:int=25; jobs:int=50; ops_min:int=5; ops_max:int=20
@dataclass
class MaintCfg: blocks_per_machine:int=1; block_len_min:int=3; block_len_max:int=8
@dataclass
class NeedsCfg: cap:int=10; setup_choices:tuple=(0.5,1.0,1.5); run_choices:tuple=(1.0,1.5,2.0); clean_choices:tuple=(0.5,1.0)
@dataclass
class ShiftCfg: break_bucket_offset:int=3; day_start:int=8; day_end:int=16; eve_start:int=16; eve_end:int=22; assign_evening_fraction:float=0.5

def build_time_axis(hcfg: HorizonCfg):
    buckets_per_hour = 60 // hcfg.bucket_min; buckets_per_day = (hcfg.end_hour - hcfg.start_hour) * buckets_per_hour
    T = np.arange(hcfg.days * buckets_per_day, dtype=np.int32)
    return {"T": T, "buckets_per_hour": buckets_per_hour, "buckets_per_day": buckets_per_day}

def generate_stress_dataset(seed: int, hcfg: HorizonCfg, scfg: ScaleCfg, mcfg: MaintCfg, ncfg: NeedsCfg, shcfg: ShiftCfg):
    rng = random.Random(seed); ax = build_time_axis(hcfg); T = ax["T"]; bpd = ax["buckets_per_day"]; bph = ax["buckets_per_hour"]
    Machines = [f"M{i+1}" for i in range(scfg.machines)]; Workers = [f"Ada_{i+1}" for i in range(scfg.workers)]
    process_skills = ["Drilling","Milling","Turning","Cutting","Holes","Injection"]
    machine_skills = [f"{role}_{m}" for m in Machines for role in ("setup","operate","clean")]
    Skills = process_skills + machine_skills
    A_M = np.ones((len(Machines), len(T)), dtype=bool)
    for mi, m in enumerate(Machines):
        L = rng.randint(mcfg.block_len_min, mcfg.block_len_max); day = rng.randint(0, hcfg.days-1); start_hour = rng.randint(hcfg.start_hour, hcfg.end_hour-1)
        start_bucket = day * bpd + (start_hour - hcfg.start_hour) * bph; start_bucket = min(start_bucket, len(T) - L); A_M[mi, start_bucket:start_bucket+L] = False
    A_W = np.zeros((len(Workers), len(T)), dtype=bool); worker_shift = []
    for wi, w in enumerate(Workers):
        eve = rng.random() < shcfg.assign_evening_fraction; worker_shift.append("eve" if eve else "day")
        for d in range(hcfg.days):
            base = d * bpd
            if eve: s = base + (shcfg.eve_start - hcfg.start_hour) * bph; e = base + (shcfg.eve_end   - hcfg.start_hour) * bph
            else:   s = base + (shcfg.day_start - hcfg.start_hour) * bph; e = base + (shcfg.day_end   - hcfg.start_hour) * bph
            if s < e:
                A_W[wi, s:e] = True; brk = s + shcfg.break_bucket_offset
                if brk < e: A_W[wi, brk] = False
    Q = np.zeros((len(Workers), len(Skills)), dtype=bool); s_idx = {s:i for i,s in enumerate(Skills)}
    for mi, m in enumerate(Machines):
        ops = rng.sample(Workers, k=min(len(Workers), rng.randint(3,6)))
        setups = rng.sample(Workers, k=min(len(Workers), rng.randint(2,4)))
        cleans = rng.sample(Workers, k=min(len(Workers), rng.randint(2,4)))
        for w in ops:    Q[Workers.index(w), s_idx[f"operate_{m}"]] = True
        for w in setups: Q[Workers.index(w), s_idx[f"setup_{m}"]]   = True
        for w in cleans: Q[Workers.index(w), s_idx[f"clean_{m}"]]   = True
    for wi, w in enumerate(Workers):
        for s in process_skills:
            if rng.random() < 0.35: Q[wi, s_idx[s]] = True
    Jobs = [f"J{j+1}" for j in range(scfg.jobs)]; Ops = []; JobOf = {}; PredEdges = []
    base_setup = {m: rng.randint(1,4) for m in Machines}; base_run = {m: rng.randint(6,20) for m in Machines}; base_clean = {m: rng.randint(1,3) for m in Machines}
    E = {}; D = {}; Need = {}; op_counter = 1
    for j in Jobs:
        n_ops = rng.randint(scfg.ops_min, scfg.ops_max); job_ops = []
        for k in range(n_ops):
            op = f"O{op_counter}"; op_counter += 1; job_ops.append(op); Ops.append(op); JobOf[op] = j
            k_m = rng.choice([1,2,2,3]); elig = rng.sample(Machines, k=min(k_m, len(Machines))); E[op] = {m: 1 if m in elig else 0 for m in Machines}
            D[op] = {"Setup":{}, "Run":{}, "Clean":{}}; Need[op] = {"Setup":{}, "Run":{}, "Clean":{}}
            run_proc_skill = random.choice(process_skills) if rng.random() < 0.30 else None
            for m in Machines:
                if E[op][m]==1:
                    su = max(0, base_setup[m] + rng.randint(-1,1)); rn = max(3, base_run[m] + rng.randint(-2,2)); cl = max(0, base_clean[m] + rng.randint(-1,1))
                    D[op]["Setup"][m] = su; D[op]["Run"][m] = rn; D[op]["Clean"][m] = cl
            for m in Machines:
                if E[op][m]==1:
                    Need[op]["Setup"][f"setup_{m}"]  = float(random.choice((0.5,1.0,1.5)))
                    Need[op]["Run"][f"operate_{m}"]  = float(random.choice((1.0,1.5,2.0)))
                    if run_proc_skill is not None: Need[op]["Run"][run_proc_skill] = float(random.choice((0.5,1.0)))
                    Need[op]["Clean"][f"clean_{m}"]  = float(random.choice((0.5,1.0)))
        L = rng.randint(3,5); layers = [[] for _ in range(L)]
        for op in job_ops: layers[rng.randint(0, L-1)].append(op)
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
    A_M_json = {m: list(A_M[i].astype(int)) for i,m in enumerate(Machines)}; A_W_frac = {w: list(A_W[i].astype(float)) for i,w in enumerate(Workers)}
    Q_json = {w: {s: int(Q[wi, s_idx[s]]) for s in Skills} for wi, w in enumerate(Workers)}
    def op_min_time(op):
        mins = [D[op]["Setup"][m] + D[op]["Run"][m] + D[op]["Clean"][m] for m in Machines if E[op][m]==1]
        return min(mins) if mins else 0
    Due = {}; T_len = len(T)
    for j in Jobs:
        job_ops_map = [op for op in Ops if JobOf[op]==j]; cp = sum(op_min_time(op) for op in job_ops_map)
        slack = int(round(0.20 * cp)); Due[j] = min(T_len, cp + slack + rng.randint(0, bpd))
    payload = {"Machines": Machines, "Workers": Workers, "Skills": Skills, "Phases": ["Setup","Run","Clean"], "T": list(T.astype(int)),
               "Ops": Ops, "JobOf": JobOf, "Jobs": Jobs, "Due": Due, "E": E, "D": D, "Need": Need, "Q": Q_json,
               "A_M": A_M_json, "A_W_frac": A_W_frac, "PredEdges": PredEdges, "CAP": ncfg.cap}
    return payload, ax

def _extract_intervals(row: np.ndarray) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None
    for t, flag in enumerate(row):
        if flag and start_idx is None:
            start_idx = t
        elif (not flag) and start_idx is not None:
            intervals.append((start_idx, t))
            start_idx = None
    if start_idx is not None:
        intervals.append((start_idx, row.size))
    return intervals


def build_matrices(data: Dict):
    Ops=data["Ops"]; Machines=data["Machines"]; Workers=data["Workers"]; Skills=data["Skills"]; Phases=data["Phases"]; T=np.array(data["T"],dtype=np.int32); CAP=int(data["CAP"])
    nO,nM,nW,nS,nT=len(Ops),len(Machines),len(Workers),len(Skills),len(T)
    idx={"op":{op:i for i,op in enumerate(Ops)}, "m":{m:i for i,m in enumerate(Machines)}, "w":{w:i for i,w in enumerate(Workers)},
         "s":{s:i for i,s in enumerate(Skills)}, "ph":{p:i for i,p in enumerate(Phases)}, "job":{j:i for i,j in enumerate(data["Jobs"])}}
    A_M=np.array([data["A_M"][m] for m in Machines], dtype=bool); A_W_frac=np.array([data["A_W_frac"][w] for w in Workers], dtype=float)
    C_W=np.rint(A_W_frac * CAP).astype(np.int16); Q=np.array([[data["Q"][w][s] for s in Skills] for w in Workers], dtype=bool);C_S=Q.T.astype(np.int16) @ C_W
    E=np.array([[data["E"][op][m] for m in Machines] for op in Ops], dtype=bool)
    D=np.zeros((nO, len(Phases), nM), dtype=np.int16)
    NeedKernelLists: Dict[Tuple[int, int], List[Tuple[int, int, int, int]]] = {}
    total_len={}
    total_len_arr=np.zeros((nO, nM), dtype=np.int16)
    demand_profiles: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for oi, op in enumerate(Ops):
        for mi, m in enumerate(Machines):
            if not E[oi, mi]:
                total_len[(oi,mi)]=0
                continue
            offset=0; tot=0
            phase_windows: List[Tuple[int, int, Dict[str, float]]] = []
            for pi, ph in enumerate(Phases):
                L=int(data["D"][op][ph].get(m, 0)); D[oi, pi, mi]=L
                if L>0:
                    phase_windows.append((offset, L, data["Need"][op][ph]))
                    offset+=L; tot+=L
            if tot>0:
                total_len[(oi,mi)]=tot
                total_len_arr[oi, mi]=tot
                skill_buffers: Dict[int, np.ndarray] = {}
                for start_off, L, need_map in phase_windows:
                    for s, need in need_map.items():
                        demand=int(math.ceil(float(need) * CAP)); si=idx["s"].get(s, None)
                        if si is None or demand<=0:
                            continue
                        NeedKernelLists.setdefault((oi,mi), []).append((si, start_off, L, demand))
                        buf=skill_buffers.setdefault(si, np.zeros(tot, dtype=np.int16))
                        buf[start_off:start_off+L] += demand
                if skill_buffers:
                    skill_ids=np.array(list(skill_buffers.keys()), dtype=np.int16)
                    demand_matrix=np.vstack([skill_buffers[si] for si in skill_ids])
                    demand_profiles[(oi, mi)] = (skill_ids, demand_matrix)
            else:
                total_len[(oi,mi)]=0
    NeedKernels: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, segments in NeedKernelLists.items():
        if not segments:
            continue
        seg_arr = np.array(segments, dtype=np.int32)
        NeedKernels[key] = (
            seg_arr[:, 0].astype(np.int16, copy=False),
            seg_arr[:, 1].astype(np.int32, copy=False),
            seg_arr[:, 2].astype(np.int16, copy=False),
            seg_arr[:, 3].astype(np.int16, copy=False),
        )
    preds=[(idx["op"][a], idx["op"][b]) for a,b in data["PredEdges"]]
    pred_list=[[] for _ in range(nO)]
    for a,b in preds:
        pred_list[b].append(a)
    pred_list=[np.array(p, dtype=np.int32) for p in pred_list]
    skill_workers=[np.flatnonzero(Q[:, si]).astype(np.int32, copy=False) for si in range(nS)]
    machine_windows=[_extract_intervals(A_M[mi]) for mi in range(nM)]
    machine_window_starts=[[int(seg[0]) for seg in windows] for windows in machine_windows]
    JobOf=np.array([idx["job"][data["JobOf"][op]] for op in Ops], dtype=np.int32)
    Due=np.array([data["Due"][j] for j in data["Jobs"]], dtype=np.int32)
    return {"idx":idx,"A_M":A_M,"C_W":C_W,"Q":Q,"C_S":C_S,"E":E,"D":D,"NeedKernels":NeedKernels,"NeedProfiles":demand_profiles,
            "total_len":total_len,"total_len_arr":total_len_arr,"preds":preds,"pred_list":pred_list,"skill_workers":skill_workers,
            "machine_windows":machine_windows,"machine_window_starts":machine_window_starts,
            "JobOf":JobOf,"Due":Due,
            "n":{"O":nO,"M":nM,"W":nW,"S":nS,"T":nT,"CAP":CAP},"meta":{"Ops":Ops,"Machines":Machines,"Workers":Workers,"Skills":Skills,"Phases":Phases,"T":T}}


@dataclass
class DecodeMetrics: total_ns:int=0; feas_ns:int=0; assign_ns:int=0; ops_scheduled:int=0; ops_failed:int=0; tries:int=0
def decode_schedule(mats: Dict, op_order: np.ndarray, machine_choice: np.ndarray, hard_deadlines: bool=False) -> Dict:
    t0=now_ns();
    nO, nT, CAP = mats["n"]["O"], mats["n"]["T"], mats["n"]["CAP"]
    machine_windows=[[(int(s), int(e)) for s, e in mats["machine_windows"][mi]] for mi in range(mats["n"]["M"])]
    machine_window_starts=[[int(s) for s, _ in windows] for windows in machine_windows]
    skill_free=mats["C_S"].copy(); skill_used=np.zeros_like(skill_free, dtype=np.int16)
    M_busy=np.zeros_like(mats["A_M"], dtype=bool); W_busy=np.zeros((mats["n"]["W"], nT), dtype=bool)
    worker_available_buffer=np.empty(mats["n"]["W"], dtype=bool)
    worker_free_buffer=np.empty_like(worker_available_buffer)
    total_len_arr=mats["total_len_arr"]
    demand_profiles=mats.get("NeedProfiles", {})
    pred_list=mats["pred_list"]
    skill_workers=mats["skill_workers"]
    C_W=mats["C_W"]
    start=np.full(nO, -1, dtype=np.int32); finish=np.full(nO, -1, dtype=np.int32); chosen_m=np.full(nO, -1, dtype=np.int16)
    assigned_workers={}; dm=DecodeMetrics()

    def earliest_pred_finish(oi: int) -> int:
        preds = pred_list[oi]
        if preds.size == 0:
            return 0
        valid = finish[preds]
        valid = valid[valid >= 0]
        if valid.size == 0:
            return 0
        return int(valid.max())

    def find_machine_start(mi: int, est: int, tot: int, demand_profile):
        windows = machine_windows[mi]
        starts = machine_window_starts[mi]
        if not windows:
            return None
        skills=None; demand=None
        if demand_profile is not None:
            skills, demand = demand_profile
        start_idx = 0
        if starts:
            pos = bisect_left(starts, est)
            if pos >= len(windows):
                pos = len(windows) - 1
            if pos > 0 and est < windows[pos - 1][1]:
                start_idx = pos - 1
            else:
                start_idx = pos
        for idx in range(start_idx, len(windows)):
            seg_start, seg_end = windows[idx]
            candidate = max(est, seg_start)
            if candidate + tot > seg_end:
                continue
            if skills is None:
                return candidate, idx
            while candidate + tot <= seg_end:
                feasible = True
                for row_idx, si in enumerate(skills):
                    demand_vec = demand[row_idx]
                    slice_view = skill_free[si, candidate:candidate+tot] < demand_vec
                    if slice_view.any():
                        candidate += int(slice_view.argmax()) + 1
                        feasible = False
                        break
                if feasible:
                    return candidate, idx
        return None

    skill_active_buffers: Dict[int, np.ndarray] = {}
    required_buffers: Dict[int, np.ndarray] = {}

    def worker_assign(oi: int, mi: int, t_start: int) -> bool:
        profile=demand_profiles.get((oi, mi))
        if profile is None:
            assigned_workers[oi]={}
            return True
        skill_ids, demand_matrix = profile
        tot=demand_matrix.shape[1]
        if tot == 0 or skill_ids.size == 0:
            assigned_workers[oi]={}
            return True
        active_buf = skill_active_buffers.setdefault(skill_ids.size, np.empty(skill_ids.size, dtype=bool))
        required_buf = required_buffers.setdefault(skill_ids.size, np.empty(skill_ids.size, dtype=np.int16))
        op_assign={}
        for local_t in range(tot):
            abs_t=t_start + local_t
            if abs_t >= nT:
                return False
            np.greater(demand_matrix[:, local_t], 0, out=active_buf)
            if not active_buf.any():
                continue
            np.greater(C_W[:, abs_t], 0, out=worker_available_buffer)
            np.logical_not(W_busy[:, abs_t], out=worker_free_buffer)
            np.logical_and(worker_available_buffer, worker_free_buffer, out=worker_available_buffer)
            if not worker_available_buffer.any():
                return False
            np.copyto(required_buf, demand_matrix[:, local_t], casting="unsafe")
            if CAP > 1:
                np.add(required_buf, CAP - 1, out=required_buf, casting="unsafe")
            np.floor_divide(required_buf, CAP, out=required_buf, casting="unsafe")
            counts=[]
            for row_idx, si in enumerate(skill_ids):
                if not active_buf[row_idx]:
                    continue
                pool = skill_workers[si]
                if pool.size == 0:
                    return False
                pool_available = pool[worker_available_buffer[pool]]
                counts.append((row_idx, pool_available))
            counts.sort(key=lambda x: x[1].size)
            for row_idx, pool_available in counts:
                required = int(required_buf[row_idx])
                if required <= 0:
                    continue
                if pool_available.size < required:
                    return False
                chosen=pool_available[:required]
                W_busy[chosen, abs_t]=True
                worker_available_buffer[chosen]=False
                op_assign.setdefault(abs_t, [])
                skill_id=int(skill_ids[row_idx])
                for w in chosen:
                    op_assign[abs_t].append((int(w), skill_id))
        assigned_workers[oi]=op_assign
        return True

    for oi in op_order:
        dm.ops_scheduled += 1; dm.tries += 1
        mi_req=machine_choice[oi]; elig=np.where(mats["E"][oi])[0]
        cand_machines=list(elig) if mi_req<0 or mi_req not in elig else [mi_req]
        if not cand_machines:
            dm.ops_failed += 1
            continue
        est=earliest_pred_finish(oi); best=None
        feas_start=now_ns()
        for mi in cand_machines:
            tot=int(total_len_arr[oi, mi])
            if tot <= 0:
                continue
            demand_profile=demand_profiles.get((oi, mi))
            slot=find_machine_start(mi, est, tot, demand_profile)
            if slot is None:
                continue
            t_candidate, interval_idx = slot
            if best is None or t_candidate < best[0]:
                best=(t_candidate, mi, interval_idx, tot, demand_profile)
        dm.feas_ns += now_ns() - feas_start
        if best is None:
            dm.ops_failed += 1
            continue
        best_t, best_mi, interval_idx, tot, demand_profile = best
        assign_start=now_ns()
        if not worker_assign(oi, best_mi, best_t):
            dm.assign_ns += now_ns() - assign_start
            dm.ops_failed += 1
            continue
        dm.assign_ns += now_ns() - assign_start
        seg_start, seg_end = machine_windows[best_mi].pop(interval_idx)
        machine_window_starts[best_mi].pop(interval_idx)
        if seg_start < best_t:
            machine_windows[best_mi].insert(interval_idx, (seg_start, best_t))
            machine_window_starts[best_mi].insert(interval_idx, seg_start)
            interval_idx += 1
        if best_t + tot < seg_end:
            machine_windows[best_mi].insert(interval_idx, (best_t + tot, seg_end))
            machine_window_starts[best_mi].insert(interval_idx, best_t + tot)
        M_busy[best_mi, best_t:best_t+tot] = True
        if demand_profile is not None:
            skills, demand = demand_profile
            for row_idx, si in enumerate(skills):
                skill_free[si, best_t:best_t+tot] -= demand[row_idx]
                skill_used[si, best_t:best_t+tot] += demand[row_idx]
        start[oi]=best_t; finish[oi]=best_t+tot; chosen_m[oi]=best_mi
    res={"feasible": bool(dm.ops_failed == 0) if hard_deadlines else True, "start":start, "finish":finish, "machine":chosen_m,
         "M_busy":M_busy, "W_busy":W_busy, "S_used":skill_used, "assigned":assigned_workers, "metrics":dm}
    res["metrics"].total_ns = now_ns() - t0; return res

def default_genome(mats: Dict, strategy="min_total_len"):
    E=mats["E"]; total_len=mats["total_len"]; nO=mats["n"]["O"]; op_order=np.arange(nO, dtype=np.int32); machine_choice=-np.ones(nO, dtype=np.int16)
    for oi in range(nO):
        elig=np.where(E[oi])[0]
        if elig.size==0: continue
        if strategy=="min_total_len":
            best=None; best_val=None
            for mi in elig:
                tot=total_len.get((oi,mi), 10**9)
                if best is None or tot<best_val: best, best_val = mi, tot
            machine_choice[oi]=best
        else: machine_choice[oi]=elig[0]
    return op_order, machine_choice

def random_genome(mats: Dict, rng: np.random.Generator):
    nO=mats["n"]["O"]; op_order=np.arange(nO, dtype=np.int32); rng.shuffle(op_order); E=mats["E"]; machine_choice=-np.ones(nO, dtype=np.int16)
    for oi in range(nO):
        elig=np.where(E[oi])[0]
        if elig.size>0: machine_choice[oi]=int(rng.choice(elig))
    return op_order, machine_choice

def evaluate_objective(mats: Dict, schedule: Dict, alpha: float = 0.03):
    Due=mats["Due"]; JobOf=mats["JobOf"]; finish=schedule["finish"]; nJ=Due.shape[0]; job_finish=np.zeros(nJ, dtype=np.int32)
    for j in range(nJ):
        ops=np.where(JobOf==j)[0]
        if ops.size>0: job_finish[j]=np.max(finish[ops])
    tardiness=int(np.maximum(0, job_finish - Due).sum()); makespan=int(finish.max()) if finish.size>0 else 0
    score=float(tardiness + alpha * makespan); return {"tardiness":tardiness,"makespan":makespan,"score":score,"job_finish":job_finish}

@dataclass
class GASettings: pop:int=150; gens:int=40; elite:int=10; tour_k:int=4; swap_rate:float=0.30; flip_rate:float=0.20; alpha:float=0.03; seed:int=123; patience:int=8; parallel_workers:int=1

def tournament_select(scores, k, rng):
    n = len(scores)
    # Clamp k to [1, n]
    k = max(1, min(int(k), n))
    idxs = rng.choice(n, size=k, replace=False)
    best_i = int(idxs[0]); best_s = scores[best_i]
    for i in idxs[1:]:
        i = int(i)
        if scores[i] < best_s:
            best_s = scores[i]; best_i = i
    return int(best_i)
def ox_crossover(order_a, order_b, rng):
    n=order_a.size; i,j=sorted(rng.choice(n, size=2, replace=False)); child=-np.ones(n, dtype=np.int32); child[i:j+1]=order_a[i:j+1]
    used=set(child[i:j+1].tolist()); pos=(j+1)%n
    for k in range(n):
        gene=order_b[(j+1+k)%n]
        if gene not in used: child[pos]=gene; used.add(gene); pos=(pos+1)%n
    return child
def one_point_crossover(mc_a, mc_b, rng):
    n=mc_a.size; cut=int(rng.integers(0,n)) if n>0 else 0; child=mc_a.copy()
    if n>0: child[cut:]=mc_b[cut:]; return child
def mutate_genome(mats, op_order, machine_choice, rng, swap_rate=0.3, flip_rate=0.2):
    n=op_order.size
    if rng.random()<swap_rate and n>=2:
        i,j=rng.choice(n, size=2, replace=False); op_order[i],op_order[j]=op_order[j],op_order[i]
    if rng.random()<flip_rate:
        oi=int(rng.integers(0,n)); elig=np.where(mats["E"][oi])[0]
        if elig.size>0:
            if rng.random()<0.25: machine_choice[oi]=-1
            else: machine_choice[oi]=int(rng.choice(elig))
    return op_order, machine_choice
def build_initial_population(mats, pop, rng):
    pop_genomes=[]; base_oo, base_mc=default_genome(mats, strategy="min_total_len"); pop_genomes.append((base_oo.copy(), base_mc.copy()))
    if pop>1: oo2,mc2=random_genome(mats, rng); pop_genomes.append((oo2,mc2))
    while len(pop_genomes)<pop: pop_genomes.append(random_genome(mats, rng)); return pop_genomes
def run_ga(mats, settings: GASettings, hard_deadlines=False, progress_cb=None):
    rng=np.random.default_rng(settings.seed); pop_list=build_initial_population(mats, settings.pop, rng)
    decode_cache: Dict[Tuple[bytes, bytes], Tuple[float, Dict, Dict]] = {}

    def genome_key(order: np.ndarray, machine_choice: np.ndarray) -> Tuple[bytes, bytes]:
        return (order.tobytes(), machine_choice.tobytes())

    def decode_and_score(order: np.ndarray, machine_choice: np.ndarray) -> Tuple[float, Dict, Dict]:
        sched=decode_schedule(mats, order, machine_choice, hard_deadlines=hard_deadlines)
        ev=evaluate_objective(mats, sched, alpha=settings.alpha)
        return float(ev["score"]), sched, ev

    def evaluate_population(pop_genomes: List[Tuple[np.ndarray, np.ndarray]]):
        results: List[Tuple[float, Dict, Dict]] = [None] * len(pop_genomes)  # type: ignore
        to_compute: List[Tuple[int, Tuple[bytes, bytes], np.ndarray, np.ndarray]] = []
        for idx, (oo, mc) in enumerate(pop_genomes):
            key=genome_key(oo, mc)
            cached=decode_cache.get(key)
            if cached is not None:
                results[idx]=cached
            else:
                to_compute.append((idx, key, oo.copy(), mc.copy()))
        worker_count=max(1, int(settings.parallel_workers))
        if to_compute:
            if worker_count > 1 and len(to_compute) > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map={executor.submit(decode_and_score, order, mc): (idx, key) for idx, key, order, mc in to_compute}
                    for future in as_completed(future_map):
                        idx, key = future_map[future]
                        res=future.result()
                        decode_cache[key]=res
                        results[idx]=res
            else:
                for idx, key, order, mc in to_compute:
                    res=decode_and_score(order, mc)
                    decode_cache[key]=res
                    results[idx]=res
        return results

    t0=time.perf_counter_ns()
    initial_results=evaluate_population(pop_list)
    first_pass_ns=time.perf_counter_ns()-t0
    scores=[res[0] for res in initial_results]
    dec_scheds=[res[1] for res in initial_results]
    evs=[res[2] for res in initial_results]
    history={"best":[], "avg":[], "time_ns":[first_pass_ns]}
    best_i=int(np.argmin(scores)); best={"score":float(scores[best_i]), "genome":pop_list[best_i], "sched":dec_scheds[best_i], "eval":evs[best_i]}
    no_improve=0
    for g in range(settings.gens):
        gen_start=time.perf_counter_ns(); eff_elite = max(0, min(int(settings.elite), len(scores)-1))
        elite_idx=np.argsort(scores)[:eff_elite]
        new_pop=[(pop_list[i][0].copy(), pop_list[i][1].copy()) for i in elite_idx]
        while len(new_pop)<settings.pop:
            pa=pop_list[tournament_select(scores, settings.tour_k, rng)]
            pb=pop_list[tournament_select(scores, settings.tour_k, rng)]
            child_order=ox_crossover(pa[0], pb[0], rng); child_mc=one_point_crossover(pa[1], pb[1], rng)
            child_order, child_mc=mutate_genome(mats, child_order, child_mc, rng, settings.swap_rate, settings.flip_rate)
            new_pop.append((child_order, child_mc))
        pop_list=new_pop
        population_results=evaluate_population(pop_list)
        scores=[res[0] for res in population_results]
        dec_scheds=[res[1] for res in population_results]
        evs=[res[2] for res in population_results]
        gen_end=time.perf_counter_ns(); history["time_ns"].append(gen_end - gen_start)
        avg=float(np.mean(scores)); best_idx=int(np.argmin(scores)); best_score=float(scores[best_idx])
        history["avg"].append(avg); history["best"].append(best_score)
        if best_score + 1e-9 < best["score"]:
            best={"score":best_score, "genome":pop_list[best_idx], "sched":dec_scheds[best_idx], "eval":evs[best_idx]}
            no_improve=0
        else:
            no_improve += 1
        if progress_cb:
            progress_cb(g, best_score, avg)
        if settings.patience and no_improve>=settings.patience:
            break
    total_ns=sum(history["time_ns"]) + history["time_ns"][0]
    stats={"total_ns": total_ns, "gens_done": len(history["best"])}
    return best, history, stats

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
    for oi in cand_ops:
        orig=int(start[oi]); delta=model.NewIntVar(-horizon, horizon, f"delta_{oi}"); model.Add(delta == s_var[oi] - orig)
        absd=model.NewIntVar(0, horizon, f"abs_{oi}"); model.AddAbsEquality(absd, delta); model.Minimize(absd)
        j=JobOf[oi]; due=int(Due[j]); late=model.NewIntVar(0, horizon, f"late_{oi}"); model.Add(late >= e_var[oi] - due); model.Minimize(late)
    solver=cp_model.CpSolver(); solver.parameters.max_time_in_seconds=float(time_limit_s); solver.parameters.num_search_workers=8
    res=solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE): return {}
    new_starts={oi:int(solver.Value(s_var[oi])) for oi in cand_ops}; return new_starts

# ========================= UI =========================
st.title("Matrix Scheduler — GA + CP-SAT")
st.caption("Vectorized decoder for speed. GA for global search. Optional CP-SAT polish on a machine window.")

with st.expander("Math"):
    st.markdown("Sliding-window checks, aggregate skill capacity, GA evolution, and CP-SAT local polish (machine-window only).")

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
    data, ax = generate_stress_dataset(int(seed), hcfg, scfg, mcfg, ncfg, shcfg); st.session_state["data"]=data; st.session_state["ax"]=ax; st.sidebar.success("Dataset generated.")

if "data" not in st.session_state:
    hcfg=HorizonCfg(); scfg=ScaleCfg(); ncfg=NeedsCfg(); mcfg=MaintCfg(); shcfg=ShiftCfg()
    data, ax = generate_stress_dataset(2025, hcfg, scfg, mcfg, ncfg, shcfg); st.session_state["data"]=data; st.session_state["ax"]=ax
data=st.session_state["data"]

t_build0=now_ns(); mats=build_matrices(data); t_build=now_ns()-t_build0
st.subheader("Scale summary")
nO,nM,nW,nS,nT=mats["n"]["O"],mats["n"]["M"],mats["n"]["W"],mats["n"]["S"],mats["n"]["T"]
colA,colB,colC,colD=st.columns(4); colA.metric("Ops", nO); colB.metric("Machines", nM); colC.metric("Workers", nW); colD.metric("Buckets", nT)
st.caption(f"Matrix build time: {ns_to_ms(t_build):.1f} ms")

tab1, tab2, tab3, tab4 = st.tabs(["Single decode", "Random benchmark", "GA optimize", "CP-SAT polish"])

with tab1:
    st.caption("Decode one heuristic genome (min total duration per op).")
    op_order, machine_choice = default_genome(mats, strategy="min_total_len")
    hard_deadlines = st.checkbox("Hard deadlines (reject if any due missed)", value=False)
    t0=now_ns(); sched=decode_schedule(mats, op_order.copy(), machine_choice.copy(), hard_deadlines=hard_deadlines); t1=now_ns()
    eva=evaluate_objective(mats, sched); c1,c2,c3,c4=st.columns(4)
    c1.metric("Decode time", f"{ns_to_ms(t1-t0):.1f} ms"); c2.metric("Feasible", str(sched["feasible"]))
    c3.metric("Tardiness", eva["tardiness"]); c4.metric("Makespan", eva["makespan"])
    st.write(f"Feasibility: {ns_to_ms(sched['metrics'].feas_ns):.1f} ms · Identity: {ns_to_ms(sched['metrics'].assign_ns):.1f} ms")

with tab2:
    st.caption("Stress the decoder with random genomes and measure throughput.")
    bench_n=st.number_input("Random decodes", 10, 2000, 200, 50); seed_b=st.number_input("Benchmark seed", value=7, step=1)
    if st.button("Run benchmark"):
        rng=np.random.default_rng(int(seed_b)); times=[]; feas_ms=[]; assign_ms=[]
        for i in range(int(bench_n)):
            oo,mc=random_genome(mats, rng); t0=now_ns(); sc=decode_schedule(mats, oo, mc, hard_deadlines=False); t1=now_ns()
            times.append(ns_to_ms(t1-t0)); feas_ms.append(ns_to_ms(sc["metrics"].feas_ns)); assign_ms.append(ns_to_ms(sc["metrics"].assign_ns))
        times=np.array(times); feas_ms=np.array(feas_ms); assign_ms=np.array(assign_ms)
        col1,col2,col3,col4,col5 = st.columns(5)
        col1.metric("Decodes", len(times)); col2.metric("Avg (ms)", f"{times.mean():.1f}"); col3.metric("Median (ms)", f"{np.median(times):.1f}")
        col4.metric("p95 (ms)", f"{np.percentile(times,95):.1f}"); col5.metric("Decodes/sec", f"{1000.0 / max(1e-6, times.mean()):.1f}")
        fig, axp=plt.subplots(); axp.hist(times, bins=30); axp.set_title("Decode time (ms)"); st.pyplot(fig)

with tab3:
    st.caption("Full GA optimization.")
    col = st.columns(2)
    with col[0]:
        pop=st.number_input("Population", 10, 1000, 150, 10); gens=st.number_input("Generations", 1, 500, 40, 1); elite=st.number_input("Elite", 0, 200, 10, 1)
        tour_k=st.number_input("Tournament k", 2, int(pop), min(4, int(pop)), 1); alpha=st.slider("Alpha (makespan weight)", 0.0, 0.2, 0.03, 0.01); patience=st.number_input("Early-stop patience", 0, 50, 8, 1)
        hard_deadlines_ga=st.checkbox("Hard deadlines", value=False, key="ga_hd")
    with col[1]:
        swap=st.slider("Mutation: swap rate", 0.0, 1.0, 0.30, 0.05); flip=st.slider("Mutation: machine flip rate", 0.0, 1.0, 0.20, 0.05); seed_ga=st.number_input("GA seed", value=123, step=1)
        run_ga_btn=st.button("Run GA")
    if run_ga_btn:
        settings=GASettings(pop=int(pop), gens=int(gens), elite=int(elite), tour_k=int(tour_k), swap_rate=float(swap), flip_rate=float(flip),
                            alpha=float(alpha), seed=int(seed_ga), patience=int(patience))
        progress=st.progress(0); status=st.empty()
        def cb(gen_idx, best_score, avg_score):
            pct=int(100*(gen_idx+1)/max(1,settings.gens)); progress.progress(min(pct,100)); status.text(f"Gen {gen_idx+1}/{settings.gens} — best {best_score:.2f} · avg {avg_score:.2f}")
        t0=time.perf_counter_ns(); best, hist, stats=run_ga(mats, settings, hard_deadlines=hard_deadlines_ga, progress_cb=cb); t1=time.perf_counter_ns()
        total_ms=(t1-t0)/1e6; st.success(f"GA completed in {total_ms:.1f} ms • gens: {len(hist['best'])} • best score: {best['score']:.2f}")
        genomes_evaluated=settings.pop * max(1, len(hist["best"]) + 1); st.write(f"≈ Genomes evaluated: {genomes_evaluated} → ~{1000.0*genomes_evaluated/max(1.0,total_ms):.1f} genomes/sec")
        fig, axp=plt.subplots(); axp.plot(hist["best"], label="best"); axp.plot(hist["avg"], label="avg"); axp.set_xlabel("gen"); axp.set_ylabel("score"); axp.legend(); st.pyplot(fig)
        Ops=mats["meta"]["Ops"]; Machines=mats["meta"]["Machines"]; best_sched=best["sched"]
        df_best=pd.DataFrame({"Op":Ops,"Machine":[Machines[m] if m>=0 else "" for m in best_sched["machine"]],"Start":best_sched["start"],"Finish":best_sched["finish"]}).sort_values(by=["Start"])
        st.dataframe(df_best.head(50))
        best_oo, best_mc = best["genome"]
        genome_json = {
            "op_order": best_oo.astype(int).tolist(),
            "machine_choice": best_mc.astype(int).tolist(),
            "alpha": settings.alpha,
        }
        genome_bytes = json.dumps(genome_json).encode("utf-8")
        csv_bytes = df_best.to_csv(index=False).encode("utf-8")
        st.session_state["ga_best_genome_json"] = genome_bytes
        st.session_state["ga_best_sched_csv"] = csv_bytes

    if st.session_state.get("ga_best_genome_json"):
        st.download_button(
            "Download best genome (JSON)",
            data=st.session_state.get("ga_best_genome_json"),
            file_name="best_genome.json",
            mime="application/json",
        )
    if st.session_state.get("ga_best_sched_csv"):
        st.download_button(
            "Download best schedule (CSV)",
            data=st.session_state.get("ga_best_sched_csv"),
            file_name="best_schedule.csv",
            mime="text/csv",
        )

with tab4:
    st.caption("CP-SAT polishing on a single machine inside a time window (workers remain greedy).")
    if not HAS_CPSAT: st.error("ortools is not installed. Run: pip install ortools")
    else:
        base_oo, base_mc = default_genome(mats, strategy="min_total_len")
        base_sched = decode_schedule(mats, base_oo.copy(), base_mc.copy(), hard_deadlines=False); base_eval=evaluate_objective(mats, base_sched)
        c1,c2,c3=st.columns(3); c1.metric("Base tardiness", base_eval["tardiness"]); c2.metric("Base makespan", base_eval["makespan"]); c3.metric("Base score", base_eval["score"])
        Machines=mats["meta"]["Machines"]; mi_name=st.selectbox("Machine to polish", Machines, index=0); mi=Machines.index(mi_name)
        window_start=st.number_input("Window start (bucket)", 0, mats["n"]["T"]-1, 0, 10); window_len=st.number_input("Window length (buckets)", 10, mats["n"]["T"], 200, 10)
        tl=st.number_input("CP-SAT time limit (seconds)", 0.1, 30.0, 3.0, 0.1); run_polish=st.button("Run CP-SAT polish")
        if run_polish:
            t0=now_ns(); new_starts=cpsat_polish_machine_window(mats, base_sched, mi, int(window_start), int(window_len), float(tl)); t1=now_ns()
            st.write(f"CP-SAT polish time: {ns_to_ms(t1-t0):.1f} ms")
            if not new_starts: st.warning("No changes (or no ops in window).")
            else:
                oo=base_oo.copy(); mc=base_mc.copy(); polished_ops=sorted(new_starts.keys(), key=lambda oi:new_starts[oi])
                mask=np.ones_like(oo, dtype=bool); mask[polished_ops]=False; rest=oo[mask]; new_order=np.array(polished_ops + rest.tolist(), dtype=np.int32)
                t2=now_ns(); new_sched=decode_schedule(mats, new_order, mc.copy(), hard_deadlines=False); t3=now_ns(); new_eval=evaluate_objective(mats, new_sched)
                d1,d2,d3=st.columns(3); d1.metric("New tardiness", new_eval["tardiness"], delta=int(new_eval["tardiness"]-base_eval["tardiness"]))
                d2.metric("New makespan", new_eval["makespan"], delta=int(new_eval["makespan"]-base_eval["makespan"]))
                d3.metric("New score", f"{new_eval['score']:.1f}", delta=float(new_eval["score"]-base_eval["score"]))
                st.write(f"Re-decode time: {ns_to_ms(t3-t2):.1f} ms")
                Ops=mats["meta"]["Ops"]; rows=[{"Op":Ops[oi], "Old start":int(base_sched["start"][oi]), "New start":int(new_starts[oi])} for oi in polished_ops]
                st.dataframe(pd.DataFrame(rows).sort_values(by="New start"))

st.markdown("---"); st.markdown("**Use GA for global search; CP-SAT to polish small windows.**")

