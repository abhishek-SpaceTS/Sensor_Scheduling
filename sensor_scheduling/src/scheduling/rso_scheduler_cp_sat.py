import math
import random
import time
from ortools.sat.python import cp_model

# Constants
T = 5700
N_RSOS = 20
D_MIN = 30
COPIES = 2
BUCKET = 60

# --- Data Generation ---

def make_windows(seed=None):
    """
    Generates synthetic visibility windows with a stable per-scenario RNG seed.
    Using the scenario id as the seed avoids reproducing the same dataset for
    every generated training sample while keeping outputs deterministic.
    """
    rng = random.Random(seed if seed is not None else time.time_ns())
    windows_by_rso = []
    for rso in range(N_RSOS):
        num_windows = rng.choice([1, 2])
        rso_windows = []
        for _ in range(num_windows):
            duration = rng.randint(300, 1500)
            start = rng.randint(0, max(0, T - duration))
            end = start + duration
            rso_windows.append((start, end))

        rso_windows.sort()
        merged = []
        for w in rso_windows:
            if not merged:
                merged.append(w)
            else:
                last_start, last_end = merged[-1]
                if w[0] <= last_end:
                    merged[-1] = (last_start, max(last_end, w[1]))
                else:
                    merged.append(w)
        windows_by_rso.append(merged)
    return windows_by_rso

def slew_time(j, k, tau):
    """
    Computes a realistic-looking attitude slew time based on angular separation.
    Phase angles of RSOs drift linearly over time.
    Return integer seconds.
    """
    if j == k:
        return 0
    
    phase_j = (j * 0.5 + tau * 0.0002 * (j % 5 + 1)) % (2 * math.pi)
    phase_k = (k * 0.5 + tau * 0.0002 * (k % 5 + 1)) % (2 * math.pi)
    
    diff = abs(phase_j - phase_k)
    if diff > math.pi:
        diff = 2 * math.pi - diff
        
    return int(15 + (diff / math.pi) * 60)

def precompute_slew_table():
    """
    Precompute a lookup table of maximum slew times in buckets of size BUCKET.
    """
    num_buckets = T // BUCKET + 1
    table = {}
    for j in range(N_RSOS):
        table[j] = {}
        for k in range(N_RSOS):
            if j == k:
                continue
            table[j][k] = []
            for b in range(num_buckets):
                start_tau = b * BUCKET
                end_tau = min(T, (b + 1) * BUCKET)

                # Check maximum slew within the bucket by sampling
                max_t = 0
                for tau in range(start_tau, end_tau + 1, max(1, BUCKET // 4)):
                    max_t = max(max_t, slew_time(j, k, tau))
                max_t = max(max_t, slew_time(j, k, start_tau))
                max_t = max(max_t, slew_time(j, k, end_tau))

                table[j][k].append(max_t)
    return table

# --- CP-SAT Model ---

class Candidate:
    def __init__(self, node_id, rso, window_idx, copy_idx, p1, p2):
        self.id = node_id
        self.rso = rso
        self.window_idx = window_idx
        self.copy_idx = copy_idx
        self.p1 = p1
        self.p2 = p2

def solve_scheduling(scenario_id, output_dir):
    import sys, os
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    devnull = open(os.devnull, 'w')
    sys.stdout = devnull

    windows_by_rso = make_windows(seed=scenario_id)
    slew_table = precompute_slew_table()

    # Flatten windows into candidates
    candidates = []
    cid = 1  # 0 is reserved for depot
    for rso in range(N_RSOS):
        for w_idx, (p1, p2) in enumerate(windows_by_rso[rso]):
            for c_idx in range(COPIES):
                candidates.append(Candidate(cid, rso, w_idx, c_idx, p1, p2))
                cid += 1
                
    model = cp_model.CpModel()
    
    # Dictionaries for decision variables
    p_vars = {}       # Presence
    s_vars = {}       # Start time
    e_vars = {}       # End time
    d_vars = {}       # Duration
    bucket_vars = {}  # Bucket for arrival time

    # 1. Variables and basic candidate constraints
    for v in candidates:
        i = v.id
        p_vars[i] = model.NewBoolVar(f'p_{i}')
        s_vars[i] = model.NewIntVar(v.p1, v.p2, f's_{i}')
        e_vars[i] = model.NewIntVar(v.p1, v.p2, f'e_{i}')
        d_vars[i] = model.NewIntVar(0, v.p2 - v.p1, f'd_{i}')
        bucket_vars[i] = model.NewIntVar(0, T // BUCKET + 1, f'bucket_{i}')

        # d_v = e_v - s_v
        model.Add(e_vars[i] - s_vars[i] == d_vars[i])

        # If present, d_v >= d_min; if absent, d_v = 0
        model.Add(d_vars[i] >= D_MIN).OnlyEnforceIf(p_vars[i])
        model.Add(d_vars[i] == 0).OnlyEnforceIf(p_vars[i].Not())

        # Lock variables if absent to reduce domain sizes
        model.Add(s_vars[i] == v.p1).OnlyEnforceIf(p_vars[i].Not())
        model.Add(e_vars[i] == v.p1).OnlyEnforceIf(p_vars[i].Not())

        # bucket_v = s_v / BUCKET
        model.AddDivisionEquality(bucket_vars[i], s_vars[i], BUCKET)

    # 2. Symmetry breaking between copies
    for v in candidates:
        if v.copy_idx > 0:
            # Find the previous copy (c-1)
            prev = next(u for u in candidates if u.rso == v.rso and u.window_idx == v.window_idx and u.copy_idx == v.copy_idx - 1)
            
            # copy c+1 present => copy c present
            model.AddImplication(p_vars[v.id], p_vars[prev.id])
            
            # s_{c+1} >= e_c if copy c+1 is present
            model.Add(s_vars[v.id] >= e_vars[prev.id]).OnlyEnforceIf(p_vars[v.id])

    # 3. Routing (Circuit) arcs
    arcs = []
    
    # Depot to candidates and candidates to depot
    for v in candidates:
        i = v.id
        arcs.append((0, i, model.NewBoolVar(f'arc_0_{i}')))
        arcs.append((i, 0, model.NewBoolVar(f'arc_{i}_0')))
        
    # Self loop for candidates that are skipped (not present)
    for v in candidates:
        i = v.id
        arcs.append((i, i, p_vars[i].Not()))
        
    # Depot self loop (for empty schedule)
    depot_self_loop = model.NewBoolVar('depot_self_loop')
    arcs.append((0, 0, depot_self_loop))
        
    # Candidate to candidate arcs
    for u in candidates:
        for v in candidates:
            if u.rso == v.rso:
                continue # No RSO is observed twice in a row
                
            # Prune impossible arcs: v's window closes before u's window opens + d_min
            if v.p2 < u.p1 + D_MIN:
                continue
                
            lit = model.NewBoolVar(f'arc_{u.id}_{v.id}')
            arcs.append((u.id, v.id, lit))
            
            # Slew time constraint: s_v >= e_u + t_uv
            # We look up t_uv from slew_table based on v's arrival time bucket
            table = slew_table[u.rso][v.rso]
            t_uv_var = model.NewIntVar(min(table), max(table), f't_{u.id}_{v.id}')
            model.AddElement(bucket_vars[v.id], table, t_uv_var)

            model.Add(s_vars[v.id] >= e_vars[u.id] + t_uv_var).OnlyEnforceIf(lit)

    model.AddCircuit(arcs)
    
    # 4. Objective
    # We do not explicitly model d_i = min(...)
    # By maximizing sum(d_v) = sum(e_v - s_v), the solver will naturally push e_v to the latest
    # possible valid time. This is either the window close time (P_2), or restricted by the 
    # slew time constraint to the next observation (s_{next} >= e_v + t_{v, next}), perfectly
    # emulating the min() logic without complex non-linear expressions.
    model.Maximize(sum(d_vars[v.id] for v in candidates))
    
    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0
    solver.parameters.num_search_workers = 8
    
    print("Starting solver...")
    start_time = time.time()
    status = solver.Solve(model)
    wall_time = time.time() - start_time
    
    # --- Output ---
    status_name = solver.StatusName(status)
    obj_val = solver.ObjectiveValue()
    best_bound = solver.BestObjectiveBound()

    print(f"\nSolver Status: {status_name}")
    print(f"Objective Value: {obj_val}")
    print(f"Best Bound: {best_bound}")
    print(f"Wall Time: {wall_time:.3f} s")
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        # Extract sequence from arcs
        next_node = {}
        for (u, v, lit) in arcs:
            if solver.BooleanValue(lit):
                next_node[u] = v
                
        seq = []
        curr = next_node[0]
        while curr != 0:
            seq.append(curr)
            curr = next_node[curr]
            
        print("\nObservation Sequence:")
        print(f"{'Idx':>3} | {'RSO':>3} | {'Window':>13} | {'Start':>5} | {'End':>5} | {'Dur':>4} | {'Gap':>4} | {'ReqSlew':>7}")
        print("-" * 75)
        
        total_time = 0
        candidate_by_id = {v.id: v for v in candidates}
        
        for idx, node_id in enumerate(seq):
            v = candidate_by_id[node_id]
            s = solver.Value(s_vars[node_id])
            e = solver.Value(e_vars[node_id])
            d = solver.Value(d_vars[node_id])
            total_time += d
            
            if idx < len(seq) - 1:
                next_id = seq[idx+1]
                next_s = solver.Value(s_vars[next_id])
                gap = next_s - e
                next_v = candidate_by_id[next_id]
                req_slew = slew_time(v.rso, next_v.rso, next_s)
                gap_str = str(gap)
                req_slew_str = str(req_slew)
            else:
                gap_str = "-"
                req_slew_str = "-"
                
            window_str = f"[{v.p1},{v.p2}]"
            print(f"{idx:3d} | {v.rso:3d} | {window_str:>13} | {s:5d} | {e:5d} | {d:4d} | {gap_str:>4} | {req_slew_str:>7}")
            
        print("-" * 75)
        print(f"Total observed time: {total_time} s")
        print(f"Percentage of orbit: {total_time / T * 100:.2f}%\n")
        
        # Independent Feasibility Check
        print("--- Feasibility Check ---")
        passed = True
        for idx, node_id in enumerate(seq):
            v = candidate_by_id[node_id]
            s = solver.Value(s_vars[node_id])
            e = solver.Value(e_vars[node_id])
            d = solver.Value(d_vars[node_id])
            
            if not (v.p1 <= s <= v.p2):
                print(f"FAIL: Obs {idx} start {s} outside window [{v.p1}, {v.p2}]")
                passed = False
            if not (v.p1 <= e <= v.p2):
                print(f"FAIL: Obs {idx} end {e} outside window [{v.p1}, {v.p2}]")
                passed = False
            if d < D_MIN:
                print(f"FAIL: Obs {idx} duration {d} < min duration {D_MIN}")
                passed = False
                
            if idx < len(seq) - 1:
                next_id = seq[idx+1]
                next_v = candidate_by_id[next_id]
                
                if v.rso == next_v.rso:
                    print(f"FAIL: RSO {v.rso} observed twice in a row (idx {idx} and {idx+1})")
                    passed = False
                    
                next_s = solver.Value(s_vars[next_id])
                gap = next_s - e
                req_slew = slew_time(v.rso, next_v.rso, next_s)
                
                if gap < req_slew:
                    print(f"FAIL: Obs {idx}->{idx+1} gap {gap} < required slew {req_slew}")
                    passed = False
                    
        if passed:
            print("Status: PASS (All checks successful)")
        else:
            print("Status: FAIL (One or more constraints violated)")

        import json
        from pathlib import Path
        output_file = output_dir / f"rso_scheduler_output_{scenario_id:04d}.json"
        
        schedule_data = []
        for idx, node_id in enumerate(seq):
            v = candidate_by_id[node_id]
            s = solver.Value(s_vars[node_id])
            e = solver.Value(e_vars[node_id])
            d = solver.Value(d_vars[node_id])
            schedule_data.append({
                "rso": v.rso,
                "window_idx": v.window_idx,
                "start": s,
                "end": e,
                "duration": d
            })
            
            
        # Convert output to match the AI's exact expected schema!
        is_scheduled = [0] * N_RSOS
        start_times = [0] * N_RSOS
        end_times = [0] * N_RSOS
        
        for data in schedule_data:
            rso = data["rso"]
            is_scheduled[rso] = 1
            start_times[rso] = data["start"]
            end_times[rso] = data["end"]
            
        # Flatten the 3D slew table into a 2D matrix for the Neural Network
        flat_slew_times = []
        for i in range(N_RSOS):
            row = []
            for j in range(N_RSOS):
                if i == j:
                    row.append(0.0)
                else:
                    row.append(max(slew_table[i][j]))
            flat_slew_times.append(row)
            
        dataset = {
            "n_objects": N_RSOS,
            "windows": windows_by_rso,
            "values": [100] * N_RSOS,
            "slew_times": flat_slew_times,
            "is_scheduled": is_scheduled,
            "start_times": start_times,
            "end_times": end_times
        }
        
        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=4)
            
    # Restore stdout and safely close file descriptor
    sys.stdout = sys.__stdout__
    devnull.close()

if __name__ == "__main__":
    from pathlib import Path
    import concurrent.futures
    import multiprocessing
    import os
    
    # Save directly to the folder the AI is looking at!
    out_dir = Path(__file__).parent.parent.parent / "data" / "processed" / "ai_training_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checking existing files in {out_dir}...")
    
    tasks = []
    for i in range(1, 1001):
        if not (out_dir / f"rso_scheduler_output_{i:04d}.json").exists():
            tasks.append((i, out_dir))
            
    print(f"Found {1000 - len(tasks)} existing files.")
    print(f"Generating the remaining {len(tasks)} datasets in PARALLEL...")
    
    # 4-core optimal settings
    num_cores = 2
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(solve_scheduling, t[0], t[1]) for t in tasks]
        
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == len(tasks):
                print(f"Completed {completed} / {len(tasks)} remaining scenarios...")
                
    print("\nDONE! All 1000 AI training labels are generated and ready for PyTorch.")
