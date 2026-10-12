import math
import random
import time
import json
import os
import argparse
from pathlib import Path
import concurrent.futures
from ortools.sat.python import cp_model

# ----------------------------------------------------------------------------
# Constants (must match the problem statement)
# ----------------------------------------------------------------------------
T = 5700          # planning horizon (one orbit), seconds
N_RSOS = 20
D_MIN = 30        # minimum useful observation time
COPIES = 2        # max visits per (RSO, window)  -> MODEL RESTRICTION, see README note
BUCKET = 60       # slew-table resolution, seconds
NUM_BUCKETS = T // BUCKET + 1
MIN_SLEW = 15     # smallest value slew_time() can return (used for pruning)


# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------
def make_windows(seed):
    """Seeded test case: 1-2 random visibility windows per RSO, merged if overlapping."""
    rng = random.Random(seed)
    windows_by_rso = []
    for _ in range(N_RSOS):
        rso_windows = []
        # Generate exactly 1 window per RSO to prevent them from merging into a longer window
        for _ in range(1):
            duration = 300
            start = rng.randint(0, max(0, T - duration))
            rso_windows.append((start, start + duration))

        rso_windows.sort()
        merged = []
        for w in rso_windows:
            if merged and w[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], w[1]))
            else:
                merged.append(w)
        windows_by_rso.append(merged)
    return windows_by_rso


def slew_time(j, k, tau):
    """Slew time (integer seconds) from RSO j to RSO k, arriving on k at time tau."""
    if j == k:
        return 0
    phase_j = (j * 0.5 + tau * 0.0002 * (j % 5 + 1)) % (2 * math.pi)
    phase_k = (k * 0.5 + tau * 0.0002 * (k % 5 + 1)) % (2 * math.pi)
    diff = abs(phase_j - phase_k)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return int(MIN_SLEW + (diff / math.pi) * 60)


def precompute_slew_table():
    """
    table[j][k][b] = MAX slew time over every integer arrival second tau in bucket b.
    Using the max makes the CP-SAT model conservative: any schedule it returns
    is also feasible under the exact slew_time(). (The old version sampled every
    15 s and could miss the peak, which could make a schedule 1 s infeasible.)
    """
    table = [[[0] * NUM_BUCKETS for _ in range(N_RSOS)] for _ in range(N_RSOS)]
    for j in range(N_RSOS):
        for k in range(N_RSOS):
            if j == k:
                continue
            for b in range(NUM_BUCKETS):
                lo = b * BUCKET
                hi = min(T, (b + 1) * BUCKET - 1)
                table[j][k][b] = max(slew_time(j, k, tau) for tau in range(lo, hi + 1))
    return table


def load_or_build_slew_table(output_dir):
    """Cache the table, but invalidate it if the constants it was built with changed."""
    path = output_dir / "slew_table.json"
    meta = {"T": T, "N_RSOS": N_RSOS, "BUCKET": BUCKET}
    if path.exists():
        with open(path) as f:
            blob = json.load(f)
        if isinstance(blob, dict) and blob.get("meta") == meta:
            print("Loading precomputed slew table...")
            return blob["table"]
        print("Slew table cache is stale/legacy -> rebuilding.")
    print("Precomputing slew table...")
    table = precompute_slew_table()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"meta": meta, "table": table}, f, separators=(",", ":"))
    os.replace(tmp, path)
    return table


# ----------------------------------------------------------------------------
# Independent verifier (uses the EXACT slew_time, not the bucket table)
# ----------------------------------------------------------------------------
def verify_schedule(windows_by_rso, schedule):
    """Returns (ok, message, total_observation_time)."""
    total = 0
    prev = None
    for i, obs in enumerate(schedule):
        r, s, e = obs["rso"], obs["start"], obs["end"]
        if not any(p1 <= s and e <= p2 for p1, p2 in windows_by_rso[r]):
            return False, f"obs {i}: outside visibility window", 0
        if e - s < D_MIN:
            return False, f"obs {i}: shorter than D_MIN", 0
        if prev is not None:
            if prev["rso"] == r:
                return False, f"obs {i}: same RSO twice in a row", 0
            if s < prev["end"] + slew_time(prev["rso"], r, s):
                return False, f"obs {i}: not enough slew time", 0
        total += e - s
        prev = obs
    return True, "ok", total


# ----------------------------------------------------------------------------
# CP-SAT model
# ----------------------------------------------------------------------------
class Candidate:
    def __init__(self, node_id, rso, window_idx, copy_idx, p1, p2):
        self.id = node_id
        self.rso = rso
        self.window_idx = window_idx
        self.copy_idx = copy_idx
        self.p1 = p1
        self.p2 = p2


def build_model(windows_by_rso, slew_table):
    candidates = []
    cid = 1  # node 0 is the depot
    for rso in range(N_RSOS):
        for w_idx, (p1, p2) in enumerate(windows_by_rso[rso]):
            for c_idx in range(COPIES):
                candidates.append(Candidate(cid, rso, w_idx, c_idx, p1, p2))
                cid += 1

    model = cp_model.CpModel()
    p_vars, s_vars, e_vars, d_vars, bucket_vars = {}, {}, {}, {}, {}

    # 1. Per-candidate variables
    for v in candidates:
        i = v.id
        p_vars[i] = model.NewBoolVar(f"p_{i}")
        s_vars[i] = model.NewIntVar(v.p1, v.p2, f"s_{i}")
        e_vars[i] = model.NewIntVar(v.p1, v.p2, f"e_{i}")
        d_vars[i] = model.NewIntVar(0, v.p2 - v.p1, f"d_{i}")
        # FIX: valid table indices are 0..NUM_BUCKETS-1 (was NUM_BUCKETS)
        bucket_vars[i] = model.NewIntVar(0, NUM_BUCKETS - 1, f"bucket_{i}")

        model.Add(e_vars[i] - s_vars[i] == d_vars[i])
        model.Add(d_vars[i] >= D_MIN).OnlyEnforceIf(p_vars[i])
        model.Add(d_vars[i] == 0).OnlyEnforceIf(p_vars[i].Not())
        model.Add(s_vars[i] == v.p1).OnlyEnforceIf(p_vars[i].Not())  # shrink domains
        model.AddDivisionEquality(bucket_vars[i], s_vars[i], BUCKET)

    # 2. Symmetry breaking between copies of the same (RSO, window)
    by_key = {(v.rso, v.window_idx, v.copy_idx): v for v in candidates}
    for v in candidates:
        if v.copy_idx > 0:
            prev = by_key[(v.rso, v.window_idx, v.copy_idx - 1)]
            model.AddImplication(p_vars[v.id], p_vars[prev.id])
            model.Add(s_vars[v.id] >= e_vars[prev.id]).OnlyEnforceIf(p_vars[v.id])

    # 3. Routing circuit
    arcs = []
    for v in candidates:
        arcs.append((0, v.id, model.NewBoolVar(f"arc_0_{v.id}")))
        arcs.append((v.id, 0, model.NewBoolVar(f"arc_{v.id}_0")))
        arcs.append((v.id, v.id, p_vars[v.id].Not()))      # skipped candidate
    arcs.append((0, 0, model.NewBoolVar("depot_self_loop")))  # empty schedule

    # PERF FIX: the slew time depends only on (rso_u, rso_v, bucket_v), NOT on
    # the copy/window of u. Create ONE element variable per (v, rso_u) and share
    # it across all arcs u->v with that rso_u (was one per arc: ~6400 -> ~1500).
    slew_vars = {}

    def get_slew_var(v, rso_u):
        key = (v.id, rso_u)
        if key not in slew_vars:
            table = slew_table[rso_u][v.rso]
            t = model.NewIntVar(min(table), max(table), f"t_{rso_u}_{v.id}")
            model.AddElement(bucket_vars[v.id], table, t)
            slew_vars[key] = t
        return slew_vars[key]

    for u in candidates:
        for v in candidates:
            if u.rso == v.rso:
                continue  # never observe the same RSO twice in a row
            # Tighter (still valid) pruning: u needs >= D_MIN, then a slew >= MIN_SLEW,
            # then v needs >= D_MIN before its window closes.
            if v.p2 < u.p1 + D_MIN + MIN_SLEW + D_MIN:
                continue
            lit = model.NewBoolVar(f"arc_{u.id}_{v.id}")
            arcs.append((u.id, v.id, lit))
            # s_v >= e_u + t_uv(arrival bucket of v)
            model.Add(s_vars[v.id] >= e_vars[u.id] + get_slew_var(v, u.rso)).OnlyEnforceIf(lit)

    model.AddCircuit(arcs)

    # Redundant constraint: one sensor, so total observation time <= T. Without it
    # the bound is ~sum of all window lengths (gap of 600-800%), which makes
    # `best_bound`/`gap` meaningless. (A NoOverlap on optional intervals also
    # tightens things but slowed the incumbent search in 20 s tests, so it is omitted.)
    model.Add(sum(d_vars[v.id] for v in candidates) <= T)

    model.Maximize(sum(d_vars[v.id] for v in candidates))

    return model, candidates, arcs, s_vars, e_vars, d_vars


def solve_scheduling_instance(index, seed, slew_table, base_time_limit,
                              max_time_limit, threads_per_job, output_dir):
    """Returns (status_string, failure_reason)."""
    windows_by_rso = make_windows(seed)
    model, candidates, arcs, s_vars, e_vars, d_vars = build_model(windows_by_rso, slew_table)
    by_id = {v.id: v for v in candidates}

    # FIX: old loop never ran if base_time_limit > max_time_limit
    max_time_limit = max(max_time_limit, base_time_limit)
    current_time_limit = base_time_limit

    while current_time_limit <= max_time_limit:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(current_time_limit)
        solver.parameters.num_workers = threads_per_job
        solver.parameters.random_seed = seed  # reproducibility (best-effort with >1 worker)

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            next_node = {u: v for (u, v, lit) in arcs if solver.BooleanValue(lit)}

            seq, curr = [], next_node[0]
            while curr != 0:
                seq.append(curr)
                curr = next_node[curr]

            schedule = []
            for node_id in seq:
                v = by_id[node_id]
                s, e = solver.Value(s_vars[node_id]), solver.Value(e_vars[node_id])
                schedule.append({"rso": v.rso, "window_idx": v.window_idx,
                                 "start": s, "end": e, "duration": e - s})

            # Never write an unverified label to a training set
            ok, msg, total = verify_schedule(windows_by_rso, schedule)
            if not ok:
                return "FAILED", f"Verifier rejected solution: {msg}"

            obj, bound = solver.ObjectiveValue(), solver.BestObjectiveBound()
            dataset = {
                "n_rsos": N_RSOS, "seed": seed, "windows": windows_by_rso,
                "schedule": schedule,
                "objective": obj, "best_bound": bound,
                "gap": (bound - obj) / max(1.0, obj),   # filter on this when training
                "status": solver.StatusName(status),
                "time_limit": current_time_limit,
                "verified": True,
            }
            out = output_dir / f"rso_scheduler_output_{index:04d}.json"
            tmp = out.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(dataset, f, indent=4)
            os.replace(tmp, out)
            return "SUCCESS", None

        if status == cp_model.UNKNOWN:
            if current_time_limit >= max_time_limit:
                return "FAILED", f"Timeout UNKNOWN at {current_time_limit}s"
            current_time_limit = min(current_time_limit + 30, max_time_limit)
        else:
            return "FAILED", f"Status {solver.StatusName(status)}"

    return "FAILED", "Max time limit reached"


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def format_time(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def generate_dataset(num_samples, output_dir, time_limit, max_time_limit, jobs, threads_per_job):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} samples into: {output_dir}")
    print(f"{jobs} parallel jobs x {threads_per_job} threads. "
          f"Time limit: {time_limit}s (max {max(time_limit, max_time_limit)}s)")

    slew_table = load_or_build_slew_table(output_dir)

    existing = set()
    for f in output_dir.glob("rso_scheduler_output_*.json"):
        try:
            existing.add(int(f.stem.split("_")[-1]))
        except ValueError:
            pass
    remaining = [i for i in range(num_samples) if i not in existing]
    if not remaining:
        print(f"Dataset generation already complete ({num_samples} samples).")
        return
    print(f"Resuming generation: {len(remaining)} samples remaining.")

    start_time = time.time()
    failures_file = output_dir / "failures.log"
    completed = failed = 0
    total_to_do = len(remaining)

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(solve_scheduling_instance, idx, 42 + idx, slew_table,
                            time_limit, max_time_limit, threads_per_job, output_dir): idx
            for idx in remaining
        }
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                status, reason = future.result()
            except Exception as e:
                status, reason = "FAILED", f"EXCEPTION {e}"
            if status == "SUCCESS":
                completed += 1
            else:
                failed += 1
                with open(failures_file, "a") as f:
                    f.write(f"Index {idx:04d} (Seed {42 + idx}): {reason}\n")

            elapsed = time.time() - start_time
            done = completed + failed
            eta = elapsed / done * (total_to_do - done)
            print(f"[{done}/{total_to_do}] Completed: {completed}, Failed: {failed} "
                  f"| Elapsed: {format_time(elapsed)} | ETA: {format_time(eta)}", end="\r")

    print(f"\nFinished! Completed {completed}, failed {failed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CP-SAT RSO scheduling dataset")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default="data/processed/ai_training_data")
    parser.add_argument("--time-limit", type=int, default=30, help="Base time limit (s)")
    parser.add_argument("--max-time-limit", type=int, default=120, help="Retry ceiling (s)")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads-per-job", type=int, default=2)
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.output_dir, args.time_limit,
                     args.max_time_limit, args.jobs, args.threads_per_job)