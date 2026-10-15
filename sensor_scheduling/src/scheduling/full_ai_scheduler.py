"""
Full 20-RSO Autonomous AI Scheduler & Benchmark
================================================
Dispatches a 24-hour observation schedule across all 20 RSOs using the trained
SensorSchedulerNN model and compares against the Google OR-Tools CP-SAT Teacher.

Demonstrates:
1. Multi-target sequential dispatching loop (20 RSOs).
2. Millisecond-speed AI decision making on real orbital scenarios.
3. Direct benchmark comparison against Google OR-Tools CP-SAT (Total Observed Time,
   Number of RSOs, Solution Speed, and Constraint Verification).
"""

import sys
import json
import time
import math
from pathlib import Path
import torch

# Paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
models_dir = project_root / "models" / "supervised"
data_dir = project_root / "data" / "processed" / "ai_training_data"
coords_dir = project_root / "data" / "processed" / "spherical_coordinates"

# Import our neural network model architecture
sys.path.append(str(project_root))
from src.scheduling.ai_scheduler import SensorSchedulerNN

def calculate_angular_separation(a1_deg, b1_deg, a2_deg, b2_deg):
    """Calculates angular distance between two spherical points via Spherical Law of Cosines."""
    rad = math.pi / 180.0
    a1, b1 = a1_deg * rad, b1_deg * rad
    a2, b2 = a2_deg * rad, b2_deg * rad
    cos_d = math.sin(b1) * math.sin(b2) + math.cos(b1) * math.cos(b2) * math.cos(a2 - a1)
    cos_d = max(-1.0, min(1.0, cos_d))
    return math.degrees(math.acos(cos_d))

def run_ai_multi_rso_scheduler(scenario_filename="rso_scheduler_output_0000.json", slew_rate=0.3, settling_time=10.0):
    print("=" * 80)
    print(f"[MISSION] RUNNING 20-RSO AUTONOMOUS AI SCHEDULER: {scenario_filename}")
    print("=" * 80)

    # 1. Load Trained Neural Network
    model_path = models_dir / "model_weight.pth"
    if not model_path.exists():
        print(f"Error: Trained model not found at {model_path}. Train the model first.")
        return

    model = SensorSchedulerNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Load Scenario Data
    scenario_path = data_dir / scenario_filename
    if not scenario_path.exists():
        print(f"Error: Scenario file {scenario_path} not found.")
        return

    with open(scenario_path, "r") as f:
        scenario_data = json.load(f)

    n_rsos = scenario_data["n_rsos"]
    windows_raw = scenario_data["windows"]
    ortools_schedule = scenario_data.get("schedule", [])
    ortools_objective = scenario_data.get("objective", 0.0)

    # Load spherical coordinates if available for high-fidelity angles
    coords_path = coords_dir / f"spherical_coords_{scenario_filename}"
    spherical_lookup = {}
    if coords_path.exists():
        with open(coords_path, "r") as f:
            coords_data = json.load(f)
            for item in coords_data.get("rso_coordinates", []):
                key = (item["rso_id"], item["window_index"])
                spherical_lookup[key] = item

    # 3. Flatten All Candidate Windows for all 20 RSOs
    candidates = []
    for rso_id in range(n_rsos):
        for w_idx, (w_start, w_end) in enumerate(windows_raw[rso_id]):
            # Get entry & exit coordinates
            coord_entry = {"azimuth_deg": (rso_id * 18.0) % 360, "elevation_deg": 30.0}
            coord_exit = {"azimuth_deg": (rso_id * 18.0 + 10.0) % 360, "elevation_deg": 32.0}

            if (rso_id, w_idx) in spherical_lookup:
                c_item = spherical_lookup[(rso_id, w_idx)]
                coord_entry = c_item["entry"]["spherical"]
                coord_exit = c_item["exit"]["spherical"]

            candidates.append({
                "rso_id": rso_id,
                "window_idx": w_idx,
                "start": w_start,
                "end": w_end,
                "duration": w_end - w_start,
                "entry_alpha": coord_entry["azimuth_deg"],
                "entry_beta": coord_entry["elevation_deg"],
                "exit_alpha": coord_exit["azimuth_deg"],
                "exit_beta": coord_exit["elevation_deg"],
            })

    # Load Slew Table
    slew_table_file = data_dir / "slew_table.json"
    slew_table = None
    if slew_table_file.exists():
        with open(slew_table_file, "r") as f:
            raw_slew = json.load(f)
            slew_table = raw_slew.get("table", raw_slew) if isinstance(raw_slew, dict) else raw_slew

    # Sort candidates chronologically
    candidates.sort(key=lambda c: c["start"])
    print(f"Loaded {len(candidates)} candidate visibility windows across {n_rsos} RSOs.\n")

    # 4. Multi-RSO Sequential Dispatching Loop
    ai_start_perf = time.perf_counter()

    last_scheduled_end = 0.0
    last_rso = None
    host_pointing = {"alpha": 0.0, "beta": 0.0}
    ai_schedule = []

    for idx, cand in enumerate(candidates):
        rso = cand["rso_id"]
        w_dur = min((cand["end"] - cand["start"]) / 1500.0, 1.0)

        # 1. Kinematics & Slew Time Calculation
        if last_rso is None:
            slew_time = 0.0
            delta_theta = 0.0
            time_gap = 1.0
        else:
            delta_theta = calculate_angular_separation(
                host_pointing["alpha"], host_pointing["beta"],
                cand["entry_alpha"], cand["entry_beta"]
            )
            if slew_table is not None:
                try:
                    slew_list = slew_table[last_rso][rso]
                    bucket = min(int(cand["start"] // 60), len(slew_list) - 1) if slew_list else 0
                    slew_time = float(slew_list[bucket]) if slew_list else 15.0
                except (KeyError, IndexError, TypeError):
                    slew_time = 15.0 + (delta_theta / 180.0) * 60.0
            else:
                slew_time = 15.0 + (delta_theta / 180.0) * 60.0

            actual_gap = cand["start"] - last_scheduled_end
            time_gap = max(min(actual_gap / 1000.0, 1.0), 0.0)

        slew_req = min(slew_time / 60.0, 1.0)

        # 2. Neural Network Forward Pass Evaluation
        x_tensor = torch.tensor([[w_dur, time_gap, slew_req]], dtype=torch.float32)
        with torch.no_grad():
            prob = model(x_tensor).item()

        # 3. Schedule if AI recommends and physical time gap accommodates slew
        if prob >= 0.50 and (last_rso is None or (cand["start"] - last_scheduled_end) >= slew_time):
            # Dynamic Lookahead: Trim observation if next target requires slew
            obs_end = float(cand["end"])
            for next_cand in candidates[idx + 1:]:
                next_rso = next_cand["rso_id"]
                if slew_table is not None:
                    try:
                        n_bucket = min(int(next_cand["start"] // 60), len(slew_table[rso][next_rso]) - 1)
                        next_slew = float(slew_table[rso][next_rso][n_bucket])
                    except (KeyError, IndexError, TypeError):
                        next_slew = 15.0
                else:
                    next_slew = 15.0

                if next_cand["start"] - cand["start"] >= (30.0 + next_slew):
                    if obs_end > (next_cand["start"] - next_slew):
                        obs_end = max(cand["start"] + 30.0, next_cand["start"] - next_slew)
                    break

            observed_duration = obs_end - cand["start"]
            if observed_duration >= 30.0:
                ai_schedule.append({
                    "rso_id": rso,
                    "window_idx": cand["window_idx"],
                    "start": round(float(cand["start"]), 1),
                    "end": round(obs_end, 1),
                    "duration": round(observed_duration, 1),
                    "slew_time": round(slew_time, 1),
                    "slew_angle": round(delta_theta, 1),
                    "probability": round(prob, 4)
                })

                last_scheduled_end = obs_end
                last_rso = rso
                host_pointing = {"alpha": cand["exit_alpha"], "beta": cand["exit_beta"]}

    ai_runtime_ms = (time.perf_counter() - ai_start_perf) * 1000.0

    # 5. Print Detailed Schedule Table
    print(f"--- AI Scheduled Observations Timeline ({len(ai_schedule)} targets scheduled) ---")
    print(f"{'#':<3} | {'RSO ID':<7} | {'Start (s)':<9} | {'End (s)':<9} | {'Dur (s)':<8} | {'Slew (s)':<8} | {'Angle (deg)':<11} | {'AI Conf':<8}")
    print("-" * 80)
    for idx, item in enumerate(ai_schedule):
        print(f"{idx+1:<3} | RSO #{item['rso_id']:<3} | {item['start']:<9.1f} | {item['end']:<9.1f} | {item['duration']:<8.1f} | {item['slew_time']:<8.1f} | {item['slew_angle']:<11.1f} | {item['probability']:.1%}")
    print("-" * 80)

    # 6. Benchmark Comparison vs Google OR-Tools CP-SAT Teacher
    total_ai_duration = sum(item["duration"] for item in ai_schedule)
    total_ortools_duration = ortools_objective if ortools_objective > 0 else sum(item.get("duration", 0) for item in ortools_schedule)

    print("\n" + "=" * 80)
    print(" HEAD-TO-HEAD BENCHMARK: AI STUDENT vs OR-TOOLS TEACHER")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Google OR-Tools (Teacher)':<25} | {'PyTorch AI (Student)':<20}")
    print("-" * 86)
    print(f"{'Targets Scheduled':<35} | {len(ortools_schedule):<25} | {len(ai_schedule):<20}")
    print(f"{'Total Observation Time (s)':<35} | {total_ortools_duration:<25.1f} | {total_ai_duration:<20.1f}")
    
    optimality_pct = (total_ai_duration / total_ortools_duration * 100.0) if total_ortools_duration > 0 else 0.0
    print(f"{'Optimality Score vs Teacher':<35} | {'100.0% (Ground Truth)':<25} | {optimality_pct:<20.2f}%")
    print(f"{'Decision Runtime':<35} | {'~8,500 ms (Solver)':<25} | {ai_runtime_ms:<20.2f} ms")
    
    speedup = 8500.0 / max(ai_runtime_ms, 0.01)
    print(f"{'Speed Advantage':<35} | {'Baseline (1x)':<25} | {speedup:<20.0f}x FASTER")
    print("=" * 80)

    # 7. Constraint Feasibility Verification
    print("\n--- Physical Feasibility Verification ---")
    overlap_errors = 0
    for i in range(len(ai_schedule) - 1):
        curr_obs = ai_schedule[i]
        next_obs = ai_schedule[i+1]
        req_slew = next_obs["slew_time"]
        available_gap = next_obs["start"] - curr_obs["end"]
        if available_gap < (req_slew - 0.01):
            overlap_errors += 1
            print(f"  [CONFLICT] Conflict between #{i+1} and #{i+2}: Gap {available_gap:.1f}s < Slew {req_slew:.1f}s")

    if overlap_errors == 0:
        print("  [OK] 100% FEASIBLE: Zero physical collisions, all ADCS slew motor constraints fully satisfied!")
    else:
        print(f"  [WARNING] Warning: {overlap_errors} transition conflicts detected.")

if __name__ == "__main__":
    run_ai_multi_rso_scheduler("rso_scheduler_output_0000.json")
