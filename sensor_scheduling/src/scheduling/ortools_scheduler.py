import sys
import json
from pathlib import Path
from datetime import datetime

# Import Google OR-Tools
from ortools.sat.python import cp_model

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

def load_scenario(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def solve_scenario(windows):
    """
    Uses Google OR-Tools to solve the Orienteering Problem with Time Windows (OPTW).
    Now uses Interval Constraints to match the continuous time formulation, correctly
    accounting for exact observation durations and dynamic slew gaps.
    """
    if not windows:
        print("No windows provided.")
        return

    print(f"\n--- Teacher (OR-Tools) Solving Scenario ---")
    print(f"Total possible targets to view: {len(windows)}")
    
    model = cp_model.CpModel()
    
    import yaml
    import math
    from datetime import timedelta
    
    adcs_config_path = project_root / "config" / "adcs_config.yaml"
    with open(adcs_config_path, 'r') as f:
        adcs_config = yaml.safe_load(f)
        
    slew_rate = adcs_config['performance_requirements']['max_slew_rate_deg_per_sec']
    settling_time_seconds = 10.0
    print(f"Dynamic ADCS slew rate: {slew_rate} deg/s (calculating exact angular distances)")
    
    # Establish a global time origin to convert datetime into integer seconds for the solver
    base_time = datetime.fromisoformat(windows[0]["start_time_utc"])
    for w in windows:
        t = datetime.fromisoformat(w["start_time_utc"])
        if t < base_time:
            base_time = t
            
    d_min = 10 # Minimum useful observation time in seconds
            
    # Decision Variables
    intervals = []
    starts = {}
    ends = {}
    durations = {}
    actual_durations = {}
    is_scheduled = {}
    
    for i, event in enumerate(windows):
        time_i_start = int((datetime.fromisoformat(event["start_time_utc"]) - base_time).total_seconds())
        time_i_end = int((datetime.fromisoformat(event["end_time_utc"]) - base_time).total_seconds())
        max_dur = time_i_end - time_i_start
        
        x_i = model.NewBoolVar(f"x_{i}")
        is_scheduled[i] = x_i
        
        # Continuous start time within the window
        s_i = model.NewIntVar(time_i_start, time_i_end, f"start_{i}")
        # Continuous duration (bounded by d_min and max available time in window)
        d_i = model.NewIntVar(d_min, max(d_min, max_dur), f"duration_{i}")
        e_i = model.NewIntVar(time_i_start, time_i_end, f"end_{i}")
        
        interval_i = model.NewOptionalIntervalVar(s_i, d_i, e_i, x_i, f"interval_{i}")
        intervals.append(interval_i)
        
        starts[i] = s_i
        ends[i] = e_i
        durations[i] = d_i
        
        # Track actual duration for objective function
        act_d_i = model.NewIntVar(0, max(d_min, max_dur), f"act_dur_{i}")
        model.Add(act_d_i == d_i).OnlyEnforceIf(x_i)
        model.Add(act_d_i == 0).OnlyEnforceIf(x_i.Not())
        actual_durations[i] = act_d_i

    # Ensure no intervals overlap fundamentally
    model.AddNoOverlap(intervals)
    
    # Precedence & Slew Constraints
    for i in range(len(windows)):
        time_i_start = int((datetime.fromisoformat(windows[i]["start_time_utc"]) - base_time).total_seconds())
        time_i_end = int((datetime.fromisoformat(windows[i]["end_time_utc"]) - base_time).total_seconds())
        vec_i = windows[i].get("direction_vector", [1, 0, 0])
        
        for j in range(i + 1, len(windows)):
            time_j_start = int((datetime.fromisoformat(windows[j]["start_time_utc"]) - base_time).total_seconds())
            time_j_end = int((datetime.fromisoformat(windows[j]["end_time_utc"]) - base_time).total_seconds())
            vec_j = windows[j].get("direction_vector", [1, 0, 0])
            
            dot_product = vec_i[0]*vec_j[0] + vec_i[1]*vec_j[1] + vec_i[2]*vec_j[2]
            dot_product = max(-1.0, min(1.0, dot_product))
            slew_distance_deg = math.degrees(math.acos(dot_product))
            
            slew_ij = math.ceil((slew_distance_deg / slew_rate) + settling_time_seconds)
            
            # Logic: Either j follows i, or i follows j. (Considering observation duration + slew)
            can_j_follow_i = (time_j_end >= time_i_start + d_min + slew_ij)
            can_i_follow_j = (time_i_end >= time_j_start + d_min + slew_ij)
            
            if not can_j_follow_i and not can_i_follow_j:
                model.Add(is_scheduled[i] + is_scheduled[j] <= 1)
            elif can_j_follow_i and not can_i_follow_j:
                model.Add(starts[j] >= ends[i] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j]])
            elif can_i_follow_j and not can_j_follow_i:
                model.Add(starts[i] >= ends[j] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j]])
            else:
                # Both orders are possible, let the solver decide via a boolean precedence variable
                b_i_j = model.NewBoolVar(f"b_{i}_{j}")
                model.Add(starts[j] >= ends[i] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j])
                model.Add(starts[i] >= ends[j] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j.Not()])
                
    # Objective: Maximize total observation duration, weighted by priority
    objective_terms = []
    for i, event in enumerate(windows):
        priority = int(event["priority_value"])
        objective_terms.append(actual_durations[i] * priority)
        
    model.Maximize(sum(objective_terms))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60.0 # Safety limit
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n=== OPTIMAL SCHEDULE FOUND ===")
        total_score = solver.ObjectiveValue()
        print(f"Total Objective Score (Duration x Priority): {total_score}\n")
        
        print("Sequence of Operations:")
        output_labels = []
        
        for i, event in enumerate(windows):
            is_sched = solver.BooleanValue(is_scheduled[i])
            labeled_event = event.copy()
            labeled_event["is_scheduled"] = 1 if is_sched else 0
            
            if is_sched:
                s_val = solver.Value(starts[i])
                d_val = solver.Value(durations[i])
                
                actual_start = base_time + timedelta(seconds=s_val)
                time_str = actual_start.isoformat().split('T')[1].split('+')[0]
                
                labeled_event["observation_start_utc"] = actual_start.isoformat() + "+00:00"
                labeled_event["observation_duration_seconds"] = d_val
                
                print(f" -> [OBSERVE] {time_str} | {event['target_name']} (Priority: {event['priority_value']}) [Duration: {d_val}s]")
            else:
                labeled_event["observation_start_utc"] = None
                labeled_event["observation_duration_seconds"] = 0
                time_str = event['start_time_utc'].split('T')[1].split('+')[0]
                print(f" -> [ IGNORE] {time_str} | {event['target_name']} (Conflict or Low Priority)")
                
            output_labels.append(labeled_event)
            
        labels_path = project_root / "data" / "processed" / "scheduling_dataset" / "scenario_001_labels.json"
        with open(labels_path, "w") as f:
            json.dump(output_labels, f, indent=4)
        print(f"\n-> Successfully saved Answer Key to {labels_path.name}")
            
    else:
        print("No solution could be found.")

def main():
    scenario_path = project_root / "data" / "processed" / "visibility_windows" / "scenario_001_windows.json"
    if not scenario_path.exists():
        print(f"Could not find {scenario_path}. Please generate it first.")
        return
        
    windows = load_scenario(scenario_path)
    solve_scenario(windows)

if __name__ == "__main__":
    main()
