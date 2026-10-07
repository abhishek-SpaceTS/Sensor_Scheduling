import json
from pathlib import Path
from ortools.sat.python import cp_model
import time

def solve_scenario(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    num_objects = data["n_objects"]
    windows = data["windows"]
    slew_times = data["slew_times"] # The precomputed matrix!
    values = data["values"]
    dwell = data["dwell"]
    
    model = cp_model.CpModel()
    
    intervals = []
    starts = {}
    ends = {}
    durations = {}
    actual_durations = {}
    is_scheduled = {}
    
    for i in range(num_objects):
        window_start = int(windows[i][0][0])
        window_end = int(windows[i][0][1])
        max_dur = window_end - window_start
        d_min = int(dwell)
        
        x_i = model.NewBoolVar(f'x_{i}')
        is_scheduled[i] = x_i
        
        if max_dur < d_min:
            model.Add(x_i == 0)
            
        s_i = model.NewIntVar(window_start, window_end, f'start_{i}')
        d_i = model.NewIntVar(d_min, max(d_min, max_dur), f'dur_{i}')
        e_i = model.NewIntVar(window_start, window_end, f'end_{i}')
        
        interval_i = model.NewOptionalIntervalVar(s_i, d_i, e_i, x_i, f'interval_{i}')
        intervals.append(interval_i)
        
        starts[i] = s_i
        ends[i] = e_i
        durations[i] = d_i
        
        act_d_i = model.NewIntVar(0, max(d_min, max_dur), f'act_dur_{i}')
        model.Add(act_d_i == d_i).OnlyEnforceIf(x_i)
        model.Add(act_d_i == 0).OnlyEnforceIf(x_i.Not())
        actual_durations[i] = act_d_i
        
    model.AddNoOverlap(intervals)
    
    for i in range(num_objects):
        window_start_i = int(windows[i][0][0])
        window_end_i = int(windows[i][0][1])
        d_min = int(dwell)
        
        for j in range(i + 1, num_objects):
            window_start_j = int(windows[j][0][0])
            window_end_j = int(windows[j][0][1])
            
            slew_ij = int(slew_times[i][j])
            slew_ji = int(slew_times[j][i])
            
            # Logic: Either j follows i, or i follows j. (Considering observation duration + slew)
            can_j_follow_i = (window_end_j >= window_start_i + d_min + slew_ij)
            can_i_follow_j = (window_end_i >= window_start_j + d_min + slew_ji)
            
            if not can_j_follow_i and not can_i_follow_j:
                model.Add(is_scheduled[i] + is_scheduled[j] <= 1)
            elif can_j_follow_i and not can_i_follow_j:
                model.Add(starts[j] >= ends[i] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j]])
            elif can_i_follow_j and not can_j_follow_i:
                model.Add(starts[i] >= ends[j] + slew_ji).OnlyEnforceIf([is_scheduled[i], is_scheduled[j]])
            else:
                b_i_j = model.NewBoolVar(f"b_{i}_{j}")
                model.Add(starts[j] >= ends[i] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j])
                model.Add(starts[i] >= ends[j] + slew_ji).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j.Not()])
                
    objective_terms = []
    for i in range(num_objects):
        # We strictly maximize the Duration (d_i) as requested in Problem Statement (1).docx
        objective_terms.append(actual_durations[i])
        
    model.Maximize(sum(objective_terms))
    
    solver = cp_model.CpSolver()
    # Add a safety timeout since we are processing 1000 files in bulk
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        data["is_scheduled"] = [int(solver.BooleanValue(is_scheduled[i])) for i in range(num_objects)]
        return data
    else:
        # If mathematically infeasible or solver timeout, return array of 0s
        data["is_scheduled"] = [0] * num_objects
        return data

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    
    input_dir = project_root / "complex_scenarios"
    output_dir = project_root / "data" / "processed" / "complex_labels"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(input_dir.glob("*.json"))
    print(f"Found {len(files)} JSON scenarios. Starting Google OR-Tools batch solver...")
    
    start_time = time.time()
    success_count = 0
    
    for idx, fpath in enumerate(files):
        labeled_data = solve_scenario(fpath)
        if labeled_data:
            out_file = output_dir / fpath.name
            with open(out_file, 'w') as out_f:
                json.dump(labeled_data, out_f, indent=4)
            success_count += 1
            
        if (idx + 1) % 100 == 0:
            print(f"Solved {idx + 1} / {len(files)}...")
            
    elapsed = time.time() - start_time
    print(f"\n--- BATCH SOLVE COMPLETE ---")
    print(f"Successfully solved and labeled: {success_count} scenarios.")
    print(f"Total time taken: {elapsed:.2f} seconds.")
    print(f"Teacher Answer Keys saved to: {output_dir}")
