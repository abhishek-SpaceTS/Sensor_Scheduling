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
    Uses Google OR-Tools to solve the Orienteering Problem (find the best schedule).
    """
    print(f"\n--- Teacher (OR-Tools) Solving Scenario ---")
    print(f"Total possible targets to view: {len(windows)}")
    
    # 1. Initialize the CP-SAT model
    model = cp_model.CpModel()
    
    # 2. Create Decision Variables
    # x[i] will be 1 if we choose to look at target i, and 0 if we ignore it
    x = {}
    for i, event in enumerate(windows):
        x[i] = model.NewBoolVar(f"x_{i}")
        
    # 3. Add Constraints (The Rules)
    # We must enforce "Slew Time". For this model, M4V needs a minimum of 
    # 60 seconds to turn its camera from one target to the next.
    min_slew_gap_seconds = 60.0
    
    for i in range(len(windows)):
        time_i = datetime.fromisoformat(windows[i]["start_time_utc"])
        
        for j in range(i + 1, len(windows)):
            time_j = datetime.fromisoformat(windows[j]["start_time_utc"])
            
            # Calculate the time difference between the two targets
            gap_seconds = (time_j - time_i).total_seconds()
            
            # If the gap is less than our required slew time (e.g. they overlap)
            # then M4V CANNOT look at both. It must choose one or the other!
            if gap_seconds < min_slew_gap_seconds:
                model.Add(x[i] + x[j] <= 1)
                
    # 4. Set the Objective (The Goal)
    # Maximize the total sum of priority points collected
    objective_terms = []
    for i, event in enumerate(windows):
        priority = int(event["priority_value"])
        objective_terms.append(x[i] * priority)
        
    model.Maximize(sum(objective_terms))
    
    # 5. Run the Solver
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # 6. Print the Results
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("\n=== OPTIMAL SCHEDULE FOUND ===")
        total_score = solver.ObjectiveValue()
        print(f"Total Priority Score Achieved: {total_score}\n")
        
        print("Sequence of Operations:")
        output_labels = []
        
        for i, event in enumerate(windows):
            is_scheduled = 1 if solver.BooleanValue(x[i]) else 0
            
            # Save the label data
            labeled_event = event.copy()
            labeled_event["is_scheduled"] = is_scheduled
            output_labels.append(labeled_event)
            
            time_str = event['start_time_utc'].split('T')[1].split('+')[0]
            if is_scheduled:
                print(f" -> [OBSERVE] {time_str} | {event['target_name']} (Priority: {event['priority_value']}) [Span: {event['duration_seconds']}s]")
            else:
                print(f" -> [ IGNORE] {time_str} | {event['target_name']} (Conflict or Low Priority)")
                
        # Save to disk!
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
