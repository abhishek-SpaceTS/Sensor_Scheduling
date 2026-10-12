import json
import time
from pathlib import Path
from ortools.sat.python import cp_model
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- METHOD 1: CP-SAT Solver ---
def solve_with_cpsat(data):
    num_objects = data["n_objects"]
    windows = data["windows"]
    slew_times = data["slew_times"]
    dwell = int(data["dwell"])
    
    model = cp_model.CpModel()
    intervals = []
    starts = {}
    ends = {}
    actual_durations = {}
    is_scheduled = {}
    
    for i in range(num_objects):
        w_start = int(windows[i][0][0])
        w_end = int(windows[i][0][1])
        max_dur = w_end - w_start
        
        x_i = model.NewBoolVar(f'x_{i}')
        is_scheduled[i] = x_i
        
        if max_dur < dwell:
            model.Add(x_i == 0)
            
        s_i = model.NewIntVar(w_start, w_end, f'start_{i}')
        d_i = model.NewIntVar(dwell, max(dwell, max_dur), f'dur_{i}')
        e_i = model.NewIntVar(w_start, w_end, f'end_{i}')
        
        intervals.append(model.NewOptionalIntervalVar(s_i, d_i, e_i, x_i, f'int_{i}'))
        starts[i] = s_i
        ends[i] = e_i
        
        act_d = model.NewIntVar(0, max(dwell, max_dur), f'act_d_{i}')
        model.Add(act_d == d_i).OnlyEnforceIf(x_i)
        model.Add(act_d == 0).OnlyEnforceIf(x_i.Not())
        actual_durations[i] = act_d
        
    model.AddNoOverlap(intervals)
    
    for i in range(num_objects):
        for j in range(i + 1, num_objects):
            slew_ij = int(slew_times[i][j])
            slew_ji = int(slew_times[j][i])
            
            b_i_j = model.NewBoolVar(f"b_{i}_{j}")
            model.Add(starts[j] >= ends[i] + slew_ij).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j])
            model.Add(starts[i] >= ends[j] + slew_ji).OnlyEnforceIf([is_scheduled[i], is_scheduled[j], b_i_j.Not()])
                
    # EXACT OBJECTIVE: Maximize sum of Durations (d_i)
    model.Maximize(sum(actual_durations[i] for i in range(num_objects)))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    
    t0 = time.time()
    status = solver.Solve(model)
    t1 = time.time()
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        score = int(solver.ObjectiveValue())
        scheduled_count = sum(1 for i in range(num_objects) if solver.Value(is_scheduled[i]))
        
        schedule_details = []
        for i in range(num_objects):
            if solver.Value(is_scheduled[i]):
                schedule_details.append({
                    "id": i,
                    "start": solver.Value(starts[i]),
                    "duration": solver.Value(actual_durations[i])
                })
        return score, t1 - t0, scheduled_count, schedule_details
    return 0, t1 - t0, 0, []

# --- METHOD 2: Routing Solver (VRP) ---
def solve_with_routing(data):
    num_objects = data["n_objects"]
    windows = data["windows"]
    slew_times = data["slew_times"]
    dwell = int(data["dwell"])
    
    num_nodes = num_objects + 2
    manager = pywrapcp.RoutingIndexManager(num_nodes, 1, [0], [1])
    routing = pywrapcp.RoutingModel(manager)
    
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        if from_node == 0 or to_node == 1 or to_node == 0:
            return 0
        return dwell + int(slew_times[from_node - 2][to_node - 2])
        
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    routing.AddDimension(transit_callback_index, 50000, 50000, False, 'Time')
    time_dimension = routing.GetDimensionOrDie('Time')
    
    for i in range(num_objects):
        index = manager.NodeToIndex(i + 2)
        w_start, w_end = int(windows[i][0][0]), int(windows[i][0][1])
        max_start = max(w_start, w_end - dwell)
        if max_start < w_start:
            time_dimension.CumulVar(index).SetRange(0, 0)
        else:
            time_dimension.CumulVar(index).SetRange(w_start, max_start)
            
        # EXACT OBJECTIVE: Penalty for dropping is Dwell Time (Since we want to maximize sum of dwell)
        routing.AddDisjunction([index], dwell)
        
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 10
    
    t0 = time.time()
    solution = routing.SolveWithParameters(search_parameters)
    t1 = time.time()
    
    if solution:
        index = routing.Start(0)
        score = 0
        scheduled_count = 0
        schedule_details = []
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= 2:
                target_idx = node - 2
                score += dwell # VRP only knows fixed Dwell
                scheduled_count += 1
                
                start_time = solution.Min(time_dimension.CumulVar(index))
                schedule_details.append({
                    "id": target_idx,
                    "start": start_time,
                    "duration": dwell
                })
            index = solution.Value(routing.NextVar(index))
        return score, t1 - t0, scheduled_count, schedule_details
    return 0, t1 - t0, 0, []

# --- METHOD 3: Greedy Heuristic ---
def solve_with_greedy(data):
    num_objects = data["n_objects"]
    windows = data["windows"]
    slew_times = data["slew_times"]
    dwell = int(data["dwell"])
    
    # Sort targets by Maximum Possible Duration (Largest First)
    durations = []
    for i in range(num_objects):
        max_dur = int(windows[i][0][1]) - int(windows[i][0][0])
        durations.append((i, max_dur))
    durations = sorted(durations, key=lambda x: x[1], reverse=True)
    
    t0 = time.time()
    schedule_details = []
    score = 0
    
    for target_idx, max_dur in durations:
        w_start, w_end = int(windows[target_idx][0][0]), int(windows[target_idx][0][1])
        
        best_start = w_start
        can_schedule = True
        
        for scheduled in sorted(schedule_details, key=lambda x: x['start']):
            prev_idx = scheduled['id']
            prev_end = scheduled['start'] + scheduled['duration']
            
            slew_to = int(slew_times[prev_idx][target_idx])
            slew_from = int(slew_times[target_idx][prev_idx])
            
            if best_start + max_dur + slew_from > scheduled['start'] and scheduled['start'] + scheduled['duration'] + slew_to > best_start:
                best_start = max(best_start, prev_end + slew_to)
                
        if best_start + dwell > w_end:
            can_schedule = False
            
        if can_schedule:
            actual_dur = min(max_dur, w_end - best_start)
            schedule_details.append({
                "id": target_idx,
                "start": best_start,
                "duration": actual_dur
            })
            score += actual_dur
            
    t1 = time.time()
    return score, t1 - t0, len(schedule_details), schedule_details

def plot_comparison(cp_schedule, rt_schedule, gr_schedule, num_objects, output_path):
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # CP-SAT
    for item in cp_schedule:
        ax1.barh(item["id"] + 1, item["duration"], left=item["start"], color='limegreen', edgecolor='black')
    ax1.set_title('Method 1: CP-SAT Constraint Solver (Global Optimum)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Target ID')
    ax1.set_yticks(range(1, num_objects + 1))
    ax1.invert_yaxis()
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Routing
    for item in rt_schedule:
        ax2.barh(item["id"] + 1, item["duration"], left=item["start"], color='dodgerblue', edgecolor='black')
    ax2.set_title('Method 2: Vehicle Routing Solver (Local Search)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Target ID')
    ax2.set_yticks(range(1, num_objects + 1))
    ax2.invert_yaxis()
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Greedy
    for item in gr_schedule:
        ax3.barh(item["id"] + 1, item["duration"], left=item["start"], color='tomato', edgecolor='black')
    ax3.set_title('Method 3: Greedy Duration Heuristic (Sub-Optimal)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (Seconds)')
    ax3.set_ylabel('Target ID')
    ax3.set_yticks(range(1, num_objects + 1))
    ax3.invert_yaxis()
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent
    scenario_path = project_root / 'complex_scenarios' / 'scenario_0001.json'
    
    with open(scenario_path, 'r') as f:
        data = json.load(f)
        
    print("="*60)
    print(f" OPTIMIZATION METHOD COMPARISON (Scenario N={data['n_objects']})")
    print(f" Exact Mathematical Objective: Maximize Sum(d_i) -> Total Duration")
    print("="*60)
    
    cp_score, cp_time, cp_targets, cp_details = solve_with_cpsat(data)
    rt_score, rt_time, rt_targets, rt_details = solve_with_routing(data)
    gr_score, gr_time, gr_targets, gr_details = solve_with_greedy(data)
    
    print(f"{'Method Name':<25} | {'Function Value (Optimum)':<25} | {'Targets Scheduled':<18} | {'Compute Time'}")
    print("-" * 90)
    print(f"{'1. CP-SAT (Global)':<25} | {cp_score:<25} | {cp_targets:<18} | {cp_time:.4f}s")
    print(f"{'2. VRP Routing (Local)':<25} | {rt_score:<25} | {rt_targets:<18} | {rt_time:.4f}s")
    print(f"{'3. Greedy (Heuristic)':<25} | {gr_score:<25} | {gr_targets:<18} | {gr_time:.4f}s")
    print("="*60)
    
    plot_path = project_root / 'solver_comparison_plot.png'
    plot_comparison(cp_details, rt_details, gr_details, data["n_objects"], plot_path)
