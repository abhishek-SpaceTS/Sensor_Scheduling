import json
import random
import math
import sys
from pathlib import Path

# Add project root to sys.path so we can import our SlewModel
script_dir = Path(__file__).parent.resolve()
project_root = script_dir
sys.path.append(str(project_root))

from src.adcs.slew_model import SlewModel

def generate_random_vector():
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    mag = math.sqrt(x*x + y*y + z*z)
    if mag == 0: return [1.0, 0.0, 0.0]
    return [x/mag, y/mag, z/mag]

def check_overlap(windows, threshold=4):
    events = []
    for w in windows:
        # w is a list of windows for this object. e.g. [[start, end]]
        start = w[0][0]
        end = w[0][1]
        events.append((start, 1))
        events.append((end, -1))
        
    events.sort(key=lambda x: (x[0], x[1]))
    
    current_overlap = 0
    for time, change in events:
        current_overlap += change
        if current_overlap >= threshold:
            return True
    return False

def generate_rl_datasets(n=1000):
    out_dir = project_root / "complex_scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the ADCS physics model
    slew_model = SlewModel()
    
    saved_count = 0
    attempts = 0
    
    print(f"Generating {n} JSON Dictionary scenarios...")
    
    while saved_count < n:
        attempts += 1
        num_objects = 20
        
        # Random starting Azimuth (0-360) and Elevation (-90 to 90)
        start_pointing = [round(random.uniform(0, 360), 2), round(random.uniform(-90, 90), 2)]
        
        directions = []
        windows = []
        deadlines = []
        values = []
        
        for _ in range(num_objects):
            start_time = round(random.uniform(0, 5400), 2)
            duration = random.choice([30.0, 60.0, 90.0, 120.0])
            end_time = round(start_time + duration, 2)
            
            directions.append(generate_random_vector())
            windows.append([[start_time, end_time]])
            deadlines.append(end_time)
            values.append(random.randint(1, 100))
            
        if check_overlap(windows, threshold=4):
            # Precompute the massive N x N Slew Times Matrix!
            slew_times = []
            for i in range(num_objects):
                row = []
                for j in range(num_objects):
                    if i == j:
                        row.append(0.0) # 0 seconds to turn to yourself
                    else:
                        time_req = slew_model.calculate_slew_time(directions[i], directions[j])
                        row.append(round(time_req, 2))
                slew_times.append(row)
                
            # Build the exact JSON schema required by the RL Environment
            scenario = {
                "n_objects": num_objects,
                "start_pointing": start_pointing,
                "directions": directions,
                "slew_times": slew_times,
                "windows": windows,
                "deadlines": deadlines,
                "values": values,
                "dwell": 10
            }
            
            saved_count += 1
            file_path = out_dir / f"scenario_{saved_count:04d}.json"
            with open(file_path, "w") as f:
                json.dump(scenario, f, indent=4)
                
            if saved_count % 100 == 0:
                print(f"Progress: {saved_count} / {n} scenarios saved...")
                
    print("\n--- GENERATION COMPLETE ---")
    print(f"Successfully generated {n} JSON dictionary scenarios.")

if __name__ == "__main__":
    n_datasets = 1000
    if len(sys.argv) > 1:
        try:
            n_datasets = int(sys.argv[1])
        except ValueError:
            pass
            
    generate_rl_datasets(n_datasets)
