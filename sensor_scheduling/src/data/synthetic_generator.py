import json
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
output_dir = project_root / "data" / "processed" / "visibility_windows"
output_dir.mkdir(parents=True, exist_ok=True)

def generate_random_vector():
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    mag = math.sqrt(x*x + y*y + z*z)
    if mag == 0:
        return [1.0, 0.0, 0.0]
    return [x/mag, y/mag, z/mag]

def generate_synthetic_scenario(scenario_id, num_targets):
    base_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    windows = []
    
    for i in range(num_targets):
        # Random start time within 24 hours
        random_seconds = random.randint(0, 86400)
        start_time = base_time + timedelta(seconds=random_seconds)
        duration = random.choice([30.0, 60.0, 90.0, 120.0])
        end_time = start_time + timedelta(seconds=duration)
        
        windows.append({
            "target_name": f"SYNTH-SAT-{i:03d}",
            "start_time_utc": start_time.isoformat(),
            "end_time_utc": end_time.isoformat(),
            "duration_seconds": duration,
            "min_distance_km": round(random.uniform(10.0, 50.0), 2),
            "priority_value": random.randint(1, 100),
            "direction_vector": generate_random_vector()
        })
        
    # Sort chronologically
    windows.sort(key=lambda x: x["start_time_utc"])
    
    file_path = output_dir / f"synthetic_{scenario_id:03d}_windows.json"
    with open(file_path, "w") as f:
        json.dump(windows, f, indent=4)
    print(f"Generated {file_path.name} with {num_targets} targets.")

if __name__ == "__main__":
    print("--- Generating 20 Synthetic Scenarios ---")
    for i in range(1, 21):
        num_targets = random.randint(30, 45) # Average 37, like real life
        generate_synthetic_scenario(i, num_targets)
