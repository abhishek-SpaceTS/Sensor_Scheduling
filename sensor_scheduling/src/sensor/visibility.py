import sys
import json
import math
import random
from pathlib import Path
from datetime import timedelta

# Add the project root to the Python path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from src.sensor.sensor_model import load_sensors
from src.orbit.sgp4_propagator import SpacecraftPropagator
from skyfield.api import EarthSatellite

def load_random_subset_targets(ts, tle_path, num_targets=20):
    """Loads a completely randomized benchmark subset of targets from the full TLE file."""
    targets = []
    try:
        with open(tle_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            
        # Group the massive file into chunks of 3 lines (Name, Line 1, Line 2)
        tle_chunks = []
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
                tle_chunks.append((lines[i], lines[i+1], lines[i+2]))
                
        # Randomly pull exactly 'num_targets' out of the 1,624 pool
        sampled_chunks = random.sample(tle_chunks, min(num_targets, len(tle_chunks)))
            
        for name, line1, line2 in sampled_chunks:
            sat = EarthSatellite(line1, line2, name, ts)
            targets.append(sat)
            
    except FileNotFoundError:
        print(f"Error: Could not find target file at {tle_path}")
    return targets

def generate_scenario(scenario_id, propagator, m4v_sat, max_range_km, tle_path):
    """Generates a unique scenario with random targets and a random start time."""
    output_path = project_root / "data" / "processed" / "visibility_windows" / f"scenario_{scenario_id:03d}_windows.json"
    print(f"\n=== Generating Scenario {scenario_id:03d} ===")
    
    # 1. Grab all 1624 satellites!
    targets = load_random_subset_targets(propagator.ts, tle_path, num_targets=1624)
    if not targets:
        return
    print(f"Selected all 1,624 targets from the pool.")
    
    # 2. Pick a random start time within the next 30 days
    random_days_ahead = random.uniform(0, 30)
    start_time = propagator.ts.now() + timedelta(days=random_days_ahead)
    
    # 3. Simulate a 24-hour window stepping every 30 seconds (2880 steps)
    time_steps = [start_time + timedelta(seconds=i*30) for i in range(2880)]
    
    print(f"Start Time: {start_time.utc_datetime().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Simulating full 24-hour window (This will take about 6 minutes)...")
    
    active_windows = {}
    completed_windows = []
    
    # 4. Run the Physics Engine
    for t in time_steps:
        m4v_pos = m4v_sat.at(t).position.km
        current_seen = set()
        
        for target in targets:
            target_pos = target.at(t).position.km
            distance_km = math.sqrt(
                (m4v_pos[0] - target_pos[0])**2 + 
                (m4v_pos[1] - target_pos[1])**2 + 
                (m4v_pos[2] - target_pos[2])**2
            )
            
            if distance_km <= max_range_km:
                name = target.name
                current_seen.add(name)
                
                dx = target_pos[0] - m4v_pos[0]
                dy = target_pos[1] - m4v_pos[1]
                dz = target_pos[2] - m4v_pos[2]
                mag = max(distance_km, 0.0001) # prevent div by zero
                dir_vec = [dx/mag, dy/mag, dz/mag]
                
                # If we just saw this target for the first time
                if name not in active_windows:
                    active_windows[name] = {
                        "target_name": name,
                        "start_time": t,
                        "end_time": t,
                        "min_distance_km": distance_km,
                        "priority_value": random.randint(1, 100),
                        "direction_vector": dir_vec
                    }
                    print(f"  [ACCESS START] {name} entered range at {distance_km:.2f} km")
                else:
                    # We are still seeing it, just update the end time and track the closest approach
                    active_windows[name]["end_time"] = t
                    if distance_km < active_windows[name]["min_distance_km"]:
                        active_windows[name]["min_distance_km"] = distance_km
                        active_windows[name]["direction_vector"] = dir_vec
                        
        # Check if any targets left the 50km bubble
        for name in list(active_windows.keys()):
            if name not in current_seen:
                w = active_windows.pop(name)
                duration = (w["end_time"].utc_datetime() - w["start_time"].utc_datetime()).total_seconds() + 30.0
                
                completed_windows.append({
                    "target_name": w["target_name"],
                    "start_time_utc": w["start_time"].utc_datetime().isoformat(),
                    "end_time_utc": (w["end_time"].utc_datetime() + timedelta(seconds=30)).isoformat(),
                    "duration_seconds": duration,
                    "min_distance_km": round(w["min_distance_km"], 2),
                    "priority_value": w["priority_value"],
                    "direction_vector": w["direction_vector"]
                })
                print(f"  [ACCESS END] {name} left range. Total duration: {duration}s")
                
    # Close any windows that were still active when the simulation ended
    for name, w in active_windows.items():
        duration = (w["end_time"].utc_datetime() - w["start_time"].utc_datetime()).total_seconds() + 30.0
        completed_windows.append({
            "target_name": w["target_name"],
            "start_time_utc": w["start_time"].utc_datetime().isoformat(),
            "end_time_utc": (w["end_time"].utc_datetime() + timedelta(seconds=30)).isoformat(),
            "duration_seconds": duration,
            "min_distance_km": round(w["min_distance_km"], 2),
            "priority_value": w["priority_value"],
            "direction_vector": w["direction_vector"]
        })
        
    # 5. Save results
    if completed_windows:
        # Sort them chronologically
        completed_windows.sort(key=lambda x: x["start_time_utc"])
        with open(output_path, "w") as f:
            json.dump(completed_windows, f, indent=4)
        print(f"-> Saved {len(completed_windows)} consolidated windows to {output_path.name}")
    else:
        print(f"-> No intercepts found in this random scenario.")

def main():
    print("--- Starting Scenario Generator ---")
    
    tle_path = project_root / "data" / "raw" / "celestrak" / "leo_550_rso.tle"
    
    sensors = load_sensors()
    if not sensors:
        return
    m4v_config = sensors[0]
    max_range_km = m4v_config.get_param('max_tracking_range_km', 50.0)
    
    propagator = SpacecraftPropagator()
    m4v_sat = propagator.create_custom_sensor_satellite(m4v_config.__dict__)
    
    # Let's auto-generate 1 completely unique scenario (with 200 targets) as a test
    for i in range(1, 2):
        generate_scenario(i, propagator, m4v_sat, max_range_km, tle_path)

if __name__ == "__main__":
    main()
