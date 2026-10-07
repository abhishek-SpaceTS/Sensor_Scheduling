import sys
from pathlib import Path
import json

# Add project root to python path
script_dir = Path(__file__).parent.resolve()
sys.path.append(str(script_dir))

# Import pipeline modules
from src.scheduling import ortools_scheduler
from src.scheduling import ai_scheduler

def main():
    print("=====================================================")
    print(" Space Domain Awareness (SDA) Sensor Scheduling Pipeline")
    print("=====================================================\n")
    
    # ---------------------------------------------------------
    # PHASE 1: THE TEACHER (Exact Math Solver)
    # ---------------------------------------------------------
    print(">>> [Phase 1] Starting OR-Tools Teacher Scheduler...")
    scenario_path = script_dir / "data" / "processed" / "visibility_windows" / "scenario_001_windows.json"
    
    if not scenario_path.exists():
        print(f"Error: Could not find {scenario_path}.")
        print("Please ensure visibility windows are generated first.")
        return
        
    with open(scenario_path, 'r') as f:
        windows = json.load(f)
        
    # Generate the perfect Answer Key labels based on the new interval math
    ortools_scheduler.solve_scenario(windows)
    
    print("\n=====================================================\n")
    
    # ---------------------------------------------------------
    # PHASE 2: THE STUDENT (PyTorch Neural Network)
    # ---------------------------------------------------------
    # print(">>> [Phase 2] Starting AI Student Training...")
    # try:
    #     # Train the AI to copy the Teacher's exact math decisions
    #     # ai_scheduler.train_student()
    #     pass
    # except Exception as e:
    #     print(f"Error running AI Scheduler: {e}")
    #     print("Note: Ensure you have PyTorch installed (`pip install torch`).")
        
    print("\n=====================================================")
    print(" Pipeline Complete.")
    print("=====================================================")

if __name__ == "__main__":
    main()
