import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add the project root to sys.path so we can import the model architecture
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from src.scheduling.ai_scheduler import SensorSchedulerNN

def plot_model_behavior():
    print("Loading model weights...")
    model_path = project_root / "models" / "supervised" / "model_weight.pth"
    
    if not model_path.exists():
        print(f"Error: Could not find model at {model_path}")
        return

    # Initialize model and load weights
    model = SensorSchedulerNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode

    print("Generating predictions...")
    # We will generate 3 plots to show how the AI reacts to different inputs
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('AI Scheduler Decision Making Analysis', fontsize=16)

    # 1. Effect of Slew Required (Fix Duration=0.2, TimeGap=1.0)
    slew_values = np.linspace(0, 1.0, 100)
    predictions_slew = []
    with torch.no_grad():
        for slew in slew_values:
            pred = model(torch.tensor([[0.2, 1.0, float(slew)]], dtype=torch.float32))
            predictions_slew.append(pred.item())
            
    axes[0].plot(slew_values * 50, predictions_slew, color='red', linewidth=2)
    axes[0].set_title('Impact of Slew Time')
    axes[0].set_xlabel('Slew Required (seconds)')
    axes[0].set_ylabel('Probability to Schedule')
    axes[0].grid(True, alpha=0.3)

    # 2. Effect of Time Gap (Fix Duration=0.2, SlewRequired=0.2)
    gap_values = np.linspace(0, 1.0, 100)
    predictions_gap = []
    with torch.no_grad():
        for gap in gap_values:
            pred = model(torch.tensor([[0.2, float(gap), 0.2]], dtype=torch.float32))
            predictions_gap.append(pred.item())
            
    axes[1].plot(gap_values * 100, predictions_gap, color='green', linewidth=2)
    axes[1].set_title('Impact of Time Gap')
    axes[1].set_xlabel('Time Since Last Target (seconds)')
    axes[1].grid(True, alpha=0.3)

    # 3. Effect of Duration (Fix TimeGap=1.0, SlewRequired=0.1)
    duration_values = np.linspace(0, 1.0, 100)
    predictions_duration = []
    with torch.no_grad():
        for duration in duration_values:
            pred = model(torch.tensor([[float(duration), 1.0, 0.1]], dtype=torch.float32))
            predictions_duration.append(pred.item())
            
    axes[2].plot(duration_values * 120, predictions_duration, color='blue', linewidth=2)
    axes[2].set_title('Impact of Task Duration')
    axes[2].set_xlabel('Duration (seconds)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save the plot
    save_path = project_root / "models" / "supervised" / "decision_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully to: {save_path}")
    print("You can open this image to see exactly how your AI thinks!")

if __name__ == "__main__":
    plot_model_behavior()
