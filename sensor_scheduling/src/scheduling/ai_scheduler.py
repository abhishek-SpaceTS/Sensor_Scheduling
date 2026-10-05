import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# Get paths
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.parent
labels_path = project_root / "data" / "processed" / "scheduling_dataset" / "scenario_001_labels.json"

class SensorSchedulerNN(nn.Module):
    """
    The Student Neural Network. 
    It takes 3 features: [Priority, Duration, Distance]
    It outputs 1 probability: [Chance of being scheduled (0.0 to 1.0)]
    """
    def __init__(self):
        super(SensorSchedulerNN, self).__init__()
        # 3 input features -> 16 hidden nodes -> 8 hidden nodes -> 1 output
        self.network = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Squishes the output between 0 and 1
        )

    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(json_path):
    """
    Reads the Teacher's JSON file and extracts the Features (X) and Labels (Y).
    Normalizes the features to be between 0 and 1.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    X_features = []
    Y_labels = []
    
    for event in data:
        # Extract features and normalize them
        priority = event["priority_value"] / 100.0          # Max priority is 100
        duration = event["duration_seconds"] / 300.0        # Assume max duration is ~300s
        distance = event["min_distance_km"] / 50.0          # Max distance is 50km
        
        X_features.append([priority, duration, distance])
        
        # The Label is exactly what OR-Tools decided (1 for scheduled, 0 for ignored)
        Y_labels.append([float(event["is_scheduled"])])
        
    # Convert lists to PyTorch Tensors
    X_tensor = torch.tensor(X_features, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_labels, dtype=torch.float32)
    
    return X_tensor, Y_tensor

def train_student():
    print("--- Training the AI Student ---")
    
    if not labels_path.exists():
        print(f"Error: {labels_path.name} not found! Run the OR-Tools Teacher first.")
        return
        
    # 1. Load Data
    X, Y = load_and_preprocess_data(labels_path)
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Initialize the AI
    model = SensorSchedulerNN()
    
    # 3. Define the Teacher's grading tools
    # Binary Cross Entropy Loss (Perfect for Yes/No scheduling)
    criterion = nn.BCELoss()
    # Adam Optimizer helps the AI learn from its mistakes
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Train loop (Let it practice 50 times)
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            
            # The AI makes a guess
            predictions = model(batch_X)
            
            # The Teacher grades the guess against the true labels
            loss = criterion(predictions, batch_Y)
            total_loss += loss.item()
            
            # The AI learns from its mistakes
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")
            
    print("\nTraining Complete! The AI has learned the Teacher's pattern.")
    
    # 5. Let's test the AI on a fake target
    print("\n--- AI Final Exam ---")
    # High Priority (95), Long Duration (120s), Very Close (10km)
    perfect_target = torch.tensor([[0.95, 120.0/300.0, 10.0/50.0]])
    prediction = model(perfect_target)
    print(f"AI Probability to schedule a Perfect Target: {prediction.item():.2%}")
    
    # Low Priority (10), Short Duration (30s), Far Away (49km)
    garbage_target = torch.tensor([[0.10, 30.0/300.0, 49.0/50.0]])
    prediction = model(garbage_target)
    print(f"AI Probability to schedule a Garbage Target: {prediction.item():.2%}")

if __name__ == "__main__":
    train_student()
