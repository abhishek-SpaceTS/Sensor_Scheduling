import yaml
from pathlib import Path

class Sensor:
    """
    A dynamic sensor class that loads all its attributes directly from a dictionary.
    This ensures that if new parameters (like tracking speeds, battery limits, etc.) 
    are added to the YAML config in the future, they are automatically supported 
    without needing to rewrite this code.
    """
    def __init__(self, config_dict):
        # Dynamically set all properties from the YAML dictionary
        for key, value in config_dict.items():
            setattr(self, key, value)
            
    def __repr__(self):
        return f"<Sensor {getattr(self, 'id', 'Unknown')} | Type: {getattr(self, 'type', 'Unknown')}>"
        
    def get_param(self, param_name, default=None):
        """Safely retrieve a parameter, returning a default if it doesn't exist."""
        return getattr(self, param_name, default)

def load_sensors(config_path=None):
    """
    Loads all sensors defined in the YAML configuration file dynamically.
    """
    if config_path is None:
        # Default to the config/sensor_config.yaml file relative to this script
        script_dir = Path(__file__).parent.resolve()
        config_path = script_dir.parent.parent / "config" / "sensor_config.yaml"
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return []
        
    sensors = []
    
    # Load any space-based sensors
    if config_data and "space_sensors" in config_data:
        for sensor_data in config_data["space_sensors"]:
            sensors.append(Sensor(sensor_data))
            
    # Load any ground-based sensors (if added in the future)
    if config_data and "ground_sensors" in config_data:
        for sensor_data in config_data["ground_sensors"]:
            sensors.append(Sensor(sensor_data))
            
    return sensors

if __name__ == "__main__":
    # Test loading the sensors to prove it adapts to the YAML dynamically
    loaded_sensors = load_sensors()
    print(f"Successfully loaded {len(loaded_sensors)} sensor(s).\n")
    
    for s in loaded_sensors:
        print(f"--- Sensor: {s.name} ({s.id}) ---")
        # Print out all dynamic attributes that were absorbed from the YAML
        for attr, value in s.__dict__.items():
            print(f"{attr}: {value}")
