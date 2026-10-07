import math
import yaml
from pathlib import Path

class SlewModel:
    def __init__(self, config_path=None):
        """
        Initializes the ADCS Slew Model.
        If a config path is provided, it dynamically loads the hardware constraints.
        Otherwise, it defaults to the M4V specs (0.3 deg/s, 10s settling).
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                adcs_config = yaml.safe_load(f)
            self.slew_rate = adcs_config['performance_requirements']['max_slew_rate_deg_per_sec']
            self.settling_time = adcs_config.get('settling_time_sec', 10.0)
        else:
            self.slew_rate = 0.3
            self.settling_time = 10.0

    @staticmethod
    def calculate_angular_distance(vec_a, vec_b):
        """
        Calculates the exact angular distance (in degrees) between two 3D unit vectors.
        """
        # Linear Algebra Dot Product
        dot = vec_a[0]*vec_b[0] + vec_a[1]*vec_b[1] + vec_a[2]*vec_b[2]
        
        # Protect against micro-floating point errors (e.g. 1.00000001)
        dot = max(-1.0, min(1.0, dot))
        
        angle_rad = math.acos(dot)
        return math.degrees(angle_rad)

    def calculate_slew_time(self, vec_a, vec_b):
        """
        Returns the exact number of seconds required to physically turn the camera 
        from vec_a to vec_b and stabilize to pointing accuracy.
        """
        angle_deg = self.calculate_angular_distance(vec_a, vec_b)
        slew_time = (angle_deg / self.slew_rate) + self.settling_time
        return slew_time
