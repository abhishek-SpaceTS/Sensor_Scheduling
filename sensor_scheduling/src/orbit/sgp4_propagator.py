import math
import json
from pathlib import Path

try:
    from skyfield.api import EarthSatellite, load
except ImportError:
    print("Error: The 'skyfield' library is required. Run: pip install skyfield")
    import sys
    sys.exit(1)


class SpacecraftPropagator:
    """
    Uses the SGP4 algorithm (via the Skyfield library) to propagate 
    satellite orbits and calculate exact positions in space.
    """
    def __init__(self):
        # Load astronomical data required for accurate propagation
        self.ts = load.timescale()

    def load_target_from_json(self, json_omm_data):
        """
        Creates a satellite object directly from CelesTrak JSON data.
        """
        # Skyfield natively supports OMM JSON dictionaries!
        return EarthSatellite.from_omm(self.ts, json_omm_data)

    def create_custom_sensor_satellite(self, sensor_config):
        """
        Our M4V sensor doesn't have a real TLE since we invented it.
        This function takes the custom altitude/inclination from our YAML 
        and generates a mathematical orbit for it to fly in.
        """
        orbit = sensor_config.get('custom_orbit', {})
        alt_km = orbit.get('altitude_km', 550.0)
        
        # Specific Sun-Synchronous inclination
        inc_deg = 97.590    
        
        # Kepler's Third Law to find Mean Motion (Revs per day)
        mu = 398600.4418
        r_earth = 6378.137
        a_km = r_earth + alt_km
        n_rad_sec = math.sqrt(mu / (a_km ** 3))
        n_rev_day = n_rad_sec * 86400.0 / (2 * math.pi)

        # Generate a mock TLE for the M4V
        name = sensor_config.get('name', 'M4V')
        norad_id = 99999
        
        # Line 1: Dummy epoch, zero drag
        line1 = f"1 {norad_id:05d}U 23001A   26277.00000000  .00000000  00000-0  00000-0 0  9999"
        
        # Line 2: Exact orbital elements
        # (RAAN and Eccentricity are just hardcoded to zero here since we don't have them yet)
        line2 = f"2 {norad_id:05d} {inc_deg:8.4f} 000.0000 0000000 000.0000 000.0000 {n_rev_day:11.8f}    15"
        
        return EarthSatellite(line1, line2, name, self.ts)


if __name__ == "__main__":
    print("--- SGP4 Propagator Initialized ---")
    
    # 1. Initialize the propagator
    propagator = SpacecraftPropagator()
    
    # 2. Let's create our M4V sensor using the exact parameters from our YAML
    mock_sensor_config = {
        'name': 'M4V Spacecraft',
        'custom_orbit': {
            'altitude_km': 550.0,

        }
    }
    m4v_sat = propagator.create_custom_sensor_satellite(mock_sensor_config)
    
    # 3. Fast forward time to right now
    now = propagator.ts.now()
    
    # 4. Ask SGP4 where our M4V satellite is right now
    position = m4v_sat.at(now).position.km
    
    print(f"\nCalculated M4V Position (X, Y, Z in km) from Earth Center:")
    print(f"X: {position[0]:.2f} km")
    print(f"Y: {position[1]:.2f} km")
    print(f"Z: {position[2]:.2f} km")
    
    # Distance from center of Earth should be approx 6378 + 550 = 6928 km
    dist = math.sqrt(sum([p**2 for p in position]))
    print(f"Distance from Earth Center: {dist:.2f} km")
    print(f"Calculated Altitude: {dist - 6378.137:.2f} km")
