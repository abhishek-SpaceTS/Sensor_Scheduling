import math
from pathlib import Path

# Constants for Orbital Math
MU = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6378.137   # Earth's equatorial radius (km)

def calculate_altitude(mean_motion):
    """Calculates approximate average altitude in km from Mean Motion (revs/day)."""
    # Convert revs/day to radians/second
    n_rad_sec = mean_motion * (2 * math.pi) / 86400.0
    
    # Calculate semi-major axis (meters) using Kepler's Third Law
    a_meters = (MU / (n_rad_sec ** 2)) ** (1/3)
    
    # Convert to km
    a_km = a_meters / 1000.0
    
    # Calculate average altitude by subtracting Earth's radius
    avg_altitude = a_km - R_EARTH
    return avg_altitude

def filter_tle_file():
    # Resolve directories
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir.parent.parent / "data" / "raw" / "celestrak"
    
    input_tle = data_dir / "active_rso.tle"
    output_tle = data_dir / "leo_550_rso.tle"
    
    if not input_tle.exists():
        print(f"Error: Could not find {input_tle}")
        return
        
    print(f"Reading TLE file: {input_tle}")
    
    # Our camera bounds (550km spacecraft +/- 50km camera range)
    target_alt_min = 500.0
    target_alt_max = 600.0
    
    with open(input_tle, "r", encoding="utf-8") as f:
        # Read all non-empty lines
        lines = [line.strip() for line in f.readlines() if line.strip()]
        
    filtered_lines = []
    
    # CelesTrak TLEs always come in chunks of 3 lines: Name, Line 1, Line 2
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
            
        name = lines[i]
        line1 = lines[i+1]
        line2 = lines[i+2]
        
        try:
            # The "Mean Motion" value in a standard TLE is on Line 2, characters 52 to 63
            mean_motion_str = line2[52:63].strip()
            mean_motion = float(mean_motion_str)
            
            # Calculate altitude
            altitude = calculate_altitude(mean_motion)
            
            # Filter based on our camera range
            if target_alt_min <= altitude <= target_alt_max:
                filtered_lines.extend([name, line1, line2])
        except Exception as e:
            continue
            
    # Save the filtered results to a new TLE file
    with open(output_tle, "w", encoding="utf-8") as f:
        for line in filtered_lines:
            f.write(line + "\n")
            
    print(f"Total satellites in original TLE: {len(lines)//3}")
    print(f"Satellites in camera range ({target_alt_min}km - {target_alt_max}km): {len(filtered_lines)//3}")
    print(f"\nSaved filtered targets to:\n{output_tle}")

if __name__ == "__main__":
    filter_tle_file()