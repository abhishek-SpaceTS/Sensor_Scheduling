import math
import requests
import json
from pathlib import Path

# Constants for Orbital Math
MU = 3.986004418e14  # Earth's gravitational parameter (m^3/s^2)
R_EARTH = 6378.137   # Earth's equatorial radius (km)

# API Configuration
URL = "https://celestrak.org/NORAD/elements/gp.php"
PARAMS = {
    "GROUP": "weather",  # Using 'weather' because 'active' and 'starlink' are currently locked by CelesTrak
    "FORMAT": "JSON"
}
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# Directories (Resolved dynamically so it works from anywhere)
script_dir = Path(__file__).parent.resolve()
output_dir = script_dir.parent.parent / "data" / "raw" / "celestrak"
output_dir.mkdir(parents=True, exist_ok=True)

full_catalog_file = output_dir / "active_rso.json"
filtered_file = output_dir / "leo_550_rso.json"


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


def get_active_catalog():
    """Fetches the active catalog from CelesTrak or loads the local cache."""
    print("Attempting to fetch latest active catalog from CelesTrak...")
    
    try:
        response = requests.get(URL, params=PARAMS, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            print("Successfully downloaded new catalog. Saving locally...")
            data = response.json()
            with open(full_catalog_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            return data
            
        elif response.status_code == 403 and "not updated since" in response.text:
            print("Data on CelesTrak hasn't updated since last download.")
            print("Loading local catalog instead...")
            if full_catalog_file.exists():
                with open(full_catalog_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                print("Error: Local file doesn't exist, but CelesTrak blocked download.")
                return []
        else:
            response.raise_for_status()
            
    except Exception as e:
        print(f"Error fetching catalog: {e}")
        return []


def main():
    # 1. Get the full catalog
    catalog = get_active_catalog()
    if not catalog:
        print("Failed to get catalog data.")
        return
        
    print(f"\nTotal active satellites found: {len(catalog)}")
    
    # 2. Filter for LEO 
    target_alt_min = 500.0
    target_alt_max = 600.0
    
    filtered_satellites = []
    
    for sat in catalog:
        # Ensure we have the required orbital data
        if "MEAN_MOTION" not in sat:
            continue
            
        mean_motion = sat["MEAN_MOTION"]
        altitude = calculate_altitude(mean_motion)
        
        # Check if the altitude falls within our 475-625km window
        if target_alt_min <= altitude <= target_alt_max:
            # Inject the calculated altitude into the data for reference
            sat["_CALCULATED_ALTITUDE_KM"] = round(altitude, 2)
            filtered_satellites.append(sat)
            
    print(f"Satellites in {target_alt_min}km - {target_alt_max}km range: {len(filtered_satellites)}")
    
    # 3. Save the filtered result
    with open(filtered_file, "w", encoding="utf-8") as f:
        json.dump(filtered_satellites, f, indent=4)
        
    print(f"\nSaved filtered satellites to:")
    print(filtered_file)


if __name__ == "__main__":
    main()