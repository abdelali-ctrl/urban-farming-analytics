"""
Data Generator for Urban Farming Optimization System
Generates 4000+ additional farming records with global regions including Morocco
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# REGION CONFIGURATIONS
# ============================================================================

REGIONS = {
    # Morocco regions (more emphasis - 30% of data)
    "Casablanca-Settat, Morocco": {
        "lat_range": (32.5, 33.8), "lon_range": (-8.5, -7.0),
        "climate": "mediterranean", "temp_range": (15, 32), "rainfall_range": (200, 500)
    },
    "Marrakech-Safi, Morocco": {
        "lat_range": (31.0, 32.5), "lon_range": (-9.5, -7.5),
        "climate": "semi_arid", "temp_range": (18, 38), "rainfall_range": (150, 350)
    },
    "Fès-Meknès, Morocco": {
        "lat_range": (33.5, 34.5), "lon_range": (-5.5, -4.0),
        "climate": "mediterranean", "temp_range": (12, 35), "rainfall_range": (300, 600)
    },
    "Souss-Massa, Morocco": {
        "lat_range": (29.5, 31.0), "lon_range": (-10.0, -8.5),
        "climate": "semi_arid", "temp_range": (15, 35), "rainfall_range": (100, 300)
    },
    "Rabat-Salé-Kénitra, Morocco": {
        "lat_range": (33.5, 34.5), "lon_range": (-7.0, -6.0),
        "climate": "mediterranean", "temp_range": (14, 30), "rainfall_range": (400, 600)
    },
    "Tanger-Tétouan, Morocco": {
        "lat_range": (35.0, 36.0), "lon_range": (-6.0, -5.0),
        "climate": "mediterranean", "temp_range": (12, 28), "rainfall_range": (500, 900)
    },
    
    # Europe (15% of data)
    "Andalusia, Spain": {
        "lat_range": (36.5, 38.5), "lon_range": (-6.5, -2.5),
        "climate": "mediterranean", "temp_range": (12, 38), "rainfall_range": (200, 600)
    },
    "Provence, France": {
        "lat_range": (43.0, 44.5), "lon_range": (4.5, 7.0),
        "climate": "mediterranean", "temp_range": (8, 32), "rainfall_range": (400, 800)
    },
    "Tuscany, Italy": {
        "lat_range": (42.5, 44.0), "lon_range": (10.5, 12.5),
        "climate": "mediterranean", "temp_range": (6, 32), "rainfall_range": (600, 1000)
    },
    "Bavaria, Germany": {
        "lat_range": (47.5, 49.5), "lon_range": (10.5, 13.5),
        "climate": "continental", "temp_range": (2, 28), "rainfall_range": (600, 1200)
    },
    
    # Americas (15% of data)
    "California, USA": {
        "lat_range": (34.0, 38.5), "lon_range": (-122.5, -118.0),
        "climate": "mediterranean", "temp_range": (10, 35), "rainfall_range": (200, 800)
    },
    "Midwest, USA": {
        "lat_range": (39.0, 43.0), "lon_range": (-95.0, -87.0),
        "climate": "continental", "temp_range": (-5, 32), "rainfall_range": (600, 1100)
    },
    "Minas Gerais, Brazil": {
        "lat_range": (-20.0, -17.5), "lon_range": (-46.0, -42.5),
        "climate": "tropical", "temp_range": (18, 32), "rainfall_range": (1000, 1800)
    },
    "Pampas, Argentina": {
        "lat_range": (-38.0, -34.0), "lon_range": (-63.0, -58.0),
        "climate": "temperate", "temp_range": (5, 30), "rainfall_range": (600, 1000)
    },
    
    # Africa (15% of data)
    "Nile Delta, Egypt": {
        "lat_range": (30.0, 31.5), "lon_range": (30.0, 32.5),
        "climate": "arid", "temp_range": (15, 40), "rainfall_range": (20, 100)
    },
    "Kenya Highlands": {
        "lat_range": (-1.5, 1.0), "lon_range": (36.0, 38.0),
        "climate": "tropical_highland", "temp_range": (12, 28), "rainfall_range": (800, 1500)
    },
    "South Africa Cape": {
        "lat_range": (-34.5, -33.0), "lon_range": (18.0, 20.0),
        "climate": "mediterranean", "temp_range": (10, 30), "rainfall_range": (400, 800)
    },
    "Nigeria North": {
        "lat_range": (10.0, 13.0), "lon_range": (7.0, 12.0),
        "climate": "savanna", "temp_range": (22, 40), "rainfall_range": (500, 1000)
    },
    
    # Asia (15% of data)
    "Punjab, India": {
        "lat_range": (29.5, 32.0), "lon_range": (74.0, 77.0),
        "climate": "semi_arid", "temp_range": (10, 42), "rainfall_range": (300, 800)
    },
    "Shandong, China": {
        "lat_range": (34.5, 38.0), "lon_range": (115.0, 122.0),
        "climate": "continental", "temp_range": (0, 32), "rainfall_range": (500, 900)
    },
    "Central Thailand": {
        "lat_range": (13.5, 16.0), "lon_range": (99.5, 101.5),
        "climate": "tropical", "temp_range": (24, 36), "rainfall_range": (1000, 1600)
    },
    "Java, Indonesia": {
        "lat_range": (-8.0, -6.0), "lon_range": (106.0, 114.0),
        "climate": "tropical", "temp_range": (24, 34), "rainfall_range": (1500, 3000)
    },
    
    # Australia & Oceania (5% of data)
    "Queensland, Australia": {
        "lat_range": (-27.0, -20.0), "lon_range": (145.0, 153.0),
        "climate": "subtropical", "temp_range": (15, 35), "rainfall_range": (400, 1200)
    },
    "Murray-Darling, Australia": {
        "lat_range": (-36.0, -32.0), "lon_range": (140.0, 148.0),
        "climate": "semi_arid", "temp_range": (8, 38), "rainfall_range": (200, 500)
    },
    
    # Original regions (keep some for compatibility - 5%)
    "North India": {
        "lat_range": (26.0, 32.0), "lon_range": (74.0, 88.0),
        "climate": "continental", "temp_range": (10, 40), "rainfall_range": (300, 1000)
    },
    "South India": {
        "lat_range": (10.0, 18.0), "lon_range": (74.0, 82.0),
        "climate": "tropical", "temp_range": (22, 38), "rainfall_range": (600, 2000)
    },
    "East Africa": {
        "lat_range": (-5.0, 5.0), "lon_range": (32.0, 42.0),
        "climate": "tropical", "temp_range": (18, 32), "rainfall_range": (500, 1500)
    },
}

# Region distribution weights (30% Morocco, rest distributed)
REGION_WEIGHTS = {
    # Morocco - 30%
    "Casablanca-Settat, Morocco": 0.08,
    "Marrakech-Safi, Morocco": 0.06,
    "Fès-Meknès, Morocco": 0.05,
    "Souss-Massa, Morocco": 0.05,
    "Rabat-Salé-Kénitra, Morocco": 0.04,
    "Tanger-Tétouan, Morocco": 0.02,
    
    # Europe - 15%
    "Andalusia, Spain": 0.05,
    "Provence, France": 0.04,
    "Tuscany, Italy": 0.03,
    "Bavaria, Germany": 0.03,
    
    # Americas - 15%
    "California, USA": 0.05,
    "Midwest, USA": 0.04,
    "Minas Gerais, Brazil": 0.03,
    "Pampas, Argentina": 0.03,
    
    # Africa - 15%
    "Nile Delta, Egypt": 0.04,
    "Kenya Highlands": 0.04,
    "South Africa Cape": 0.04,
    "Nigeria North": 0.03,
    
    # Asia - 15%
    "Punjab, India": 0.04,
    "Shandong, China": 0.04,
    "Central Thailand": 0.04,
    "Java, Indonesia": 0.03,
    
    # Australia - 5%
    "Queensland, Australia": 0.03,
    "Murray-Darling, Australia": 0.02,
    
    # Original - 5%
    "North India": 0.02,
    "South India": 0.02,
    "East Africa": 0.01,
}

# Crop configurations
CROPS = {
    "Wheat": {"growing_days": (90, 150), "optimal_temp": (15, 24), "water_needs": "medium"},
    "Maize": {"growing_days": (90, 140), "optimal_temp": (20, 30), "water_needs": "high"},
    "Rice": {"growing_days": (100, 150), "optimal_temp": (22, 32), "water_needs": "very_high"},
    "Soybean": {"growing_days": (80, 120), "optimal_temp": (20, 30), "water_needs": "medium"},
    "Cotton": {"growing_days": (150, 180), "optimal_temp": (22, 35), "water_needs": "medium"},
    "Barley": {"growing_days": (80, 120), "optimal_temp": (12, 22), "water_needs": "low"},
    "Olives": {"growing_days": (180, 240), "optimal_temp": (15, 30), "water_needs": "low"},
    "Tomatoes": {"growing_days": (60, 100), "optimal_temp": (18, 27), "water_needs": "high"},
    "Potatoes": {"growing_days": (70, 120), "optimal_temp": (15, 22), "water_needs": "medium"},
    "Citrus": {"growing_days": (200, 300), "optimal_temp": (15, 30), "water_needs": "medium"},
}

IRRIGATION_TYPES = ["Drip", "Sprinkler", "Manual", "No Irrigation", "Flood", "Center Pivot"]
FERTILIZER_TYPES = ["Organic", "Inorganic", "Mixed", "Biofertilizer"]
DISEASE_STATUS = ["No Disease", "Mild", "Moderate", "Severe"]


def generate_farm_id(start_num: int) -> str:
    return f"FARM{start_num:04d}"


def generate_sensor_id(start_num: int) -> str:
    return f"SENS{start_num:04d}"


def get_region_for_farm() -> str:
    """Select region based on weights"""
    regions = list(REGION_WEIGHTS.keys())
    weights = list(REGION_WEIGHTS.values())
    return random.choices(regions, weights=weights, k=1)[0]


def generate_coordinates(region: str) -> tuple:
    """Generate lat/lon within region bounds"""
    config = REGIONS[region]
    lat = np.random.uniform(config["lat_range"][0], config["lat_range"][1])
    lon = np.random.uniform(config["lon_range"][0], config["lon_range"][1])
    return round(lat, 6), round(lon, 6)


def get_crop_for_region(region: str) -> str:
    """Select appropriate crop for region's climate"""
    climate = REGIONS[region]["climate"]
    
    if climate in ["mediterranean", "semi_arid"]:
        crops = ["Wheat", "Barley", "Olives", "Cotton", "Citrus", "Tomatoes"]
    elif climate in ["tropical", "tropical_highland"]:
        crops = ["Rice", "Maize", "Cotton", "Soybean", "Tomatoes"]
    elif climate in ["continental", "temperate"]:
        crops = ["Wheat", "Maize", "Soybean", "Barley", "Potatoes"]
    elif climate == "arid":
        crops = ["Wheat", "Cotton", "Barley", "Citrus"]
    elif climate == "savanna":
        crops = ["Maize", "Cotton", "Soybean", "Rice"]
    else:
        crops = list(CROPS.keys())
    
    return random.choice(crops)


def generate_environmental_data(region: str, crop: str) -> dict:
    """Generate environmental metrics based on region and crop"""
    config = REGIONS[region]
    crop_config = CROPS[crop]
    
    # Temperature based on region
    temp = np.random.uniform(config["temp_range"][0], config["temp_range"][1])
    
    # Rainfall based on region
    rainfall = np.random.uniform(config["rainfall_range"][0], config["rainfall_range"][1])
    
    # Soil metrics
    soil_moisture = np.random.uniform(10, 45)
    soil_ph = np.random.uniform(5.5, 7.5)
    
    # Adjust for climate
    if config["climate"] == "arid":
        soil_moisture = np.random.uniform(8, 25)
        rainfall = min(rainfall, 150)
    elif config["climate"] in ["tropical", "tropical_highland"]:
        soil_moisture = np.random.uniform(25, 45)
    
    humidity = np.random.uniform(40, 90)
    sunlight = np.random.uniform(4, 10)
    
    return {
        "temperature_C": round(temp, 2),
        "rainfall_mm": round(rainfall, 2),
        "soil_moisture_%": round(soil_moisture, 2),
        "soil_pH": round(soil_ph, 2),
        "humidity_%": round(humidity, 2),
        "sunlight_hours": round(sunlight, 2),
    }


def generate_dates() -> tuple:
    """Generate sowing and harvest dates"""
    # Sowing in 2024
    sowing_month = random.randint(1, 4)  # Jan-Apr typical sowing
    sowing_day = random.randint(1, 28)
    sowing_date = datetime(2024, sowing_month, sowing_day)
    
    # Growing period 90-180 days
    growing_days = random.randint(90, 180)
    harvest_date = sowing_date + timedelta(days=growing_days)
    
    return (
        sowing_date.strftime("%Y-%m-%d"),
        harvest_date.strftime("%Y-%m-%d"),
        growing_days,
        sowing_month,
        harvest_date.month
    )


def calculate_yield(crop: str, env_data: dict, disease: str, irrigation: str) -> float:
    """Calculate yield based on conditions"""
    crop_config = CROPS[crop]
    
    # Base yield (2000-6000 kg/ha)
    base_yield = np.random.uniform(2000, 6000)
    
    # Temperature impact
    optimal_temp = crop_config["optimal_temp"]
    temp = env_data["temperature_C"]
    if optimal_temp[0] <= temp <= optimal_temp[1]:
        temp_factor = 1.0
    else:
        temp_factor = 0.7 + 0.3 * max(0, 1 - abs(temp - sum(optimal_temp)/2) / 15)
    
    # Water impact
    water_needs = crop_config["water_needs"]
    moisture = env_data["soil_moisture_%"]
    if water_needs == "very_high":
        water_factor = min(1.0, moisture / 35)
    elif water_needs == "high":
        water_factor = min(1.0, moisture / 30)
    elif water_needs == "medium":
        water_factor = min(1.0, moisture / 25)
    else:
        water_factor = min(1.0, moisture / 15)
    
    # Disease impact
    disease_factors = {"No Disease": 1.0, "Mild": 0.85, "Moderate": 0.7, "Severe": 0.5}
    disease_factor = disease_factors.get(disease, 0.8)
    
    # Irrigation impact
    if irrigation in ["Drip", "Center Pivot"]:
        irr_factor = 1.1
    elif irrigation == "Sprinkler":
        irr_factor = 1.05
    elif irrigation == "Flood":
        irr_factor = 0.95
    else:
        irr_factor = 0.9
    
    yield_kg = base_yield * temp_factor * water_factor * disease_factor * irr_factor
    return round(max(1500, min(6500, yield_kg)), 2)


def generate_record(farm_num: int) -> dict:
    """Generate a single farm record"""
    region = get_region_for_farm()
    crop = get_crop_for_region(region)
    lat, lon = generate_coordinates(region)
    env_data = generate_environmental_data(region, crop)
    
    sowing_date, harvest_date, total_days, sowing_month, harvest_month = generate_dates()
    
    irrigation = random.choice(IRRIGATION_TYPES)
    fertilizer = random.choice(FERTILIZER_TYPES)
    disease = random.choices(
        DISEASE_STATUS, 
        weights=[0.45, 0.25, 0.18, 0.12]
    )[0]
    
    pesticide = round(np.random.uniform(5, 50), 2)
    ndvi = round(np.random.uniform(0.3, 0.9), 2)
    
    yield_kg = calculate_yield(crop, env_data, disease, irrigation)
    
    # Timestamp during growing period
    timestamp_date = datetime.strptime(sowing_date, "%Y-%m-%d") + timedelta(
        days=random.randint(30, min(total_days - 10, 120))
    )
    
    return {
        "farm_id": generate_farm_id(farm_num),
        "region": region,
        "crop_type": crop,
        "soil_moisture_%": env_data["soil_moisture_%"],
        "soil_pH": env_data["soil_pH"],
        "temperature_C": env_data["temperature_C"],
        "rainfall_mm": env_data["rainfall_mm"],
        "humidity_%": env_data["humidity_%"],
        "sunlight_hours": env_data["sunlight_hours"],
        "irrigation_type": irrigation,
        "fertilizer_type": fertilizer,
        "pesticide_usage_ml": pesticide,
        "sowing_date": sowing_date,
        "harvest_date": harvest_date,
        "total_days": total_days,
        "yield_kg_per_hectare": yield_kg,
        "sensor_id": generate_sensor_id(farm_num),
        "timestamp": timestamp_date.strftime("%Y-%m-%d"),
        "latitude": lat,
        "longitude": lon,
        "NDVI_index": ndvi,
        "crop_disease_status": disease,
        "sowing_month": sowing_month,
        "harvest_month": harvest_month,
    }


def main():
    print("=" * 60)
    print("  URBAN FARMING DATA GENERATOR")
    print("  Generating 4000 records with global regions")
    print("=" * 60)
    
    # Load existing data
    existing_path = "data/Smart_farming_post_processing.csv"
    try:
        existing_df = pd.read_csv(existing_path)
        start_num = len(existing_df) + 1
        print(f"\n[INFO] Found {len(existing_df)} existing records")
    except FileNotFoundError:
        existing_df = pd.DataFrame()
        start_num = 1
        print("\n[INFO] No existing data found, starting fresh")
    
    # Generate new records
    num_records = 4000
    print(f"[INFO] Generating {num_records} new records...")
    
    records = []
    for i in range(num_records):
        record = generate_record(start_num + i)
        records.append(record)
        
        if (i + 1) % 500 == 0:
            print(f"  - Generated {i + 1} / {num_records} records...")
    
    new_df = pd.DataFrame(records)
    
    # Combine with existing
    if len(existing_df) > 0:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Save results
    output_path = "data/Smart_farming_extended.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Saved {len(combined_df)} total records to {output_path}")
    
    # Show region distribution
    print("\n[INFO] Region Distribution in New Data:")
    region_counts = new_df['region'].value_counts()
    for region, count in region_counts.head(15).items():
        print(f"  - {region}: {count} ({100*count/len(new_df):.1f}%)")
    
    # Morocco summary
    morocco_regions = [r for r in new_df['region'].unique() if 'Morocco' in r]
    morocco_count = new_df[new_df['region'].isin(morocco_regions)].shape[0]
    print(f"\n[INFO] Total Morocco records: {morocco_count} ({100*morocco_count/len(new_df):.1f}%)")
    
    print("\n[DONE] Data generation complete!")
    return output_path


if __name__ == "__main__":
    main()
