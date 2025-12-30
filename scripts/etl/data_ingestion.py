"""
ETL Data Ingestion Pipeline for Urban Farming Optimization System
=================================================================
This script handles:
1. Extraction: Load data from CSV files (Smart Farming, Commodity Prices)
2. Transformation: Normalize, handle missing values, add rice prices, merge data
3. Loading: Insert data into SQL Server database

Author: Data Mining Project
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os

# Optional SQL Server dependencies
try:
    import pyodbc
    from sqlalchemy import create_engine, text
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False
    print("[INFO] SQL Server modules not installed. Run: pip install pyodbc sqlalchemy")
    print("[INFO] Running in CSV-only mode.")

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration settings for the ETL pipeline."""
    
    # File paths
    DATA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    FARMING_DATA_PATH = os.path.join(DATA_DIR, 'data', 'Smart_farming_extended.csv')  # Extended global dataset
    COMMODITY_DATA_PATH = os.path.join(DATA_DIR, 'data', 'commodity_futures.csv')
    OUTPUT_PATH = os.path.join(DATA_DIR, 'data', 'farming_with_prices_extended.csv')
    
    # SQL Server connection settings
    SQL_SERVER = 'localhost'  # Change to your SQL Server instance
    SQL_DATABASE = 'UrbanFarmingDW'
    SQL_DRIVER = 'ODBC Driver 17 for SQL Server'
    SQL_TRUSTED_CONNECTION = True  # Windows Authentication
    # For SQL authentication, set these:
    SQL_USERNAME = None
    SQL_PASSWORD = None
    
    # Rice price simulation parameters (since Rice not in futures data)
    RICE_BASE_PRICE_EUR_PER_KG = 0.45  # Based on FAO average
    RICE_PRICE_VOLATILITY = 0.10  # ±10% variation
    
    # Crop price mapping: Smart Farming crop -> Futures column
    # Crops with None use static 2024 prices from NEW_CROP_PRICES
    CROP_PRICE_MAPPING = {
        'Maize': 'CORN',
        'Wheat': 'WHEAT',
        'Soybean': 'SOYBEANS',
        'Cotton': 'COTTON',
        'Rice': None,      # Will use static price
        'Barley': None,    # New crop - static price
        'Olives': None,    # New crop - static price
        'Citrus': None,    # New crop - static price
        'Tomatoes': None,  # New crop - static price
        'Potatoes': None,  # New crop - static price
    }
    
    # 2024 Market Prices for crops not in commodity futures (EUR/kg)
    # Sources: FAO, Eurostat, EU Fresh Market Data, Morocco Agri Reports
    NEW_CROP_PRICES = {
        'Rice': 0.45,      # FAO global average
        'Barley': 0.14,    # Global price ~$136/ton (FRED/FAO)
        'Olives': 1.50,    # EU table olives producer price €0.90-2.40/kg avg
        'Citrus': 0.85,    # EU oranges avg €0.73-0.93/kg (2024)
        'Tomatoes': 1.45,  # EU average €1.28-1.72/kg (2024)
        'Potatoes': 0.25,  # EU futures/retail €0.06-0.40/kg avg
    }
    
    # Unit conversion factors (to convert to EUR/kg)
    UNIT_CONVERSIONS = {
        'CORN': 25.4,      # 1 bushel = 25.4 kg
        'WHEAT': 27.2,     # 1 bushel = 27.2 kg
        'SOYBEANS': 27.2,  # 1 bushel = 27.2 kg
        'COTTON': 0.453,   # 1 lb = 0.453 kg
    }


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_farming_data(filepath: str) -> pd.DataFrame:
    """
    Extract farming data from CSV file.
    
    Args:
        filepath: Path to the Smart Farming CSV file
        
    Returns:
        DataFrame with farming data
    """
    print(f"[EXTRACT] Loading farming data from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"[EXTRACT] Loaded {len(df)} farming records with {len(df.columns)} columns")
    print(f"[EXTRACT] Columns: {list(df.columns)}")
    
    return df


def extract_commodity_data(filepath: str) -> pd.DataFrame:
    """
    Extract commodity futures data from CSV file.
    
    Args:
        filepath: Path to commodity futures CSV file
        
    Returns:
        DataFrame with commodity price data
    """
    print(f"\n[EXTRACT] Loading commodity data from: {filepath}")
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"[EXTRACT] Loaded {len(df)} commodity records")
    print(f"[EXTRACT] Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Keep only relevant columns
    relevant_cols = ['Date', 'CORN', 'WHEAT', 'SOYBEANS', 'COTTON']
    df_filtered = df[relevant_cols].copy()
    
    print(f"[EXTRACT] Filtered to columns: {relevant_cols}")
    
    return df_filtered


# ============================================================================
# TRANSFORMATION FUNCTIONS
# ============================================================================

def simulate_rice_prices(start_date: str, end_date: str, 
                         base_price: float = 0.45, 
                         volatility: float = 0.10) -> pd.DataFrame:
    """
    Simulate rice prices since they are not in the commodity futures data.
    Uses seasonal variation and random noise to create realistic prices.
    
    Args:
        start_date: Start date for simulation
        end_date: End date for simulation
        base_price: Base price in EUR/kg (~$0.50/kg FAO average)
        volatility: Price volatility as percentage
        
    Returns:
        DataFrame with simulated rice prices
    """
    print(f"\n[TRANSFORM] Simulating rice prices (base: €{base_price}/kg)")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Seasonal variation (higher prices during planting/harvest seasons)
    # Rice prices typically peak in March-April and September-October
    seasonal = volatility * np.sin(2 * np.pi * dates.dayofyear / 365)
    
    # Add random noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, volatility * 0.5, len(dates))
    
    # Calculate prices
    prices = base_price * (1 + seasonal + noise)
    prices = np.clip(prices, base_price * 0.7, base_price * 1.3)  # Limit extreme values
    
    rice_df = pd.DataFrame({
        'Date': dates,
        'RICE_EUR_PER_KG': prices
    })
    
    print(f"[TRANSFORM] Generated {len(rice_df)} rice price records")
    print(f"[TRANSFORM] Rice price range: €{prices.min():.3f} - €{prices.max():.3f}/kg")
    
    return rice_df


def normalize_commodity_prices(df: pd.DataFrame, conversions: dict) -> pd.DataFrame:
    """
    Convert commodity prices to EUR/kg for standardization.
    
    Note: Commodity futures data is in USD cents per bushel (grains) or cents per pound (cotton).
    We need to:
    1. Convert from cents to dollars (divide by 100)
    2. Convert from USD to EUR (multiply by ~0.92)
    3. Convert from bushels/pounds to kg (divide by kg_per_unit)
    
    Args:
        df: Commodity prices DataFrame
        conversions: Dictionary of unit conversion factors (kg per trading unit)
        
    Returns:
        DataFrame with normalized prices in EUR/kg
    """
    print("\n[TRANSFORM] Normalizing commodity prices to EUR/kg")
    
    # 2024 average exchange rate
    USD_TO_EUR = 0.92
    
    df_normalized = df.copy()
    
    for commodity, kg_per_unit in conversions.items():
        if commodity in df_normalized.columns:
            col_name = f'{commodity}_EUR_PER_KG'
            # Futures prices are in cents, so divide by 100 first
            # Then convert USD to EUR, then divide by kg per unit
            df_normalized[col_name] = (
                df_normalized[commodity] / 100  # cents to dollars
                * USD_TO_EUR                    # USD to EUR
                / kg_per_unit                   # per unit to per kg
            )
            print(f"  - {commodity}: (cents/100) * {USD_TO_EUR} EUR/USD / {kg_per_unit} kg = EUR/kg")
    
    return df_normalized


def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: 'interpolate', 'ffill', 'mean', or 'drop'
        
    Returns:
        DataFrame with missing values handled
    """
    print(f"\n[TRANSFORM] Handling missing values (strategy: {strategy})")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        print(f"[TRANSFORM] Columns with missing values:")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count} missing ({100*count/len(df):.1f}%)")
    else:
        print("[TRANSFORM] No missing values found")
        return df
    
    df_clean = df.copy()
    
    if strategy == 'interpolate':
        # For numeric columns, use linear interpolation
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].interpolate(method='linear')
        # Forward fill any remaining NaNs at edges
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()
    elif strategy == 'ffill':
        df_clean = df_clean.ffill().bfill()
    elif strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"[TRANSFORM] Remaining missing values: {remaining_missing}")
    
    return df_clean


def calculate_average_prices(commodity_df: pd.DataFrame, year: int = 2024) -> dict:
    """
    Calculate average prices for a specific year (or use latest available).
    
    Args:
        commodity_df: Normalized commodity prices DataFrame
        year: Target year
        
    Returns:
        Dictionary of average prices per crop
    """
    print(f"\n[TRANSFORM] Calculating average prices for year {year}")
    
    # Filter for the target year (or use 2023 as proxy for 2024)
    df_year = commodity_df[commodity_df['Date'].dt.year == year]
    
    if len(df_year) == 0:
        print(f"[TRANSFORM] No data for {year}, using 2023 as proxy")
        df_year = commodity_df[commodity_df['Date'].dt.year == 2023]
    
    price_cols = [col for col in df_year.columns if col.endswith('_EUR_PER_KG')]
    avg_prices = {}
    
    for col in price_cols:
        crop = col.replace('_EUR_PER_KG', '')
        avg_prices[crop] = df_year[col].mean()
        print(f"  - {crop}: €{avg_prices[crop]:.3f}/kg")
    
    return avg_prices


def merge_farming_with_prices(farming_df: pd.DataFrame, 
                               avg_prices: dict,
                               crop_mapping: dict) -> pd.DataFrame:
    """
    Add price information to farming data based on crop type.
    
    Args:
        farming_df: Farming data DataFrame
        avg_prices: Dictionary of average prices per commodity
        crop_mapping: Mapping from farming crop names to commodity names
        
    Returns:
        DataFrame with price columns added
    """
    print("\n[TRANSFORM] Merging farming data with prices")
    
    df = farming_df.copy()
    
    def get_price(crop_type):
        commodity = crop_mapping.get(crop_type)
        if commodity is None:
            # Use static 2024 prices for crops not in commodity futures
            if crop_type in Config.NEW_CROP_PRICES:
                return Config.NEW_CROP_PRICES[crop_type]
            # Fallback for unknown crops
            return avg_prices.get('RICE', Config.RICE_BASE_PRICE_EUR_PER_KG)
        return avg_prices.get(commodity, 0)
    
    # Add price per kg
    df['price_per_kg_EUR'] = df['crop_type'].apply(get_price)
    
    # Calculate economic metrics
    df['revenue_per_hectare_EUR'] = df['yield_kg_per_hectare'] * df['price_per_kg_EUR']
    
    print(f"[TRANSFORM] Added price columns to {len(df)} records")
    print(f"[TRANSFORM] Price per kg range: €{df['price_per_kg_EUR'].min():.3f} - €{df['price_per_kg_EUR'].max():.3f}")
    print(f"[TRANSFORM] Revenue per hectare range: €{df['revenue_per_hectare_EUR'].min():.2f} - €{df['revenue_per_hectare_EUR'].max():.2f}")
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional derived features useful for analysis.
    
    Args:
        df: Merged farming data DataFrame
        
    Returns:
        DataFrame with additional features
    """
    print("\n[TRANSFORM] Adding derived features")
    
    df = df.copy()
    
    # Efficiency metrics
    df['yield_per_day'] = df['yield_kg_per_hectare'] / df['total_days']
    df['water_efficiency'] = df['yield_kg_per_hectare'] / (df['rainfall_mm'] + 1)  # +1 to avoid division by zero
    df['pesticide_efficiency'] = df['yield_kg_per_hectare'] / (df['pesticide_usage_ml'] + 1)
    
    # Environmental stress indicators
    df['heat_stress'] = (df['temperature_C'] > 32).astype(int)
    df['drought_stress'] = (df['soil_moisture_%'] < 20).astype(int)
    
    # Growing conditions score (0-100)
    df['growing_conditions_score'] = (
        (df['soil_moisture_%'] / 45 * 25) +  # Optimal ~45%
        ((1 - abs(df['soil_pH'] - 6.5) / 1.5) * 25) +  # Optimal pH ~6.5
        ((1 - abs(df['temperature_C'] - 25) / 10) * 25) +  # Optimal temp ~25
        (df['NDVI_index'] * 25)
    ).clip(0, 100)
    
    # Season classification based on sowing month
    def get_season(month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'
    
    df['sowing_season'] = df['sowing_month'].apply(get_season)
    
    # Extract country from region (for map visualizations)
    def extract_country(region):
        if ',' in region:
            return region.split(',')[-1].strip()
        # Handle regions without country suffix
        country_mapping = {
            'North India': 'India',
            'South India': 'India',
            'Punjab, India': 'India',
            'East Africa': 'Kenya',
            'Kenya Highlands': 'Kenya',
            'Nigeria North': 'Nigeria',
            'Nile Delta, Egypt': 'Egypt',
            'Central USA': 'USA',
            'South USA': 'USA',
            'Midwest, USA': 'USA',
            'Central Thailand': 'Thailand',
            'South Africa Cape': 'South Africa',
        }
        return country_mapping.get(region, region)
    
    df['country'] = df['region'].apply(extract_country)
    
    print(f"[TRANSFORM] Added features: yield_per_day, water_efficiency, pesticide_efficiency,")
    print(f"             heat_stress, drought_stress, growing_conditions_score, sowing_season, country")
    
    return df


# ============================================================================
# LOADING FUNCTIONS
# ============================================================================

def get_sql_connection_string(config: Config) -> str:
    """
    Build SQL Server connection string.
    
    Args:
        config: Configuration object
        
    Returns:
        Connection string for SQL Server
    """
    if config.SQL_TRUSTED_CONNECTION:
        conn_str = (
            f"mssql+pyodbc://@{config.SQL_SERVER}/{config.SQL_DATABASE}"
            f"?driver={config.SQL_DRIVER.replace(' ', '+')}"
            f"&Trusted_Connection=yes"
        )
    else:
        conn_str = (
            f"mssql+pyodbc://{config.SQL_USERNAME}:{config.SQL_PASSWORD}"
            f"@{config.SQL_SERVER}/{config.SQL_DATABASE}"
            f"?driver={config.SQL_DRIVER.replace(' ', '+')}"
        )
    return conn_str


def create_database_schema(engine) -> None:
    """
    Create database tables if they don't exist.
    
    Args:
        engine: SQLAlchemy engine
    """
    print("\n[LOAD] Creating database schema...")
    
    # Dimension: Crop
    create_dim_crop = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Dim_Crop' AND xtype='U')
    CREATE TABLE Dim_Crop (
        crop_id INT IDENTITY(1,1) PRIMARY KEY,
        crop_type NVARCHAR(50) NOT NULL UNIQUE,
        price_per_kg_EUR DECIMAL(10,4),
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Dimension: Region (with country for map visualizations)
    create_dim_region = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Dim_Region' AND xtype='U')
    CREATE TABLE Dim_Region (
        region_id INT IDENTITY(1,1) PRIMARY KEY,
        region_name NVARCHAR(100) NOT NULL UNIQUE,
        country NVARCHAR(50),
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Dimension: Time
    create_dim_time = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Dim_Time' AND xtype='U')
    CREATE TABLE Dim_Time (
        time_id INT IDENTITY(1,1) PRIMARY KEY,
        full_date DATE NOT NULL,
        year INT,
        month INT,
        day INT,
        quarter INT,
        month_name NVARCHAR(20),
        season NVARCHAR(20),
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Dimension: Irrigation
    create_dim_irrigation = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Dim_Irrigation' AND xtype='U')
    CREATE TABLE Dim_Irrigation (
        irrigation_id INT IDENTITY(1,1) PRIMARY KEY,
        irrigation_type NVARCHAR(50) NOT NULL UNIQUE,
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Dimension: Disease Status
    create_dim_disease = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Dim_Disease' AND xtype='U')
    CREATE TABLE Dim_Disease (
        disease_id INT IDENTITY(1,1) PRIMARY KEY,
        disease_status NVARCHAR(50) NOT NULL UNIQUE,
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Fact: Farm Production
    create_fact_production = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Fact_Production' AND xtype='U')
    CREATE TABLE Fact_Production (
        production_id INT IDENTITY(1,1) PRIMARY KEY,
        farm_id NVARCHAR(20) NOT NULL,
        crop_id INT FOREIGN KEY REFERENCES Dim_Crop(crop_id),
        region_id INT FOREIGN KEY REFERENCES Dim_Region(region_id),
        irrigation_id INT FOREIGN KEY REFERENCES Dim_Irrigation(irrigation_id),
        disease_id INT FOREIGN KEY REFERENCES Dim_Disease(disease_id),
        sowing_time_id INT FOREIGN KEY REFERENCES Dim_Time(time_id),
        harvest_time_id INT FOREIGN KEY REFERENCES Dim_Time(time_id),
        sowing_date DATE,
        harvest_date DATE,
        total_days INT,
        soil_moisture_pct DECIMAL(5,2),
        soil_pH DECIMAL(4,2),
        temperature_C DECIMAL(5,2),
        rainfall_mm DECIMAL(8,2),
        humidity_pct DECIMAL(5,2),
        sunlight_hours DECIMAL(5,2),
        pesticide_usage_ml DECIMAL(8,2),
        NDVI_index DECIMAL(4,2),
        yield_kg_per_hectare DECIMAL(10,2),
        price_per_kg_EUR DECIMAL(10,4),
        revenue_per_hectare_EUR DECIMAL(12,2),
        yield_per_day DECIMAL(10,4),
        water_efficiency DECIMAL(10,4),
        growing_conditions_score DECIMAL(5,2),
        latitude DECIMAL(10,6),
        longitude DECIMAL(10,6),
        sensor_id NVARCHAR(20),
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    # Fact: Commodity Prices - linked to time dimension for OLAP analysis
    create_fact_prices = """
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Fact_CommodityPrices' AND xtype='U')
    CREATE TABLE Fact_CommodityPrices (
        price_id INT IDENTITY(1,1) PRIMARY KEY,
        time_id INT FOREIGN KEY REFERENCES Dim_Time(time_id),
        price_date DATE NOT NULL,
        corn_eur_per_kg DECIMAL(10,4),
        wheat_eur_per_kg DECIMAL(10,4),
        soybean_eur_per_kg DECIMAL(10,4),
        cotton_eur_per_kg DECIMAL(10,4),
        rice_eur_per_kg DECIMAL(10,4),
        created_at DATETIME DEFAULT GETDATE()
    );
    """
    
    schemas = [
        create_dim_crop, create_dim_region, create_dim_time,
        create_dim_irrigation, create_dim_disease,
        create_fact_production, create_fact_prices
    ]
    
    with engine.begin() as conn:
        for schema in schemas:
            conn.execute(text(schema))
    
    print("[LOAD] Database schema created successfully")


def load_dimension_tables(engine, df: pd.DataFrame) -> dict:
    """
    Load dimension tables and return ID mappings.
    
    Args:
        engine: SQLAlchemy engine
        df: Farming data DataFrame
        
    Returns:
        Dictionary of dimension ID mappings
    """
    print("\n[LOAD] Loading dimension tables...")
    
    mappings = {}
    
    with engine.begin() as conn:
        # Load Dim_Crop
        crops = df[['crop_type', 'price_per_kg_EUR']].groupby('crop_type').mean().reset_index()
        for _, row in crops.iterrows():
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Crop WHERE crop_type = :crop)
                INSERT INTO Dim_Crop (crop_type, price_per_kg_EUR) VALUES (:crop, :price)
            """), {'crop': row['crop_type'], 'price': row['price_per_kg_EUR']})
        
        result = conn.execute(text("SELECT crop_id, crop_type FROM Dim_Crop"))
        mappings['crop'] = {row[1]: row[0] for row in result}
        print(f"  - Loaded {len(mappings['crop'])} crops")
        
        # Load Dim_Region (with country for map visualizations)
        regions = df[['region', 'country']].drop_duplicates()
        for _, row in regions.iterrows():
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Region WHERE region_name = :region)
                INSERT INTO Dim_Region (region_name, country) VALUES (:region, :country)
            """), {'region': row['region'], 'country': row['country']})
        
        result = conn.execute(text("SELECT region_id, region_name FROM Dim_Region"))
        mappings['region'] = {row[1]: row[0] for row in result}
        print(f"  - Loaded {len(mappings['region'])} regions")
        
        # Load Dim_Irrigation
        irrigation_types = df['irrigation_type'].unique()
        for irr_type in irrigation_types:
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Irrigation WHERE irrigation_type = :irr)
                INSERT INTO Dim_Irrigation (irrigation_type) VALUES (:irr)
            """), {'irr': irr_type})
        
        result = conn.execute(text("SELECT irrigation_id, irrigation_type FROM Dim_Irrigation"))
        mappings['irrigation'] = {row[1]: row[0] for row in result}
        print(f"  - Loaded {len(mappings['irrigation'])} irrigation types")
        
        # Load Dim_Disease
        disease_statuses = df['crop_disease_status'].unique()
        for status in disease_statuses:
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Disease WHERE disease_status = :status)
                INSERT INTO Dim_Disease (disease_status) VALUES (:status)
            """), {'status': status})
        
        result = conn.execute(text("SELECT disease_id, disease_status FROM Dim_Disease"))
        mappings['disease'] = {row[1]: row[0] for row in result}
        print(f"  - Loaded {len(mappings['disease'])} disease statuses")
        
        # Load Dim_Time - Generate time dimension from farming data dates
        # Get unique dates from sowing and harvest dates
        sowing_dates = pd.to_datetime(df['sowing_date']).dt.date.unique()
        harvest_dates = pd.to_datetime(df['harvest_date']).dt.date.unique()
        all_dates = set(sowing_dates) | set(harvest_dates)
        
        for date in all_dates:
            if pd.isna(date):
                continue
            date_obj = pd.Timestamp(date)
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_name = month_names[date_obj.month - 1]
            
            # Determine season
            if date_obj.month in [3, 4, 5]:
                season = 'Spring'
            elif date_obj.month in [6, 7, 8]:
                season = 'Summer'
            elif date_obj.month in [9, 10, 11]:
                season = 'Fall'
            else:
                season = 'Winter'
            
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Time WHERE full_date = :full_date)
                INSERT INTO Dim_Time (full_date, year, month, day, quarter, month_name, season)
                VALUES (:full_date, :year, :month, :day, :quarter, :month_name, :season)
            """), {
                'full_date': date,
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'quarter': (date_obj.month - 1) // 3 + 1,
                'month_name': month_name,
                'season': season
            })
        
        result = conn.execute(text("SELECT time_id, full_date FROM Dim_Time"))
        mappings['time'] = {str(row[1]): row[0] for row in result}
        print(f"  - Loaded {len(mappings['time'])} time records")
    
    return mappings


def load_fact_production(engine, df: pd.DataFrame, mappings: dict) -> None:
    """
    Load the fact production table.
    
    Args:
        engine: SQLAlchemy engine
        df: Farming data DataFrame
        mappings: Dimension ID mappings
    """
    print("\n[LOAD] Loading Fact_Production table...")
    
    # Clear existing data (optional - comment out for incremental loads)
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM Fact_Production"))
    
    records = []
    for _, row in df.iterrows():
        # Get time dimension IDs for sowing and harvest dates
        sowing_date_str = str(pd.to_datetime(row['sowing_date']).date())
        harvest_date_str = str(pd.to_datetime(row['harvest_date']).date())
        
        record = {
            'farm_id': row['farm_id'],
            'crop_id': mappings['crop'].get(row['crop_type']),
            'region_id': mappings['region'].get(row['region']),
            'irrigation_id': mappings['irrigation'].get(row['irrigation_type']),
            'disease_id': mappings['disease'].get(row['crop_disease_status']),
            'sowing_time_id': mappings['time'].get(sowing_date_str),
            'harvest_time_id': mappings['time'].get(harvest_date_str),
            'sowing_date': row['sowing_date'],
            'harvest_date': row['harvest_date'],
            'total_days': row['total_days'],
            'soil_moisture_pct': row['soil_moisture_%'],
            'soil_pH': row['soil_pH'],
            'temperature_C': row['temperature_C'],
            'rainfall_mm': row['rainfall_mm'],
            'humidity_pct': row['humidity_%'],
            'sunlight_hours': row['sunlight_hours'],
            'pesticide_usage_ml': row['pesticide_usage_ml'],
            'NDVI_index': row['NDVI_index'],
            'yield_kg_per_hectare': row['yield_kg_per_hectare'],
            'price_per_kg_EUR': row['price_per_kg_EUR'],
            'revenue_per_hectare_EUR': row['revenue_per_hectare_EUR'],
            'yield_per_day': row['yield_per_day'],
            'water_efficiency': row['water_efficiency'],
            'growing_conditions_score': row['growing_conditions_score'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'sensor_id': row['sensor_id']
        }
        records.append(record)
    
    # Bulk insert
    fact_df = pd.DataFrame(records)
    fact_df.to_sql('Fact_Production', engine, if_exists='append', index=False)
    
    print(f"[LOAD] Loaded {len(records)} production records")


def load_fact_prices(engine, commodity_df: pd.DataFrame, rice_df: pd.DataFrame) -> None:
    """
    Load the commodity prices fact table with time dimension linkage.
    
    Args:
        engine: SQLAlchemy engine
        commodity_df: Normalized commodity prices DataFrame
        rice_df: Simulated rice prices DataFrame
    """
    print("\n[LOAD] Loading Fact_CommodityPrices table...")
    
    # Merge commodity and rice prices
    merged = commodity_df.merge(rice_df, on='Date', how='left')
    
    # First, populate Dim_Time with all commodity price dates
    print("  - Populating Dim_Time with commodity price dates...")
    with engine.begin() as conn:
        for date in merged['Date'].unique():
            date_obj = pd.Timestamp(date)
            month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_name = month_names[date_obj.month - 1]
            
            if date_obj.month in [3, 4, 5]:
                season = 'Spring'
            elif date_obj.month in [6, 7, 8]:
                season = 'Summer'
            elif date_obj.month in [9, 10, 11]:
                season = 'Fall'
            else:
                season = 'Winter'
            
            conn.execute(text("""
                IF NOT EXISTS (SELECT 1 FROM Dim_Time WHERE full_date = :full_date)
                INSERT INTO Dim_Time (full_date, year, month, day, quarter, month_name, season)
                VALUES (:full_date, :year, :month, :day, :quarter, :month_name, :season)
            """), {
                'full_date': date_obj.date(),
                'year': date_obj.year,
                'month': date_obj.month,
                'day': date_obj.day,
                'quarter': (date_obj.month - 1) // 3 + 1,
                'month_name': month_name,
                'season': season
            })
    
    # Get time dimension mappings
    with engine.begin() as conn:
        result = conn.execute(text("SELECT time_id, full_date FROM Dim_Time"))
        time_mappings = {str(row[1]): row[0] for row in result}
    print(f"  - Dim_Time now has {len(time_mappings)} records")
    
    # Build prices dataframe with time_id
    prices_df = pd.DataFrame()
    prices_df['time_id'] = merged['Date'].apply(lambda x: time_mappings.get(str(pd.Timestamp(x).date())))
    prices_df['price_date'] = merged['Date']
    prices_df['corn_eur_per_kg'] = merged.get('CORN_EUR_PER_KG')
    prices_df['wheat_eur_per_kg'] = merged.get('WHEAT_EUR_PER_KG')
    prices_df['soybean_eur_per_kg'] = merged.get('SOYBEANS_EUR_PER_KG')
    prices_df['cotton_eur_per_kg'] = merged.get('COTTON_EUR_PER_KG')
    prices_df['rice_eur_per_kg'] = merged.get('RICE_EUR_PER_KG')
    
    # Clear existing data
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM Fact_CommodityPrices"))
    
    prices_df.to_sql('Fact_CommodityPrices', engine, if_exists='append', index=False)
    
    print(f"[LOAD] Loaded {len(prices_df)} price records with time dimension linkage")


def save_to_csv(df: pd.DataFrame, filepath: str) -> None:
    """
    Save the merged dataset to CSV for backup/verification.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
    """
    print(f"\n[EXPORT] Saving merged data to: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"[EXPORT] Saved {len(df)} records")


# ============================================================================
# MAIN ETL PIPELINE
# ============================================================================

def run_etl_pipeline(load_to_sql: bool = True, save_csv: bool = True) -> pd.DataFrame:
    """
    Execute the complete ETL pipeline.
    
    Args:
        load_to_sql: Whether to load data to SQL Server
        save_csv: Whether to save merged data to CSV
        
    Returns:
        Final merged DataFrame
    """
    print("=" * 70)
    print("  URBAN FARMING ETL PIPELINE")
    print("=" * 70)
    print(f"  Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    config = Config()
    
    # =========================================
    # EXTRACTION
    # =========================================
    print("\n" + "=" * 30 + " EXTRACTION " + "=" * 30)
    
    farming_df = extract_farming_data(config.FARMING_DATA_PATH)
    commodity_df = extract_commodity_data(config.COMMODITY_DATA_PATH)
    
    # =========================================
    # TRANSFORMATION
    # =========================================
    print("\n" + "=" * 28 + " TRANSFORMATION " + "=" * 28)
    
    # Handle missing values in commodity data
    commodity_df = handle_missing_values(commodity_df, strategy='interpolate')
    
    # Normalize commodity prices to EUR/kg
    commodity_normalized = normalize_commodity_prices(commodity_df, config.UNIT_CONVERSIONS)
    
    # Simulate rice prices
    rice_df = simulate_rice_prices(
        start_date=commodity_df['Date'].min().strftime('%Y-%m-%d'),
        end_date=commodity_df['Date'].max().strftime('%Y-%m-%d'),
        base_price=config.RICE_BASE_PRICE_EUR_PER_KG,
        volatility=config.RICE_PRICE_VOLATILITY
    )
    
    # Calculate average prices
    avg_prices = calculate_average_prices(commodity_normalized, year=2024)
    avg_prices['RICE'] = rice_df['RICE_EUR_PER_KG'].mean()
    print(f"  - RICE (simulated): €{avg_prices['RICE']:.3f}/kg")
    
    # Merge farming data with prices
    merged_df = merge_farming_with_prices(farming_df, avg_prices, config.CROP_PRICE_MAPPING)
    
    # Add derived features
    final_df = add_derived_features(merged_df)
    
    # =========================================
    # LOADING
    # =========================================
    print("\n" + "=" * 32 + " LOADING " + "=" * 32)
    
    if save_csv:
        save_to_csv(final_df, config.OUTPUT_PATH)
    
    if load_to_sql:
        try:
            conn_str = get_sql_connection_string(config)
            engine = create_engine(conn_str)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print(f"[LOAD] Connected to SQL Server: {config.SQL_SERVER}/{config.SQL_DATABASE}")
            
            # Create schema
            create_database_schema(engine)
            
            # Load dimension tables
            mappings = load_dimension_tables(engine, final_df)
            
            # Load fact tables
            load_fact_production(engine, final_df, mappings)
            load_fact_prices(engine, commodity_normalized, rice_df)
            
            print("\n[LOAD] [SUCCESS] SQL Server load complete!")
            
        except Exception as e:
            print(f"\n[LOAD] [ERROR] SQL Server connection failed: {e}")
            print("[LOAD] Data saved to CSV only. Configure SQL Server settings in Config class.")
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("  ETL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Records Processed: {len(final_df)}")
    print(f"  Columns: {len(final_df.columns)}")
    print(f"  Output File: {config.OUTPUT_PATH}")
    print("=" * 70)
    
    return final_df


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Run the ETL pipeline
    # SQL Server is now configured
    result_df = run_etl_pipeline(load_to_sql=True, save_csv=True)
    
    # Display sample output
    print("\n[SAMPLE] Merged data preview:")
    print(result_df[['farm_id', 'crop_type', 'region', 'yield_kg_per_hectare', 
                     'price_per_kg_EUR', 'revenue_per_hectare_EUR']].head(10))
