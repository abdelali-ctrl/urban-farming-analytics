import pandas as pd
import numpy as np
import os
from config import Config

class DataLoader:
    def __init__(self):
        self.config = Config()
        self.df_smart_farming = None
        self.df_commodity = None
        self.df_fruit = None
        self.df_vegetable = None
    
    def load_smart_farming_data(self):
        """Load smart farming dataset"""
        try:
            df = pd.read_csv(self.config.SMART_FARMING_DATA)
            
            # Convert date columns if they exist
            date_cols = ['sowing_date', 'harvest_date', 'timestamp']
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Fill categorical missing values
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            self.df_smart_farming = df
            print(f"Loaded Smart Farming data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            print(f"Error loading Smart Farming data: {e}")
            # Create sample data if file not found
            return self._create_sample_data()
    
    def load_commodity_data(self):
        """Load commodity futures data"""
        try:
            df = pd.read_csv(self.config.COMMODITY_DATA)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            self.df_commodity = df
            print(f"Loaded Commodity data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading Commodity data: {e}")
            return pd.DataFrame()
    
    def load_fruit_data(self):
        """Load fruit prices data"""
        try:
            df = pd.read_csv(self.config.FRUIT_DATA)
            self.df_fruit = df
            print(f"Loaded Fruit data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading Fruit data: {e}")
            return pd.DataFrame()
    
    def load_vegetable_data(self):
        """Load vegetable prices data"""
        try:
            df = pd.read_csv(self.config.VEGETABLE_DATA)
            self.df_vegetable = df
            print(f"Loaded Vegetable data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading Vegetable data: {e}")
            return pd.DataFrame()
    
    def get_comprehensive_stats(self, df):
        """Get comprehensive statistics for the dataset"""
        stats = {
            'dataset_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'date_range': {}
            },
            'yield_statistics': {
                'mean': round(df['yield_kg_per_hectare'].mean(), 2),
                'median': round(df['yield_kg_per_hectare'].median(), 2),
                'std': round(df['yield_kg_per_hectare'].std(), 2),
                'min': round(df['yield_kg_per_hectare'].min(), 2),
                'max': round(df['yield_kg_per_hectare'].max(), 2),
                'q1': round(df['yield_kg_per_hectare'].quantile(0.25), 2),
                'q3': round(df['yield_kg_per_hectare'].quantile(0.75), 2)
            },
            'categorical_counts': {},
            'numerical_stats': {}
        }
        
        # Date range if date columns exist
        date_cols = ['sowing_date', 'harvest_date']
        for col in date_cols:
            if col in df.columns:
                stats['dataset_info']['date_range'][col] = {
                    'min': df[col].min().strftime('%Y-%m-%d') if pd.notnull(df[col].min()) else 'N/A',
                    'max': df[col].max().strftime('%Y-%m-%d') if pd.notnull(df[col].max()) else 'N/A'
                }
        
        # Categorical columns counts
        categorical_cols = ['region', 'crop_type', 'irrigation_type', 
                           'fertilizer_type', 'crop_disease_status']
        for col in categorical_cols:
            if col in df.columns:
                stats['categorical_counts'][col] = {
                    'unique_values': int(df[col].nunique()),
                    'top_value': df[col].mode()[0] if not df[col].mode().empty else 'N/A',
                    'top_count': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
                }
        
        # Numerical columns statistics
        numerical_cols = ['soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
                         'humidity_%', 'sunlight_hours', 'pesticide_usage_ml',
                         'NDVI_index', 'total_days']
        for col in numerical_cols:
            if col in df.columns:
                stats['numerical_stats'][col] = {
                    'mean': round(df[col].mean(), 2),
                    'std': round(df[col].std(), 2),
                    'min': round(df[col].min(), 2),
                    'max': round(df[col].max(), 2),
                    'median': round(df[col].median(), 2)
                }
        
        return stats
    
    def _create_sample_data(self):
        """Create sample data for testing"""
        print("Creating sample data...")
        np.random.seed(42)
        
        n_samples = 1000
        regions = ['North', 'South', 'East', 'West', 'Central']
        crop_types = ['Wheat', 'Corn', 'Rice', 'Soybean', 'Barley']
        irrigation_types = ['Drip', 'Sprinkler', 'Flood', 'No Irrigation']
        fertilizer_types = ['Organic', 'NPK', 'Urea', 'Mixed']
        disease_status = ['No Disease', 'Mild', 'Moderate', 'Severe']
        
        data = {
            'farm_id': range(1, n_samples + 1),
            'region': np.random.choice(regions, n_samples),
            'crop_type': np.random.choice(crop_types, n_samples),
            'irrigation_type': np.random.choice(irrigation_types, n_samples),
            'fertilizer_type': np.random.choice(fertilizer_types, n_samples),
            'crop_disease_status': np.random.choice(disease_status, n_samples),
            'soil_moisture_%': np.random.uniform(20, 80, n_samples),
            'soil_pH': np.random.uniform(5.5, 7.5, n_samples),
            'temperature_C': np.random.uniform(15, 35, n_samples),
            'rainfall_mm': np.random.uniform(200, 1200, n_samples),
            'humidity_%': np.random.uniform(40, 90, n_samples),
            'sunlight_hours': np.random.uniform(4, 12, n_samples),
            'pesticide_usage_ml': np.random.uniform(0, 500, n_samples),
            'NDVI_index': np.random.uniform(0.3, 0.9, n_samples),
            'total_days': np.random.randint(90, 180, n_samples),
            'yield_kg_per_hectare': np.random.uniform(2000, 8000, n_samples)
        }
        
        df = pd.DataFrame(data)
        self.df_smart_farming = df
        return df