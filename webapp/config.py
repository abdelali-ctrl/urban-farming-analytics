"""
Configuration for the Urban Farming Flask Web Application
==========================================================
Defines paths to data files and ML models for the dashboard.
"""

import os

class Config:
    """Application configuration with paths to project resources."""
    
    # Base directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.dirname(BASE_DIR)  # Parent of webapp
    DATA_DIR = os.path.join(PROJECT_DIR, 'data')
    MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
    REPORTS_DIR = os.path.join(PROJECT_DIR, 'reports')
    
    # Data file paths
    SMART_FARMING_DATA = os.path.join(DATA_DIR, 'farming_with_prices_extended.csv')
    COMMODITY_DATA = os.path.join(DATA_DIR, 'commodity_futures.csv')
    FRUIT_DATA = os.path.join(DATA_DIR, 'fruit-prices-cleaned.csv')
    VEGETABLE_DATA = os.path.join(DATA_DIR, 'vegetable-prices-cleaned.csv')
    
    # ML Model paths (trained models saved by scripts/ml/yield_prediction.py)
    PIPELINE_MODEL = os.path.join(MODELS_DIR, 'yield_prediction_pipeline.pkl')
    YIELD_MODEL = os.path.join(MODELS_DIR, 'best_yield_model.pkl')
    
    # Flask settings
    SECRET_KEY = 'urban-farming-dashboard-secret-key-2024'
    DEBUG = True
    
    # API settings
    JSON_SORT_KEYS = False
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure required directories exist."""
        for dir_path in [cls.MODELS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
