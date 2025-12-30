"""
Yield Prediction Module for Urban Farming
==========================================
Predicts crop yield using Pipeline-based ML models with:
- OneHotEncoder for categorical features
- StandardScaler for numerical features
- Random Forest, Gradient Boosting, and Linear Regression

Adapted from coworker's approach with extended features.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import pandas as pd

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'reports', 'ml')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# FEATURE CONFIGURATION
# ================================

# Base numerical features (from coworker's approach)
BASE_NUM_FEATURES = [
    'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
    'humidity_%', 'sunlight_hours', 'pesticide_usage_ml',
    'NDVI_index', 'total_days'
]

# Extended numerical features (your extended dataset)
# NOTE: Removed yield_per_day as it causes data leakage (calculated from yield)
EXTENDED_NUM_FEATURES = [
    'water_efficiency', 'pesticide_efficiency',
    'heat_stress', 'drought_stress', 'growing_conditions_score'
]

# Categorical features
CAT_FEATURES = [
    'region', 'crop_type', 'irrigation_type',
    'fertilizer_type', 'crop_disease_status'
]

# Add seasonal features if available
SEASONAL_FEATURES = ['sowing_season']

TARGET = 'yield_kg_per_hectare'


def load_data():
    """Load and prepare the farming dataset."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} columns")
    print(f"\nRegions: {df['region'].nunique()}")
    print(f"Crop types: {df['crop_type'].nunique()}")
    
    return df


def get_available_features(df):
    """Identify available features from the dataset."""
    # Numerical features
    num_features = [f for f in BASE_NUM_FEATURES if f in df.columns]
    extended = [f for f in EXTENDED_NUM_FEATURES if f in df.columns]
    num_features.extend(extended)
    
    # Categorical features
    cat_features = [f for f in CAT_FEATURES if f in df.columns]
    seasonal = [f for f in SEASONAL_FEATURES if f in df.columns]
    cat_features.extend(seasonal)
    
    print(f"\n✓ Numerical features: {len(num_features)}")
    print(f"✓ Categorical features: {len(cat_features)}")
    
    return num_features, cat_features


def prepare_data(df, num_features, cat_features):
    """Prepare features and target for modeling."""
    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)
    
    # Select features
    all_features = num_features + cat_features
    X = df[all_features].copy()
    y = df[TARGET].copy()
    
    # Handle missing values in numerical features
    for col in num_features:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())
    
    # Handle missing values in categorical features
    for col in cat_features:
        if col in X.columns:
            mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
            X[col] = X[col].fillna(mode_val)
    
    print(f"✓ X shape: {X.shape}")
    print(f"✓ y shape: {y.shape}")
    
    return X, y


def create_preprocessor(num_features, cat_features):
    """Create ColumnTransformer for preprocessing."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    return preprocessor


def train_models(X_train, X_test, y_train, y_test, num_features, cat_features):
    """Train and compare multiple models using Pipeline."""
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    results = []
    models = {}
    predictions = {}
    
    # Create preprocessor
    preprocessor = create_preprocessor(num_features, cat_features)
    
    # ---- 1. Linear Regression (Baseline)
    print("\nTraining Linear Regression...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    
    results.append([
        "Linear Regression",
        r2_score(y_test, y_pred_lr),
        np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        mean_absolute_error(y_test, y_pred_lr)
    ])
    models["Linear Regression"] = lr_pipeline
    predictions["Linear Regression"] = y_pred_lr
    
    # ---- 2. Random Forest (Main Model - coworker's config)
    print("Training Random Forest (300 trees, depth 18)...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    
    results.append([
        "Random Forest",
        r2_score(y_test, y_pred_rf),
        np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        mean_absolute_error(y_test, y_pred_rf)
    ])
    models["Random Forest"] = rf_pipeline
    predictions["Random Forest"] = y_pred_rf
    
    # ---- 3. Gradient Boosting
    print("Training Gradient Boosting...")
    gbr_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ))
    ])
    gbr_pipeline.fit(X_train, y_train)
    y_pred_gbr = gbr_pipeline.predict(X_test)
    
    results.append([
        "Gradient Boosting",
        r2_score(y_test, y_pred_gbr),
        np.sqrt(mean_squared_error(y_test, y_pred_gbr)),
        mean_absolute_error(y_test, y_pred_gbr)
    ])
    models["Gradient Boosting"] = gbr_pipeline
    predictions["Gradient Boosting"] = y_pred_gbr
    
    # Create results DataFrame
    results_df = pd.DataFrame(
        results,
        columns=["Model", "R2", "RMSE", "MAE"]
    ).round(3)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Select best model based on R2
    best_idx = results_df['R2'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_r2 = results_df.loc[best_idx, 'R2']
    
    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
    
    return models, results_df, predictions, best_model_name


def analyze_feature_importance(model, num_features, cat_features, model_name):
    """Analyze and visualize feature importance for tree-based models."""
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    
    # Get the actual model from pipeline
    actual_model = model.named_steps['model']
    
    if not hasattr(actual_model, 'feature_importances_'):
        print("⚠ Feature importance not available for this model type")
        return None
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessing']
    
    # Get transformed feature names
    num_feature_names = num_features
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
    all_feature_names = num_feature_names + cat_feature_names
    
    # Get importances
    importances = actual_model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Print top 10
    print("\nTop 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:40}: {row['importance']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    colors = plt.cm.RdYlGn(top_features['importance'] / top_features['importance'].max())
    plt.barh(top_features['feature'], top_features['importance'], color=colors)
    plt.xlabel('Importance')
    plt.title(f'Top 15 Important Features ({model_name})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    print(f"\n✓ Feature importance plot saved to {OUTPUT_DIR}/feature_importance.png")
    
    return importance_df


def plot_predictions(y_test, y_pred, model_name):
    """Plot actual vs predicted values and residuals."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5, c='steelblue')
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Yield (kg/ha)')
    axes[0].set_ylabel('Predicted Yield (kg/ha)')
    axes[0].set_title(f'{model_name}: Actual vs Predicted')
    
    # Residuals
    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white')
    axes[1].axvline(x=0, color='red', linestyle='--')
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Prediction Residuals')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predictions_analysis.png'), dpi=150)
    plt.close()
    print(f"✓ Predictions plot saved to {OUTPUT_DIR}/predictions_analysis.png")


def save_models(models, best_model_name, results_df, importance_df):
    """Save trained models and results."""
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    
    # Save best model as pipeline
    best_model = models[best_model_name]
    pipeline_path = os.path.join(MODEL_DIR, 'yield_prediction_pipeline.pkl')
    joblib.dump(best_model, pipeline_path)
    print(f"✓ Best model pipeline saved: {pipeline_path}")
    
    # Also save as best_yield_model.pkl (for coworker's interface compatibility)
    best_path = os.path.join(MODEL_DIR, 'best_yield_model.pkl')
    joblib.dump(best_model, best_path)
    print(f"✓ Best model saved: {best_path}")
    
    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✓ Model comparison saved: {results_path}")
    
    if importance_df is not None:
        importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"✓ Feature importance saved: {importance_path}")
    
    return pipeline_path


def predict_yield(input_data: dict, model_path=None):
    """
    Make yield prediction for new data.
    
    Args:
        input_data: Dictionary with feature values
        model_path: Path to saved model pipeline
        
    Returns:
        Predicted yield in kg/ha
    """
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'yield_prediction_pipeline.pkl')
    
    pipeline = joblib.load(model_path)
    
    # Prepare input as DataFrame
    df_input = pd.DataFrame([input_data])
    
    # Predict
    prediction = pipeline.predict(df_input)[0]
    
    return prediction


def main():
    """Run complete ML pipeline."""
    print("\n" + "=" * 70)
    print("YIELD PREDICTION MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    df = load_data()
    
    # Get available features
    num_features, cat_features = get_available_features(df)
    
    # Prepare data
    X, y = prepare_data(df, num_features, cat_features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    models, results_df, predictions, best_model_name = train_models(
        X_train, X_test, y_train, y_test, num_features, cat_features
    )
    
    # Feature importance for best model
    importance_df = analyze_feature_importance(
        models[best_model_name], num_features, cat_features, best_model_name
    )
    
    # Plot predictions for best model
    plot_predictions(y_test.values, predictions[best_model_name], best_model_name)
    
    # Save models
    save_models(models, best_model_name, results_df, importance_df)
    
    # Sample prediction
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTION")
    print("=" * 70)
    sample = X.iloc[[0]]
    predicted_yield = models[best_model_name].predict(sample)[0]
    actual_yield = y.iloc[0]
    print(f"Sample farm predicted yield: {predicted_yield:.0f} kg/ha")
    print(f"Actual yield: {actual_yield:.0f} kg/ha")
    
    print("\n✓ Training complete!")
    
    return models[best_model_name], results_df, importance_df


if __name__ == "__main__":
    model, results, importance = main()
