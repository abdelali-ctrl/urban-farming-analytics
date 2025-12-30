"""
Disease Diagnosis Module using KNN
===================================
Diagnoses crop disease risk based on environmental conditions.
"""

import pandas as pd
import numpy as np
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')  # Extended dataset
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'reports', 'ml')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load and prepare data."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    return df


def prepare_classification_data(df):
    """Prepare data for disease classification."""
    # Check for disease column
    disease_col = 'crop_disease_status'
    if disease_col not in df.columns:
        print(f"Column {disease_col} not found. Creating synthetic labels based on conditions.")
        # Create synthetic disease risk based on conditions
        df['disease_risk'] = 'Low'
        df.loc[(df['humidity_%'] > 80) & (df['temperature_C'] > 30), 'disease_risk'] = 'High'
        df.loc[(df['humidity_%'] > 70) | (df['temperature_C'] > 28), 'disease_risk'] = 'Medium'
        disease_col = 'disease_risk'
    
    # Features for disease prediction
    feature_cols = [
        'soil_moisture_%', 'soil_pH', 'temperature_C', 'rainfall_mm',
        'humidity_%', 'sunlight_hours', 'NDVI_index'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[disease_col])
    
    print(f"Classes: {le.classes_}")
    print(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, feature_cols, le


def find_optimal_k(X_train, y_train, max_k=20):
    """Find optimal k for KNN using cross-validation."""
    k_range = range(1, min(max_k, len(X_train) // 5) + 1)
    cv_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, cv_scores, 'bo-')
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN: Finding Optimal k')
    plt.grid(True)
    
    optimal_k = k_range[np.argmax(cv_scores)]
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_k.png'), dpi=150)
    plt.close()
    
    print(f"Optimal k: {optimal_k} (Accuracy: {max(cv_scores):.4f})")
    return optimal_k


def train_knn(X_train, X_test, y_train, y_test, k=5):
    """Train KNN classifier."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = knn.predict(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return knn, scaler, y_pred


def plot_confusion_matrix(y_test, y_pred, le):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Disease Classification - Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'disease_confusion_matrix.png'), dpi=150)
    plt.close()


def save_model(model, scaler, le):
    """Save trained model."""
    joblib.dump(model, os.path.join(MODEL_DIR, 'disease_knn_model.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'disease_scaler.joblib'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'disease_label_encoder.joblib'))
    print(f"\nDisease model saved to {MODEL_DIR}/")


def diagnose_disease(input_data: dict):
    """
    Diagnose disease risk for new data.
    
    Args:
        input_data: Dictionary with feature values
        
    Returns:
        Disease risk prediction and probability
    """
    model = joblib.load(os.path.join(MODEL_DIR, 'disease_knn_model.joblib'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'disease_scaler.joblib'))
    le = joblib.load(os.path.join(MODEL_DIR, 'disease_label_encoder.joblib'))
    
    # Prepare input
    df_input = pd.DataFrame([input_data])
    X = scaler.transform(df_input.select_dtypes(include=[np.number]))
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    result = {
        'prediction': le.inverse_transform([prediction])[0],
        'probabilities': {le.classes_[i]: p for i, p in enumerate(probabilities)}
    }
    
    return result


def main():
    """Run complete disease diagnosis pipeline."""
    print("=" * 60)
    print("DISEASE DIAGNOSIS MODEL (KNN)")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Prepare data
    X, y, feature_cols, le = prepare_classification_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Find optimal k
    optimal_k = find_optimal_k(X_train, y_train)
    
    # Train
    model, scaler, y_pred = train_knn(X_train, X_test, y_train, y_test, k=optimal_k)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, le)
    
    # Save
    save_model(model, scaler, le)
    
    return model, le


if __name__ == "__main__":
    model, le = main()
