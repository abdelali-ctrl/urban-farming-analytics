"""
Clustering Analysis for Urban Farming
======================================
K-Means clustering to identify farm performance groups.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')  # Extended dataset
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'reports', 'mining')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Load the farming dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    return df


def prepare_features(df):
    """Prepare features for clustering."""
    # Select numerical features for clustering
    feature_cols = [
        'yield_kg_per_hectare',
        'soil_moisture_%',
        'soil_pH',
        'temperature_C',
        'rainfall_mm',
        'humidity_%',
        'sunlight_hours',
        'NDVI_index',
        'water_efficiency',
        'growing_conditions_score'
    ]
    
    # Filter to available columns
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"Using features: {available_cols}")
    
    X = df[available_cols].copy()
    X = X.fillna(X.median())
    
    return X, available_cols


def find_optimal_clusters(X_scaled, max_k=10):
    """Find optimal number of clusters using elbow method and silhouette score."""
    inertias = []
    silhouettes = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Plot elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(K_range, inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].grid(True)
    
    axes[1].plot(K_range, silhouettes, 'ro-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'optimal_clusters.png'), dpi=150)
    plt.close()
    
    # Return optimal k (highest silhouette)
    optimal_k = K_range[np.argmax(silhouettes)]
    print(f"Optimal clusters (silhouette): {optimal_k}")
    return optimal_k


def perform_clustering(df, X, feature_cols, n_clusters=4):
    """Perform K-Means clustering."""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal clusters
    n_clusters = find_optimal_clusters(X_scaled, max_k=8)
    
    # Fit final model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster centers (unscaled)
    centers_scaled = kmeans.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=feature_cols)
    centers_df['cluster'] = range(n_clusters)
    
    return df, centers_df, kmeans, X_scaled


def analyze_clusters(df, centers_df, feature_cols):
    """Analyze and visualize clusters."""
    
    # 1. Cluster distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Cluster sizes
    cluster_counts = df['cluster'].value_counts().sort_index()
    axes[0, 0].bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette('viridis', len(cluster_counts)))
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Number of Farms')
    axes[0, 0].set_title('Farm Distribution by Cluster')
    
    # Average yield by cluster
    yield_by_cluster = df.groupby('cluster')['yield_kg_per_hectare'].mean()
    colors = ['green' if y > df['yield_kg_per_hectare'].mean() else 'red' for y in yield_by_cluster]
    axes[0, 1].bar(yield_by_cluster.index, yield_by_cluster.values, color=colors)
    axes[0, 1].axhline(y=df['yield_kg_per_hectare'].mean(), color='blue', linestyle='--', label='Overall Mean')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Average Yield (kg/ha)')
    axes[0, 1].set_title('Yield Performance by Cluster')
    axes[0, 1].legend()
    
    # Crop distribution in clusters
    crop_cluster = pd.crosstab(df['cluster'], df['crop_type'], normalize='index')
    crop_cluster.plot(kind='bar', stacked=True, ax=axes[1, 0], colormap='Set2')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Proportion')
    axes[1, 0].set_title('Crop Distribution by Cluster')
    axes[1, 0].legend(title='Crop', bbox_to_anchor=(1.02, 1))
    axes[1, 0].tick_params(axis='x', rotation=0)
    
    # Region distribution in clusters
    region_cluster = pd.crosstab(df['cluster'], df['region'], normalize='index')
    region_cluster.plot(kind='bar', stacked=True, ax=axes[1, 1], colormap='Set1')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Proportion')
    axes[1, 1].set_title('Region Distribution by Cluster')
    axes[1, 1].legend(title='Region', bbox_to_anchor=(1.02, 1))
    axes[1, 1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Cluster profiles heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    profile_cols = ['yield_kg_per_hectare', 'water_efficiency', 'growing_conditions_score', 
                    'temperature_C', 'rainfall_mm', 'soil_pH']
    profile_cols = [c for c in profile_cols if c in centers_df.columns]
    
    profile = centers_df[profile_cols].T
    profile.columns = [f'Cluster {i}' for i in range(len(centers_df))]
    
    # Normalize for heatmap
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    sns.heatmap(profile_norm, annot=profile.round(1), fmt='', cmap='RdYlGn', ax=ax)
    ax.set_title('Cluster Profiles (Normalized)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cluster_profiles.png'), dpi=150)
    plt.close()
    
    return cluster_counts


def generate_cluster_labels(df, centers_df):
    """Generate descriptive labels for each cluster."""
    labels = {}
    overall_mean = df['yield_kg_per_hectare'].mean()
    
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        avg_yield = cluster_data['yield_kg_per_hectare'].mean()
        avg_water_eff = cluster_data['water_efficiency'].mean() if 'water_efficiency' in df.columns else 0
        
        if avg_yield > overall_mean * 1.1:
            performance = "High Performers"
        elif avg_yield < overall_mean * 0.9:
            performance = "Underperformers"
        else:
            performance = "Average Performers"
        
        labels[cluster] = performance
    
    return labels


def main():
    """Run complete clustering analysis."""
    print("=" * 60)
    print("URBAN FARMING CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data()
    X, feature_cols = prepare_features(df)
    
    # Perform clustering
    df, centers_df, model, X_scaled = perform_clustering(df, X, feature_cols)
    
    # Analyze clusters
    cluster_counts = analyze_clusters(df, centers_df, feature_cols)
    
    # Generate labels
    labels = generate_cluster_labels(df, centers_df)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTER SUMMARY")
    print("=" * 60)
    
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster]
        print(f"\nCluster {cluster} - {labels[cluster]}:")
        print(f"  Farms: {len(cluster_data)}")
        print(f"  Avg Yield: {cluster_data['yield_kg_per_hectare'].mean():.1f} kg/ha")
        if 'water_efficiency' in df.columns:
            print(f"  Avg Water Efficiency: {cluster_data['water_efficiency'].mean():.2f}")
        print(f"  Top Crops: {cluster_data['crop_type'].value_counts().head(2).index.tolist()}")
    
    # Save clustered data
    output_file = os.path.join(OUTPUT_DIR, 'farming_clustered.csv')
    df.to_csv(output_file, index=False)
    print(f"\nClustered data saved to: {output_file}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    
    return df, centers_df, labels


if __name__ == "__main__":
    df, centers, labels = main()
