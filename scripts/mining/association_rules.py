"""
Association Rule Mining for Urban Farming
==========================================
Discover patterns between growing conditions and high yields.
"""

import pandas as pd
import numpy as np
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns
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


def discretize_features(df):
    """Convert continuous variables to categorical for association mining."""
    df_disc = df.copy()
    
    # Yield categories
    yield_q = df['yield_kg_per_hectare'].quantile([0.33, 0.66])
    df_disc['yield_level'] = pd.cut(
        df['yield_kg_per_hectare'],
        bins=[-np.inf, yield_q[0.33], yield_q[0.66], np.inf],
        labels=['Low_Yield', 'Medium_Yield', 'High_Yield']
    )
    
    # Temperature categories
    df_disc['temp_category'] = pd.cut(
        df['temperature_C'],
        bins=[-np.inf, 20, 30, np.inf],
        labels=['Cool', 'Moderate_Temp', 'Hot']
    )
    
    # Rainfall categories
    df_disc['rainfall_category'] = pd.cut(
        df['rainfall_mm'],
        bins=[-np.inf, 100, 300, np.inf],
        labels=['Low_Rain', 'Moderate_Rain', 'High_Rain']
    )
    
    # Soil moisture
    df_disc['moisture_category'] = pd.cut(
        df['soil_moisture_%'],
        bins=[-np.inf, 40, 60, np.inf],
        labels=['Dry_Soil', 'Moist_Soil', 'Wet_Soil']
    )
    
    # Soil pH
    df_disc['ph_category'] = pd.cut(
        df['soil_pH'],
        bins=[-np.inf, 6.0, 7.5, np.inf],
        labels=['Acidic_Soil', 'Neutral_Soil', 'Alkaline_Soil']
    )
    
    # NDVI (vegetation health)
    if 'NDVI_index' in df.columns:
        df_disc['ndvi_category'] = pd.cut(
            df['NDVI_index'],
            bins=[-np.inf, 0.4, 0.7, np.inf],
            labels=['Low_NDVI', 'Medium_NDVI', 'High_NDVI']
        )
    
    # Water efficiency
    if 'water_efficiency' in df.columns:
        eff_q = df['water_efficiency'].quantile([0.33, 0.66])
        df_disc['water_eff_category'] = pd.cut(
            df['water_efficiency'],
            bins=[-np.inf, eff_q[0.33], eff_q[0.66], np.inf],
            labels=['Low_Water_Eff', 'Medium_Water_Eff', 'High_Water_Eff']
        )
    
    return df_disc


def create_transactions(df_disc):
    """Create transaction dataset for apriori algorithm."""
    # Select categorical columns for transactions
    cat_cols = [
        'crop_type', 'region', 'irrigation_type', 'fertilizer_type',
        'yield_level', 'temp_category', 'rainfall_category',
        'moisture_category', 'ph_category'
    ]
    
    if 'ndvi_category' in df_disc.columns:
        cat_cols.append('ndvi_category')
    if 'water_eff_category' in df_disc.columns:
        cat_cols.append('water_eff_category')
    if 'sowing_season' in df_disc.columns:
        cat_cols.append('sowing_season')
    
    # Filter to available columns
    available_cols = [c for c in cat_cols if c in df_disc.columns]
    
    # Create transaction list
    transactions = []
    for _, row in df_disc[available_cols].iterrows():
        transaction = [f"{col}={row[col]}" for col in available_cols if pd.notna(row[col])]
        transactions.append(transaction)
    
    # Encode transactions
    te = TransactionEncoder()
    te_data = te.fit_transform(transactions)
    df_encoded = pd.DataFrame(te_data, columns=te.columns_)
    
    print(f"Created {len(transactions)} transactions with {len(te.columns_)} unique items")
    return df_encoded, transactions


def mine_association_rules(df_encoded, min_support=0.05, min_confidence=0.5):
    """Mine association rules using Apriori algorithm."""
    print(f"\nMining rules with min_support={min_support}, min_confidence={min_confidence}")
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    print(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    if len(frequent_itemsets) == 0:
        print("No frequent itemsets found. Try lowering min_support.")
        return None, None
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values('lift', ascending=False)
    print(f"Generated {len(rules)} association rules")
    
    return frequent_itemsets, rules


def filter_high_yield_rules(rules):
    """Filter rules that predict high yield."""
    if rules is None or len(rules) == 0:
        return None
    
    # Filter rules where consequent contains high yield
    high_yield_rules = rules[
        rules['consequents'].apply(lambda x: any('High_Yield' in str(item) for item in x))
    ].copy()
    
    print(f"Found {len(high_yield_rules)} rules predicting high yield")
    return high_yield_rules


def visualize_rules(rules, top_n=15):
    """Visualize association rules."""
    if rules is None or len(rules) == 0:
        print("No rules to visualize")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Support vs Confidence colored by Lift
    top_rules = rules.head(min(50, len(rules)))
    scatter = axes[0].scatter(
        top_rules['support'],
        top_rules['confidence'],
        c=top_rules['lift'],
        cmap='RdYlGn',
        s=100 * top_rules['lift'],
        alpha=0.7
    )
    axes[0].set_xlabel('Support')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Association Rules: Support vs Confidence')
    plt.colorbar(scatter, ax=axes[0], label='Lift')
    
    # Top rules by lift
    top_lift = rules.head(top_n).copy()
    top_lift['rule'] = top_lift.apply(
        lambda x: f"{', '.join([str(i).split('=')[1] if '=' in str(i) else str(i) for i in x['antecedents']])} → "
                  f"{', '.join([str(i).split('=')[1] if '=' in str(i) else str(i) for i in x['consequents']])}",
        axis=1
    )
    
    colors = plt.cm.RdYlGn(top_lift['lift'] / top_lift['lift'].max())
    axes[1].barh(range(len(top_lift)), top_lift['lift'], color=colors)
    axes[1].set_yticks(range(len(top_lift)))
    axes[1].set_yticklabels(top_lift['rule'], fontsize=8)
    axes[1].set_xlabel('Lift')
    axes[1].set_title(f'Top {top_n} Rules by Lift')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'association_rules.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_insights(high_yield_rules):
    """Generate actionable insights from high yield rules."""
    insights = []
    
    if high_yield_rules is None or len(high_yield_rules) == 0:
        return ["No significant patterns found for high yield prediction."]
    
    for _, rule in high_yield_rules.head(10).iterrows():
        antecedents = list(rule['antecedents'])
        conditions = [a.split('=')[1] if '=' in str(a) else str(a) for a in antecedents]
        
        insight = f"When: {', '.join(conditions)} → High Yield (Confidence: {rule['confidence']:.1%}, Lift: {rule['lift']:.2f})"
        insights.append(insight)
    
    return insights


def main():
    """Run complete association rule mining analysis."""
    print("=" * 60)
    print("ASSOCIATION RULE MINING FOR URBAN FARMING")
    print("=" * 60)
    
    # Load and prepare data
    df = load_data()
    df_disc = discretize_features(df)
    
    # Create transactions
    df_encoded, transactions = create_transactions(df_disc)
    
    # Mine rules
    frequent_itemsets, rules = mine_association_rules(df_encoded, min_support=0.03, min_confidence=0.4)
    
    if rules is not None:
        # Filter for high yield
        high_yield_rules = filter_high_yield_rules(rules)
        
        # Visualize
        visualize_rules(rules)
        
        # Generate insights
        insights = generate_insights(high_yield_rules)
        
        # Print results
        print("\n" + "=" * 60)
        print("KEY INSIGHTS FOR HIGH YIELD")
        print("=" * 60)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Save rules
        if high_yield_rules is not None and len(high_yield_rules) > 0:
            # Convert frozensets to strings for saving
            save_rules = high_yield_rules.copy()
            save_rules['antecedents'] = save_rules['antecedents'].apply(lambda x: ', '.join(map(str, x)))
            save_rules['consequents'] = save_rules['consequents'].apply(lambda x: ', '.join(map(str, x)))
            save_rules.to_csv(os.path.join(OUTPUT_DIR, 'high_yield_rules.csv'), index=False)
            print(f"\nRules saved to: {os.path.join(OUTPUT_DIR, 'high_yield_rules.csv')}")
        
        return rules, high_yield_rules, insights
    
    return None, None, []


if __name__ == "__main__":
    rules, high_yield_rules, insights = main()
