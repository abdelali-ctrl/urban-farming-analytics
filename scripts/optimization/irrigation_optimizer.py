"""
Irrigation Optimization Module
==============================
Linear programming model to optimize water usage and maximize yield.
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
import matplotlib.pyplot as plt
import os

# Try to use PuLP if available
try:
    from pulp import *
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("PuLP not installed. Using scipy.optimize instead.")

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')  # Extended dataset
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'reports', 'optimization')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class IrrigationOptimizer:
    """
    Optimizes irrigation schedule to minimize water usage while maintaining yield.
    
    Decision Variables:
        - Water allocation per crop type per region
        
    Objective:
        - Minimize total water usage
        OR
        - Maximize yield per unit water (water efficiency)
        
    Constraints:
        - Minimum yield threshold per crop
        - Maximum water availability
        - Minimum/maximum water per crop
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.crops = df['crop_type'].unique()
        self.regions = df['region'].unique()
        self.results = None
    
    def estimate_water_yield_relationship(self):
        """Estimate water-yield coefficients from historical data."""
        # Group by crop and calculate average yield per rainfall
        coefficients = {}
        
        for crop in self.crops:
            crop_data = self.df[self.df['crop_type'] == crop]
            
            # Simple linear relationship: yield = a * rainfall + b
            if len(crop_data) > 10:
                from scipy.stats import linregress
                slope, intercept, r, p, se = linregress(
                    crop_data['rainfall_mm'], 
                    crop_data['yield_kg_per_hectare']
                )
                coefficients[crop] = {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r**2,
                    'min_water': crop_data['rainfall_mm'].quantile(0.1),
                    'max_water': crop_data['rainfall_mm'].quantile(0.9),
                    'avg_yield': crop_data['yield_kg_per_hectare'].mean()
                }
            else:
                coefficients[crop] = {
                    'slope': 5,
                    'intercept': 2000,
                    'r_squared': 0,
                    'min_water': 100,
                    'max_water': 400,
                    'avg_yield': 3000
                }
        
        self.coefficients = coefficients
        return coefficients
    
    def optimize_water_allocation(self, total_water_budget: float, min_yield_pct: float = 0.8):
        """
        Optimize water allocation across crops.
        
        Args:
            total_water_budget: Total water available (mm equivalent)
            min_yield_pct: Minimum yield as percentage of average
            
        Returns:
            Optimal allocation dict
        """
        if not hasattr(self, 'coefficients'):
            self.estimate_water_yield_relationship()
        
        if PULP_AVAILABLE:
            return self._optimize_with_pulp(total_water_budget, min_yield_pct)
        else:
            return self._optimize_with_scipy(total_water_budget, min_yield_pct)
    
    def _optimize_with_pulp(self, total_water: float, min_yield_pct: float):
        """Optimize using PuLP (linear programming)."""
        # Create problem
        prob = LpProblem("Irrigation_Optimization", LpMaximize)
        
        # Decision variables: water allocation per crop
        water_vars = {crop: LpVariable(f"water_{crop}", lowBound=0) for crop in self.crops}
        
        # Objective: Maximize total yield (water efficiency)
        prob += lpSum([
            self.coefficients[crop]['slope'] * water_vars[crop] + 
            self.coefficients[crop]['intercept']
            for crop in self.crops
        ]), "Total_Yield"
        
        # Constraint: Total water budget
        prob += lpSum([water_vars[crop] for crop in self.crops]) <= total_water, "Total_Water_Budget"
        
        # Constraints: Min/max water per crop
        for crop in self.crops:
            min_w = self.coefficients[crop]['min_water']
            max_w = self.coefficients[crop]['max_water']
            prob += water_vars[crop] >= min_w, f"Min_Water_{crop}"
            prob += water_vars[crop] <= max_w, f"Max_Water_{crop}"
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract results
        allocation = {crop: water_vars[crop].varValue for crop in self.crops}
        total_allocated = sum(allocation.values())
        
        expected_yields = {
            crop: self.coefficients[crop]['slope'] * allocation[crop] + 
                  self.coefficients[crop]['intercept']
            for crop in self.crops
        }
        
        self.results = {
            'status': LpStatus[prob.status],
            'allocation': allocation,
            'total_water_used': total_allocated,
            'water_saved': total_water - total_allocated,
            'expected_yields': expected_yields,
            'total_expected_yield': sum(expected_yields.values())
        }
        
        return self.results
    
    def _optimize_with_scipy(self, total_water: float, min_yield_pct: float):
        """Optimize using scipy (fallback)."""
        n_crops = len(self.crops)
        crop_list = list(self.crops)
        
        # Objective: Minimize negative yield (maximize yield)
        c = [-self.coefficients[crop]['slope'] for crop in crop_list]
        
        # Constraint: Total water <= budget
        A_ub = [[1] * n_crops]
        b_ub = [total_water]
        
        # Bounds: min/max water per crop
        bounds = [
            (self.coefficients[crop]['min_water'], self.coefficients[crop]['max_water'])
            for crop in crop_list
        ]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            allocation = {crop_list[i]: result.x[i] for i in range(n_crops)}
            total_allocated = sum(allocation.values())
            
            expected_yields = {
                crop: self.coefficients[crop]['slope'] * allocation[crop] + 
                      self.coefficients[crop]['intercept']
                for crop in crop_list
            }
            
            self.results = {
                'status': 'Optimal',
                'allocation': allocation,
                'total_water_used': total_allocated,
                'water_saved': total_water - total_allocated,
                'expected_yields': expected_yields,
                'total_expected_yield': sum(expected_yields.values())
            }
        else:
            self.results = {'status': 'Failed', 'message': result.message}
        
        return self.results
    
    def visualize_results(self):
        """Visualize optimization results."""
        if self.results is None or self.results['status'] != 'Optimal':
            print("No valid results to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # 1. Water allocation
        crops = list(self.results['allocation'].keys())
        allocations = list(self.results['allocation'].values())
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(crops)))
        axes[0].bar(crops, allocations, color=colors)
        axes[0].set_xlabel('Crop')
        axes[0].set_ylabel('Water Allocation (mm)')
        axes[0].set_title('Optimal Water Allocation by Crop')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 2. Expected yields
        yields = list(self.results['expected_yields'].values())
        axes[1].bar(crops, yields, color='green', alpha=0.7)
        axes[1].set_xlabel('Crop')
        axes[1].set_ylabel('Expected Yield (kg/ha)')
        axes[1].set_title('Expected Yields with Optimal Allocation')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 3. Water efficiency
        efficiency = [y / w if w > 0 else 0 for y, w in zip(yields, allocations)]
        axes[2].bar(crops, efficiency, color='steelblue')
        axes[2].set_xlabel('Crop')
        axes[2].set_ylabel('Yield per mm Water')
        axes[2].set_title('Water Use Efficiency')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'irrigation_optimization.png'), dpi=150)
        plt.close()
        
        print(f"Visualization saved to {OUTPUT_DIR}/")
    
    def generate_report(self):
        """Generate optimization report."""
        if self.results is None:
            return "No results available"
        
        report = f"""
# Irrigation Optimization Report

## Summary
- **Status**: {self.results['status']}
- **Total Water Used**: {self.results.get('total_water_used', 0):.1f} mm
- **Water Saved**: {self.results.get('water_saved', 0):.1f} mm
- **Total Expected Yield**: {self.results.get('total_expected_yield', 0):,.0f} kg/ha

## Optimal Allocation

| Crop | Water (mm) | Expected Yield (kg/ha) | Efficiency |
|------|------------|------------------------|------------|
"""
        if 'allocation' in self.results:
            for crop in self.results['allocation']:
                water = self.results['allocation'][crop]
                yield_val = self.results['expected_yields'][crop]
                eff = yield_val / water if water > 0 else 0
                report += f"| {crop} | {water:.1f} | {yield_val:,.0f} | {eff:.1f} |\n"
        
        return report


def main():
    """Run irrigation optimization."""
    print("=" * 60)
    print("IRRIGATION OPTIMIZATION")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    
    # Initialize optimizer
    optimizer = IrrigationOptimizer(df)
    
    # Estimate coefficients
    coefficients = optimizer.estimate_water_yield_relationship()
    print("\nWater-Yield Coefficients:")
    for crop, coef in coefficients.items():
        print(f"  {crop}: slope={coef['slope']:.2f}, R²={coef['r_squared']:.3f}")
    
    # Optimize with water budget
    total_budget = 1000  # Total water budget in mm
    results = optimizer.optimize_water_allocation(total_budget)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    
    if results['status'] == 'Optimal':
        print(f"\nWater Allocation:")
        for crop, water in results['allocation'].items():
            print(f"  {crop}: {water:.1f} mm")
        
        print(f"\nTotal Water Used: {results['total_water_used']:.1f} mm")
        print(f"Water Saved: {results['water_saved']:.1f} mm ({results['water_saved']/total_budget*100:.1f}%)")
        print(f"Total Expected Yield: {results['total_expected_yield']:,.0f} kg/ha")
        
        # Visualize
        optimizer.visualize_results()
        
        # Save report
        report = optimizer.generate_report()
        report_path = os.path.join(OUTPUT_DIR, 'irrigation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
    
    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()
