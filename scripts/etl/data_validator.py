"""
Data Validator Module using Great Expectations
===============================================
Validates farming data quality before loading to warehouse.
"""

import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Data quality validator for Urban Farming data.
    Validates schema, ranges, nulls, and business rules.
    """
    
    # Expected schema for farming data
    FARMING_SCHEMA = {
        'farm_id': 'object',
        'region': 'object',
        'crop_type': 'object',
        'soil_moisture_%': 'float64',
        'soil_pH': 'float64',
        'temperature_C': 'float64',
        'rainfall_mm': 'float64',
        'humidity_%': 'float64',
        'sunlight_hours': 'float64',
        'irrigation_type': 'object',
        'fertilizer_type': 'object',
        'pesticide_usage_ml': 'float64',
        'yield_kg_per_hectare': 'float64',
    }
    
    # Valid categorical values
    VALID_CROPS = ['Wheat', 'Maize', 'Rice', 'Soybean', 'Cotton']
    VALID_REGIONS = ['North India', 'South India', 'Central USA', 'South USA', 'East Africa']
    VALID_IRRIGATION = ['Drip', 'Sprinkler', 'Manual', 'None', 'No Irrigation']
    VALID_FERTILIZER = ['Organic', 'Inorganic', 'Mixed']
    VALID_DISEASE_STATUS = ['None', 'No Disease', 'Mild', 'Moderate', 'Severe']
    
    # Numeric ranges (min, max)
    NUMERIC_RANGES = {
        'soil_moisture_%': (0, 100),
        'soil_pH': (0, 14),
        'temperature_C': (-20, 50),
        'rainfall_mm': (0, 1000),
        'humidity_%': (0, 100),
        'sunlight_hours': (0, 24),
        'pesticide_usage_ml': (0, 100),
        'yield_kg_per_hectare': (0, 10000),
        'NDVI_index': (0, 1),
    }
    
    def __init__(self):
        self.validation_results: List[Dict] = []
        self.is_valid = True
        
    def validate(self, df: pd.DataFrame, dataset_name: str = "dataset") -> Tuple[bool, Dict]:
        """
        Run all validations on the dataset.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging purposes
            
        Returns:
            Tuple of (is_valid, results_dict)
        """
        logger.info(f"Starting validation for {dataset_name} ({len(df)} rows)")
        self.validation_results = []
        self.is_valid = True
        
        # Run validations
        self._check_not_empty(df, dataset_name)
        self._check_required_columns(df)
        self._check_nulls(df)
        self._check_duplicates(df, 'farm_id')
        self._check_categorical_values(df)
        self._check_numeric_ranges(df)
        self._check_date_validity(df)
        self._check_business_rules(df)
        
        # Generate summary
        passed = sum(1 for r in self.validation_results if r['passed'])
        failed = len(self.validation_results) - passed
        
        summary = {
            'dataset': dataset_name,
            'total_rows': len(df),
            'validations_passed': passed,
            'validations_failed': failed,
            'is_valid': self.is_valid,
            'timestamp': datetime.now().isoformat(),
            'details': self.validation_results
        }
        
        if self.is_valid:
            logger.info(f"✅ Validation PASSED for {dataset_name}")
        else:
            logger.warning(f"❌ Validation FAILED for {dataset_name} ({failed} issues)")
            
        return self.is_valid, summary
    
    def _add_result(self, check_name: str, passed: bool, message: str, severity: str = 'error'):
        """Add a validation result."""
        result = {
            'check': check_name,
            'passed': passed,
            'message': message,
            'severity': severity
        }
        self.validation_results.append(result)
        if not passed and severity == 'error':
            self.is_valid = False
        log_func = logger.info if passed else (logger.warning if severity == 'warning' else logger.error)
        log_func(f"  {'✓' if passed else '✗'} {check_name}: {message}")
    
    def _check_not_empty(self, df: pd.DataFrame, name: str):
        """Check that DataFrame is not empty."""
        passed = len(df) > 0
        self._add_result(
            'not_empty',
            passed,
            f"{name} has {len(df)} rows" if passed else f"{name} is empty"
        )
    
    def _check_required_columns(self, df: pd.DataFrame):
        """Check that required columns exist."""
        required = ['farm_id', 'crop_type', 'region', 'yield_kg_per_hectare']
        missing = [col for col in required if col not in df.columns]
        passed = len(missing) == 0
        self._add_result(
            'required_columns',
            passed,
            "All required columns present" if passed else f"Missing columns: {missing}"
        )
    
    def _check_nulls(self, df: pd.DataFrame):
        """Check for null values in critical columns."""
        critical_cols = ['farm_id', 'crop_type', 'yield_kg_per_hectare']
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                passed = null_count == 0
                self._add_result(
                    f'no_nulls_{col}',
                    passed,
                    f"No nulls in {col}" if passed else f"{null_count} nulls in {col}"
                )
    
    def _check_duplicates(self, df: pd.DataFrame, key_column: str):
        """Check for duplicate keys."""
        if key_column in df.columns:
            dup_count = df[key_column].duplicated().sum()
            passed = dup_count == 0
            self._add_result(
                f'no_duplicates_{key_column}',
                passed,
                f"No duplicate {key_column}" if passed else f"{dup_count} duplicate {key_column}",
                severity='warning'  # Duplicates might be acceptable
            )
    
    def _check_categorical_values(self, df: pd.DataFrame):
        """Check categorical columns have valid values."""
        checks = [
            ('crop_type', self.VALID_CROPS),
            ('region', self.VALID_REGIONS),
            ('fertilizer_type', self.VALID_FERTILIZER),
        ]
        
        for col, valid_values in checks:
            if col in df.columns:
                invalid = df[~df[col].isin(valid_values)][col].unique()
                passed = len(invalid) == 0
                self._add_result(
                    f'valid_values_{col}',
                    passed,
                    f"All {col} values valid" if passed else f"Invalid {col} values: {list(invalid)[:5]}"
                )
    
    def _check_numeric_ranges(self, df: pd.DataFrame):
        """Check numeric columns are within expected ranges."""
        for col, (min_val, max_val) in self.NUMERIC_RANGES.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                passed = len(out_of_range) == 0
                self._add_result(
                    f'range_{col}',
                    passed,
                    f"{col} values in range [{min_val}, {max_val}]" if passed 
                    else f"{len(out_of_range)} {col} values out of range",
                    severity='warning'
                )
    
    def _check_date_validity(self, df: pd.DataFrame):
        """Check date columns are valid."""
        date_cols = ['sowing_date', 'harvest_date']
        for col in date_cols:
            if col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    self._add_result(f'valid_date_{col}', True, f"{col} dates are valid")
                except Exception as e:
                    self._add_result(f'valid_date_{col}', False, f"Invalid dates in {col}: {str(e)}")
    
    def _check_business_rules(self, df: pd.DataFrame):
        """Check business logic rules."""
        # Rule 1: harvest_date > sowing_date
        if 'sowing_date' in df.columns and 'harvest_date' in df.columns:
            try:
                sowing = pd.to_datetime(df['sowing_date'])
                harvest = pd.to_datetime(df['harvest_date'])
                invalid = (harvest <= sowing).sum()
                passed = invalid == 0
                self._add_result(
                    'harvest_after_sowing',
                    passed,
                    "All harvest dates after sowing" if passed else f"{invalid} records with invalid date sequence"
                )
            except:
                pass
        
        # Rule 2: total_days should match date difference
        if all(col in df.columns for col in ['sowing_date', 'harvest_date', 'total_days']):
            try:
                sowing = pd.to_datetime(df['sowing_date'])
                harvest = pd.to_datetime(df['harvest_date'])
                calculated = (harvest - sowing).dt.days
                mismatch = (calculated != df['total_days']).sum()
                passed = mismatch == 0
                self._add_result(
                    'total_days_consistency',
                    passed,
                    "total_days matches date difference" if passed else f"{mismatch} mismatched total_days",
                    severity='warning'
                )
            except:
                pass


def validate_farming_data(filepath: str) -> Tuple[bool, Dict]:
    """
    Convenience function to validate farming data file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (is_valid, validation_summary)
    """
    df = pd.read_csv(filepath)
    validator = DataValidator()
    return validator.validate(df, dataset_name=filepath.split('/')[-1])


def generate_validation_report(summary: Dict, output_path: Optional[str] = None) -> str:
    """
    Generate a markdown validation report.
    
    Args:
        summary: Validation summary dict
        output_path: Optional path to save report
        
    Returns:
        Markdown report string
    """
    report = f"""# Data Validation Report

**Dataset:** {summary['dataset']}  
**Timestamp:** {summary['timestamp']}  
**Total Rows:** {summary['total_rows']}  
**Status:** {'✅ PASSED' if summary['is_valid'] else '❌ FAILED'}

## Summary
- Validations Passed: {summary['validations_passed']}
- Validations Failed: {summary['validations_failed']}

## Details

| Check | Status | Message | Severity |
|-------|--------|---------|----------|
"""
    for detail in summary['details']:
        status = '✓' if detail['passed'] else '✗'
        report += f"| {detail['check']} | {status} | {detail['message']} | {detail.get('severity', 'error')} |\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


if __name__ == "__main__":
    # Example usage
    import os
    import sys
    
    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'data')
    
    # Validate post-processed data
    filepath = os.path.join(data_dir, 'Smart_farming_post_processing.csv')
    if os.path.exists(filepath):
        is_valid, summary = validate_farming_data(filepath)
        report = generate_validation_report(summary)
        print("\n" + report)
    else:
        print(f"File not found: {filepath}")
