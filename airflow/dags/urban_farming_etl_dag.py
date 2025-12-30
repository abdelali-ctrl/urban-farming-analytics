"""
Urban Farming ETL Pipeline DAG
==============================
Orchestrates the complete ETL pipeline for urban farming data.

Schedule: Daily at 6:00 AM
Tasks:
    1. Validate source data (check file exists)
    2. Run CSV processing (Python callable)
    3. Retrain ML models
    4. Log pipeline success

Note: This DAG runs inside Docker container with access to project files
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os
import sys

# Project directory inside the container
PROJECT_DIR = '/opt/airflow/project'

# Default arguments for all tasks
default_args = {
    'owner': 'urban_farming',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=30),  # Increased for ML training
}


def validate_data(**context):
    """Task 1: Validate source data exists and is readable."""
    import pandas as pd
    
    # Use the extended dataset
    data_path = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Try to read the file
    df = pd.read_csv(data_path)
    row_count = len(df)
    col_count = len(df.columns)
    
    print(f"✅ Data validation passed!")
    print(f"   - File: {data_path}")
    print(f"   - Rows: {row_count}")
    print(f"   - Columns: {col_count}")
    print(f"   - Crop types: {df['crop_type'].nunique()}")
    print(f"   - Regions: {df['region'].nunique()}")
    
    # Push to XCom
    context['ti'].xcom_push(key='row_count', value=row_count)
    context['ti'].xcom_push(key='is_valid', value=True)
    
    return f"Validated {row_count} rows"


def process_data(**context):
    """Task 2: Process and transform the farming data."""
    import pandas as pd
    import numpy as np
    
    data_path = os.path.join(PROJECT_DIR, 'data', 'farming_with_prices_extended.csv')
    output_path = os.path.join(PROJECT_DIR, 'data', 'farming_processed_airflow.csv')
    
    # Read data
    df = pd.read_csv(data_path)
    
    # Add processing timestamp
    df['processed_at'] = datetime.now().isoformat()
    df['processed_by'] = 'airflow'
    
    # Calculate derived metrics if not present
    if 'yield_per_day' not in df.columns:
        df['yield_per_day'] = df['yield_kg_per_hectare'] / df['total_days']
    
    if 'water_efficiency' not in df.columns:
        df['water_efficiency'] = df['yield_kg_per_hectare'] / (df['rainfall_mm'] + 1)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    
    print(f"✅ Data processing complete!")
    print(f"   - Input: {len(df)} rows")
    print(f"   - Output: {output_path}")
    
    return f"Processed {len(df)} rows"


def retrain_models(**context):
    """Task 3: Retrain ML models with the latest data."""
    import subprocess
    import sys
    
    # Path to the ML training script
    ml_script = os.path.join(PROJECT_DIR, 'scripts', 'ml', 'yield_prediction.py')
    
    if not os.path.exists(ml_script):
        print(f"⚠️ ML script not found: {ml_script}")
        print("   Skipping model retraining...")
        return "ML script not found - skipped"
    
    print(f"🔄 Starting ML model retraining...")
    print(f"   - Script: {ml_script}")
    
    try:
        # Run the ML training script
        result = subprocess.run(
            [sys.executable, ml_script],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ ML model retraining complete!")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            
            # Push model path to XCom
            model_path = os.path.join(PROJECT_DIR, 'models', 'yield_prediction_pipeline.pkl')
            context['ti'].xcom_push(key='model_path', value=model_path)
            
            return "ML models retrained successfully"
        else:
            print(f"❌ ML training failed with code {result.returncode}")
            print(f"   Error: {result.stderr}")
            raise Exception(f"ML training failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        raise Exception("ML training timed out after 10 minutes")
    except Exception as e:
        print(f"❌ Error during ML training: {e}")
        raise


def log_pipeline_success(**context):
    """Task 4: Log successful pipeline completion."""
    import json
    
    row_count = context['ti'].xcom_pull(key='row_count', task_ids='validate_source_data')
    model_path = context['ti'].xcom_pull(key='model_path', task_ids='retrain_models')
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'SUCCESS',
        'dag_id': 'urban_farming_etl',
        'execution_date': str(context['execution_date']),
        'rows_processed': row_count,
        'model_retrained': model_path is not None,
        'model_path': model_path,
    }
    
    log_path = os.path.join(PROJECT_DIR, 'logs', 'airflow_pipeline_runs.jsonl')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"✅ Pipeline logged successfully!")
    print(f"   - Log file: {log_path}")
    print(f"   - Model retrained: {model_path is not None}")
    
    return "Pipeline logged"


# Define the DAG
with DAG(
    'urban_farming_etl',
    default_args=default_args,
    description='Daily ETL pipeline for Urban Farming data processing and ML model retraining',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['urban-farming', 'etl', 'data-warehouse', 'ml-training'],
) as dag:
    
    # Task 1: Validate source data
    validate_task = PythonOperator(
        task_id='validate_source_data',
        python_callable=validate_data,
        provide_context=True,
    )
    
    # Task 2: Process data
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        provide_context=True,
    )
    
    # Task 3: Retrain ML models
    retrain_task = PythonOperator(
        task_id='retrain_models',
        python_callable=retrain_models,
        provide_context=True,
    )
    
    # Task 4: Log success
    log_task = PythonOperator(
        task_id='log_pipeline_success',
        python_callable=log_pipeline_success,
        provide_context=True,
    )
    
    # Define task dependencies
    # validate -> process -> retrain -> log
    validate_task >> process_task >> retrain_task >> log_task

