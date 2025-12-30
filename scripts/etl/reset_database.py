"""
Reset database tables for schema update.
Drops existing tables and allows them to be recreated with the new schema.
"""

import pyodbc

def reset_tables():
    conn_str = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=UrbanFarmingDB;"
        "Trusted_Connection=yes;"
    )
    
    conn = pyodbc.connect(conn_str)
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("[INFO] Dropping existing tables to apply new schema...")
    
    # Drop in correct order (fact tables first, then dimensions)
    tables = [
        'Fact_CommodityPrices',
        'Fact_Production',
        'Dim_Disease',
        'Dim_Irrigation',
        'Dim_Time',
        'Dim_Region',
        'Dim_Crop'
    ]
    
    for table in tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            print(f"  - Dropped {table}")
        except Exception as e:
            print(f"  - Could not drop {table}: {e}")
    
    conn.close()
    print("[SUCCESS] Tables dropped. Run data_ingestion.py to recreate with new schema.")


if __name__ == "__main__":
    reset_tables()
