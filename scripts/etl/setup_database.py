"""
Database Setup Script for Urban Farming ETL Pipeline
Creates the UrbanFarmingDB database if it doesn't exist.
"""

import pyodbc

def create_database():
    """Create the UrbanFarmingDB database if it doesn't exist."""
    
    # Connect to master database
    conn_str = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=master;"
        "Trusted_Connection=yes;"
    )
    
    try:
        conn = pyodbc.connect(conn_str)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT name FROM sys.databases WHERE name = 'UrbanFarmingDB'")
        result = cursor.fetchone()
        
        if result:
            print("[INFO] Database 'UrbanFarmingDB' already exists.")
        else:
            print("[INFO] Creating database 'UrbanFarmingDB'...")
            cursor.execute("CREATE DATABASE UrbanFarmingDB")
            print("[SUCCESS] Database 'UrbanFarmingDB' created successfully!")
        
        conn.close()
        return True
        
    except pyodbc.Error as e:
        print(f"[ERROR] SQL Server connection failed: {e}")
        print("\nPossible fixes:")
        print("1. Make sure SQL Server is running")
        print("2. Try: Server=.\\SQLEXPRESS if using SQL Server Express")
        print("3. Check Windows Authentication is enabled")
        return False


if __name__ == "__main__":
    create_database()
