"""
Database Verification Script - Validates the complete data warehouse schema.
"""

import pyodbc

def verify_database():
    conn_str = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost;"
        "Database=UrbanFarmingDW;"
        "Trusted_Connection=yes;"
    )
    
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    print("=" * 70)
    print("  SQL SERVER DATA WAREHOUSE VERIFICATION")
    print("=" * 70)
    
    # Check all tables
    tables = [
        ('Dim_Crop', 'crop_type'),
        ('Dim_Region', 'region_name'),
        ('Dim_Irrigation', 'irrigation_type'),
        ('Dim_Disease', 'disease_status'),
        ('Dim_Time', 'full_date'),
        ('Fact_Production', 'farm_id'),
        ('Fact_CommodityPrices', 'price_date')
    ]
    
    print("\n--- TABLE RECORD COUNTS ---")
    for table, col in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:25} : {count:,} records")
    
    # Show Dim_Time sample
    print("\n--- DIM_TIME SAMPLE ---")
    cursor.execute("SELECT TOP 5 time_id, full_date, year, month, quarter, season FROM Dim_Time ORDER BY full_date")
    for row in cursor.fetchall():
        print(f"  ID:{row[0]:4} | {row[1]} | Year:{row[2]} | Month:{row[3]:2} | Q{row[4]} | {row[5]}")
    
    # Show Fact_Production with all dimension joins
    print("\n--- FACT_PRODUCTION WITH DIMENSION JOINS ---")
    cursor.execute("""
        SELECT TOP 5 
            fp.farm_id,
            dc.crop_type,
            dr.region_name,
            di.irrigation_type,
            dd.disease_status,
            t1.season as sowing_season,
            t2.season as harvest_season,
            fp.yield_kg_per_hectare,
            fp.revenue_per_hectare_EUR
        FROM Fact_Production fp
        JOIN Dim_Crop dc ON fp.crop_id = dc.crop_id
        JOIN Dim_Region dr ON fp.region_id = dr.region_id
        JOIN Dim_Irrigation di ON fp.irrigation_id = di.irrigation_id
        JOIN Dim_Disease dd ON fp.disease_id = dd.disease_id
        LEFT JOIN Dim_Time t1 ON fp.sowing_time_id = t1.time_id
        LEFT JOIN Dim_Time t2 ON fp.harvest_time_id = t2.time_id
    """)
    
    print(f"  {'Farm':<10} {'Crop':<8} {'Region':<12} {'Irrigation':<10} {'Disease':<10} {'SowSeason':<10} {'Yield':>10}")
    print("  " + "-" * 80)
    for row in cursor.fetchall():
        print(f"  {row[0]:<10} {row[1]:<8} {row[2]:<12} {row[3]:<10} {row[4]:<10} {row[5] or 'N/A':<10} {row[7]:>10,.0f}")
    
    # Show Fact_CommodityPrices with time dimension join
    print("\n--- FACT_COMMODITYPRICES WITH TIME DIMENSION ---")
    cursor.execute("""
        SELECT TOP 5
            fcp.price_date,
            dt.year,
            dt.quarter,
            dt.season,
            fcp.corn_eur_per_kg,
            fcp.wheat_eur_per_kg,
            fcp.rice_eur_per_kg
        FROM Fact_CommodityPrices fcp
        JOIN Dim_Time dt ON fcp.time_id = dt.time_id
        ORDER BY fcp.price_date DESC
    """)
    
    print(f"  {'Date':<12} {'Year':<6} {'Q':<3} {'Season':<8} {'Corn':>8} {'Wheat':>8} {'Rice':>8}")
    print("  " + "-" * 60)
    for row in cursor.fetchall():
        print(f"  {str(row[0]):<12} {row[1]:<6} Q{row[2]:<2} {row[3]:<8} {row[4]:>8.3f} {row[5]:>8.3f} {row[6] or 0:>8.3f}")
    
    # Summary statistics
    print("\n--- ANALYTICS SAMPLE: Average Yield by Season ---")
    cursor.execute("""
        SELECT 
            t1.season,
            AVG(fp.yield_kg_per_hectare) as avg_yield,
            COUNT(*) as farm_count
        FROM Fact_Production fp
        LEFT JOIN Dim_Time t1 ON fp.sowing_time_id = t1.time_id
        WHERE t1.season IS NOT NULL
        GROUP BY t1.season
        ORDER BY avg_yield DESC
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]:<10}: {row[1]:,.0f} kg/ha avg (from {row[2]} farms)")
    
    conn.close()
    print("\n" + "=" * 70)
    print("  [SUCCESS] Data warehouse verification complete!")
    print("=" * 70)


if __name__ == "__main__":
    verify_database()
