# run_final_sql.py
import pandas as pd
import sqlite3

print("="*60)
print("FINAL SQL ANALYSIS WITH FIXED PRICES")
print("="*60)

# Load the FIXED data
df = pd.read_csv('data/house_data_PRICE_FIXED.csv')

# Create SQLite database
conn = sqlite3.connect(':memory:')
df.to_sql('house_data', conn, index=False, if_exists='replace')

print(f"✓ Data loaded: {len(df):,} properties")
print(f"✓ Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
print(f"✓ Average price: ${df['price'].mean():,.0f}")

# ============ FINAL QUERY 1: PRICE CATEGORIES ============

query1 = """
SELECT 
    CASE 
        WHEN price < 300000 THEN '< $300K'
        WHEN price BETWEEN 300000 AND 500000 THEN '$300K-500K'
        WHEN price BETWEEN 500000 AND 650000 THEN '$500K-650K'
        WHEN price BETWEEN 650000 AND 1000000 THEN '$650K-1M'
        ELSE '> $1M'
    END AS price_category,
    COUNT(*) AS property_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM house_data), 2) AS percentage,
    ROUND(AVG(price), 0) AS avg_price_in_category,
    ROUND(MIN(price), 0) AS min_price_in_category,
    ROUND(MAX(price), 0) AS max_price_in_category
FROM house_data
GROUP BY price_category
ORDER BY 
    CASE price_category
        WHEN '< $300K' THEN 1
        WHEN '$300K-500K' THEN 2
        WHEN '$500K-650K' THEN 3
        WHEN '$650K-1M' THEN 4
        ELSE 5
    END;
"""

print("\n" + "="*60)
print("1. PRICE CATEGORY DISTRIBUTION")
print("="*60)
result1 = pd.read_sql_query(query1, conn)
print(result1.to_string(index=False))
result1.to_csv('data/final_price_categories.csv', index=False)

# ============ FINAL QUERY 2: HIGH VALUE ANALYSIS ============

query2 = """
SELECT 
    CASE 
        WHEN price >= 650000 THEN 'High Value (≥$650K)' 
        ELSE 'Regular (<$650K)' 
    END AS property_type,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM house_data), 2) AS market_share_pct,
    ROUND(AVG(price), 0) AS avg_price,
    ROUND(AVG(sqft_living), 0) AS avg_living_area,
    ROUND(AVG(grade), 2) AS avg_grade,
    ROUND(AVG(bathrooms), 2) AS avg_bathrooms,
    ROUND(AVG(bedrooms), 2) AS avg_bedrooms,
    ROUND(SUM(CASE WHEN waterfront = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS waterfront_pct,
    ROUND(AVG(price / NULLIF(sqft_living, 0)), 0) AS avg_price_per_sqft
FROM house_data
GROUP BY property_type;
"""

print("\n" + "="*60)
print("2. HIGH VALUE PROPERTIES ANALYSIS")
print("="*60)
result2 = pd.read_sql_query(query2, conn)
print(result2.to_string(index=False))
result2.to_csv('data/final_high_value_analysis.csv', index=False)

# ============ FINAL QUERY 3: TOP CORRELATIONS ============

query3 = """
WITH correlations AS (
    SELECT 'sqft_living' AS feature, CORR(price, sqft_living) AS correlation FROM house_data
    UNION ALL SELECT 'grade', CORR(price, grade) FROM house_data
    UNION ALL SELECT 'bathrooms', CORR(price, bathrooms) FROM house_data
    UNION ALL SELECT 'sqft_above', CORR(price, sqft_above) FROM house_data
    UNION ALL SELECT 'view', CORR(price, view) FROM house_data
    UNION ALL SELECT 'waterfront', CORR(price, waterfront) FROM house_data
    UNION ALL SELECT 'bedrooms', CORR(price, bedrooms) FROM house_data
    UNION ALL SELECT 'sqft_lot', CORR(price, sqft_lot) FROM house_data
)
SELECT 
    feature,
    ROUND(correlation, 3) AS correlation_with_price,
    CASE 
        WHEN ABS(correlation) >= 0.7 THEN 'Very Strong'
        WHEN ABS(correlation) >= 0.5 THEN 'Strong'
        WHEN ABS(correlation) >= 0.3 THEN 'Moderate'
        WHEN ABS(correlation) >= 0.1 THEN 'Weak'
        ELSE 'Very Weak'
    END AS strength,
    CASE 
        WHEN correlation > 0 THEN 'Positive' 
        WHEN correlation < 0 THEN 'Negative' 
        ELSE 'No Correlation' 
    END AS direction
FROM correlations
ORDER BY ABS(correlation) DESC
LIMIT 10;
"""

print("\n" + "="*60)
print("3. TOP CORRELATIONS WITH PRICE")
print("="*60)
result3 = pd.read_sql_query(query3, conn)
print(result3.to_string(index=False))
result3.to_csv('data/final_correlations.csv', index=False)

# ============ FINAL QUERY 4: WATERFRONT PREMIUM ============

query4 = """
SELECT 
    CASE WHEN waterfront = 1 THEN 'Waterfront' ELSE 'Non-Waterfront' END AS waterfront_status,
    COUNT(*) AS property_count,
    ROUND(AVG(price), 0) AS avg_price,
    ROUND(AVG(price) - (SELECT AVG(price) FROM house_data WHERE waterfront = 0), 0) AS price_premium,
    ROUND((AVG(price) / (SELECT AVG(price) FROM house_data WHERE waterfront = 0) - 1) * 100, 2) AS premium_pct,
    ROUND(AVG(sqft_living), 0) AS avg_living_area,
    ROUND(AVG(grade), 2) AS avg_grade
FROM house_data
GROUP BY waterfront_status;
"""

print("\n" + "="*60)
print("4. WATERFRONT PREMIUM")
print("="*60)
result4 = pd.read_sql_query(query4, conn)
print(result4.to_string(index=False))
result4.to_csv('data/final_waterfront_premium.csv', index=False)

# ============ FINAL QUERY 5: EXECUTIVE SUMMARY ============

query5 = """
SELECT 
    -- Market Size
    (SELECT COUNT(*) FROM house_data) AS total_properties,
    
    -- Price Statistics
    (SELECT ROUND(AVG(price), 0) FROM house_data) AS avg_price,
    (SELECT ROUND(MIN(price), 0) FROM house_data) AS min_price,
    (SELECT ROUND(MAX(price), 0) FROM house_data) AS max_price,
    
    -- High-Value Market
    (SELECT COUNT(*) FROM house_data WHERE price >= 650000) AS high_value_count,
    (SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM house_data), 2) 
     FROM house_data WHERE price >= 650000) AS high_value_market_share,
    
    -- Waterfront Premium
    (SELECT ROUND((AVG(price) / 
        (SELECT AVG(price) FROM house_data WHERE waterfront = 0) - 1) * 100, 2)
     FROM house_data WHERE waterfront = 1) AS waterfront_premium_pct,
    
    -- Most Common Price Range
    (SELECT price_category FROM (
        SELECT 
            CASE 
                WHEN price < 300000 THEN '< $300K'
                WHEN price BETWEEN 300000 AND 500000 THEN '$300K-500K'
                WHEN price BETWEEN 500000 AND 650000 THEN '$500K-650K'
                WHEN price BETWEEN 650000 AND 1000000 THEN '$650K-1M'
                ELSE '> $1M'
            END AS price_category,
            COUNT(*) as cnt
        FROM house_data
        GROUP BY price_category
        ORDER BY cnt DESC
        LIMIT 1
    )) AS most_common_price_range
FROM house_data
LIMIT 1;
"""

print("\n" + "="*60)
print("5. EXECUTIVE SUMMARY")
print("="*60)
result5 = pd.read_sql_query(query5, conn)
print(result5.to_string(index=False))
result5.to_csv('data/final_executive_summary.csv', index=False)

conn.close()

print("\n" + "="*60)
print("ALL FINAL REPORTS SAVED TO /data/ FOLDER:")
print("="*60)
print("✓ final_price_categories.csv")
print("✓ final_high_value_analysis.csv")
print("✓ final_correlations.csv")
print("✓ final_waterfront_premium.csv")
print("✓ final_executive_summary.csv")

print("\n✅ FINAL SQL ANALYSIS COMPLETE!")