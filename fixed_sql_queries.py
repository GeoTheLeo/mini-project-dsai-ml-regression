# fixed_sql_queries.py
import pandas as pd
import sqlite3
import numpy as np

print("="*60)
print("FIXED SQL QUERIES - CORRECTING PRICE CALCULATIONS")
print("="*60)

# Load your original data
df = pd.read_csv('data/house_data_tableau_fixed.csv')

# Identify the actual price column
price_col = None
for col in df.columns:
    if 'price' in str(col).lower():
        price_col = col
        print(f"✓ Found price column: '{price_col}'")
        break

if not price_col:
    # Find numeric column with highest values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].mean() > 50000:  # Looks like a price
            price_col = col
            print(f"✓ Using '{col}' as price column (avg: ${df[col].mean():,.0f})")
            break

if not price_col:
    print("❌ Could not find price column!")
    exit()

# Create SQLite database
conn = sqlite3.connect(':memory:')
df.to_sql('house_data', conn, index=False, if_exists='replace')

print(f"\n✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")
print(f"✓ Price column: '{price_col}' (range: ${df[price_col].min():,.0f} - ${df[price_col].max():,.0f})")

# ============ FIXED QUERY 1: PRICE CATEGORY DISTRIBUTION ============

print("\n" + "="*60)
print("FIXED QUERY 1: PRICE CATEGORY DISTRIBUTION")
print("="*60)

query1 = f"""
SELECT 
    CASE 
        WHEN {price_col} < 300000 THEN '< $300K'
        WHEN {price_col} BETWEEN 300000 AND 500000 THEN '$300K-500K'
        WHEN {price_col} BETWEEN 500000 AND 650000 THEN '$500K-650K'
        WHEN {price_col} BETWEEN 650000 AND 1000000 THEN '$650K-1M'
        ELSE '> $1M'
    END AS price_category,
    COUNT(*) AS property_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM house_data), 2) AS percentage,
    ROUND(AVG({price_col}), 2) AS avg_price_in_category,
    ROUND(MIN({price_col}), 2) AS min_price_in_category,
    ROUND(MAX({price_col}), 2) AS max_price_in_category
FROM house_data
GROUP BY 
    CASE 
        WHEN {price_col} < 300000 THEN '< $300K'
        WHEN {price_col} BETWEEN 300000 AND 500000 THEN '$300K-500K'
        WHEN {price_col} BETWEEN 500000 AND 650000 THEN '$500K-650K'
        WHEN {price_col} BETWEEN 650000 AND 1000000 THEN '$650K-1M'
        ELSE '> $1M'
    END
ORDER BY 
    CASE 
        WHEN {price_col} < 300000 THEN 1
        WHEN {price_col} BETWEEN 300000 AND 500000 THEN 2
        WHEN {price_col} BETWEEN 500000 AND 650000 THEN 3
        WHEN {price_col} BETWEEN 650000 AND 1000000 THEN 4
        ELSE 5
    END;
"""

print("Running Query 1...")
result1 = pd.read_sql_query(query1, conn)
print(result1.to_string(index=False))
result1.to_csv('data/price_categories_FIXED.csv', index=False)
print(f"✓ Saved to: data/price_categories_FIXED.csv")

# ============ FIXED QUERY 2: HIGH VALUE ANALYSIS ============

print("\n" + "="*60)
print("FIXED QUERY 2: HIGH VALUE ANALYSIS (≥ $650K)")
print("="*60)

# Check what columns are available for analysis
print("Available columns for analysis:")
key_columns = []
for col in df.columns:
    if col != price_col and df[col].dtype in [np.int64, np.float64]:
        if df[col].nunique() > 1:  # Not constant
            key_columns.append(col)

print(f"  • Price: {price_col}")
for col in key_columns[:10]:  # Show first 10
    print(f"  • {col}")

query2 = f"""
SELECT 
    CASE 
        WHEN {price_col} >= 650000 THEN 'High Value (≥$650K)' 
        ELSE 'Regular (<$650K)' 
    END AS property_type,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM house_data), 2) AS market_share_pct,
    ROUND(AVG({price_col}), 2) AS avg_price,
    ROUND(AVG({price_col}) / 1000, 0) * 1000 AS avg_price_rounded
FROM house_data
GROUP BY 
    CASE 
        WHEN {price_col} >= 650000 THEN 'High Value (≥$650K)' 
        ELSE 'Regular (<$650K)' 
    END;
"""

print("\nRunning Query 2...")
result2 = pd.read_sql_query(query2, conn)
print(result2.to_string(index=False))
result2.to_csv('data/high_value_analysis_FIXED.csv', index=False)
print(f"✓ Saved to: data/high_value_analysis_FIXED.csv")

# ============ FIXED QUERY 3: DETAILED HIGH VALUE CHARACTERISTICS ============

print("\n" + "="*60)
print("FIXED QUERY 3: DETAILED HIGH VALUE CHARACTERISTICS")
print("="*60)

# Find living area column
living_col = None
for col in df.columns:
    if any(term in str(col).lower() for term in ['sqft', 'area', 'living', 'size']):
        if df[col].dtype in [np.int64, np.float64]:
            living_col = col
            print(f"✓ Using '{living_col}' for living area")
            break

if living_col:
    query3 = f"""
    SELECT 
        CASE WHEN {price_col} >= 650000 THEN 'High Value (≥$650K)' ELSE 'Regular (<$650K)' END AS property_type,
        COUNT(*) AS count,
        ROUND(AVG({price_col}), 2) AS avg_price,
        ROUND(AVG({living_col}), 0) AS avg_living_area,
        ROUND(AVG({price_col} / NULLIF({living_col}, 0)), 2) AS avg_price_per_sqft
    FROM house_data
    GROUP BY property_type;
    """
    
    print("Running Query 3...")
    result3 = pd.read_sql_query(query3, conn)
    print(result3.to_string(index=False))
    result3.to_csv('data/high_value_detailed_FIXED.csv', index=False)
    print(f"✓ Saved to: data/high_value_detailed_FIXED.csv")

# ============ FIXED QUERY 4: WATERFRONT PREMIUM ============

if 'waterfront' in df.columns:
    print("\n" + "="*60)
    print("FIXED QUERY 4: WATERFRONT PREMIUM ANALYSIS")
    print("="*60)
    
    query4 = f"""
    SELECT 
        CASE WHEN waterfront = 1 THEN 'Waterfront' ELSE 'Non-Waterfront' END AS waterfront_status,
        COUNT(*) AS property_count,
        ROUND(AVG({price_col}), 2) AS avg_price,
        ROUND(AVG({price_col}) - (SELECT AVG({price_col}) FROM house_data WHERE waterfront = 0), 2) AS price_premium,
        ROUND((AVG({price_col}) / (SELECT AVG({price_col}) FROM house_data WHERE waterfront = 0) - 1) * 100, 2) AS premium_pct
    FROM house_data
    GROUP BY waterfront_status;
    """
    
    print("Running Query 4...")
    result4 = pd.read_sql_query(query4, conn)
    print(result4.to_string(index=False))
    result4.to_csv('data/waterfront_analysis_FIXED.csv', index=False)
    print(f"✓ Saved to: data/waterfront_analysis_FIXED.csv")

# ============ SUMMARY REPORT ============

print("\n" + "="*60)
print("SUMMARY REPORT - FIXED CALCULATIONS")
print("="*60)

# Calculate key metrics
total_properties = len(df)
high_value_count = df[df[price_col] >= 650000].shape[0]
high_value_pct = (high_value_count / total_properties) * 100

print(f"Total Properties: {total_properties:,}")
print(f"High-Value Properties (≥$650K): {high_value_count:,} ({high_value_pct:.1f}%)")
print(f"Average Price: ${df[price_col].mean():,.0f}")
print(f"Median Price: ${df[price_col].median():,.0f}")
print(f"Price Range: ${df[price_col].min():,.0f} - ${df[price_col].max():,.0f}")

# Price distribution summary
print(f"\nPrice Distribution Summary:")
bins = [0, 300000, 500000, 650000, 1000000, float('inf')]
labels = ['<$300K', '$300K-500K', '$500K-650K', '$650K-1M', '>$1M']

for i in range(len(labels)):
    if i == 0:
        count = df[df[price_col] < bins[1]].shape[0]
    elif i == len(labels) - 1:
        count = df[df[price_col] >= bins[i]].shape[0]
    else:
        count = df[(df[price_col] >= bins[i]) & (df[price_col] < bins[i+1])].shape[0]
    
    pct = (count / total_properties) * 100
    avg_price = df[(df[price_col] >= bins[i]) & (df[price_col] < bins[i+1])][price_col].mean() if count > 0 else 0
    print(f"  {labels[i]}: {count:,} properties ({pct:.1f}%), Avg: ${avg_price:,.0f}")

conn.close()

print("\n" + "="*60)
print("ALL FIXED FILES SAVED TO /data/ FOLDER:")
print("="*60)
print("✓ price_categories_FIXED.csv")
print("✓ high_value_analysis_FIXED.csv")
if living_col:
    print("✓ high_value_detailed_FIXED.csv")
if 'waterfront' in df.columns:
    print("✓ waterfront_analysis_FIXED.csv")
print("\n✅ FIXED SQL QUERIES COMPLETE!")