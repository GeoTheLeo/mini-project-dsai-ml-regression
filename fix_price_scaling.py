# fix_price_scaling.py
import pandas as pd
import numpy as np

print("="*60)
print("FIXING PRICE SCALING ISSUE")
print("="*60)

# Load your Tableau-ready data
df = pd.read_csv('data/house_data_tableau_fixed.csv')

print(f"Original data shape: {df.shape}")

# Find the price column
price_col = 'price'  # Based on your output

print(f"\nBEFORE FIX - Price column '{price_col}':")
print(f"  Min: ${df[price_col].min():,.2f}")
print(f"  Max: ${df[price_col].max():,.2f}")
print(f"  Mean: ${df[price_col].mean():,.2f}")
print(f"  Median: ${df[price_col].median():,.2f}")

# Check if this is the scaling issue
print(f"\nDIAGNOSIS:")
print(f"  • Your prices range from ${df[price_col].min():,.0f} to ${df[price_col].max():,.0f}")
print(f"  • Typical house prices should be $100,000+")
print(f"  • SUSPECTED ISSUE: Prices are in $100,000 units (e.g., '3' means $300,000)")

# Ask for confirmation
print(f"\nPOSSIBLE SCALING FACTORS:")
print(f"  1. Multiply by 100,000: ${df[price_col].min()*100000:,.0f} - ${df[price_col].max()*100000:,.0f}")
print(f"  2. Multiply by 10,000: ${df[price_col].min()*10000:,.0f} - ${df[price_col].max()*10000:,.0f}")
print(f"  3. Multiply by 1,000: ${df[price_col].min()*1000:,.0f} - ${df[price_col].max()*1000:,.0f}")

# Based on your earlier output showing $3 = $300,000, use 100,000 multiplier
scaling_factor = 100000
df[price_col] = df[price_col] * scaling_factor

print(f"\nAFTER FIX - Price column '{price_col}' (×{scaling_factor:,}):")
print(f"  Min: ${df[price_col].min():,.0f}")
print(f"  Max: ${df[price_col].max():,.0f}")
print(f"  Mean: ${df[price_col].mean():,.0f}")
print(f"  Median: ${df[price_col].median():,.0f}")

# Verify the fix makes sense
print(f"\nVERIFICATION:")
print(f"  • Min price: ${df[price_col].min():,.0f} → Reasonable? {'✓' if df[price_col].min() > 50000 else '✗'}")
print(f"  • Max price: ${df[price_col].max():,.0f} → Reasonable? {'✓' if df[price_col].max() < 10000000 else '✗'}")
print(f"  • Median: ${df[price_col].median():,.0f} → Reasonable? {'✓' if 200000 < df[price_col].median() < 1000000 else '✗'}")

# Save the fixed data
df.to_csv('data/house_data_PRICE_FIXED.csv', index=False)
print(f"\n✓ Saved fixed data to: data/house_data_PRICE_FIXED.csv")

# Quick analysis with fixed prices
print(f"\n" + "="*60)
print("QUICK ANALYSIS WITH FIXED PRICES")
print("="*60)

total = len(df)
high_value = df[df[price_col] >= 650000]
high_value_count = len(high_value)
high_value_pct = (high_value_count / total) * 100

print(f"Total Properties: {total:,}")
print(f"High-Value (≥$650K): {high_value_count:,} ({high_value_pct:.1f}%)")
print(f"Average Price: ${df[price_col].mean():,.0f}")
print(f"Median Price: ${df[price_col].median():,.0f}")

# Price distribution
bins = [0, 300000, 500000, 650000, 1000000, float('inf')]
labels = ['<$300K', '$300K-500K', '$500K-650K', '$650K-1M', '>$1M']

print(f"\nPrice Distribution:")
for i in range(len(labels)):
    if i == 0:
        mask = df[price_col] < bins[1]
    elif i == len(labels) - 1:
        mask = df[price_col] >= bins[i]
    else:
        mask = (df[price_col] >= bins[i]) & (df[price_col] < bins[i+1])
    
    count = df[mask].shape[0]
    pct = (count / total) * 100
    avg = df[mask][price_col].mean() if count > 0 else 0
    print(f"  {labels[i]}: {count:,} ({pct:.1f}%), Avg: ${avg:,.0f}")