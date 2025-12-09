# view_fixed_results.py
import pandas as pd

print("="*60)
print("VIEWING FIXED SQL RESULTS")
print("="*60)

# Load fixed files
print("\n1. FIXED PRICE CATEGORIES:")
pc_fixed = pd.read_csv('data/price_categories_FIXED.csv')
print(pc_fixed.to_string(index=False))

print("\n2. FIXED HIGH VALUE ANALYSIS:")
hv_fixed = pd.read_csv('data/high_value_analysis_FIXED.csv')
print(hv_fixed.to_string(index=False))

# Extract exact insights
print("\n" + "="*60)
print("EXACT INSIGHTS FOR MANAGEMENT")
print("="*60)

# From price categories
print("\n💰 PRICE DISTRIBUTION:")
for _, row in pc_fixed.iterrows():
    print(f"• {row['price_category']}: {row['property_count']:,} properties ({row['percentage']:.1f}%)")
    print(f"  Average price: ${row['avg_price_in_category']:,.0f}")

# From high value analysis
print("\n🏆 HIGH-VALUE PROPERTIES (≥ $650K):")
high_value_row = hv_fixed[hv_fixed['property_type'].str.contains('High')]
if not high_value_row.empty:
    hv = high_value_row.iloc[0]
    print(f"• Count: {int(hv['count']):,} properties")
    print(f"• Market Share: {hv['market_share_pct']:.1f}%")
    print(f"• Average Price: ${hv['avg_price']:,.0f}")

regular_row = hv_fixed[hv_fixed['property_type'].str.contains('Regular')]
if not regular_row.empty:
    reg = regular_row.iloc[0]
    print(f"\n📊 REGULAR PROPERTIES (< $650K):")
    print(f"• Count: {int(reg['count']):,} properties")
    print(f"• Market Share: {reg['market_share_pct']:.1f}%")
    print(f"• Average Price: ${reg['avg_price']:,.0f}")

# Calculate premium
if not high_value_row.empty and not regular_row.empty:
    premium = (hv['avg_price'] / reg['avg_price'] - 1) * 100
    print(f"\n📈 HIGH-VALUE PREMIUM: {premium:.1f}% higher than regular properties")

# Summary
print("\n" + "="*60)
print("KEY BUSINESS INSIGHTS")
print("="*60)

# Find which price category has most properties
max_category = pc_fixed.loc[pc_fixed['property_count'].idxmax()]
print(f"1. Largest Market Segment: {max_category['price_category']} ({max_category['percentage']:.1f}% of properties)")

# Find highest average price category
max_avg_price = pc_fixed.loc[pc_fixed['avg_price_in_category'].idxmax()]
print(f"2. Most Expensive Segment: {max_avg_price['price_category']} (avg: ${max_avg_price['avg_price_in_category']:,.0f})")

# Calculate market concentration
top_2_segments = pc_fixed.nlargest(2, 'property_count')
top_2_pct = top_2_segments['percentage'].sum()
print(f"3. Market Concentration: Top 2 segments represent {top_2_pct:.1f}% of market")

# High-value market opportunity
if not high_value_row.empty:
    print(f"4. High-Value Opportunity: {hv['market_share_pct']:.1f}% of properties are ≥$650K")
    
print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print("• Focus marketing on the {max_category['price_category']} segment (largest market)")
print("• Target {max_avg_price['price_category']} segment for premium offerings")
print(f"• High-value properties represent significant opportunity at {hv['market_share_pct']:.1f}% of market")