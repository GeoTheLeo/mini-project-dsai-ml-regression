# view_final_insights.py
import pandas as pd

print("="*60)
print("FINAL SQL INSIGHTS FOR MANAGEMENT")
print("="*60)

# Load all final results
print("\n📊 LOADING FINAL RESULTS...")

# 1. Price Categories
pc = pd.read_csv('data/final_price_categories.csv')
print("\n1. PRICE MARKET SEGMENTATION:")
for _, row in pc.iterrows():
    print(f"   • {row['price_category']}: {row['property_count']:,} properties ({row['percentage']}%)")
    print(f"     Avg price: ${row['avg_price_in_category']:,.0f}")

# 2. High Value Analysis
hv = pd.read_csv('data/final_high_value_analysis.csv')
print("\n2. HIGH-VALUE PROPERTIES (≥ $650K):")
high_value = hv[hv['property_type'].str.contains('High')]
regular = hv[hv['property_type'].str.contains('Regular')]

if not high_value.empty:
    hv_row = high_value.iloc[0]
    print(f"   • Count: {int(hv_row['count']):,} properties")
    print(f"   • Market Share: {hv_row['market_share_pct']}%")
    print(f"   • Average Price: ${hv_row['avg_price']:,.0f}")
    print(f"   • Average Size: {hv_row['avg_living_area']:,.0f} sqft")
    print(f"   • Average Grade: {hv_row['avg_grade']}")
    print(f"   • Waterfront: {hv_row['waterfront_pct']}%")

if not regular.empty:
    reg_row = regular.iloc[0]
    print(f"\n3. REGULAR PROPERTIES (< $650K):")
    print(f"   • Count: {int(reg_row['count']):,} properties")
    print(f"   • Market Share: {reg_row['market_share_pct']}%")
    print(f"   • Average Price: ${reg_row['avg_price']:,.0f}")

# 3. Waterfront Premium
wf = pd.read_csv('data/final_waterfront_premium.csv')
print("\n4. WATERFRONT PREMIUM:")
if not wf.empty and len(wf) > 1:
    waterfront = wf[wf['waterfront_status'] == 'Waterfront']
    non_waterfront = wf[wf['waterfront_status'] == 'Non-Waterfront']
    
    if not waterfront.empty and not non_waterfront.empty:
        wf_row = waterfront.iloc[0]
        nwf_row = non_waterfront.iloc[0]
        print(f"   • Waterfront properties: {wf_row['property_count']:,}")
        print(f"   • Price Premium: {wf_row['premium_pct']}%")
        print(f"   • Avg Waterfront Price: ${wf_row['avg_price']:,.0f}")
        print(f"   • Avg Non-Waterfront Price: ${nwf_row['avg_price']:,.0f}")

# 4. Correlations
corr = pd.read_csv('data/final_correlations.csv')
print("\n5. TOP PRICE PREDICTORS:")
for _, row in corr.head(3).iterrows():
    print(f"   • {row['feature']}: {row['correlation_with_price']} ({row['strength']}, {row['direction']})")

# 5. Executive Summary
exec_sum = pd.read_csv('data/final_executive_summary.csv')
if not exec_sum.empty:
    print("\n6. EXECUTIVE SUMMARY:")
    es = exec_sum.iloc[0]
    print(f"   • Total Properties: {int(es['total_properties']):,}")
    print(f"   • Average Price: ${es['avg_price']:,.0f}")
    print(f"   • High-Value Properties: {int(es['high_value_count']):,} ({es['high_value_market_share']}%)")
    print(f"   • Waterfront Premium: {es['waterfront_premium_pct']}%")
    print(f"   • Most Common Price Range: {es['most_common_price_range']}")

print("\n" + "="*60)
print("KEY BUSINESS INSIGHTS")
print("="*60)

# Calculate key insights
total_props = int(pc['property_count'].sum())
high_value_count = int(hv[hv['property_type'].str.contains('High')]['count'].sum()) if not hv.empty else 0
high_value_pct = (high_value_count / total_props * 100) if total_props > 0 else 0

print(f"1. Market Size: {total_props:,} properties analyzed")
print(f"2. High-Value Opportunity: {high_value_count:,} properties ≥$650K ({high_value_pct:.1f}% of market)")
print(f"3. Largest Segment: {pc.loc[pc['property_count'].idxmax()]['price_category']}")

if not corr.empty:
    top_predictor = corr.iloc[0]['feature']
    print(f"4. Top Price Driver: {top_predictor}")

print("\n✅ READY FOR MANAGEMENT PRESENTATION!")