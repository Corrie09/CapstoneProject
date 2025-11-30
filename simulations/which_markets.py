import pandas as pd

df = pd.read_csv('notebooks/outputs/Final/okcupid_final_analysis_ready.csv')

# Check what location and orientation columns exist
print("Columns in data:")
print([col for col in df.columns if 'location' in col.lower() or 'orient' in col.lower()])

# Group by location and orientation
markets = df.groupby(['location', 'orientation']).agg({
    'is_ltr_oriented': ['mean', 'count'],
    'target_avg_effort': 'mean',
    'rating_index': 'mean'
}).round(3)

markets.columns = ['LTR_share', 'N_users', 'Market_clarity', 'Avg_rating']
markets = markets[markets['N_users'] >= 50]  # Only markets with 50+ users
markets = markets.sort_values('LTR_share')

print("\nMarkets in your data:")
print(markets)

# Find low, medium, high LTR markets
print("\n" + "="*80)
print("CANDIDATE MARKETS:")
print("="*80)

print("\nLOW LTR markets:")
print(markets.head(3))

print("\nMEDIUM LTR markets:")
median_idx = len(markets) // 2
print(markets.iloc[median_idx-1:median_idx+2])

print("\nHIGH LTR markets:")
print(markets.tail(3))