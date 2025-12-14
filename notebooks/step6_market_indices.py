"""
Step 6: Create Market-Level Indices
OkCupid Dating App Project

This script calculates market-level metrics for each market (location × sex × orientation):
1. Market seriousness (ρ_m): % of LTR-oriented users in the TARGET market
2. Signal clarity: average profile completeness/effort of potential partners
3. Competition metrics: gender ratios, market size, scarcity
4. Average rating: mean rating of potential partners

These market indices are used to test the frustration hypothesis:
- Do users in markets with low ρ_m (few LTR partners) show lower effort?
- Do markets with unclear signals show more adapted behavior?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("STEP 6: CREATE MARKET-LEVEL INDICES")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading dataset with relationship goals...")

try:
    df = pd.read_csv('okcupid_with_goals.csv')
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'okcupid_with_goals.csv' not found.")
    print("Please run step5_relationship_goals.py first!")
    exit()

# ============================================================================
# 1. UNDERSTAND MARKET STRUCTURE
# ============================================================================
print("\n[2] Understanding market structure...")

print(f"  Total unique market_ids: {df['market_id'].nunique()}")
print(f"  Total unique target_markets: {df['target_market'].nunique()}")

# Markets with sufficient data (≥20 profiles)
market_sizes = df['market_id'].value_counts()
large_markets = market_sizes[market_sizes >= 20]
print(f"  Markets with ≥20 profiles: {len(large_markets)}")

# Show top markets
print("\n  Top 10 markets by size:")
print(market_sizes.head(10).to_string())

# ============================================================================
# 2. CALCULATE TARGET MARKET METRICS
# ============================================================================
print("\n[3] Calculating target market characteristics...")

# For each user, we need to know the characteristics of their TARGET market
# (the market they are looking at, not their own market)

# Create target market statistics
target_market_stats = df.groupby('target_market').agg({
    'is_ltr_oriented': 'mean',  # Market seriousness (ρ_m)
    'effort_index': 'mean',     # Signal clarity (avg effort)
    'rating_index': 'mean',     # Average rating of potential partners
    'demographic_completeness': 'mean',  # Profile completeness
    'age': 'mean',
    'education_level': 'mean',
    'market_id': 'count'  # Market size
}).round(4)

target_market_stats.columns = [
    'target_ltr_share',         # ρ_m: share of LTR users in target market
    'target_avg_effort',        # Average effort of potential partners
    'target_avg_rating',        # Average rating of potential partners
    'target_avg_completeness',  # Signal clarity
    'target_avg_age',
    'target_avg_education',
    'target_market_size'        # Number of potential partners
]

target_market_stats = target_market_stats.reset_index()

print(f"  ✓ Target market statistics calculated for {len(target_market_stats)} markets")
print(f"\n  Summary of target market characteristics:")
print(target_market_stats[['target_ltr_share', 'target_avg_effort', 
                           'target_avg_rating', 'target_market_size']].describe().round(3).to_string())

# ============================================================================
# 3. CALCULATE COMPETITION METRICS
# ============================================================================
print("\n[4] Calculating competition metrics...")

# For straight users: calculate opposite-sex ratio in location
# For gay/lesbian: calculate same-sex ratio

def calculate_competition_metrics(df_local):
    """Calculate competition metrics by location and orientation"""
    
    location_orient_stats = []
    
    for location in df_local['location'].unique():
        if pd.isna(location):
            continue
            
        loc_data = df_local[df_local['location'] == location]
        
        # For straight users
        straight_data = loc_data[loc_data['orientation'] == 'straight']
        if len(straight_data) > 0:
            males = (straight_data['sex'] == 'm').sum()
            females = (straight_data['sex'] == 'f').sum()
            
            # Male perspective: ratio of males to females (competition for females)
            male_female_ratio = males / females if females > 0 else np.nan
            
            # Female perspective: ratio of females to males (competition for males)
            female_male_ratio = females / males if males > 0 else np.nan
            
            location_orient_stats.append({
                'location': location,
                'orientation': 'straight',
                'sex': 'm',
                'competition_ratio': male_female_ratio,
                'market_size': males + females
            })
            
            location_orient_stats.append({
                'location': location,
                'orientation': 'straight',
                'sex': 'f',
                'competition_ratio': female_male_ratio,
                'market_size': males + females
            })
        
        # For gay users
        for orientation in ['gay', 'bisexual']:
            orient_data = loc_data[loc_data['orientation'] == orientation]
            if len(orient_data) > 0:
                males = (orient_data['sex'] == 'm').sum()
                females = (orient_data['sex'] == 'f').sum()
                total = males + females
                
                location_orient_stats.append({
                    'location': location,
                    'orientation': orientation,
                    'sex': 'm',
                    'competition_ratio': 1.0,  # Same pool
                    'market_size': males if orientation == 'gay' else total
                })
                
                location_orient_stats.append({
                    'location': location,
                    'orientation': orientation,
                    'sex': 'f',
                    'competition_ratio': 1.0,
                    'market_size': females if orientation == 'gay' else total
                })
    
    return pd.DataFrame(location_orient_stats)

competition_df = calculate_competition_metrics(df)

print(f"  ✓ Competition metrics calculated")
print(f"\n  Competition ratio statistics (straight users):")
straight_comp = competition_df[competition_df['orientation'] == 'straight']
print(straight_comp[['sex', 'competition_ratio']].groupby('sex').describe().round(3).to_string())

# ============================================================================
# 4. MERGE MARKET METRICS BACK TO INDIVIDUAL LEVEL
# ============================================================================
print("\n[5] Merging market metrics to individual profiles...")

# Merge target market characteristics
df = df.merge(target_market_stats, on='target_market', how='left')

# Merge competition metrics
df = df.merge(competition_df, on=['location', 'orientation', 'sex'], how='left')

print(f"  ✓ Market metrics merged")
print(f"    - Profiles with target market data: {df['target_ltr_share'].notna().sum():,}")
print(f"    - Profiles with competition data: {df['competition_ratio'].notna().sum():,}")

# ============================================================================
# 5. CREATE MARKET QUALITY CATEGORIES
# ============================================================================
print("\n[6] Creating market quality categories...")

# Categorize markets by LTR share (market seriousness)
df['market_seriousness_category'] = pd.cut(
    df['target_ltr_share'],
    bins=[0, 0.45, 0.55, 1.0],
    labels=['low_ltr', 'medium_ltr', 'high_ltr'],
    include_lowest=True
)

# Categorize by competition
df['competition_category'] = pd.cut(
    df['competition_ratio'],
    bins=[0, 0.9, 1.1, 10],
    labels=['favorable', 'balanced', 'competitive'],
    include_lowest=True
)

# Categorize by signal clarity (target avg effort)
df['signal_clarity_category'] = pd.cut(
    df['target_avg_effort'],
    bins=[0, 0.60, 0.70, 1.0],
    labels=['low_clarity', 'medium_clarity', 'high_clarity'],
    include_lowest=True
)

print(f"  ✓ Market categories created")
print(f"\n  Market seriousness distribution:")
print(df['market_seriousness_category'].value_counts().to_string())

print(f"\n  Competition distribution:")
print(df['competition_category'].value_counts().to_string())

print(f"\n  Signal clarity distribution:")
print(df['signal_clarity_category'].value_counts().to_string())

# ============================================================================
# 6. ANALYZE EFFORT BY MARKET CHARACTERISTICS
# ============================================================================
print("\n[7] Analyzing effort by market characteristics...")

# Effort by market seriousness (KEY TEST!)
print("\n  Mean effort by market seriousness:")
effort_by_seriousness = df.groupby('market_seriousness_category')['effort_index'].agg(['mean', 'std', 'count'])
print(effort_by_seriousness.round(3).to_string())

# For LTR-oriented users specifically
print("\n  Mean effort by market seriousness (LTR-oriented users only):")
ltr_users = df[df['is_ltr_oriented'] == 1]
effort_by_seriousness_ltr = ltr_users.groupby('market_seriousness_category')['effort_index'].agg(['mean', 'std', 'count'])
print(effort_by_seriousness_ltr.round(3).to_string())

# Effort by competition
print("\n  Mean effort by competition level:")
effort_by_competition = df.groupby('competition_category')['effort_index'].agg(['mean', 'std', 'count'])
print(effort_by_competition.round(3).to_string())

# Effort by signal clarity
print("\n  Mean effort by signal clarity:")
effort_by_clarity = df.groupby('signal_clarity_category')['effort_index'].agg(['mean', 'std', 'count'])
print(effort_by_clarity.round(3).to_string())

# ============================================================================
# 7. INTERACTION EFFECTS
# ============================================================================
print("\n[8] Analyzing interaction effects...")

# High-rated users in low-LTR markets (frustration hypothesis!)
print("\n  Testing frustration hypothesis: High-rated users in low-LTR markets")

high_rated = df[df['rating_quintile'].isin(['high', 'top_20'])]
low_rated = df[df['rating_quintile'].isin(['bottom_20', 'low'])]

print("\n  High-rated users:")
high_effort = high_rated.groupby('market_seriousness_category')['effort_index'].mean()
print(high_effort.to_string())

print("\n  Low-rated users:")
low_effort = low_rated.groupby('market_seriousness_category')['effort_index'].mean()
print(low_effort.to_string())

# Difference (frustration effect)
print("\n  Effort gap (high-rated minus low-rated) by market:")
effort_gap = high_effort - low_effort
print(effort_gap.to_string())

# ============================================================================
# 8. CORRELATION ANALYSIS
# ============================================================================
print("\n[9] Correlation analysis...")

# Correlations between market characteristics and individual effort
market_vars = ['target_ltr_share', 'target_avg_rating', 'target_avg_effort',
               'competition_ratio', 'target_market_size']

correlations = []
for var in market_vars:
    if var in df.columns:
        corr = df[['effort_index', var]].corr().iloc[0, 1]
        correlations.append({
            'Market Variable': var,
            'Correlation with Effort': corr
        })

corr_df = pd.DataFrame(correlations).sort_values('Correlation with Effort', ascending=False)
print("\nMarket characteristics correlation with effort:")
print(corr_df.to_string(index=False))

# For LTR users only
print("\nFor LTR-oriented users only:")
ltr_correlations = []
for var in market_vars:
    if var in ltr_users.columns:
        corr = ltr_users[['effort_index', var]].corr().iloc[0, 1]
        ltr_correlations.append({
            'Market Variable': var,
            'Correlation with Effort': corr
        })

ltr_corr_df = pd.DataFrame(ltr_correlations).sort_values('Correlation with Effort', ascending=False)
print(ltr_corr_df.to_string(index=False))

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================
print("\n[10] Creating visualizations...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of market seriousness (target LTR share)
    axes[0, 0].hist(df['target_ltr_share'].dropna(), bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Market Seriousness (ρ_m)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Target Market LTR Share')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(df['target_ltr_share'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['target_ltr_share'].mean():.3f}")
    axes[0, 0].legend()
    
    # 2. Effort by market seriousness
    effort_by_seriousness.plot(y='mean', kind='bar', ax=axes[0, 1], 
                               color='steelblue', legend=False, yerr=effort_by_seriousness['std'])
    axes[0, 1].set_title('Mean Effort by Market Seriousness', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Market Seriousness (LTR Share)')
    axes[0, 1].set_ylabel('Mean Effort Index')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Effort by market seriousness (LTR users only)
    effort_by_seriousness_ltr.plot(y='mean', kind='bar', ax=axes[0, 2],
                                   color='green', legend=False, yerr=effort_by_seriousness_ltr['std'])
    axes[0, 2].set_title('Mean Effort by Market Seriousness\n(LTR-oriented users only)', 
                         fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Market Seriousness (LTR Share)')
    axes[0, 2].set_ylabel('Mean Effort Index')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Scatter: Target LTR share vs Effort
    sample = df.dropna(subset=['target_ltr_share', 'effort_index']).sample(min(5000, len(df)))
    axes[1, 0].scatter(sample['target_ltr_share'], sample['effort_index'], alpha=0.3, s=10)
    axes[1, 0].set_title('Individual Effort vs Market Seriousness', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Target Market LTR Share (ρ_m)')
    axes[1, 0].set_ylabel('Effort Index')
    
    # Add trend line
    z = np.polyfit(sample['target_ltr_share'], sample['effort_index'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(sample['target_ltr_share'].min(), sample['target_ltr_share'].max(), 100)
    axes[1, 0].plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend')
    axes[1, 0].legend()
    
    # 5. Effort by competition level
    effort_by_competition.plot(y='mean', kind='bar', ax=axes[1, 1],
                               color='orange', legend=False, yerr=effort_by_competition['std'])
    axes[1, 1].set_title('Mean Effort by Competition Level', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Competition Level')
    axes[1, 1].set_ylabel('Mean Effort Index')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Interaction: Rating × Market Seriousness
    interaction_data = df.groupby(['rating_quintile', 'market_seriousness_category'])['effort_index'].mean().unstack()
    interaction_data = interaction_data.reindex(['bottom_20', 'low', 'middle', 'high', 'top_20'])
    interaction_data.plot(kind='bar', ax=axes[1, 2], width=0.8)
    axes[1, 2].set_title('Effort: Rating × Market Seriousness', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Rating Quintile')
    axes[1, 2].set_ylabel('Mean Effort Index')
    axes[1, 2].legend(title='Market LTR', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('market_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Visualizations saved to 'market_analysis.png'")
    
except Exception as e:
    print(f"  ⚠ Visualization error (non-critical): {e}")

# ============================================================================
# 10. CREATE SUMMARY TABLES FOR REPORT
# ============================================================================
print("\n[11] Creating summary tables for report...")

# Table 1: Descriptive statistics by market type
desc_table = df.groupby('market_seriousness_category').agg({
    'effort_index': ['mean', 'std'],
    'rating_index': ['mean', 'std'],
    'age': 'mean',
    'education_level': 'mean',
    'is_ltr_oriented': 'mean',
    'market_id': 'count'
}).round(3)

desc_table.columns = ['_'.join(col).strip() for col in desc_table.columns]
desc_table = desc_table.reset_index()

# Save to CSV
desc_table.to_csv('descriptive_stats_by_market.csv', index=False)
print("  ✓ Descriptive statistics table saved: 'descriptive_stats_by_market.csv'")

# Table 2: Market-level statistics
market_summary = target_market_stats.describe().round(3)
market_summary.to_csv('market_level_summary_stats.csv')
print("  ✓ Market summary statistics saved: 'market_level_summary_stats.csv'")

# ============================================================================
# 11. SAVE FINAL DATASET
# ============================================================================
print("\n[12] Saving final dataset...")

df.to_csv('okcupid_final_analysis_ready.csv', index=False)
print(f"  ✓ Final dataset saved: 'okcupid_final_analysis_ready.csv'")
print(f"    Shape: {df.shape}")

# Save market indices separately for easy reference
market_indices = df[['market_id', 'target_market', 'target_ltr_share', 'target_avg_effort',
                     'target_avg_rating', 'target_market_size', 'competition_ratio',
                     'market_seriousness_category', 'competition_category', 
                     'signal_clarity_category']].drop_duplicates()
market_indices.to_csv('market_indices.csv', index=False)
print(f"  ✓ Market indices saved: 'market_indices.csv'")

# Save summary report
with open('market_analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("MARKET-LEVEL ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(df):,} profiles across {df['market_id'].nunique()} markets\n\n")
    
    f.write("KEY MARKET METRICS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Mean target market LTR share (ρ_m): {df['target_ltr_share'].mean():.3f}\n")
    f.write(f"  Mean target market effort: {df['target_avg_effort'].mean():.3f}\n")
    f.write(f"  Mean target market rating: {df['target_avg_rating'].mean():.3f}\n")
    f.write(f"  Mean competition ratio: {df['competition_ratio'].mean():.3f}\n")
    f.write(f"  Mean target market size: {df['target_market_size'].mean():.0f}\n\n")
    
    f.write("EFFORT BY MARKET SERIOUSNESS:\n")
    f.write("-"*80 + "\n")
    for category in ['low_ltr', 'medium_ltr', 'high_ltr']:
        if category in effort_by_seriousness.index:
            mean_effort = effort_by_seriousness.loc[category, 'mean']
            count = effort_by_seriousness.loc[category, 'count']
            f.write(f"  {category:12s}: {mean_effort:.3f} (n={count:,.0f})\n")
    
    f.write("\nEFFORT BY MARKET SERIOUSNESS (LTR USERS ONLY):\n")
    f.write("-"*80 + "\n")
    for category in ['low_ltr', 'medium_ltr', 'high_ltr']:
        if category in effort_by_seriousness_ltr.index:
            mean_effort = effort_by_seriousness_ltr.loc[category, 'mean']
            count = effort_by_seriousness_ltr.loc[category, 'count']
            f.write(f"  {category:12s}: {mean_effort:.3f} (n={count:,.0f})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS FOR FRUSTRATION HYPOTHESIS:\n")
    f.write("-"*80 + "\n")
    
    # Calculate key statistics
    low_ltr_effort = df[df['market_seriousness_category'] == 'low_ltr']['effort_index'].mean()
    high_ltr_effort = df[df['market_seriousness_category'] == 'high_ltr']['effort_index'].mean()
    effort_diff = high_ltr_effort - low_ltr_effort
    
    f.write(f"  1. Users in low-LTR markets have effort: {low_ltr_effort:.3f}\n")
    f.write(f"  2. Users in high-LTR markets have effort: {high_ltr_effort:.3f}\n")
    f.write(f"  3. Difference: {effort_diff:.3f} {'(higher in high-LTR markets)' if effort_diff > 0 else '(higher in low-LTR markets)'}\n")
    f.write(f"  4. Correlation (target LTR share, effort): {df[['target_ltr_share', 'effort_index']].corr().iloc[0,1]:.3f}\n")
    
    # For LTR users specifically
    ltr_low = ltr_users[ltr_users['market_seriousness_category'] == 'low_ltr']['effort_index'].mean()
    ltr_high = ltr_users[ltr_users['market_seriousness_category'] == 'high_ltr']['effort_index'].mean()
    ltr_diff = ltr_high - ltr_low
    
    f.write(f"\n  For LTR-oriented users specifically:\n")
    f.write(f"  - Effort in low-LTR markets: {ltr_low:.3f}\n")
    f.write(f"  - Effort in high-LTR markets: {ltr_high:.3f}\n")
    f.write(f"  - Difference: {ltr_diff:.3f}\n")
    
    f.write("\n" + "="*80 + "\n")

print("  ✓ Summary saved: 'market_analysis_summary.txt'")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("MARKET-LEVEL ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFiles created:")
print(f"  1. okcupid_final_analysis_ready.csv (complete dataset with all variables)")
print(f"  2. market_indices.csv (market-level characteristics)")
print(f"  3. descriptive_stats_by_market.csv (summary table for report)")
print(f"  4. market_level_summary_stats.csv (market statistics)")
print(f"  5. market_analysis.png (visualizations)")
print(f"  6. market_analysis_summary.txt (key findings)")
print(f"\nKey variables added:")
print(f"  - target_ltr_share (ρ_m) - MARKET SERIOUSNESS")
print(f"  - target_avg_effort - SIGNAL CLARITY")
print(f"  - target_avg_rating - AVERAGE PARTNER QUALITY")
print(f"  - competition_ratio - SCARCITY/COMPETITION")
print(f"  - market_seriousness_category, competition_category, signal_clarity_category")
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE! Ready for:")
print("  - Descriptive statistics (for your report)")
print("  - Hypothesis testing (frustration effects)")
print("  - Comparative statics & simulations")
print("="*80)
