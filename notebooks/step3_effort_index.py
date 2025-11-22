"""
Step 3: Calculate Effort Index
OkCupid Dating App Project

This script calculates the effort index (E_i) from:
1. Essay word counts
2. Number of essays completed
3. Profile completeness (demographic fields)

The effort index is a key variable for testing the frustration hypothesis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

print("="*80)
print("STEP 3: CALCULATE EFFORT INDEX")
print("="*80)

# ============================================================================
# LOAD CLEANED DATA
# ============================================================================
print("\n[1] Loading cleaned dataset...")

try:
    df = pd.read_csv('data/okcupid_cleaned.csv')
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'okcupid_cleaned.csv' not found.")
    print("Please run step2_data_cleaning.py first!")
    exit()

# ============================================================================
# 1. CALCULATE ESSAY METRICS
# ============================================================================
print("\n[2] Calculating essay metrics...")

essay_columns = [f'essay{i}' for i in range(10)]

def count_words(text):
    """Count words in text, handling NaN"""
    if pd.isna(text):
        return 0
    # Remove HTML tags and special characters, then count words
    text_clean = re.sub(r'<[^>]+>', ' ', str(text))  # Remove HTML
    text_clean = re.sub(r'&[a-z]+;', ' ', text_clean)  # Remove HTML entities like &rsquo;
    words = text_clean.split()
    return len(words)

# Count words in each essay
print("  - Counting words in each essay...")
for col in essay_columns:
    df[f'{col}_word_count'] = df[col].apply(count_words)

# Total word count across all essays
df['total_essay_words'] = df[[f'{col}_word_count' for col in essay_columns]].sum(axis=1)

# Number of essays completed (non-empty)
df['num_essays_completed'] = df[essay_columns].notna().sum(axis=1)

# Average words per completed essay
df['avg_words_per_essay'] = df.apply(
    lambda row: row['total_essay_words'] / row['num_essays_completed'] if row['num_essays_completed'] > 0 else 0,
    axis=1
)

print(f"  ✓ Essay metrics calculated")
print(f"    - Mean total words: {df['total_essay_words'].mean():.0f}")
print(f"    - Median total words: {df['total_essay_words'].median():.0f}")
print(f"    - Mean essays completed: {df['num_essays_completed'].mean():.1f} / 10")
print(f"    - Mean words per essay: {df['avg_words_per_essay'].mean():.0f}")

# ============================================================================
# 2. CALCULATE DEMOGRAPHIC PROFILE COMPLETENESS
# ============================================================================
print("\n[3] Calculating demographic profile completeness...")

# Key demographic fields (from cleaned columns)
demographic_fields = [
    'age', 'sex', 'orientation', 'status',  # Always complete
    'body_type_clean', 'diet_type', 'drinks', 'drugs', 'smokes',
    'education_level', 'ethnicity_primary', 'height_clean', 
    'offspring', 'pets', 'religion_type', 'zodiac_sign', 'speaks',
    'job', 'income_clean'
]

# Count how many fields are filled
df['demographic_fields_filled'] = df[demographic_fields].notna().sum(axis=1)
df['demographic_completeness'] = df['demographic_fields_filled'] / len(demographic_fields)

print(f"  ✓ Demographic completeness calculated")
print(f"    - Mean fields filled: {df['demographic_fields_filled'].mean():.1f} / {len(demographic_fields)}")
print(f"    - Mean completeness: {df['demographic_completeness'].mean():.1%}")

# ============================================================================
# 3. CREATE OVERALL EFFORT INDEX
# ============================================================================
print("\n[4] Creating overall effort index...")

# Normalize components to 0-1 scale
# Essay effort: based on total words (use percentile to handle outliers)
df['essay_effort_raw'] = df['total_essay_words'].rank(pct=True)

# Essay completion: number of essays completed / 10
df['essay_completion'] = df['num_essays_completed'] / 10

# Demographic completeness: already 0-1

# Combined effort index: weighted average
# Weights: 40% essay words, 30% essay completion, 30% demographic completeness
WEIGHT_WORDS = 0.40
WEIGHT_COMPLETION = 0.30
WEIGHT_DEMOGRAPHIC = 0.30

df['effort_index'] = (
    WEIGHT_WORDS * df['essay_effort_raw'] +
    WEIGHT_COMPLETION * df['essay_completion'] +
    WEIGHT_DEMOGRAPHIC * df['demographic_completeness']
)

print(f"  ✓ Effort index created")
print(f"    - Weights: {WEIGHT_WORDS:.0%} words, {WEIGHT_COMPLETION:.0%} completion, {WEIGHT_DEMOGRAPHIC:.0%} demographic")
print(f"    - Mean effort index: {df['effort_index'].mean():.3f}")
print(f"    - Median effort index: {df['effort_index'].median():.3f}")
print(f"    - Std effort index: {df['effort_index'].std():.3f}")

# ============================================================================
# 4. CREATE EFFORT CATEGORIES
# ============================================================================
print("\n[5] Creating effort categories...")

# Categorize effort levels
df['effort_category'] = pd.cut(
    df['effort_index'],
    bins=[0, 0.25, 0.5, 0.75, 1.0],
    labels=['very_low', 'low', 'medium', 'high'],
    include_lowest=True
)

print(f"  ✓ Effort categories created")
print(f"    - Distribution:")
for cat in ['very_low', 'low', 'medium', 'high']:
    count = (df['effort_category'] == cat).sum()
    pct = count / len(df) * 100
    print(f"      {cat:12s}: {count:6,} ({pct:5.1f}%)")

# ============================================================================
# 5. ANALYZE EFFORT BY DEMOGRAPHICS
# ============================================================================
print("\n[6] Analyzing effort by demographics...")

# By sex
print("\n  By Sex:")
for sex in df['sex'].unique():
    mean_effort = df[df['sex'] == sex]['effort_index'].mean()
    print(f"    {sex}: {mean_effort:.3f}")

# By orientation
print("\n  By Orientation:")
for orient in df['orientation'].unique():
    mean_effort = df[df['orientation'] == orient]['effort_index'].mean()
    print(f"    {orient:10s}: {mean_effort:.3f}")

# By age group
df['age_group'] = pd.cut(df['age'], bins=[18, 25, 30, 35, 40, 50, 120], 
                          labels=['18-25', '26-30', '31-35', '36-40', '41-50', '50+'])
print("\n  By Age Group:")
for age in ['18-25', '26-30', '31-35', '36-40', '41-50', '50+']:
    mean_effort = df[df['age_group'] == age]['effort_index'].mean()
    print(f"    {age}: {mean_effort:.3f}")

# By education level
print("\n  By Education Level:")
for edu in sorted(df['education_level'].dropna().unique()):
    mean_effort = df[df['education_level'] == edu]['effort_index'].mean()
    print(f"    Level {edu:.0f}: {mean_effort:.3f}")

# ============================================================================
# 6. ANALYZE EFFORT BY MARKET
# ============================================================================
print("\n[7] Analyzing effort by market...")

# Calculate market-level statistics
market_stats = df.groupby('market_id').agg({
    'effort_index': ['mean', 'std', 'count'],
    'total_essay_words': 'mean',
    'num_essays_completed': 'mean',
    'demographic_completeness': 'mean'
}).round(3)

market_stats.columns = ['_'.join(col).strip() for col in market_stats.columns]
market_stats = market_stats.reset_index()

# Filter markets with at least 20 profiles
market_stats_filtered = market_stats[market_stats['effort_index_count'] >= 20].copy()

print(f"  ✓ Market statistics calculated")
print(f"    - Total markets: {len(market_stats)}")
print(f"    - Markets with ≥20 profiles: {len(market_stats_filtered)}")

print("\n  Top 5 markets by mean effort:")
top_effort = market_stats_filtered.nlargest(5, 'effort_index_mean')[
    ['market_id', 'effort_index_mean', 'effort_index_count']
]
print(top_effort.to_string(index=False))

print("\n  Bottom 5 markets by mean effort:")
bottom_effort = market_stats_filtered.nsmallest(5, 'effort_index_mean')[
    ['market_id', 'effort_index_mean', 'effort_index_count']
]
print(bottom_effort.to_string(index=False))

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================
print("\n[8] Summary statistics...")

summary_stats = pd.DataFrame({
    'Metric': [
        'Total Essay Words',
        'Essays Completed (out of 10)',
        'Avg Words per Essay',
        'Demographic Fields Filled (out of 19)',
        'Demographic Completeness',
        'Effort Index'
    ],
    'Mean': [
        f"{df['total_essay_words'].mean():.0f}",
        f"{df['num_essays_completed'].mean():.2f}",
        f"{df['avg_words_per_essay'].mean():.0f}",
        f"{df['demographic_fields_filled'].mean():.1f}",
        f"{df['demographic_completeness'].mean():.1%}",
        f"{df['effort_index'].mean():.3f}"
    ],
    'Median': [
        f"{df['total_essay_words'].median():.0f}",
        f"{df['num_essays_completed'].median():.0f}",
        f"{df['avg_words_per_essay'].median():.0f}",
        f"{df['demographic_fields_filled'].median():.0f}",
        f"{df['demographic_completeness'].median():.1%}",
        f"{df['effort_index'].median():.3f}"
    ],
    'Std': [
        f"{df['total_essay_words'].std():.0f}",
        f"{df['num_essays_completed'].std():.2f}",
        f"{df['avg_words_per_essay'].std():.0f}",
        f"{df['demographic_fields_filled'].std():.1f}",
        f"{df['demographic_completeness'].std():.1%}",
        f"{df['effort_index'].std():.3f}"
    ]
})

print("\n")
print(summary_stats.to_string(index=False))

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[9] Creating visualizations...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of total essay words
    df['total_essay_words'].hist(bins=50, ax=axes[0, 0], edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Essay Words', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Total Words')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(df['total_essay_words'].mean(), color='red', linestyle='--', 
                       label=f"Mean: {df['total_essay_words'].mean():.0f}")
    axes[0, 0].legend()
    
    # 2. Distribution of essays completed
    essay_counts = df['num_essays_completed'].value_counts().sort_index()
    axes[0, 1].bar(essay_counts.index, essay_counts.values, edgecolor='black')
    axes[0, 1].set_title('Number of Essays Completed', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Essays Completed (out of 10)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks(range(0, 11))
    
    # 3. Distribution of effort index
    df['effort_index'].hist(bins=50, ax=axes[0, 2], edgecolor='black', color='green', alpha=0.7)
    axes[0, 2].set_title('Distribution of Effort Index', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Effort Index')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].axvline(df['effort_index'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df['effort_index'].mean():.3f}")
    axes[0, 2].legend()
    
    # 4. Effort by sex
    effort_by_sex = df.groupby('sex')['effort_index'].mean().sort_values()
    axes[1, 0].bar(effort_by_sex.index, effort_by_sex.values, color=['skyblue', 'salmon'])
    axes[1, 0].set_title('Mean Effort Index by Sex', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Sex')
    axes[1, 0].set_ylabel('Mean Effort Index')
    axes[1, 0].set_ylim([0, 1])
    
    # 5. Effort by orientation
    effort_by_orient = df.groupby('orientation')['effort_index'].mean().sort_values()
    axes[1, 1].bar(range(len(effort_by_orient)), effort_by_orient.values, 
                   tick_label=effort_by_orient.index, color='steelblue')
    axes[1, 1].set_title('Mean Effort Index by Orientation', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Orientation')
    axes[1, 1].set_ylabel('Mean Effort Index')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Effort by age group
    effort_by_age = df.groupby('age_group')['effort_index'].mean()
    axes[1, 2].bar(range(len(effort_by_age)), effort_by_age.values, color='orange')
    axes[1, 2].set_title('Mean Effort Index by Age Group', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Age Group')
    axes[1, 2].set_ylabel('Mean Effort Index')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].set_xticks(range(len(effort_by_age)))
    axes[1, 2].set_xticklabels(effort_by_age.index, rotation=45)
    
    plt.tight_layout()
    plt.savefig('effort_analysis_plots.png', dpi=150, bbox_inches='tight')
    print("  ✓ Visualizations saved to 'effort_analysis_plots.png'")
    
except Exception as e:
    print(f"  ⚠ Visualization error (non-critical): {e}")

# ============================================================================
# 9. CORRELATION ANALYSIS
# ============================================================================
print("\n[10] Correlation analysis...")

# Correlations with effort index
effort_correlations = pd.DataFrame({
    'Variable': [
        'Age',
        'Education Level',
        'Income (reported)',
        'Height',
        'Is Multiracial',
        'Income Reported (flag)'
    ],
    'Correlation': [
        df[['age', 'effort_index']].corr().iloc[0, 1],
        df[['education_level', 'effort_index']].corr().iloc[0, 1],
        df[['income_clean', 'effort_index']].corr().iloc[0, 1],
        df[['height_clean', 'effort_index']].corr().iloc[0, 1],
        df[['is_multiracial', 'effort_index']].corr().iloc[0, 1],
        df[['income_reported', 'effort_index']].corr().iloc[0, 1]
    ]
})

effort_correlations = effort_correlations.sort_values('Correlation', ascending=False)
print("\nCorrelations with Effort Index:")
print(effort_correlations.to_string(index=False))

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n[11] Saving results...")

# Save updated dataset with effort metrics
df.to_csv('okcupid_with_effort.csv', index=False)
print(f"  ✓ Dataset with effort metrics saved: 'okcupid_with_effort.csv'")

# Save market-level statistics
market_stats_filtered.to_csv('market_effort_statistics.csv', index=False)
print(f"  ✓ Market statistics saved: 'market_effort_statistics.csv'")

# Save summary report with UTF-8 encoding (or replace symbols)
with open('effort_analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("EFFORT INDEX ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(df):,} profiles\n\n")
    
    f.write("EFFORT INDEX COMPONENTS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  1. Essay Words (40% weight): Mean = {df['total_essay_words'].mean():.0f}, Median = {df['total_essay_words'].median():.0f}\n")
    f.write(f"  2. Essay Completion (30% weight): Mean = {df['num_essays_completed'].mean():.1f}/10\n")
    f.write(f"  3. Demographic Completeness (30% weight): Mean = {df['demographic_completeness'].mean():.1%}\n\n")
    
    f.write("EFFORT INDEX STATISTICS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Mean:   {df['effort_index'].mean():.3f}\n")
    f.write(f"  Median: {df['effort_index'].median():.3f}\n")
    f.write(f"  Std:    {df['effort_index'].std():.3f}\n")
    f.write(f"  Min:    {df['effort_index'].min():.3f}\n")
    f.write(f"  Max:    {df['effort_index'].max():.3f}\n\n")
    
    f.write("EFFORT DISTRIBUTION:\n")
    f.write("-"*80 + "\n")
    for cat in ['very_low', 'low', 'medium', 'high']:
        count = (df['effort_category'] == cat).sum()
        pct = count / len(df) * 100
        f.write(f"  {cat:12s}: {count:6,} ({pct:5.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  - Average user completes {df['num_essays_completed'].mean():.1f} out of 10 essays\n")
    f.write(f"  - Average essay length: {df['avg_words_per_essay'].mean():.0f} words\n")
    f.write(f"  - {(df['effort_category'].isin(['very_low', 'low'])).sum():,} profiles ({(df['effort_category'].isin(['very_low', 'low'])).sum()/len(df)*100:.1f}%) show low effort\n")
    f.write(f"  - Education level correlates {df[['education_level', 'effort_index']].corr().iloc[0, 1]:.3f} with effort\n")
    f.write(f"  - {len(market_stats_filtered)} markets have sufficient data (>=20 profiles)\n")  # Changed >= to >=
    f.write("\n" + "="*80 + "\n")

print("✓ Summary file created successfully with UTF-8 encoding!")
print("✓ File: effort_analysis_summary.txt")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("EFFORT ANALYSIS COMPLETE!")
print("="*80)
print(f"\nFiles created:")
print(f"  1. okcupid_with_effort.csv (dataset with {df.shape[1]} columns)")
print(f"  2. market_effort_statistics.csv (market-level metrics)")
print(f"  3. effort_analysis_summary.txt (summary report)")
print(f"  4. effort_analysis_plots.png (visualizations)")
print(f"\nKey effort metrics added:")
print(f"  - total_essay_words, num_essays_completed, avg_words_per_essay")
print(f"  - demographic_completeness")
print(f"  - effort_index (0-1 scale)")
print(f"  - effort_category (very_low, low, medium, high)")
print("\nNext step: Step 4 - Build rating index (r_i)")
print("="*80)
