"""
Step 4: Build Rating Index (r_i)
OkCupid Dating App Project

This script creates a desirability rating index (r_i) for each user based on:
- Education level
- Income (when reported)
- Body type
- Height (relative to gender norms)
- Age (younger generally preferred in dating markets)
- Profile effort/completeness

The rating index will be normalized to [0,1] and used for:
1. Identifying user's position in the market
2. Testing frustration hypothesis (high-rated users in bad markets)
3. Simulations and comparative statics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats

print("="*80)
print("STEP 4: BUILD RATING INDEX (r_i)")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading dataset with effort metrics...")

try:
    df = pd.read_csv('data/okcupid_with_effort.csv')
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'okcupid_with_effort.csv' not found.")
    print("Please run step3_effort_index.py first!")
    exit()

# ============================================================================
# 1. PREPARE RATING COMPONENTS
# ============================================================================
print("\n[2] Preparing rating components...")

# Create a copy for rating calculation
df_rating = df.copy()

# ----- EDUCATION (already ordinal 0-5) -----
df_rating['education_score'] = df_rating['education_level'].copy()
print(f"  ✓ Education score: mean = {df_rating['education_score'].mean():.2f}")

# ----- INCOME -----
# Log-transform income to handle skewness, then normalize
# Only for those who reported income
df_rating['income_score'] = np.nan
income_mask = df_rating['income_clean'].notna() & (df_rating['income_clean'] > 0)

if income_mask.sum() > 0:
    # Log transform
    df_rating.loc[income_mask, 'income_score'] = np.log1p(df_rating.loc[income_mask, 'income_clean'])
    # Normalize to 0-5 scale (to match education)
    income_min = df_rating.loc[income_mask, 'income_score'].min()
    income_max = df_rating.loc[income_mask, 'income_score'].max()
    df_rating.loc[income_mask, 'income_score'] = 5 * (df_rating.loc[income_mask, 'income_score'] - income_min) / (income_max - income_min)

print(f"  ✓ Income score: mean = {df_rating['income_score'].mean():.2f} (for {income_mask.sum()} who reported)")

# ----- BODY TYPE -----
# Assign desirability scores based on typical dating preferences
body_type_scores = {
    'fit': 5,
    'athletic': 5,  # In case it wasn't consolidated
    'thin': 4,
    'average': 3,
    'curvy': 3,
    'a little extra': 2,
    'overweight': 1
}

df_rating['body_score'] = df_rating['body_type_clean'].map(body_type_scores)
print(f"  ✓ Body type score: mean = {df_rating['body_score'].mean():.2f}")
print(f"    Distribution: {df_rating['body_score'].value_counts().sort_index().to_dict()}")

# ----- HEIGHT (relative to gender norms) -----
# Taller is generally preferred, but penalize extremes
# Normalize by gender since height preferences differ

def calculate_height_score(row):
    """Height score relative to gender norms"""
    if pd.isna(row['height_clean']):
        return np.nan
    
    height = row['height_clean']
    sex = row['sex']
    
    # Gender-specific ideal ranges (in inches)
    if sex == 'm':
        # Men: ideal 69-74 inches (5'9" - 6'2")
        if 69 <= height <= 74:
            score = 5
        elif 67 <= height < 69 or 74 < height <= 76:
            score = 4
        elif 65 <= height < 67 or 76 < height <= 78:
            score = 3
        elif 63 <= height < 65 or 78 < height <= 80:
            score = 2
        else:
            score = 1
    else:  # female
        # Women: ideal 63-68 inches (5'3" - 5'8")
        if 63 <= height <= 68:
            score = 5
        elif 61 <= height < 63 or 68 < height <= 70:
            score = 4
        elif 59 <= height < 61 or 70 < height <= 72:
            score = 3
        elif 57 <= height < 59 or 72 < height <= 74:
            score = 2
        else:
            score = 1
    
    return score

df_rating['height_score'] = df_rating.apply(calculate_height_score, axis=1)
print(f"  ✓ Height score: mean = {df_rating['height_score'].mean():.2f}")

# ----- AGE -----
# Younger is generally preferred in dating markets, but not too young
# Penalize very young (18-20) and older ages
# Peak desirability: 25-32

def calculate_age_score(age):
    """Age score - peak desirability in mid-to-late 20s"""
    if pd.isna(age):
        return np.nan
    
    if 25 <= age <= 32:
        score = 5
    elif 22 <= age < 25 or 32 < age <= 35:
        score = 4
    elif 20 <= age < 22 or 35 < age <= 40:
        score = 3
    elif 18 <= age < 20 or 40 < age <= 45:
        score = 2
    else:  # <18 or >45
        score = 1
    
    return score

df_rating['age_score'] = df_rating['age'].apply(calculate_age_score)
print(f"  ✓ Age score: mean = {df_rating['age_score'].mean():.2f}")
print(f"    Distribution: {df_rating['age_score'].value_counts().sort_index().to_dict()}")

# ----- EFFORT/COMPLETENESS -----
# Already have effort_index (0-1), scale to 0-5
df_rating['effort_score'] = df_rating['effort_index'] * 5
print(f"  ✓ Effort score: mean = {df_rating['effort_score'].mean():.2f}")

# ============================================================================
# 2. HANDLE MISSING VALUES IN COMPONENTS
# ============================================================================
print("\n[3] Handling missing values...")

rating_components = ['education_score', 'income_score', 'body_score', 
                     'height_score', 'age_score', 'effort_score']

for component in rating_components:
    missing_count = df_rating[component].isna().sum()
    missing_pct = missing_count / len(df_rating) * 100
    print(f"  {component:20s}: {missing_count:6,} missing ({missing_pct:5.1f}%)")

# Strategy: Impute missing values with median (by sex and orientation group)
print("\n  Imputing missing values with group medians (by sex × orientation)...")

for component in rating_components:
    # Calculate group medians
    group_medians = df_rating.groupby(['sex', 'orientation'])[component].median()
    
    # Impute missing values
    for (sex, orientation), median_value in group_medians.items():
        mask = (df_rating['sex'] == sex) & \
               (df_rating['orientation'] == orientation) & \
               (df_rating[component].isna())
        df_rating.loc[mask, component] = median_value

print(f"  ✓ Missing values imputed")

# ============================================================================
# 3. CALCULATE COMPOSITE RATING INDEX
# ============================================================================
print("\n[4] Calculating composite rating index...")

# Weights for each component
# Income gets lower weight since only 19% report it
WEIGHTS = {
    'education_score': 0.20,
    'income_score': 0.10,    # Lower weight due to missingness
    'body_score': 0.20,
    'height_score': 0.15,
    'age_score': 0.15,
    'effort_score': 0.20
}

print(f"  Component weights:")
for component, weight in WEIGHTS.items():
    print(f"    {component:20s}: {weight:.0%}")

# Calculate weighted average
df_rating['rating_raw'] = sum(
    df_rating[component] * weight 
    for component, weight in WEIGHTS.items()
)

print(f"\n  ✓ Raw rating calculated")
print(f"    - Mean: {df_rating['rating_raw'].mean():.3f}")
print(f"    - Median: {df_rating['rating_raw'].median():.3f}")
print(f"    - Std: {df_rating['rating_raw'].std():.3f}")
print(f"    - Min: {df_rating['rating_raw'].min():.3f}")
print(f"    - Max: {df_rating['rating_raw'].max():.3f}")

# ============================================================================
# 4. NORMALIZE TO [0, 1] AND CREATE RATING CATEGORIES
# ============================================================================
print("\n[5] Normalizing rating to [0,1] scale...")

# Normalize to [0, 1]
rating_min = df_rating['rating_raw'].min()
rating_max = df_rating['rating_raw'].max()
df_rating['rating_index'] = (df_rating['rating_raw'] - rating_min) / (rating_max - rating_min)

print(f"  ✓ Rating index (r_i) normalized")
print(f"    - Mean: {df_rating['rating_index'].mean():.3f}")
print(f"    - Median: {df_rating['rating_index'].median():.3f}")
print(f"    - Std: {df_rating['rating_index'].std():.3f}")

# Create rating categories (quintiles)
df_rating['rating_quintile'] = pd.qcut(
    df_rating['rating_index'], 
    q=5, 
    labels=['bottom_20', 'low', 'middle', 'high', 'top_20']
)

print(f"\n  Rating quintiles:")
for quintile in ['bottom_20', 'low', 'middle', 'high', 'top_20']:
    count = (df_rating['rating_quintile'] == quintile).sum()
    mean_rating = df_rating[df_rating['rating_quintile'] == quintile]['rating_index'].mean()
    print(f"    {quintile:10s}: {count:6,} profiles (mean rating: {mean_rating:.3f})")

# ============================================================================
# 5. ANALYZE RATING BY DEMOGRAPHICS
# ============================================================================
print("\n[6] Analyzing rating by demographics...")

# By sex
print("\n  By Sex:")
for sex in df_rating['sex'].unique():
    mean_rating = df_rating[df_rating['sex'] == sex]['rating_index'].mean()
    print(f"    {sex}: {mean_rating:.3f}")

# By orientation
print("\n  By Orientation:")
for orient in df_rating['orientation'].unique():
    mean_rating = df_rating[df_rating['orientation'] == orient]['rating_index'].mean()
    print(f"    {orient:10s}: {mean_rating:.3f}")

# By education level
print("\n  By Education Level:")
for edu in sorted(df_rating['education_level'].dropna().unique()):
    mean_rating = df_rating[df_rating['education_level'] == edu]['rating_index'].mean()
    count = (df_rating['education_level'] == edu).sum()
    print(f"    Level {edu:.0f}: {mean_rating:.3f} (n={count:,})")

# By body type
print("\n  By Body Type:")
body_order = ['overweight', 'a little extra', 'average', 'curvy', 'thin', 'fit']
for body in body_order:
    if body in df_rating['body_type_clean'].values:
        mean_rating = df_rating[df_rating['body_type_clean'] == body]['rating_index'].mean()
        count = (df_rating['body_type_clean'] == body).sum()
        print(f"    {body:15s}: {mean_rating:.3f} (n={count:,})")

# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================
print("\n[7] Correlation analysis...")

correlations = pd.DataFrame({
    'Component': [
        'Education Score',
        'Income Score',
        'Body Score',
        'Height Score',
        'Age Score',
        'Effort Score'
    ],
    'Correlation_with_Rating': [
        df_rating[['education_score', 'rating_index']].corr().iloc[0, 1],
        df_rating[['income_score', 'rating_index']].corr().iloc[0, 1],
        df_rating[['body_score', 'rating_index']].corr().iloc[0, 1],
        df_rating[['height_score', 'rating_index']].corr().iloc[0, 1],
        df_rating[['age_score', 'rating_index']].corr().iloc[0, 1],
        df_rating[['effort_score', 'rating_index']].corr().iloc[0, 1]
    ]
})

correlations = correlations.sort_values('Correlation_with_Rating', ascending=False)
print("\nComponent correlations with rating index:")
print(correlations.to_string(index=False))

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================
print("\n[8] Creating visualizations...")

try:
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    
    # 1. Distribution of rating index
    axes[0, 0].hist(df_rating['rating_index'], bins=50, edgecolor='black', color='purple', alpha=0.7)
    axes[0, 0].set_title('Distribution of Rating Index', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Rating Index (r_i)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(df_rating['rating_index'].mean(), color='red', linestyle='--',
                       label=f"Mean: {df_rating['rating_index'].mean():.3f}")
    axes[0, 0].legend()
    
    # 2. Rating by sex
    rating_by_sex = df_rating.groupby('sex')['rating_index'].mean()
    axes[0, 1].bar(rating_by_sex.index, rating_by_sex.values, color=['skyblue', 'salmon'])
    axes[0, 1].set_title('Mean Rating by Sex', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Sex')
    axes[0, 1].set_ylabel('Mean Rating Index')
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Rating by orientation
    rating_by_orient = df_rating.groupby('orientation')['rating_index'].mean()
    axes[0, 2].bar(range(len(rating_by_orient)), rating_by_orient.values, color='steelblue')
    axes[0, 2].set_title('Mean Rating by Orientation', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Orientation')
    axes[0, 2].set_ylabel('Mean Rating Index')
    axes[0, 2].set_ylim([0, 1])
    axes[0, 2].set_xticks(range(len(rating_by_orient)))
    axes[0, 2].set_xticklabels(rating_by_orient.index, rotation=45)
    
    # 4. Rating by education level
    rating_by_edu = df_rating.groupby('education_level')['rating_index'].mean().sort_index()
    axes[1, 0].bar(range(len(rating_by_edu)), rating_by_edu.values, color='green', alpha=0.7)
    axes[1, 0].set_title('Mean Rating by Education Level', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Education Level')
    axes[1, 0].set_ylabel('Mean Rating Index')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].set_xticks(range(len(rating_by_edu)))
    axes[1, 0].set_xticklabels([f"{int(x)}" for x in rating_by_edu.index])
    
    # 5. Rating by body type
    body_order = ['overweight', 'a little extra', 'average', 'curvy', 'thin', 'fit']
    rating_by_body = df_rating[df_rating['body_type_clean'].isin(body_order)].groupby('body_type_clean')['rating_index'].mean()
    rating_by_body = rating_by_body.reindex(body_order)
    axes[1, 1].bar(range(len(rating_by_body)), rating_by_body.values, color='orange', alpha=0.7)
    axes[1, 1].set_title('Mean Rating by Body Type', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Body Type')
    axes[1, 1].set_ylabel('Mean Rating Index')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].set_xticks(range(len(rating_by_body)))
    axes[1, 1].set_xticklabels(rating_by_body.index, rotation=45, ha='right')
    
    # 6. Rating by age group
    rating_by_age = df_rating.groupby('age_group')['rating_index'].mean()
    axes[1, 2].bar(range(len(rating_by_age)), rating_by_age.values, color='coral')
    axes[1, 2].set_title('Mean Rating by Age Group', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Age Group')
    axes[1, 2].set_ylabel('Mean Rating Index')
    axes[1, 2].set_ylim([0, 1])
    axes[1, 2].set_xticks(range(len(rating_by_age)))
    axes[1, 2].set_xticklabels(rating_by_age.index, rotation=45)
    
    # 7. Scatter: Rating vs Effort
    sample = df_rating.sample(min(5000, len(df_rating)))
    axes[2, 0].scatter(sample['effort_index'], sample['rating_index'], alpha=0.3, s=10)
    axes[2, 0].set_title('Rating vs Effort Index', fontsize=12, fontweight='bold')
    axes[2, 0].set_xlabel('Effort Index')
    axes[2, 0].set_ylabel('Rating Index')
    
    # 8. Scatter: Rating vs Age
    axes[2, 1].scatter(sample['age'], sample['rating_index'], alpha=0.3, s=10, color='green')
    axes[2, 1].set_title('Rating vs Age', fontsize=12, fontweight='bold')
    axes[2, 1].set_xlabel('Age')
    axes[2, 1].set_ylabel('Rating Index')
    
    # 9. Correlation heatmap
    corr_data = df_rating[['education_score', 'income_score', 'body_score', 
                            'height_score', 'age_score', 'effort_score', 'rating_index']].corr()
    im = axes[2, 2].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[2, 2].set_xticks(range(len(corr_data.columns)))
    axes[2, 2].set_yticks(range(len(corr_data.columns)))
    axes[2, 2].set_xticklabels(['Edu', 'Inc', 'Body', 'Hgt', 'Age', 'Eff', 'Rating'], rotation=45, ha='right')
    axes[2, 2].set_yticklabels(['Edu', 'Inc', 'Body', 'Hgt', 'Age', 'Eff', 'Rating'])
    axes[2, 2].set_title('Component Correlations', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[2, 2])
    
    plt.tight_layout()
    plt.savefig('rating_index_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Visualizations saved to 'rating_index_analysis.png'")
    
except Exception as e:
    print(f"  ⚠ Visualization error (non-critical): {e}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n[9] Saving results...")

# Add rating columns to original dataframe
rating_cols = ['education_score', 'income_score', 'body_score', 'height_score', 
               'age_score', 'effort_score', 'rating_raw', 'rating_index', 'rating_quintile']

for col in rating_cols:
    df[col] = df_rating[col]

# Save full dataset
df.to_csv('okcupid_with_ratings.csv', index=False)
print(f"  ✓ Dataset with ratings saved: 'okcupid_with_ratings.csv'")

# Save summary
with open('rating_index_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RATING INDEX SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(df):,} profiles\n\n")
    
    f.write("RATING INDEX COMPONENTS & WEIGHTS:\n")
    f.write("-"*80 + "\n")
    for component, weight in WEIGHTS.items():
        mean_val = df_rating[component].mean()
        f.write(f"  {component:20s}: {weight:5.0%} weight (mean = {mean_val:.2f})\n")
    
    f.write("\nRATING INDEX STATISTICS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Mean:   {df['rating_index'].mean():.3f}\n")
    f.write(f"  Median: {df['rating_index'].median():.3f}\n")
    f.write(f"  Std:    {df['rating_index'].std():.3f}\n")
    f.write(f"  Min:    {df['rating_index'].min():.3f}\n")
    f.write(f"  Max:    {df['rating_index'].max():.3f}\n\n")
    
    f.write("RATING QUINTILES:\n")
    f.write("-"*80 + "\n")
    for quintile in ['bottom_20', 'low', 'middle', 'high', 'top_20']:
        count = (df['rating_quintile'] == quintile).sum()
        mean_rating = df[df['rating_quintile'] == quintile]['rating_index'].mean()
        f.write(f"  {quintile:12s}: {count:6,} profiles (mean = {mean_rating:.3f})\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  - Education is the strongest predictor (r = {df[['education_score', 'rating_index']].corr().iloc[0, 1]:.3f})\n")
    f.write(f"  - Body type highly correlated (r = {df[['body_score', 'rating_index']].corr().iloc[0, 1]:.3f})\n")
    f.write(f"  - Effort/completeness matters (r = {df[['effort_score', 'rating_index']].corr().iloc[0, 1]:.3f})\n")
    f.write(f"  - Mean rating: Males = {df[df['sex']=='m']['rating_index'].mean():.3f}, Females = {df[df['sex']=='f']['rating_index'].mean():.3f}\n")
    f.write("\n" + "="*80 + "\n")

print("  ✓ Summary saved: 'rating_index_summary.txt'")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("RATING INDEX COMPLETE!")
print("="*80)
print(f"\nFiles created:")
print(f"  1. okcupid_with_ratings.csv (dataset with rating index)")
print(f"  2. rating_index_summary.txt (summary report)")
print(f"  3. rating_index_analysis.png (visualizations)")
print(f"\nKey variables added:")
print(f"  - Component scores: education_score, income_score, body_score, height_score, age_score, effort_score")
print(f"  - rating_index (0-1 scale) - THE KEY DESIRABILITY MEASURE")
print(f"  - rating_quintile (bottom_20, low, middle, high, top_20)")
print("\nNext step: Step 5 - Classify relationship goals from essays (g_i)")
print("="*80)
