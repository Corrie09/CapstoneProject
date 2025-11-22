"""
Step 5: Classify Relationship Goals (g_i)
OkCupid Dating App Project

This script classifies users' relationship goals as:
- LTR (Long-term relationship oriented)
- Casual (Short-term/hookup oriented)
- Ambiguous (unclear or mixed signals)

Uses keyword analysis and pattern matching in essay text.
This is the g_i variable in your theoretical framework.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import matplotlib.pyplot as plt

print("="*80)
print("STEP 5: CLASSIFY RELATIONSHIP GOALS (g_i)")
print("="*80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1] Loading dataset with ratings...")

try:
    df = pd.read_csv('data/okcupid_with_ratings.csv')
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'okcupid_with_ratings.csv' not found.")
    print("Please run step4_rating_index.py first!")
    exit()

# ============================================================================
# 1. DEFINE KEYWORD DICTIONARIES
# ============================================================================
print("\n[2] Setting up keyword dictionaries...")

# Long-term relationship keywords
LTR_KEYWORDS = [
    # Direct mentions
    'long term', 'long-term', 'relationship', 'serious', 'committed', 'commitment',
    'partner', 'life partner', 'settle down', 'settling down', 'marriage', 'marry',
    'wife', 'husband', 'future', 'family', 'kids', 'children',
    
    # Emotional/deep connection
    'genuine', 'authentic', 'meaningful', 'deep connection', 'real connection',
    'intimacy', 'emotional', 'trust', 'loyalty', 'faithful',
    
    # Stability indicators
    'stable', 'mature', 'grown up', 'adult', 'responsible',
    
    # Dating with intent
    'looking for something real', 'looking for someone special', 'the one',
    'soulmate', 'life together', 'build a life', 'grow together'
]

# Casual/short-term keywords
CASUAL_KEYWORDS = [
    # Direct mentions
    'casual', 'hookup', 'hook up', 'one night', 'fling', 'no strings',
    'friends with benefits', 'fwb', 'short term', 'short-term',
    
    # Fun/non-committal
    'just fun', 'good time', 'keeping it light', 'nothing serious',
    'not looking for anything serious', 'no commitment', 'no drama',
    
    # Physical focus
    'physical', 'attraction', 'chemistry', 'sexy', 'hot',
    
    # Explicit casual indicators
    'right now', 'for now', 'seeing where it goes', 'go with the flow',
    'whatever happens', 'no expectations'
]

# Ambiguous/mixed signals
AMBIGUOUS_KEYWORDS = [
    'open minded', 'open-minded', 'flexible', 'depends', 'not sure',
    'figure it out', 'see what happens', "let's see", 'maybe', 'possibly',
    'new friends', 'activity partners', 'activity partner', 'new people'
]

print(f"  ✓ LTR keywords: {len(LTR_KEYWORDS)}")
print(f"  ✓ Casual keywords: {len(CASUAL_KEYWORDS)}")
print(f"  ✓ Ambiguous keywords: {len(AMBIGUOUS_KEYWORDS)}")

# ============================================================================
# 2. COMBINE ALL ESSAYS INTO ONE TEXT PER USER
# ============================================================================
print("\n[3] Combining essay text...")

essay_columns = [f'essay{i}' for i in range(10)]

def combine_essays(row):
    """Combine all essays into one text string"""
    texts = []
    for col in essay_columns:
        if pd.notna(row[col]):
            texts.append(str(row[col]))
    return ' '.join(texts).lower()

df['all_essays'] = df.apply(combine_essays, axis=1)

# Count users with no essay text
no_essays = (df['all_essays'].str.len() == 0).sum()
print(f"  ✓ Essays combined")
print(f"    - Profiles with no essay text: {no_essays:,} ({no_essays/len(df)*100:.1f}%)")

# ============================================================================
# 3. COUNT KEYWORD MATCHES
# ============================================================================
print("\n[4] Counting keyword matches...")

def count_keywords(text, keywords):
    """Count how many times any keyword appears in text"""
    if pd.isna(text) or len(str(text)) == 0:
        return 0
    
    text_lower = str(text).lower()
    count = 0
    
    for keyword in keywords:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword) + r'\b'
        matches = len(re.findall(pattern, text_lower))
        count += matches
    
    return count

# Count keyword matches for each category
print("  - Counting LTR keywords...")
df['ltr_keyword_count'] = df['all_essays'].apply(lambda x: count_keywords(x, LTR_KEYWORDS))

print("  - Counting Casual keywords...")
df['casual_keyword_count'] = df['all_essays'].apply(lambda x: count_keywords(x, CASUAL_KEYWORDS))

print("  - Counting Ambiguous keywords...")
df['ambiguous_keyword_count'] = df['all_essays'].apply(lambda x: count_keywords(x, AMBIGUOUS_KEYWORDS))

print(f"  ✓ Keyword counting complete")
print(f"    - Mean LTR keywords per profile: {df['ltr_keyword_count'].mean():.2f}")
print(f"    - Mean Casual keywords per profile: {df['casual_keyword_count'].mean():.2f}")
print(f"    - Mean Ambiguous keywords per profile: {df['ambiguous_keyword_count'].mean():.2f}")

# ============================================================================
# 4. CLASSIFY RELATIONSHIP GOALS
# ============================================================================
print("\n[5] Classifying relationship goals...")

def classify_goal(row):
    """
    Classify relationship goal based on keyword counts
    
    Rules:
    1. If no essays: use 'status' field as backup
    2. If LTR keywords >> Casual keywords: LTR
    3. If Casual keywords >> LTR keywords: Casual
    4. If both low or similar: Ambiguous
    5. If ambiguous keywords dominate: Ambiguous
    """
    
    ltr_count = row['ltr_keyword_count']
    casual_count = row['casual_keyword_count']
    ambig_count = row['ambiguous_keyword_count']
    
    # No essay text - use status as backup
    if len(str(row['all_essays'])) < 10:
        status = str(row['status']).lower()
        if 'single' in status or 'available' in status:
            return 'ambiguous'  # Could be either
        elif 'seeing someone' in status or 'married' in status:
            return 'ltr'  # Likely relationship-oriented
        else:
            return 'ambiguous'
    
    # Strong LTR signal (LTR keywords at least 2x casual, and >0)
    if ltr_count >= 2 * casual_count and ltr_count > 0:
        return 'ltr'
    
    # Strong Casual signal (Casual keywords at least 2x LTR, and >0)
    if casual_count >= 2 * ltr_count and casual_count > 0:
        return 'casual'
    
    # Ambiguous keywords dominate
    if ambig_count > ltr_count and ambig_count > casual_count:
        return 'ambiguous'
    
    # Similar counts or both low - ambiguous
    if abs(ltr_count - casual_count) <= 1:
        return 'ambiguous'
    
    # Slight LTR preference
    if ltr_count > casual_count:
        return 'ltr'
    
    # Slight Casual preference
    if casual_count > ltr_count:
        return 'casual'
    
    # Default
    return 'ambiguous'

df['relationship_goal'] = df.apply(classify_goal, axis=1)

print(f"  ✓ Classification complete")
print(f"\n  Distribution of relationship goals:")
goal_dist = df['relationship_goal'].value_counts()
for goal, count in goal_dist.items():
    pct = count / len(df) * 100
    print(f"    {goal:10s}: {count:6,} ({pct:5.1f}%)")

# ============================================================================
# 5. VALIDATE CLASSIFICATION WITH STATUS FIELD
# ============================================================================
print("\n[6] Cross-checking with 'status' field...")

# Check if classification makes sense with status
status_by_goal = pd.crosstab(df['relationship_goal'], df['status'], normalize='index') * 100
print("\n  Status distribution by relationship goal (%):")
print(status_by_goal.round(1).to_string())

# ============================================================================
# 6. ANALYZE GOALS BY DEMOGRAPHICS
# ============================================================================
print("\n[7] Analyzing relationship goals by demographics...")

# By sex
print("\n  By Sex:")
for sex in df['sex'].unique():
    ltr_pct = (df[df['sex'] == sex]['relationship_goal'] == 'ltr').sum() / (df['sex'] == sex).sum() * 100
    casual_pct = (df[df['sex'] == sex]['relationship_goal'] == 'casual').sum() / (df['sex'] == sex).sum() * 100
    print(f"    {sex}: LTR={ltr_pct:.1f}%, Casual={casual_pct:.1f}%")

# By orientation
print("\n  By Orientation:")
for orient in df['orientation'].unique():
    ltr_pct = (df[df['orientation'] == orient]['relationship_goal'] == 'ltr').sum() / (df['orientation'] == orient).sum() * 100
    casual_pct = (df[df['orientation'] == orient]['relationship_goal'] == 'casual').sum() / (df['orientation'] == orient).sum() * 100
    print(f"    {orient:10s}: LTR={ltr_pct:.1f}%, Casual={casual_pct:.1f}%")

# By age group
print("\n  By Age Group:")
for age in ['18-25', '26-30', '31-35', '36-40', '41-50', '50+']:
    if age in df['age_group'].values:
        ltr_pct = (df[df['age_group'] == age]['relationship_goal'] == 'ltr').sum() / (df['age_group'] == age).sum() * 100
        casual_pct = (df[df['age_group'] == age]['relationship_goal'] == 'casual').sum() / (df['age_group'] == age).sum() * 100
        print(f"    {age}: LTR={ltr_pct:.1f}%, Casual={casual_pct:.1f}%")

# By rating quintile
print("\n  By Rating Quintile:")
for quintile in ['bottom_20', 'low', 'middle', 'high', 'top_20']:
    ltr_pct = (df[df['rating_quintile'] == quintile]['relationship_goal'] == 'ltr').sum() / (df['rating_quintile'] == quintile).sum() * 100
    casual_pct = (df[df['rating_quintile'] == quintile]['relationship_goal'] == 'casual').sum() / (df['rating_quintile'] == quintile).sum() * 100
    print(f"    {quintile:12s}: LTR={ltr_pct:.1f}%, Casual={casual_pct:.1f}%")

# ============================================================================
# 7. RELATIONSHIP BETWEEN GOALS, EFFORT, AND RATING
# ============================================================================
print("\n[8] Analyzing relationship between goals, effort, and rating...")

goal_stats = df.groupby('relationship_goal').agg({
    'effort_index': ['mean', 'std'],
    'rating_index': ['mean', 'std'],
    'age': 'mean',
    'education_level': 'mean'
}).round(3)

print("\n  Mean effort and rating by relationship goal:")
print(goal_stats.to_string())

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================
print("\n[9] Creating visualizations...")

try:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Distribution of relationship goals
    goal_counts = df['relationship_goal'].value_counts()
    axes[0, 0].bar(goal_counts.index, goal_counts.values, color=['green', 'gray', 'red'])
    axes[0, 0].set_title('Distribution of Relationship Goals', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Relationship Goal')
    axes[0, 0].set_ylabel('Count')
    for i, (goal, count) in enumerate(goal_counts.items()):
        axes[0, 0].text(i, count, f'{count:,}\n({count/len(df)*100:.1f}%)', 
                       ha='center', va='bottom')
    
    # 2. Goals by sex
    goal_by_sex = pd.crosstab(df['sex'], df['relationship_goal'], normalize='index') * 100
    goal_by_sex.plot(kind='bar', ax=axes[0, 1], color=['green', 'gray', 'red'])
    axes[0, 1].set_title('Relationship Goals by Sex (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Sex')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].legend(title='Goal')
    axes[0, 1].tick_params(axis='x', rotation=0)
    
    # 3. Goals by orientation
    goal_by_orient = pd.crosstab(df['orientation'], df['relationship_goal'], normalize='index') * 100
    goal_by_orient.plot(kind='bar', ax=axes[0, 2], color=['green', 'gray', 'red'])
    axes[0, 2].set_title('Relationship Goals by Orientation (%)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Orientation')
    axes[0, 2].set_ylabel('Percentage')
    axes[0, 2].legend(title='Goal')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Effort by goal
    effort_by_goal = df.groupby('relationship_goal')['effort_index'].mean()
    axes[1, 0].bar(effort_by_goal.index, effort_by_goal.values, color=['green', 'gray', 'red'])
    axes[1, 0].set_title('Mean Effort Index by Relationship Goal', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Relationship Goal')
    axes[1, 0].set_ylabel('Mean Effort Index')
    axes[1, 0].set_ylim([0, 1])
    
    # 5. Rating by goal
    rating_by_goal = df.groupby('relationship_goal')['rating_index'].mean()
    axes[1, 1].bar(rating_by_goal.index, rating_by_goal.values, color=['green', 'gray', 'red'])
    axes[1, 1].set_title('Mean Rating Index by Relationship Goal', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Relationship Goal')
    axes[1, 1].set_ylabel('Mean Rating Index')
    axes[1, 1].set_ylim([0, 1])
    
    # 6. Goals by rating quintile
    goal_by_rating = pd.crosstab(df['rating_quintile'], df['relationship_goal'], normalize='index') * 100
    goal_by_rating = goal_by_rating.reindex(['bottom_20', 'low', 'middle', 'high', 'top_20'])
    goal_by_rating.plot(kind='bar', ax=axes[1, 2], color=['green', 'gray', 'red'])
    axes[1, 2].set_title('Relationship Goals by Rating Quintile (%)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Rating Quintile')
    axes[1, 2].set_ylabel('Percentage')
    axes[1, 2].legend(title='Goal')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('relationship_goals_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Visualizations saved to 'relationship_goals_analysis.png'")
    
except Exception as e:
    print(f"  ⚠ Visualization error (non-critical): {e}")

# ============================================================================
# 9. CREATE BINARY LTR FLAG (for easier analysis)
# ============================================================================
print("\n[10] Creating binary LTR flag...")

df['is_ltr_oriented'] = (df['relationship_goal'] == 'ltr').astype(int)
df['is_casual_oriented'] = (df['relationship_goal'] == 'casual').astype(int)

print(f"  ✓ Binary flags created")
print(f"    - LTR-oriented: {df['is_ltr_oriented'].sum():,} ({df['is_ltr_oriented'].mean()*100:.1f}%)")
print(f"    - Casual-oriented: {df['is_casual_oriented'].sum():,} ({df['is_casual_oriented'].mean()*100:.1f}%)")

# ============================================================================
# 10. SAVE RESULTS
# ============================================================================
print("\n[11] Saving results...")

# Save full dataset
df.to_csv('okcupid_with_goals.csv', index=False)
print(f"  ✓ Dataset with relationship goals saved: 'okcupid_with_goals.csv'")

# Save summary
with open('relationship_goals_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("RELATIONSHIP GOALS CLASSIFICATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset: {len(df):,} profiles\n\n")
    
    f.write("CLASSIFICATION METHOD:\n")
    f.write("-"*80 + "\n")
    f.write(f"  - LTR keywords: {len(LTR_KEYWORDS)} terms\n")
    f.write(f"  - Casual keywords: {len(CASUAL_KEYWORDS)} terms\n")
    f.write(f"  - Ambiguous keywords: {len(AMBIGUOUS_KEYWORDS)} terms\n")
    f.write("  - Classification based on keyword frequency analysis\n")
    f.write("  - Backup: status field for profiles with no essays\n\n")
    
    f.write("DISTRIBUTION OF RELATIONSHIP GOALS:\n")
    f.write("-"*80 + "\n")
    for goal, count in df['relationship_goal'].value_counts().items():
        pct = count / len(df) * 100
        f.write(f"  {goal:12s}: {count:6,} ({pct:5.1f}%)\n")
    
    f.write("\nMEAN EFFORT AND RATING BY GOAL:\n")
    f.write("-"*80 + "\n")
    for goal in ['ltr', 'ambiguous', 'casual']:
        effort_mean = df[df['relationship_goal'] == goal]['effort_index'].mean()
        rating_mean = df[df['relationship_goal'] == goal]['rating_index'].mean()
        f.write(f"  {goal:12s}: Effort={effort_mean:.3f}, Rating={rating_mean:.3f}\n")
    
    f.write("\nGOALS BY DEMOGRAPHICS:\n")
    f.write("-"*80 + "\n")
    
    f.write("  By Sex:\n")
    for sex in df['sex'].unique():
        ltr_pct = (df[df['sex'] == sex]['relationship_goal'] == 'ltr').sum() / (df['sex'] == sex).sum() * 100
        f.write(f"    {sex}: {ltr_pct:.1f}% LTR-oriented\n")
    
    f.write("\n  By Orientation:\n")
    for orient in df['orientation'].unique():
        ltr_pct = (df[df['orientation'] == orient]['relationship_goal'] == 'ltr').sum() / (df['orientation'] == orient).sum() * 100
        f.write(f"    {orient:10s}: {ltr_pct:.1f}% LTR-oriented\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  - Most users are ambiguous ({(df['relationship_goal']=='ambiguous').sum()/len(df)*100:.1f}%)\n")
    f.write(f"  - LTR-oriented users: {df['is_ltr_oriented'].sum():,} ({df['is_ltr_oriented'].mean()*100:.1f}%)\n")
    f.write(f"  - Mean keyword counts: LTR={df['ltr_keyword_count'].mean():.2f}, Casual={df['casual_keyword_count'].mean():.2f}\n")
    f.write(f"  - LTR users have higher effort: {df[df['relationship_goal']=='ltr']['effort_index'].mean():.3f} vs ambiguous {df[df['relationship_goal']=='ambiguous']['effort_index'].mean():.3f}\n")
    f.write("\n" + "="*80 + "\n")

print("  ✓ Summary saved: 'relationship_goals_summary.txt'")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("RELATIONSHIP GOALS CLASSIFICATION COMPLETE!")
print("="*80)
print(f"\nFiles created:")
print(f"  1. okcupid_with_goals.csv (dataset with relationship goals)")
print(f"  2. relationship_goals_summary.txt (summary report)")
print(f"  3. relationship_goals_analysis.png (visualizations)")
print(f"\nKey variables added:")
print(f"  - relationship_goal (ltr / casual / ambiguous) - THE g_i VARIABLE")
print(f"  - is_ltr_oriented (binary flag)")
print(f"  - is_casual_oriented (binary flag)")
print(f"  - ltr_keyword_count, casual_keyword_count, ambiguous_keyword_count")
print("\nNext step: Step 6 - Create market-level indices")
print("  (market seriousness, signal clarity, competition metrics)")
print("="*80)
