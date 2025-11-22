"""
Step 2: Data Cleaning and Standardization
OkCupid Dating App Project

This script cleans and standardizes all messy categorical variables
and prepares them for rating index construction and analysis.
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

print("="*80)
print("STEP 2: DATA CLEANING AND STANDARDIZATION")
print("="*80)

# ============================================================================
# LOAD THE DATA
# ============================================================================
print("\n[1] Loading the dataset...")

# UPDATE THIS PATH TO YOUR CSV FILE LOCATION
file_path = "data/okcupid_profiles.csv"  # <-- CHANGE THIS TO YOUR FILE PATH

try:
    df = pd.read_csv(file_path)
    print(f"✓ Data loaded successfully! Shape: {df.shape}")
except FileNotFoundError:
    print(f"✗ File not found: {file_path}")
    print("Please update the 'file_path' variable with the correct path to your CSV file.")
    exit()

# Create a copy for cleaning
df_clean = df.copy()

print(f"Starting with {len(df_clean):,} profiles")

# ============================================================================
# 1. CLEAN EDUCATION
# ============================================================================
print("\n[2] Cleaning EDUCATION...")

def clean_education(edu):
    """Convert education to ordinal level (0-5)"""
    if pd.isna(edu):
        return np.nan
    
    edu_lower = str(edu).lower()
    
    # Space camp = joke/missing
    if 'space camp' in edu_lower:
        return np.nan
    
    # PhD, Law, Med school = 5
    if any(x in edu_lower for x in ['ph.d', 'phd', 'law school', 'med school']):
        return 5
    
    # Masters = 4
    if 'masters' in edu_lower:
        return 4
    
    # College/University = 3
    if 'college' in edu_lower or 'university' in edu_lower:
        # But if "two-year college" = 2
        if 'two-year' in edu_lower or 'two year' in edu_lower:
            return 2
        return 3
    
    # High school = 1
    if 'high school' in edu_lower:
        return 1
    
    # Anything else
    return np.nan

df_clean['education_level'] = df_clean['education'].apply(clean_education)

# Also create education status (working on, graduated, dropped out)
def get_education_status(edu):
    """Extract education status"""
    if pd.isna(edu):
        return np.nan
    
    edu_lower = str(edu).lower()
    
    if 'graduated' in edu_lower:
        return 'graduated'
    elif 'working on' in edu_lower:
        return 'working_on'
    elif 'dropped out' in edu_lower:
        return 'dropped_out'
    else:
        return 'unknown'

df_clean['education_status'] = df_clean['education'].apply(get_education_status)

print(f"  ✓ Education cleaned")
print(f"    - Education levels: {df_clean['education_level'].value_counts().sort_index().to_dict()}")
print(f"    - Missing: {df_clean['education_level'].isna().sum():,} ({df_clean['education_level'].isna().sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 2. CLEAN ETHNICITY
# ============================================================================
print("\n[3] Cleaning ETHNICITY...")

def extract_primary_ethnicity(eth):
    """Extract the first mentioned ethnicity"""
    if pd.isna(eth):
        return np.nan
    
    eth_str = str(eth).lower().strip()
    
    # Split by comma and take first
    parts = [p.strip() for p in eth_str.split(',')]
    primary = parts[0]
    
    # Standardize
    if 'white' in primary:
        return 'white'
    elif 'asian' in primary:
        return 'asian'
    elif 'black' in primary:
        return 'black'
    elif 'hispanic' in primary or 'latin' in primary:
        return 'hispanic_latin'
    elif 'indian' in primary:
        return 'indian'
    elif 'middle eastern' in primary:
        return 'middle_eastern'
    elif 'pacific islander' in primary:
        return 'pacific_islander'
    elif 'native american' in primary:
        return 'native_american'
    elif 'other' in primary:
        return 'other'
    else:
        return 'other'

def is_multiracial(eth):
    """Check if multiple ethnicities are listed"""
    if pd.isna(eth):
        return np.nan
    return 1 if ',' in str(eth) else 0

df_clean['ethnicity_primary'] = df_clean['ethnicity'].apply(extract_primary_ethnicity)
df_clean['is_multiracial'] = df_clean['ethnicity'].apply(is_multiracial)

print(f"  ✓ Ethnicity cleaned")
print(f"    - Primary ethnicities: {df_clean['ethnicity_primary'].value_counts().to_dict()}")
print(f"    - Multiracial: {df_clean['is_multiracial'].sum():,} ({df_clean['is_multiracial'].sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 3. CLEAN BODY TYPE
# ============================================================================
print("\n[4] Cleaning BODY_TYPE...")

def clean_body_type(body):
    """Standardize body type categories"""
    if pd.isna(body):
        return np.nan
    
    body_lower = str(body).lower().strip()
    
    # Thin/skinny → thin
    if body_lower in ['thin', 'skinny']:
        return 'thin'
    
    # Fit/athletic/jacked → fit
    if body_lower in ['fit', 'athletic', 'jacked']:
        return 'fit'
    
    # Overweight/full figured → overweight
    if body_lower in ['overweight', 'full figured']:
        return 'overweight'
    
    # Keep as is
    if body_lower in ['average', 'curvy', 'a little extra']:
        return body_lower
    
    # Used up, rather not say → missing
    if body_lower in ['used up', 'rather not say']:
        return np.nan
    
    return np.nan

df_clean['body_type_clean'] = df_clean['body_type'].apply(clean_body_type)

print(f"  ✓ Body type cleaned")
print(f"    - Categories: {df_clean['body_type_clean'].value_counts().to_dict()}")
print(f"    - Missing: {df_clean['body_type_clean'].isna().sum():,} ({df_clean['body_type_clean'].isna().sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 4. CLEAN DIET
# ============================================================================
print("\n[5] Cleaning DIET...")

def clean_diet(diet_str):
    """Extract main diet type and strictness"""
    if pd.isna(diet_str):
        return np.nan, np.nan
    
    diet_lower = str(diet_str).lower().strip()
    
    # Determine strictness
    if 'strictly' in diet_lower:
        strictness = 'strict'
    elif 'mostly' in diet_lower:
        strictness = 'mostly'
    else:
        strictness = 'not_strict'
    
    # Determine diet type
    if 'vegan' in diet_lower:
        diet_type = 'vegan'
    elif 'vegetarian' in diet_lower:
        diet_type = 'vegetarian'
    elif 'kosher' in diet_lower:
        diet_type = 'kosher'
    elif 'halal' in diet_lower:
        diet_type = 'halal'
    elif 'anything' in diet_lower:
        diet_type = 'omnivore'
    elif 'other' in diet_lower:
        diet_type = 'other'
    else:
        diet_type = 'other'
    
    return diet_type, strictness

# Apply the function
diet_results = df_clean['diet'].apply(lambda x: pd.Series(clean_diet(x)))
df_clean['diet_type'] = diet_results[0]
df_clean['diet_strictness'] = diet_results[1]

print(f"  ✓ Diet cleaned")
print(f"    - Diet types: {df_clean['diet_type'].value_counts().to_dict()}")
print(f"    - Missing: {df_clean['diet_type'].isna().sum():,} ({df_clean['diet_type'].isna().sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 5. CLEAN OFFSPRING
# ============================================================================
print("\n[6] Cleaning OFFSPRING...")

def clean_offspring(off):
    """Extract has_children and wants_children status"""
    if pd.isna(off):
        return np.nan, np.nan
    
    off_lower = str(off).lower().strip()
    
    # Has children?
    if 'has a kid' in off_lower or 'has kids' in off_lower:
        has_children = 1
    elif "doesn't have kids" in off_lower or "doesn&rsquo;t have kids" in off_lower:
        has_children = 0
    else:
        has_children = np.nan
    
    # Wants children?
    if 'wants them' in off_lower or 'and wants' in off_lower:
        wants_children = 'yes'
    elif 'might want' in off_lower:
        wants_children = 'maybe'
    elif "doesn't want" in off_lower or "doesn&rsquo;t want" in off_lower:
        wants_children = 'no'
    elif "doesn't want more" in off_lower or "doesn&rsquo;t want more" in off_lower:
        wants_children = 'no_more'
    else:
        wants_children = np.nan
    
    return has_children, wants_children

offspring_results = df_clean['offspring'].apply(lambda x: pd.Series(clean_offspring(x)))
df_clean['has_children'] = offspring_results[0]
df_clean['wants_children'] = offspring_results[1]

print(f"  ✓ Offspring cleaned")
print(f"    - Has children: {df_clean['has_children'].value_counts().to_dict()}")
print(f"    - Wants children: {df_clean['wants_children'].value_counts().to_dict()}")

# ============================================================================
# 6. CLEAN PETS
# ============================================================================
print("\n[7] Cleaning PETS...")

def clean_pets(pet_str):
    """Extract dog and cat preferences and ownership"""
    if pd.isna(pet_str):
        return np.nan, np.nan, np.nan, np.nan
    
    pet_lower = str(pet_str).lower().strip()
    
    # Dog ownership
    has_dogs = 1 if 'has dogs' in pet_lower else 0
    
    # Cat ownership
    has_cats = 1 if 'has cats' in pet_lower else 0
    
    # Dog preference
    if 'likes dogs' in pet_lower:
        dog_pref = 'likes'
    elif 'dislikes dogs' in pet_lower:
        dog_pref = 'dislikes'
    elif 'has dogs' in pet_lower:
        dog_pref = 'owns'
    else:
        dog_pref = 'neutral'
    
    # Cat preference
    if 'likes cats' in pet_lower:
        cat_pref = 'likes'
    elif 'dislikes cats' in pet_lower:
        cat_pref = 'dislikes'
    elif 'has cats' in pet_lower:
        cat_pref = 'owns'
    else:
        cat_pref = 'neutral'
    
    return has_dogs, has_cats, dog_pref, cat_pref

pets_results = df_clean['pets'].apply(lambda x: pd.Series(clean_pets(x)))
df_clean['has_dogs'] = pets_results[0]
df_clean['has_cats'] = pets_results[1]
df_clean['dog_preference'] = pets_results[2]
df_clean['cat_preference'] = pets_results[3]

print(f"  ✓ Pets cleaned")
print(f"    - Has dogs: {df_clean['has_dogs'].sum():,}")
print(f"    - Has cats: {df_clean['has_cats'].sum():,}")
print(f"    - Dog preferences: {df_clean['dog_preference'].value_counts().to_dict()}")

# ============================================================================
# 7. CLEAN RELIGION
# ============================================================================
print("\n[8] Cleaning RELIGION...")

def clean_religion(rel):
    """Separate religion type and seriousness"""
    if pd.isna(rel):
        return np.nan, np.nan
    
    rel_lower = str(rel).lower().strip()
    
    # Extract seriousness
    if 'very serious' in rel_lower:
        seriousness = 'very_serious'
    elif 'somewhat serious' in rel_lower:
        seriousness = 'somewhat_serious'
    elif 'not too serious' in rel_lower:
        seriousness = 'not_serious'
    elif 'laughing' in rel_lower:
        seriousness = 'laughing'
    else:
        seriousness = 'unspecified'
    
    # Extract religion type
    if 'atheism' in rel_lower or 'atheist' in rel_lower:
        rel_type = 'atheism'
    elif 'agnosticism' in rel_lower or 'agnostic' in rel_lower:
        rel_type = 'agnosticism'
    elif 'christianity' in rel_lower or 'christian' in rel_lower:
        rel_type = 'christianity'
    elif 'catholicism' in rel_lower or 'catholic' in rel_lower:
        rel_type = 'catholicism'
    elif 'judaism' in rel_lower or 'jewish' in rel_lower:
        rel_type = 'judaism'
    elif 'buddhism' in rel_lower or 'buddhist' in rel_lower:
        rel_type = 'buddhism'
    elif 'islam' in rel_lower or 'muslim' in rel_lower:
        rel_type = 'islam'
    elif 'hinduism' in rel_lower or 'hindu' in rel_lower:
        rel_type = 'hinduism'
    elif 'other' in rel_lower:
        rel_type = 'other'
    else:
        rel_type = 'other'
    
    return rel_type, seriousness

religion_results = df_clean['religion'].apply(lambda x: pd.Series(clean_religion(x)))
df_clean['religion_type'] = religion_results[0]
df_clean['religion_seriousness'] = religion_results[1]

print(f"  ✓ Religion cleaned")
print(f"    - Religion types: {df_clean['religion_type'].value_counts().head(10).to_dict()}")
print(f"    - Seriousness: {df_clean['religion_seriousness'].value_counts().to_dict()}")

# ============================================================================
# 8. CLEAN SIGN (optional - extract zodiac only)
# ============================================================================
print("\n[9] Cleaning SIGN...")

def extract_zodiac(sign_str):
    """Extract just the zodiac sign, ignore attitude"""
    if pd.isna(sign_str):
        return np.nan
    
    sign_lower = str(sign_str).lower().strip()
    
    # Remove HTML entities
    sign_lower = sign_lower.replace('&rsquo;', "'").replace('&nbsp;', ' ')
    
    # Extract sign
    signs = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo', 
             'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces']
    
    for sign in signs:
        if sign in sign_lower:
            return sign
    
    return np.nan

df_clean['zodiac_sign'] = df_clean['sign'].apply(extract_zodiac)

print(f"  ✓ Sign cleaned")
print(f"    - Missing: {df_clean['zodiac_sign'].isna().sum():,} ({df_clean['zodiac_sign'].isna().sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 9. CLEAN SPEAKS (languages)
# ============================================================================
print("\n[10] Cleaning SPEAKS...")

def parse_languages(speaks_str):
    """Extract number of languages and English proficiency"""
    if pd.isna(speaks_str):
        return 0, np.nan, 0
    
    speaks_lower = str(speaks_str).lower().strip()
    
    # Count languages (by counting commas + 1)
    num_languages = speaks_lower.count(',') + 1
    
    # English proficiency
    if 'english (fluently)' in speaks_lower or speaks_lower == 'english':
        english_prof = 'fluent'
    elif 'english (okay)' in speaks_lower:
        english_prof = 'okay'
    elif 'english (poorly)' in speaks_lower:
        english_prof = 'poor'
    else:
        english_prof = 'unspecified'
    
    # Speaks Spanish?
    speaks_spanish = 1 if 'spanish' in speaks_lower else 0
    
    return num_languages, english_prof, speaks_spanish

speaks_results = df_clean['speaks'].apply(lambda x: pd.Series(parse_languages(x)))
df_clean['num_languages'] = speaks_results[0]
df_clean['english_proficiency'] = speaks_results[1]
df_clean['speaks_spanish'] = speaks_results[2]

print(f"  ✓ Speaks cleaned")
print(f"    - Avg languages: {df_clean['num_languages'].mean():.2f}")
print(f"    - English proficiency: {df_clean['english_proficiency'].value_counts().to_dict()}")
print(f"    - Speaks Spanish: {df_clean['speaks_spanish'].sum():,} ({df_clean['speaks_spanish'].sum()/len(df_clean)*100:.1f}%)")

# ============================================================================
# 10. CLEAN HEIGHT (remove outliers)
# ============================================================================
print("\n[11] Cleaning HEIGHT...")

# Original stats
print(f"  Original height range: {df_clean['height'].min():.0f} - {df_clean['height'].max():.0f} inches")

# Clean: reasonable range is 48-84 inches (4 feet to 7 feet)
df_clean['height_clean'] = df_clean['height'].copy()
df_clean.loc[(df_clean['height_clean'] < 48) | (df_clean['height_clean'] > 84), 'height_clean'] = np.nan

print(f"  Cleaned height range: {df_clean['height_clean'].min():.0f} - {df_clean['height_clean'].max():.0f} inches")
print(f"  Outliers removed: {(df_clean['height'].notna() & df_clean['height_clean'].isna()).sum()}")

# ============================================================================
# 11. CLEAN INCOME (create flag for reported income)
# ============================================================================
print("\n[12] Cleaning INCOME...")

df_clean['income_reported'] = (df_clean['income'] > 0).astype(int)
df_clean['income_clean'] = df_clean['income'].copy()
df_clean.loc[df_clean['income_clean'] == -1, 'income_clean'] = np.nan

print(f"  ✓ Income cleaned")
print(f"    - Reported income: {df_clean['income_reported'].sum():,} ({df_clean['income_reported'].sum()/len(df_clean)*100:.1f}%)")
print(f"    - Mean income (reported only): ${df_clean['income_clean'].mean():,.0f}")
print(f"    - Median income (reported only): ${df_clean['income_clean'].median():,.0f}")

# ============================================================================
# 12. EXTRACT LOCATION (city and state)
# ============================================================================
print("\n[13] Extracting LOCATION components...")

def parse_location(loc):
    """Extract city and state from location string"""
    if pd.isna(loc):
        return np.nan, np.nan
    
    # Format is typically "city, state"
    parts = str(loc).split(',')
    
    if len(parts) >= 2:
        city = parts[0].strip().lower()
        state = parts[1].strip().lower()
        return city, state
    else:
        return np.nan, np.nan

location_results = df_clean['location'].apply(lambda x: pd.Series(parse_location(x)))
df_clean['city'] = location_results[0]
df_clean['state'] = location_results[1]

print(f"  ✓ Location parsed")
print(f"    - Unique cities: {df_clean['city'].nunique()}")
print(f"    - Unique states: {df_clean['state'].nunique()}")
print(f"    - Top cities: {df_clean['city'].value_counts().head(5).to_dict()}")

# ============================================================================
# 13. CREATE MARKET ID (location × sex × orientation)
# ============================================================================
print("\n[14] Creating MARKET_ID...")

def create_market_id(row):
    """
    Create market ID for matching
    For straight: same location, opposite sex
    For gay/lesbian: same location, same sex
    For bisexual: might match with both (keep both orientations)
    """
    if pd.isna(row['city']) or pd.isna(row['state']):
        return np.nan
    
    location = f"{row['city']}_{row['state']}".replace(' ', '_')
    sex = str(row['sex']).lower()
    orientation = str(row['orientation']).lower()
    
    return f"{location}_{sex}_{orientation}"

df_clean['market_id'] = df_clean.apply(create_market_id, axis=1)

# Also create target market (who they are looking for)
def create_target_market(row):
    """Who is this person looking for?"""
    if pd.isna(row['city']) or pd.isna(row['state']):
        return np.nan
    
    location = f"{row['city']}_{row['state']}".replace(' ', '_')
    sex = str(row['sex']).lower()
    orientation = str(row['orientation']).lower()
    
    # Straight people look for opposite sex
    if orientation == 'straight':
        target_sex = 'f' if sex == 'm' else 'm'
        return f"{location}_{target_sex}_straight"
    
    # Gay/lesbian look for same sex
    elif orientation == 'gay':
        return f"{location}_{sex}_gay"
    
    # Bisexual can match with both (for now, keep as is)
    elif orientation == 'bisexual':
        return f"{location}_both_bisexual"
    
    return np.nan

df_clean['target_market'] = df_clean.apply(create_target_market, axis=1)

print(f"  ✓ Market IDs created")
print(f"    - Unique markets: {df_clean['market_id'].nunique()}")
print(f"    - Top markets: {df_clean['market_id'].value_counts().head(5).to_dict()}")

# ============================================================================
# 14. SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("CLEANING SUMMARY")
print("="*80)

new_columns = [
    'education_level', 'education_status', 'ethnicity_primary', 'is_multiracial',
    'body_type_clean', 'diet_type', 'diet_strictness', 'has_children', 'wants_children',
    'has_dogs', 'has_cats', 'dog_preference', 'cat_preference', 'religion_type', 
    'religion_seriousness', 'zodiac_sign', 'num_languages', 'english_proficiency',
    'speaks_spanish', 'height_clean', 'income_reported', 'income_clean',
    'city', 'state', 'market_id', 'target_market'
]

print(f"\nNew cleaned columns created: {len(new_columns)}")
print(f"Original columns: {df.shape[1]}")
print(f"Total columns now: {df_clean.shape[1]}")

# Missing values in new columns
print("\nMissing values in key cleaned columns:")
missing_summary = pd.DataFrame({
    'Column': new_columns[:15],  # First 15
    'Missing_Count': [df_clean[col].isna().sum() for col in new_columns[:15]],
    'Missing_Percent': [f"{df_clean[col].isna().sum()/len(df_clean)*100:.1f}%" for col in new_columns[:15]]
})
print(missing_summary.to_string(index=False))

# ============================================================================
# 15. SAVE CLEANED DATA
# ============================================================================
print("\n[15] Saving cleaned dataset...")

# Save full dataset
df_clean.to_csv('okcupid_cleaned.csv', index=False)
print(f"  ✓ Full cleaned dataset saved: 'okcupid_cleaned.csv'")

# Save just the key columns (smaller file)
key_columns = [
    # Original key columns
    'age', 'sex', 'orientation', 'height_clean', 'income_clean', 'income_reported',
    # Cleaned demographic columns
    'education_level', 'education_status', 'ethnicity_primary', 'is_multiracial',
    'body_type_clean', 'diet_type', 'has_children', 'wants_children',
    'religion_type', 'religion_seriousness', 'num_languages', 'english_proficiency',
    # Market columns
    'city', 'state', 'market_id', 'target_market',
    # Essays (for later text analysis)
    'essay0', 'essay1', 'essay2', 'essay3', 'essay4', 
    'essay5', 'essay6', 'essay7', 'essay8', 'essay9',
    # Original columns for reference
    'status', 'location'
]

df_key = df_clean[key_columns].copy()
df_key.to_csv('okcupid_cleaned_key_columns.csv', index=False)
print(f"  ✓ Key columns dataset saved: 'okcupid_cleaned_key_columns.csv'")
print(f"    (Contains {len(key_columns)} columns instead of {df_clean.shape[1]})")

# Save cleaning summary
with open('data_cleaning_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("DATA CLEANING SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    f.write(f"Cleaned dataset: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns\n\n")
    
    f.write("NEW COLUMNS CREATED:\n")
    f.write("-"*80 + "\n")
    for col in new_columns:
        missing = df_clean[col].isna().sum()
        missing_pct = missing / len(df_clean) * 100
        f.write(f"  {col:30s} - {missing:6,} missing ({missing_pct:5.1f}%)\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("READY FOR NEXT STEPS:\n")
    f.write("  - Step 3: Calculate effort index from essays\n")
    f.write("  - Step 4: Build rating index from cleaned variables\n")
    f.write("  - Step 5: Classify relationship goals from essays\n")
    f.write("  - Step 6: Create market-level indices\n")
    f.write("="*80 + "\n")

print("  ✓ Summary saved: 'data_cleaning_summary.txt'")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("CLEANING COMPLETE!")
print("="*80)
print(f"\nCleaned data saved to:")
print(f"  1. okcupid_cleaned.csv (full dataset with {df_clean.shape[1]} columns)")
print(f"  2. okcupid_cleaned_key_columns.csv (key columns only - {len(key_columns)} columns)")
print(f"  3. data_cleaning_summary.txt (summary of changes)")
print("\nNext step: Step 3 - Calculate effort index from essays")
print("="*80)
