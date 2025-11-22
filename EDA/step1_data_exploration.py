"""
Step 1: Initial Data Loading and Exploration
OkCupid Dating App Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

print("="*80)
print("STEP 1: LOADING AND EXPLORING THE OKCUPID DATASET")
print("="*80)

# ============================================================================
# 1. LOAD THE DATA
# ============================================================================
print("\n[1] Loading the dataset...")

# UPDATE THIS PATH TO YOUR CSV FILE LOCATION
file_path = "data/okcupid_profiles.csv"  # <-- CHANGE THIS TO YOUR FILE PATH

try:
    df = pd.read_csv(file_path)
    print(f"✓ Data loaded successfully!")
except FileNotFoundError:
    print(f"✗ File not found: {file_path}")
    print("Please update the 'file_path' variable with the correct path to your CSV file.")
    exit()

# ============================================================================
# 2. BASIC DATASET INFORMATION
# ============================================================================
print("\n[2] Basic Dataset Information")
print("-" * 80)
print(f"Number of profiles (rows): {df.shape[0]:,}")
print(f"Number of variables (columns): {df.shape[1]}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 3. COLUMN NAMES AND DATA TYPES
# ============================================================================
print("\n[3] Column Names and Data Types")
print("-" * 80)
print(df.dtypes)

# ============================================================================
# 4. FIRST FEW ROWS
# ============================================================================
print("\n[4] First 5 Rows of Data")
print("-" * 80)
print(df.head())

# ============================================================================
# 5. MISSING VALUES ANALYSIS
# ============================================================================
print("\n[5] Missing Values Analysis")
print("-" * 80)

missing_stats = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': (df.isnull().sum() / len(df) * 100).round(2),
    'Data_Type': df.dtypes
})
missing_stats = missing_stats.sort_values('Missing_Percent', ascending=False)
missing_stats = missing_stats.reset_index(drop=True)

print(missing_stats.to_string(index=False))

# Identify heavily missing columns (>50% missing)
high_missing = missing_stats[missing_stats['Missing_Percent'] > 50]
if len(high_missing) > 0:
    print(f"\n⚠ Warning: {len(high_missing)} columns have >50% missing values:")
    print(high_missing[['Column', 'Missing_Percent']].to_string(index=False))

# ============================================================================
# 6. KEY DEMOGRAPHIC VARIABLES
# ============================================================================
print("\n[6] Key Demographic Variables - Value Counts")
print("-" * 80)

# Variables that should have few categories
categorical_vars = ['sex', 'status', 'orientation', 'drinks', 'drugs', 'smokes']

for var in categorical_vars:
    if var in df.columns:
        print(f"\n{var.upper()}:")
        print(df[var].value_counts(dropna=False).head(10))
        print(f"Unique values: {df[var].nunique()}")
    else:
        print(f"\n{var.upper()}: Column not found")

# ============================================================================
# 7. CONTINUOUS VARIABLES
# ============================================================================
print("\n[7] Continuous Variables - Summary Statistics")
print("-" * 80)

continuous_vars = ['age', 'height', 'income']
if any(var in df.columns for var in continuous_vars):
    print(df[continuous_vars].describe())

# ============================================================================
# 8. MESSY CATEGORICAL VARIABLES (from Data_columns notes)
# ============================================================================
print("\n[8] Messy Categorical Variables - Sample Values")
print("-" * 80)

messy_vars = ['body_type', 'diet', 'education', 'ethnicity', 'offspring', 
              'pets', 'religion', 'sign', 'speaks']

for var in messy_vars:
    if var in df.columns:
        print(f"\n{var.upper()} (unique values: {df[var].nunique()}):")
        print(df[var].value_counts(dropna=False).head(15))
    else:
        print(f"\n{var.upper()}: Column not found")

# ============================================================================
# 9. ESSAY COLUMNS
# ============================================================================
print("\n[9] Essay Columns - Non-Empty Counts")
print("-" * 80)

essay_cols = [f'essay{i}' for i in range(10)]
essay_stats = []

for col in essay_cols:
    if col in df.columns:
        non_empty = df[col].notna().sum()
        pct_filled = (non_empty / len(df) * 100)
        essay_stats.append({
            'Essay': col,
            'Non_Empty': non_empty,
            'Percent_Filled': f"{pct_filled:.1f}%"
        })

if essay_stats:
    essay_df = pd.DataFrame(essay_stats)
    print(essay_df.to_string(index=False))

# ============================================================================
# 10. LOCATION INFORMATION
# ============================================================================
print("\n[10] Location Information")
print("-" * 80)

if 'location' in df.columns:
    print(f"Unique locations: {df['location'].nunique()}")
    print("\nTop 15 locations:")
    print(df['location'].value_counts().head(15))

# ============================================================================
# 11. INCOME SPECIAL VALUES
# ============================================================================
print("\n[11] Income Special Values (checking for -1 indicating missing)")
print("-" * 80)

if 'income' in df.columns:
    print(df['income'].value_counts(dropna=False).head(15))
    negative_income = (df['income'] == -1).sum()
    print(f"\nProfiles with income = -1 (missing): {negative_income:,} ({negative_income/len(df)*100:.1f}%)")

# ============================================================================
# 12. SAVE SUMMARY TO FILE
# ============================================================================
print("\n[12] Saving exploration summary...")

with open('data_exploration_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("OKCUPID DATASET - INITIAL EXPLORATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
    
    f.write("MISSING VALUES:\n")
    f.write("-"*80 + "\n")
    f.write(missing_stats.to_string(index=False))
    f.write("\n\n")
    
    f.write("COLUMN DATA TYPES:\n")
    f.write("-"*80 + "\n")
    f.write(df.dtypes.to_string())
    f.write("\n\n")

print("✓ Summary saved to 'data_exploration_summary.txt'")

# ============================================================================
# 13. QUICK VISUALIZATION (optional - comment out if issues)
# ============================================================================
print("\n[13] Creating basic visualizations...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age distribution
    if 'age' in df.columns:
        df['age'].hist(bins=50, ax=axes[0, 0], edgecolor='black')
        axes[0, 0].set_title('Age Distribution')
        axes[0, 0].set_xlabel('Age')
        axes[0, 0].set_ylabel('Count')
    
    # Sex distribution
    if 'sex' in df.columns:
        df['sex'].value_counts().plot(kind='bar', ax=axes[0, 1], color=['skyblue', 'salmon'])
        axes[0, 1].set_title('Sex Distribution')
        axes[0, 1].set_xlabel('Sex')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=0)
    
    # Orientation distribution
    if 'orientation' in df.columns:
        df['orientation'].value_counts().plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Orientation Distribution')
        axes[1, 0].set_xlabel('Orientation')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Income distribution (excluding -1)
    if 'income' in df.columns:
        income_clean = df[df['income'] > 0]['income']
        if len(income_clean) > 0:
            income_clean.hist(bins=30, ax=axes[1, 1], edgecolor='black')
            axes[1, 1].set_title('Income Distribution (excluding missing)')
            axes[1, 1].set_xlabel('Income')
            axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('initial_exploration_plots.png', dpi=150, bbox_inches='tight')
    print("✓ Visualizations saved to 'initial_exploration_plots.png'")
    
except Exception as e:
    print(f"⚠ Visualization error (non-critical): {e}")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "="*80)
print("EXPLORATION COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review the output above and 'data_exploration_summary.txt'")
print("2. Note which variables need cleaning (especially the 'messy' ones)")
print("3. Ready to move to Step 2: Data Cleaning")
print("="*80)
