"""
Extract Parameters from OkCupid Data for Simulation
Run this locally with your okcupid_final_analysis_ready.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import json

# ============================================================================
# LOAD DATA
# ============================================================================

print("="*80)
print("EXTRACTING SIMULATION PARAMETERS FROM OKCUPID DATA")
print("="*80)

# Change this path to wherever your CSV is located
df = pd.read_csv('notebooks/outputs/Final/okcupid_final_analysis_ready.csv')

print(f"\n✓ Loaded {len(df):,} profiles")

# ============================================================================
# EXTRACT PARAMETERS
# ============================================================================

print("\n" + "="*80)
print("EXTRACTED PARAMETERS")
print("="*80)

# 1. Rating distribution (r_i)
mean_rating = df['rating_index'].mean()
std_rating = df['rating_index'].std()

# Fit Beta distribution
alpha_rating = mean_rating * ((mean_rating * (1 - mean_rating) / (std_rating**2)) - 1)
beta_rating = (1 - mean_rating) * ((mean_rating * (1 - mean_rating) / (std_rating**2)) - 1)

print(f"\n1. RATING DISTRIBUTION (r_i):")
print(f"   Mean: {mean_rating:.3f}")
print(f"   Std:  {std_rating:.3f}")
print(f"   Beta parameters: α = {alpha_rating:.2f}, β = {beta_rating:.2f}")

# 2. Market LTR share (ρ_m)
ltr_share = df['is_ltr_oriented'].mean()

print(f"\n2. MARKET LTR SHARE (ρ_m):")
print(f"   {ltr_share:.1%}")

# 3. Market clarity (ψ_m) - from signal clarity
market_clarity = df['target_avg_effort'].mean()
market_clarity_std = df['target_avg_effort'].std()

print(f"\n3. MARKET CLARITY (ψ_m):")
print(f"   Mean: {market_clarity:.3f}")
print(f"   Std:  {market_clarity_std:.3f}")
print(f"   (proxy: average effort of potential partners)")

# 4. Effort distribution (for τ_i proxy)
mean_effort = df['effort_index'].mean()
std_effort = df['effort_index'].std()

print(f"\n4. EFFORT DISTRIBUTION:")
print(f"   Mean: {mean_effort:.3f}")
print(f"   Std:  {std_effort:.3f}")
print(f"   (can proxy user clarity τ_i)")

# 5. Self-based priors (ρ_{i0}) by goal type
ltr_users = df[df['is_ltr_oriented'] == True]
casual_users = df[df['is_ltr_oriented'] == False]

print(f"\n5. SELF-BASED PRIORS (ρ_{{i0}}):")
print(f"   Suggested for LTR users:    ρ_{{i0}} = 0.60 (optimistic self-projection)")
print(f"   Suggested for Casual users: ρ_{{i0}} = 0.40 (neutral)")

# 6. Sample size K
print(f"\n6. SAMPLE SIZE (K):")
print(f"   K = 20 (standard batch size)")

# ============================================================================
# SAVE PARAMETERS TO FILE
# ============================================================================

params = {
    'mean_rating': mean_rating,
    'std_rating': std_rating,
    'alpha_rating': alpha_rating,
    'beta_rating': beta_rating,
    'ltr_share': ltr_share,
    'market_clarity': market_clarity,
    'market_clarity_std': market_clarity_std,
    'mean_effort': mean_effort,
    'std_effort': std_effort,
    'K': 20,
    'rho_i0_ltr': 0.60,
    'rho_i0_casual': 0.40,
    'tau_i_suggested': 10
}

# Save as JSON
with open('simulation_parameters.json', 'w') as f:
    json.dump(params, f, indent=2)

print("\n" + "="*80)
print("✓ Parameters saved to: simulation_parameters.json")
print("="*80)

# ============================================================================
# SUGGESTED PARAMETER VALUES FOR SIMULATION
# ============================================================================

print("\n" + "="*80)
print("READY TO USE IN SIMULATION")
print("="*80)

print(f"""
Copy these into your simulation:

# From OkCupid data
MEAN_RATING = {mean_rating:.3f}
STD_RATING = {std_rating:.3f}
ALPHA_RATING = {alpha_rating:.2f}
BETA_RATING = {beta_rating:.2f}

RHO_M_TRUE = {ltr_share:.3f}  # True LTR share
PSI_M = {market_clarity:.3f}       # Market clarity

K = 20               # Profiles shown per period

# Self-based priors
RHO_I0_LTR = 0.60    # LTR users expect ~60% LTR market
RHO_I0_CASUAL = 0.40 # Casual users expect ~40% LTR market

# Prior strength (calibrate)
TAU_I = 10           # Can vary: higher = stronger prior

# Frustration threshold (calibrate to match exits)
P_BAR = 0.20         # Exit when p_hat < 0.20
""")

print("="*80)
print("\n✓ EXTRACTION COMPLETE!")
print("="*80)