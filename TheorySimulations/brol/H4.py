import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# PARAMETERS
# ============================================================================

K = 30
beta_0 = 0.01
beta_1 = 0.02

def beta(r):
    return beta_0 + beta_1 * r

theta_L_minus = 0.90
theta_L_plus = 1.10
theta_S_minus = 0.60
theta_S_plus = 1.40

p_bar_L = 0.02
p_bar_S = 0.01

T = 50
N_users = 1000

# Onboarding parameters
T_onboard = 5  # Number of onboarding periods
rho_boost = 0.3  # Boost LT share by this much during onboarding

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def initialize_market(rho_m, N=N_users):
    goals = np.random.choice(['LT', 'Casual'], size=N, p=[rho_m, 1-rho_m])
    ratings = np.random.normal(0.611, 0.126, N)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    for i in range(N):
        if goals[i] == 'LT':
            ratings[i] += np.random.normal(0.02, 0.01)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    # Track when users entered market (for onboarding)
    market = pd.DataFrame({
        'user_id': range(N),
        'goal': goals,
        'rating': ratings,
        'active': True,
        'exit_time': np.nan,
        'exit_type': None,
        'entry_time': 0,  # All enter at t=0 for simplicity
        'is_high_rating_LT': False
    })
    
    # Flag high-rating LT users (top 25% of LT users)
    LT_users = market[market['goal'] == 'LT']
    if len(LT_users) > 0:
        high_rating_threshold = LT_users['rating'].quantile(0.75)
        market.loc[(market['goal'] == 'LT') & (market['rating'] >= high_rating_threshold), 
                   'is_high_rating_LT'] = True
    
    return market

def calculate_acceptable_mass(user_rating, user_goal, market_df, curated=False, user_is_high_LT=False):
    """
    Calculate acceptable partner mass
    
    If curated=True and user is high-rating LT, boost the apparent LT share
    """
    active = market_df[market_df['active'] == True].copy()
    
    if len(active) == 0:
        return 0.0
    
    if user_goal == 'LT':
        rating_min = theta_L_minus * user_rating
        rating_max = theta_L_plus * user_rating
        
        acceptable = active[
            (active['goal'] == 'LT') & 
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
        
        # CURATED ONBOARDING: Boost apparent LT share for high-rating LT users
        if curated and user_is_high_LT:
            # Show them a feed with artificially higher LT concentration
            base_mass = len(acceptable) / len(active)
            # Boost by showing them more LT profiles
            boosted_mass = min(base_mass * (1 + rho_boost), 1.0)
            return boosted_mass
        
    else:  # Casual
        rating_min = theta_S_minus * user_rating
        rating_max = theta_S_plus * user_rating
        
        acceptable = active[
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
    
    return len(acceptable) / len(active)

def calculate_match_probability(user_row, market_df, curated=False):
    """Calculate match probability, potentially with curated feed"""
    A = calculate_acceptable_mass(
        user_row['rating'], 
        user_row['goal'], 
        market_df,
        curated=curated,
        user_is_high_LT=user_row['is_high_rating_LT']
    )
    p = K * beta(user_row['rating']) * A
    return min(p, 1.0)

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_market_with_onboarding(rho_m, curated_onboarding=False, T=50, N=1000):
    """
    Simulate market with optional curated onboarding
    
    If curated_onboarding=True, high-rating LT users get boosted feed for first T_onboard periods
    """
    market = initialize_market(rho_m=rho_m, N=N)
    
    history = {
        'period': [],
        'n_high_LT_active': [],
        'n_high_LT_exits_match': [],
        'n_high_LT_exits_frustration': [],
    }
    
    for t in range(T):
        active_mask = market['active'] == True
        n_active = active_mask.sum()
        
        if n_active == 0:
            break
        
        # Track high-rating LT users
        high_LT_active = market[active_mask & market['is_high_rating_LT']]
        history['period'].append(t)
        history['n_high_LT_active'].append(len(high_LT_active))
        
        exits_match_high_LT = 0
        exits_frust_high_LT = 0
        
        for idx in market[active_mask].index:
            user = market.loc[idx]
            
            # Determine if this user gets curated feed
            in_onboarding = (t - user['entry_time']) < T_onboard
            gets_curated = curated_onboarding and in_onboarding and user['is_high_rating_LT']
            
            # Calculate match probability
            p_match = calculate_match_probability(user, market, curated=gets_curated)
            
            # Check frustration threshold
            threshold = p_bar_L if user['goal'] == 'LT' else p_bar_S
            
            if p_match < threshold:
                # Frustration exit
                market.loc[idx, 'active'] = False
                market.loc[idx, 'exit_time'] = t
                market.loc[idx, 'exit_type'] = 'frustration'
                
                if user['is_high_rating_LT']:
                    exits_frust_high_LT += 1
                
            else:
                # Attempt to match
                if np.random.random() < p_match:
                    market.loc[idx, 'active'] = False
                    market.loc[idx, 'exit_time'] = t
                    market.loc[idx, 'exit_type'] = 'match'
                    
                    if user['is_high_rating_LT']:
                        exits_match_high_LT += 1
        
        history['n_high_LT_exits_match'].append(exits_match_high_LT)
        history['n_high_LT_exits_frustration'].append(exits_frust_high_LT)
    
    return market, history

# ============================================================================
# RUN SIMULATIONS FOR H4
# ============================================================================

print("Running simulations for H4 (Onboarding Design)...")
print(f"Onboarding period: {T_onboard} periods")
print(f"LT share boost for high-rating LT users: +{rho_boost*100:.0f}%\n")

np.random.seed(42)

print("  - Baseline (random feed)...")
market_baseline, hist_baseline = simulate_market_with_onboarding(
    rho_m=0.25, curated_onboarding=False, T=50, N=1000
)

print("  - Curated onboarding...")
market_curated, hist_curated = simulate_market_with_onboarding(
    rho_m=0.25, curated_onboarding=True, T=50, N=1000
)

print("✓ Simulations complete\n")

# Summary stats for high-rating LT users
high_LT_baseline = market_baseline[market_baseline['is_high_rating_LT']]
high_LT_curated = market_curated[market_curated['is_high_rating_LT']]

print("High-rating LT users (Baseline):")
print(f"  Total: {len(high_LT_baseline)}")
print(f"  Match exits: {(high_LT_baseline['exit_type'] == 'match').sum()}")
print(f"  Frustration exits: {(high_LT_baseline['exit_type'] == 'frustration').sum()}")
if (high_LT_baseline['exit_type'].notna()).sum() > 0:
    print(f"  Mean exit time: {high_LT_baseline['exit_time'].mean():.2f}")

print("\nHigh-rating LT users (Curated onboarding):")
print(f"  Total: {len(high_LT_curated)}")
print(f"  Match exits: {(high_LT_curated['exit_type'] == 'match').sum()}")
print(f"  Frustration exits: {(high_LT_curated['exit_type'] == 'frustration').sum()}")
if (high_LT_curated['exit_type'].notna()).sum() > 0:
    print(f"  Mean exit time: {high_LT_curated['exit_time'].mean():.2f}")

# ============================================================================
# FIGURE: Survival Curves for High-Rating LT Users
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Baseline survival
exit_times_base = high_LT_baseline['exit_time'].dropna().values
if len(exit_times_base) > 0:
    max_period = int(max(exit_times_base.max(), 30))
    periods = np.arange(0, max_period + 1)
    survival_base = [np.mean(exit_times_base >= t) for t in periods]
    ax.plot(periods, survival_base, label='Baseline (Random Feed)', 
           color='#d62728', linewidth=3.5, linestyle='-')

# Curated survival
exit_times_cur = high_LT_curated['exit_time'].dropna().values
if len(exit_times_cur) > 0:
    max_period = int(max(exit_times_cur.max(), 30))
    periods = np.arange(0, max_period + 1)
    survival_cur = [np.mean(exit_times_cur >= t) for t in periods]
    ax.plot(periods, survival_cur, label='Curated Onboarding', 
           color='#2ca02c', linewidth=3.5, linestyle='-')

# Mark onboarding period
ax.axvspan(0, T_onboard, alpha=0.2, color='green', label='Onboarding Period')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Proportion Still Active', fontsize=14)
ax.set_title('H4: Curated Onboarding Increases High-Rating LT User Tenure', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H4_onboarding_design.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: 'H4_onboarding_design.png'")
plt.show()