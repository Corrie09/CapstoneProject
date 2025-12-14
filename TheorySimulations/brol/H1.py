import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PARAMETERS FROM YOUR PAPER (ADJUSTED FOR REALISTIC DYNAMICS)
# ============================================================================

# Matching technology
# Matching technology
K = 30  # profiles shown per period
beta_0 = 0.01
beta_1 = 0.02

def beta(r):
    """Match probability function: β(r) = 0.01 + 0.02r"""
    return beta_0 + beta_1 * r

# Acceptance regions
theta_L_minus = 0.90  # LT lower bound
theta_L_plus = 1.10   # LT upper bound
theta_S_minus = 0.60  # Casual lower bound
theta_S_plus = 1.40   # Casual upper bound

# Frustration thresholds
p_bar_L = 0.02  # LT threshold
p_bar_S = 0.01  # Casual threshold

# Simulation parameters
T = 50  # number of periods
N_users = 1000  # users per market (adjust as needed)

# ============================================================================
# MARKET INITIALIZATION
# ============================================================================

def initialize_market(rho_m, N=N_users, rating_mean=0.611, rating_sd=0.126):
    """
    Initialize a dating market with N users.
    
    Parameters:
    -----------
    rho_m : float
        LT share in the market (fundamental)
    N : int
        Number of users
    rating_mean, rating_sd : float
        Parameters for rating distribution
        
    Returns:
    --------
    DataFrame with columns: user_id, goal, rating, active, exit_time, exit_type
    """
    
    # Assign goals: LT with probability rho_m
    goals = np.random.choice(['LT', 'Casual'], size=N, p=[rho_m, 1-rho_m])
    
    # Generate ratings from Beta distribution
    # Convert mean/sd to alpha/beta parameters for Beta distribution
    # We'll use a Beta distribution scaled to approximately match the moments
    # For simplicity, we'll use normal and clip to [0,1]
    ratings = np.random.normal(rating_mean, rating_sd, N)
    ratings = np.clip(ratings, 0.3, 0.9)  # clip to observed range
    
    # Adjust ratings slightly by goal (higher-rated users more likely LT)
    # From data: top quintile 60% LT, bottom quintile 47% LT
    # We'll add a small boost for LT users
    for i in range(N):
        if goals[i] == 'LT':
            ratings[i] += np.random.normal(0.02, 0.01)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    # Create DataFrame
    market = pd.DataFrame({
        'user_id': range(N),
        'goal': goals,
        'rating': ratings,
        'active': True,
        'exit_time': np.nan,
        'exit_type': None  # 'match' or 'frustration'
    })
    
    return market

# Test initialization
print("Testing market initialization...")
print("\n=== LT-poor market (ρ_m = 0.25) ===")
market_poor = initialize_market(rho_m=0.25)
print(f"Total users: {len(market_poor)}")
print(f"LT share: {(market_poor['goal'] == 'LT').mean():.3f}")
print(f"Mean rating: {market_poor['rating'].mean():.3f}")
print(f"LT users - mean rating: {market_poor[market_poor['goal']=='LT']['rating'].mean():.3f}")
print(f"Casual users - mean rating: {market_poor[market_poor['goal']=='Casual']['rating'].mean():.3f}")

print("\n=== Balanced market (ρ_m = 0.55) ===")
market_balanced = initialize_market(rho_m=0.55)
print(f"LT share: {(market_balanced['goal'] == 'LT').mean():.3f}")

print("\n=== LT-rich market (ρ_m = 0.85) ===")
market_rich = initialize_market(rho_m=0.85)
print(f"LT share: {(market_rich['goal'] == 'LT').mean():.3f}")

# ============================================================================
# MATCHING PROBABILITY CALCULATIONS
# ============================================================================

def calculate_acceptable_mass(user_rating, user_goal, market_df):
    """
    Calculate A^g_m,t(r_i): the mass of acceptable partners for user i
    
    Parameters:
    -----------
    user_rating : float
        Rating of the user
    user_goal : str
        'LT' or 'Casual'
    market_df : DataFrame
        Current active market
        
    Returns:
    --------
    float : proportion of acceptable partners in active pool
    """
    
    active = market_df[market_df['active'] == True].copy()
    
    if len(active) == 0:
        return 0.0
    
    if user_goal == 'LT':
        # LT users need: (1) partner is LT, (2) rating in narrow range
        rating_min = theta_L_minus * user_rating
        rating_max = theta_L_plus * user_rating
        
        acceptable = active[
            (active['goal'] == 'LT') & 
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
        
    else:  # Casual
        # Casual users need: rating in wide range (don't care about goal)
        rating_min = theta_S_minus * user_rating
        rating_max = theta_S_plus * user_rating
        
        acceptable = active[
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
    
    return len(acceptable) / len(active)


def calculate_match_probability(user_rating, user_goal, market_df):
    """
    Calculate p^g_i,t using equation (2): p ≃ K * β(r) * A^g_m,t(r)
    
    Returns:
    --------
    float : per-period match probability
    """
    
    A = calculate_acceptable_mass(user_rating, user_goal, market_df)
    p = K * beta(user_rating) * A
    
    # Cap at 1.0 (it's a probability)
    return min(p, 1.0)


# ============================================================================
# TEST MATCHING PROBABILITIES
# ============================================================================

print("\n" + "="*70)
print("TESTING MATCHING PROBABILITIES")
print("="*70)

# Test in LT-poor market
market_test = initialize_market(rho_m=0.25, N=500)

# High-rating LT user
high_LT = market_test[(market_test['goal'] == 'LT') & 
                      (market_test['rating'] > 0.75)].iloc[0]
p_high_LT = calculate_match_probability(high_LT['rating'], high_LT['goal'], market_test)

# High-rating Casual user
high_Casual = market_test[(market_test['goal'] == 'Casual') & 
                          (market_test['rating'] > 0.75)].iloc[0]
p_high_Casual = calculate_match_probability(high_Casual['rating'], high_Casual['goal'], market_test)

# Low-rating LT user
low_LT = market_test[(market_test['goal'] == 'LT') & 
                     (market_test['rating'] < 0.50)].iloc[0]
p_low_LT = calculate_match_probability(low_LT['rating'], low_LT['goal'], market_test)

print(f"\nIn LT-poor market (ρ_m = 0.25):")
print(f"High-rating LT user (r={high_LT['rating']:.3f}): p = {p_high_LT:.4f}")
print(f"High-rating Casual user (r={high_Casual['rating']:.3f}): p = {p_high_Casual:.4f}")
print(f"Low-rating LT user (r={low_LT['rating']:.3f}): p = {p_low_LT:.4f}")

print(f"\n→ High-rating LT has {p_high_LT:.4f} < threshold {p_bar_L:.2f}? {p_high_LT < p_bar_L}")
print(f"→ High-rating Casual has {p_high_Casual:.4f} < threshold {p_bar_S:.2f}? {p_high_Casual < p_bar_S}")

# Compare across market types
print("\n" + "-"*70)
print("Comparing match probabilities across market types:")
print("-"*70)

for rho, label in [(0.25, 'LT-poor'), (0.55, 'Balanced'), (0.85, 'LT-rich')]:
    market = initialize_market(rho_m=rho, N=500)
    high_LT_user = market[(market['goal'] == 'LT') & (market['rating'] > 0.75)].iloc[0]
    p = calculate_match_probability(high_LT_user['rating'], 'LT', market)
    print(f"{label:12s} (ρ={rho:.2f}): High-rating LT user p = {p:.4f}")
    
# ============================================================================
# MARKET SIMULATION
# ============================================================================

def simulate_market(rho_m, T=50, N=1000, track_metrics=True):
    """
    Simulate a dating market for T periods.
    
    Returns:
    --------
    market : DataFrame with final state
    history : dict with time series of key metrics
    """
    
    # Initialize
    market = initialize_market(rho_m=rho_m, N=N)
    
    # Track metrics over time
    history = {
        'period': [],
        'rho_t': [],  # LT share among active
        'n_active': [],
        'n_active_LT': [],
        'n_active_Casual': [],
        'mean_rating_LT': [],
        'mean_rating_Casual': [],
        'exits_match': [],
        'exits_frustration': [],
    }
    
    # Simulation loop
    for t in range(T):
        active_mask = market['active'] == True
        n_active = active_mask.sum()
        
        if n_active == 0:
            break
        
        # Track metrics
        active_users = market[active_mask]
        LT_active = active_users[active_users['goal'] == 'LT']
        Casual_active = active_users[active_users['goal'] == 'Casual']
        
        history['period'].append(t)
        history['rho_t'].append(len(LT_active) / n_active if n_active > 0 else 0)
        history['n_active'].append(n_active)
        history['n_active_LT'].append(len(LT_active))
        history['n_active_Casual'].append(len(Casual_active))
        history['mean_rating_LT'].append(LT_active['rating'].mean() if len(LT_active) > 0 else np.nan)
        history['mean_rating_Casual'].append(Casual_active['rating'].mean() if len(Casual_active) > 0 else np.nan)
        
        # Process each active user
        exits_match = 0
        exits_frustration = 0
        
        for idx in market[active_mask].index:
            user = market.loc[idx]
            
            # Calculate match probability
            p_match = calculate_match_probability(user['rating'], user['goal'], market)
            
            # Check frustration threshold
            threshold = p_bar_L if user['goal'] == 'LT' else p_bar_S
            
            if p_match < threshold:
                # Frustration exit
                market.loc[idx, 'active'] = False
                market.loc[idx, 'exit_time'] = t
                market.loc[idx, 'exit_type'] = 'frustration'
                exits_frustration += 1
                
            else:
                # Attempt to match
                if np.random.random() < p_match:
                    # Successful match exit
                    market.loc[idx, 'active'] = False
                    market.loc[idx, 'exit_time'] = t
                    market.loc[idx, 'exit_type'] = 'match'
                    exits_match += 1
        
        history['exits_match'].append(exits_match)
        history['exits_frustration'].append(exits_frustration)
    
    return market, history


# ============================================================================
# RUN SIMULATIONS FOR THREE MARKET TYPES
# ============================================================================

print("\n" + "="*70)
print("RUNNING MARKET SIMULATIONS (H1: LT Wash-out)")
print("="*70)

results = {}

for rho_m, label in [(0.25, 'LT-poor'), (0.55, 'Balanced'), (0.85, 'LT-rich')]:
    print(f"\n--- Simulating {label} market (ρ_m = {rho_m}) ---")
    market_final, history = simulate_market(rho_m=rho_m, T=50, N=1000)
    results[label] = {'market': market_final, 'history': history}
    
    # Summary statistics
    print(f"Initial active users: {history['n_active'][0]}")
    print(f"Final active users: {history['n_active'][-1]}")
    print(f"Initial LT share: {history['rho_t'][0]:.3f}")
    print(f"Final LT share: {history['rho_t'][-1]:.3f}")
    print(f"Total match exits: {sum(history['exits_match'])}")
    print(f"Total frustration exits: {sum(history['exits_frustration'])}")
    
    # Exit analysis by type
    exits = market_final[market_final['active'] == False]
    if len(exits) > 0:
        LT_exits = exits[exits['goal'] == 'LT']
        Casual_exits = exits[exits['goal'] == 'Casual']
        
        print(f"\nLT users:")
        print(f"  - Match exits: {(LT_exits['exit_type'] == 'match').sum()}")
        print(f"  - Frustration exits: {(LT_exits['exit_type'] == 'frustration').sum()}")
        if len(LT_exits) > 0:
            print(f"  - Mean exit time: {LT_exits['exit_time'].mean():.1f}")
        
        print(f"Casual users:")
        print(f"  - Match exits: {(Casual_exits['exit_type'] == 'match').sum()}")
        print(f"  - Frustration exits: {(Casual_exits['exit_type'] == 'frustration').sum()}")
        if len(Casual_exits) > 0:
            print(f"  - Mean exit time: {Casual_exits['exit_time'].mean():.1f}")

print("\n" + "="*70)
print("Simulation complete!")
print("="*70)

# ============================================================================
# CLEANER VISUALIZATIONS FOR H1
# ============================================================================

# FIGURE 1: LT Share Decline Over Time (Key H1 result)
# ============================================================================
fig1, ax = plt.subplots(1, 1, figsize=(10, 6))

for label, color in [('LT-poor', '#d62728'), ('Balanced', '#ff7f0e'), ('LT-rich', '#2ca02c')]:
    history = results[label]['history']
    initial_rho = history['rho_t'][0]
    ax.plot(history['period'], history['rho_t'], 
            label=f"{label} (ρ₀={initial_rho:.2f})", 
            color=color, linewidth=3, marker='o', markersize=5, markevery=3)
    
    # Add horizontal reference line for initial value
    ax.axhline(y=initial_rho, color=color, linestyle='--', alpha=0.3, linewidth=1)

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('LT Share Among Active Users (ρₜ)', fontsize=14)
ax.set_title('H1: LT Share Declines Over Time (Market Wash-out)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H1_LT_share_decline.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure 1 saved: 'H1_LT_share_decline.png'")
plt.show()


# FIGURE 2: Survival Curves - LT vs Casual Users
# ============================================================================
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('H1: Survival Curves by User Type Across Markets', 
              fontsize=16, fontweight='bold')

for idx, (label, title) in enumerate([('LT-poor', 'LT-Poor Market (ρ₀=0.25)'), 
                                       ('Balanced', 'Balanced Market (ρ₀=0.55)'), 
                                       ('LT-rich', 'LT-Rich Market (ρ₀=0.85)')]):
    ax = axes[idx]
    market = results[label]['market']
    
    # LT vs Casual survival
    for goal, color, linestyle in [('LT', '#1f77b4', '-'), ('Casual', '#d62728', '--')]:
        exits = market[market['goal'] == goal]
        exit_times = exits['exit_time'].dropna().values
        
        if len(exit_times) > 0:
            max_period = int(exit_times.max()) + 1
            periods = np.arange(0, max_period + 1)
            survival = [np.mean(exit_times >= t) for t in periods]
            ax.plot(periods, survival, label=f"{goal} users", 
                   color=color, linewidth=3, linestyle=linestyle)
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Proportion Still Active', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H1_survival_curves.png', dpi=300, bbox_inches='tight')
print("✓ Figure 2 saved: 'H1_survival_curves.png'")
plt.show()


# FIGURE 3: High-rating vs Low-rating LT Users (LT-poor market only)
# ============================================================================
fig3, ax = plt.subplots(1, 1, figsize=(10, 6))

market_poor = results['LT-poor']['market']
LT_users = market_poor[market_poor['goal'] == 'LT']

# Define rating quintiles
q75 = LT_users['rating'].quantile(0.75)
q25 = LT_users['rating'].quantile(0.25)

for threshold, label, color, style in [(q75, f'High-rating LT (r ≥ {q75:.2f})', '#1f77b4', '-'),
                                        (q25, f'Low-rating LT (r ≤ {q25:.2f})', '#ff7f0e', '--')]:
    if 'High' in label:
        subset = LT_users[LT_users['rating'] >= threshold]
    else:
        subset = LT_users[LT_users['rating'] <= threshold]
    
    exit_times = subset['exit_time'].dropna().values
    
    if len(exit_times) > 0:
        max_period = int(exit_times.max()) + 1
        periods = np.arange(0, max_period + 1)
        survival = [np.mean(exit_times >= t) for t in periods]
        ax.plot(periods, survival, label=label, 
               color=color, linewidth=3, linestyle=style)

# Add Casual users for comparison
casual = market_poor[market_poor['goal'] == 'Casual']
exit_times_casual = casual['exit_time'].dropna().values
if len(exit_times_casual) > 0:
    max_period = int(exit_times_casual.max()) + 1
    periods = np.arange(0, max_period + 1)
    survival = [np.mean(exit_times_casual >= t) for t in periods]
    ax.plot(periods, survival, label='Casual users (all ratings)', 
           color='#d62728', linewidth=3, linestyle=':')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Proportion Still Active', fontsize=14)
ax.set_title('H1: High-Rating LT Users Exit Faster (LT-Poor Market)', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H1_rating_effects.png', dpi=300, bbox_inches='tight')
print("✓ Figure 3 saved: 'H1_rating_effects.png'")
plt.show()

print("\n" + "="*70)
print("All H1 figures generated successfully!")
print("="*70)