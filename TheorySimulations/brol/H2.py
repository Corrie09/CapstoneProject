import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# PARAMETERS (same as H1)
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

# NEW: Belief parameters
tau_prior = 10  # Prior strength (α_0 + β_0)
optimistic_prior = 0.6  # Users expect 60% LT share

# ============================================================================
# CORE FUNCTIONS (modified for belief tracking)
# ============================================================================

def initialize_market_with_beliefs(rho_m, psi_m, N=N_users):
    """Initialize market with OPTIMISTIC belief tracking for LT users"""
    goals = np.random.choice(['LT', 'Casual'], size=N, p=[rho_m, 1-rho_m])
    ratings = np.random.normal(0.611, 0.126, N)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    for i in range(N):
        if goals[i] == 'LT':
            ratings[i] += np.random.normal(0.02, 0.01)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    # Initialize OPTIMISTIC beliefs for LT users
    # Prior mean = 0.6 (optimistic: "I expect many people like me")
    alpha_0 = tau_prior * optimistic_prior
    beta_0 = tau_prior * (1 - optimistic_prior)
    
    market = pd.DataFrame({
        'user_id': range(N),
        'goal': goals,
        'rating': ratings,
        'active': True,
        'exit_time': np.nan,
        'exit_type': None,
        # Belief parameters (only used for LT users)
        'alpha': alpha_0,
        'beta': beta_0,
        'rho_belief': optimistic_prior,  # Start optimistic
        'psi_m': psi_m  # Signal clarity
    })
    return market

def update_beliefs(user_row, current_rho_true):
    """
    Update LT user's belief about rho_m based on observed profiles
    
    Returns: new_alpha, new_beta, new_belief
    """
    if user_row['goal'] != 'LT':
        return user_row['alpha'], user_row['beta'], user_row['rho_belief']
    
    psi_m = user_row['psi_m']
    
    # Number of informative profiles this period
    K_eff = np.random.binomial(K, psi_m)
    
    if K_eff == 0:
        # No new information
        return user_row['alpha'], user_row['beta'], user_row['rho_belief']
    
    # Number that appear LT (based on true market composition)
    K_L = np.random.binomial(K_eff, current_rho_true)
    
    # Bayesian update
    new_alpha = user_row['alpha'] + K_L
    new_beta = user_row['beta'] + (K_eff - K_L)
    new_belief = new_alpha / (new_alpha + new_beta)
    
    return new_alpha, new_beta, new_belief

def calculate_acceptable_mass(user_rating, user_goal, market_df):
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
    else:
        rating_min = theta_S_minus * user_rating
        rating_max = theta_S_plus * user_rating
        acceptable = active[
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
    
    return len(acceptable) / len(active)

def calculate_match_probability_with_belief(user_row, market_df):
    """
    For LT users, use their BELIEF about rho_m to calculate perceived p
    For Casual users, use actual market composition
    """
    if user_row['goal'] == 'LT':
        # LT user uses their belief
        # Approximate: A^L ≈ rho_belief * (fraction in rating range)
        # Simplified: assume rating compatibility is ~0.2 for narrow range
        rating_compatibility = 0.2
        A_perceived = user_row['rho_belief'] * rating_compatibility
        p = K * beta(user_row['rating']) * A_perceived
    else:
        # Casual user uses actual market
        A = calculate_acceptable_mass(user_row['rating'], user_row['goal'], market_df)
        p = K * beta(user_row['rating']) * A
    
    return min(p, 1.0)

# ============================================================================
# SIMULATION WITH BELIEF TRACKING
# ============================================================================

def simulate_market_with_beliefs(rho_m, psi_m, T=50, N=1000, track_users=None):
    """
    Simulate market with belief dynamics
    
    track_users: list of user_ids to track in detail (default: 5 random LT users)
    """
    market = initialize_market_with_beliefs(rho_m=rho_m, psi_m=psi_m, N=N)
    
    # Select users to track (high-rating LT users who will be disappointed)
    if track_users is None:
        LT_users = market[market['goal'] == 'LT']
        # Track high-rating LT users who should get frustrated
        high_rating_LT = LT_users[LT_users['rating'] > LT_users['rating'].quantile(0.6)]
        if len(high_rating_LT) >= 5:
            track_users = high_rating_LT.sample(n=5)['user_id'].values
        else:
            track_users = LT_users.sample(n=min(5, len(LT_users)))['user_id'].values
    
    # Track detailed trajectories
    trajectories = {uid: {'period': [], 'belief': [], 'p_match': [], 'active': []} 
                    for uid in track_users}
    
    # Simulation loop
    for t in range(T):
        active_mask = market['active'] == True
        n_active = active_mask.sum()
        
        if n_active == 0:
            break
        
        # Current true LT share among active
        active_users = market[active_mask]
        rho_true = (active_users['goal'] == 'LT').mean()
        
        # Update beliefs and check exits
        for idx in market[active_mask].index:
            user = market.loc[idx]
            
            # Update beliefs (for LT users only)
            new_alpha, new_beta, new_belief = update_beliefs(user, rho_true)
            market.loc[idx, 'alpha'] = new_alpha
            market.loc[idx, 'beta'] = new_beta
            market.loc[idx, 'rho_belief'] = new_belief
            
            # Calculate match probability (using belief for LT, actual for Casual)
            p_match = calculate_match_probability_with_belief(
                market.loc[idx], market
            )
            
            # Track specific users
            if user['user_id'] in track_users and user['active']:
                trajectories[user['user_id']]['period'].append(t)
                trajectories[user['user_id']]['belief'].append(new_belief)
                trajectories[user['user_id']]['p_match'].append(p_match)
                trajectories[user['user_id']]['active'].append(True)
            
            # Check frustration threshold
            threshold = p_bar_L if user['goal'] == 'LT' else p_bar_S
            
            if p_match < threshold:
                # Frustration exit
                market.loc[idx, 'active'] = False
                market.loc[idx, 'exit_time'] = t
                market.loc[idx, 'exit_type'] = 'frustration'
                
                if user['user_id'] in track_users:
                    trajectories[user['user_id']]['active'].append(False)
                
            else:
                # Attempt to match
                if np.random.random() < p_match:
                    market.loc[idx, 'active'] = False
                    market.loc[idx, 'exit_time'] = t
                    market.loc[idx, 'exit_type'] = 'match'
                    
                    if user['user_id'] in track_users:
                        trajectories[user['user_id']]['active'].append(False)
    
    return market, trajectories

# ============================================================================
# RUN SIMULATIONS FOR H2
# ============================================================================

print("Running simulations for H2 (Belief-Driven Frustration with Optimistic Priors)...")
print(f"LT users start with optimistic prior: ρ̂₀ = {optimistic_prior}\n")

# Compare low vs high intent clarity in LT-poor market
np.random.seed(42)

print("  - Low clarity (ψ_m = 0.3)...")
market_low, traj_low = simulate_market_with_beliefs(rho_m=0.25, psi_m=0.3, T=50, N=1000)

print("  - High clarity (ψ_m = 0.7)...")
market_high, traj_high = simulate_market_with_beliefs(rho_m=0.25, psi_m=0.7, T=50, N=1000)

print("✓ Simulations complete\n")

# Summary stats
LT_exits_low = market_low[(market_low['goal'] == 'LT') & (market_low['active'] == False)]
LT_exits_high = market_high[(market_high['goal'] == 'LT') & (market_high['active'] == False)]

print(f"Low clarity (ψ_m=0.3):")
print(f"  LT frustration exits: {(LT_exits_low['exit_type'] == 'frustration').sum()}")
print(f"  LT match exits: {(LT_exits_low['exit_type'] == 'match').sum()}")

print(f"\nHigh clarity (ψ_m=0.7):")
print(f"  LT frustration exits: {(LT_exits_high['exit_type'] == 'frustration').sum()}")
print(f"  LT match exits: {(LT_exits_high['exit_type'] == 'match').sum()}")

# ============================================================================
# FIGURE: Belief Trajectories
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('H2: Belief Dynamics Lead to Frustration Exit (LT-Poor Market, Optimistic Priors)', 
             fontsize=15, fontweight='bold')

# Plot low clarity
ax = axes[0]
for uid, traj in list(traj_low.items())[:5]:  # Show 5 users
    if len(traj['period']) > 1:
        color = 'red' if not traj['active'][-1] else 'blue'
        linestyle = '--' if not traj['active'][-1] else '-'
        ax.plot(traj['period'], traj['belief'], linewidth=2.5, 
               marker='o', markersize=4, alpha=0.7, color=color, linestyle=linestyle)
        # Mark exit point
        if not traj['active'][-1]:
            ax.scatter(traj['period'][-1], traj['belief'][-1], 
                      s=200, marker='X', color='red', zorder=5, edgecolors='black', linewidths=2)

ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='True ρ_m = 0.25')
ax.axhline(y=optimistic_prior, color='green', linestyle=':', alpha=0.6, linewidth=2, label=f'Prior ρ̂₀ = {optimistic_prior}')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Belief about LT Share (ρ̂ᵢ,ₜ)', fontsize=12)
ax.set_title('Low Intent Clarity (ψ_m = 0.3)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.15, 0.7])

# Plot high clarity
ax = axes[1]
for uid, traj in list(traj_high.items())[:5]:
    if len(traj['period']) > 1:
        color = 'red' if not traj['active'][-1] else 'blue'
        linestyle = '--' if not traj['active'][-1] else '-'
        ax.plot(traj['period'], traj['belief'], linewidth=2.5,
               marker='o', markersize=4, alpha=0.7, color=color, linestyle=linestyle)
        if not traj['active'][-1]:
            ax.scatter(traj['period'][-1], traj['belief'][-1], 
                      s=200, marker='X', color='red', zorder=5, edgecolors='black', linewidths=2)

ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.6, linewidth=2, label='True ρ_m = 0.25')
ax.axhline(y=optimistic_prior, color='green', linestyle=':', alpha=0.6, linewidth=2, label=f'Prior ρ̂₀ = {optimistic_prior}')
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Belief about LT Share (ρ̂ᵢ,ₜ)', fontsize=12)
ax.set_title('High Intent Clarity (ψ_m = 0.7)', fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0.15, 0.7])

plt.tight_layout()
plt.savefig('H2_belief_dynamics.png', dpi=300, bbox_inches='tight')
print("\n✓ Figure saved: 'H2_belief_dynamics.png'")
plt.show()