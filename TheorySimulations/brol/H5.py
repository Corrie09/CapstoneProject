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

# Casual acceptance (constant)
theta_S_minus = 0.60
theta_S_plus = 1.40

# Frustration thresholds
p_bar_L = 0.02
p_bar_S = 0.01

T = 50
N_users = 1000

# ============================================================================
# CORE FUNCTIONS (with parameterized selectivity)
# ============================================================================

def initialize_market(rho_m, N=N_users):
    goals = np.random.choice(['LT', 'Casual'], size=N, p=[rho_m, 1-rho_m])
    ratings = np.random.normal(0.611, 0.126, N)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    for i in range(N):
        if goals[i] == 'LT':
            ratings[i] += np.random.normal(0.02, 0.01)
    ratings = np.clip(ratings, 0.3, 0.9)
    
    market = pd.DataFrame({
        'user_id': range(N),
        'goal': goals,
        'rating': ratings,
        'active': True,
        'exit_time': np.nan,
        'exit_type': None
    })
    return market

def calculate_acceptable_mass(user_rating, user_goal, market_df, theta_L_minus, theta_L_plus):
    """Calculate acceptable mass with parameterized LT selectivity"""
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
    else:  # Casual
        rating_min = theta_S_minus * user_rating
        rating_max = theta_S_plus * user_rating
        
        acceptable = active[
            (active['rating'] >= rating_min) & 
            (active['rating'] <= rating_max)
        ]
    
    return len(acceptable) / len(active)

def calculate_match_probability(user_rating, user_goal, market_df, theta_L_minus, theta_L_plus):
    """Calculate match probability with parameterized selectivity"""
    A = calculate_acceptable_mass(user_rating, user_goal, market_df, theta_L_minus, theta_L_plus)
    p = K * beta(user_rating) * A
    return min(p, 1.0)

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_market_selectivity(rho_m, theta_L_minus, theta_L_plus, T=50, N=1000):
    """
    Simulate market with specified LT selectivity parameters
    """
    market = initialize_market(rho_m=rho_m, N=N)
    
    for t in range(T):
        active_mask = market['active'] == True
        n_active = active_mask.sum()
        
        if n_active == 0:
            break
        
        for idx in market[active_mask].index:
            user = market.loc[idx]
            
            # Calculate match probability with current selectivity
            p_match = calculate_match_probability(
                user['rating'], user['goal'], market, theta_L_minus, theta_L_plus
            )
            
            # Check frustration threshold
            threshold = p_bar_L if user['goal'] == 'LT' else p_bar_S
            
            if p_match < threshold:
                # Frustration exit
                market.loc[idx, 'active'] = False
                market.loc[idx, 'exit_time'] = t
                market.loc[idx, 'exit_type'] = 'frustration'
            else:
                # Attempt to match
                if np.random.random() < p_match:
                    market.loc[idx, 'active'] = False
                    market.loc[idx, 'exit_time'] = t
                    market.loc[idx, 'exit_type'] = 'match'
    
    return market

# ============================================================================
# RUN SIMULATIONS FOR H3
# ============================================================================

print("Running simulations for H3 (Selectivity Effects)...")
print("Testing three LT selectivity levels in LT-poor market (ρ_m = 0.25)\n")

selectivity_configs = [
    (0.85, 1.15, 'Wide (±15%)', '#2ca02c'),
    (0.90, 1.10, 'Baseline (±10%)', '#ff7f0e'),
    (0.95, 1.05, 'Narrow (±5%)', '#d62728'),
]

results = {}

for theta_minus, theta_plus, label, color in selectivity_configs:
    print(f"  - {label}: θ_L ∈ [{theta_minus:.2f}, {theta_plus:.2f}]")
    
    np.random.seed(42)  # Same initial market for fair comparison
    market = simulate_market_selectivity(
        rho_m=0.25, 
        theta_L_minus=theta_minus, 
        theta_L_plus=theta_plus,
        T=50, 
        N=1000
    )
    
    results[label] = {'market': market, 'color': color}
    
    # Summary stats for LT users
    LT_users = market[market['goal'] == 'LT']
    LT_exits = LT_users[LT_users['active'] == False]
    
    print(f"    LT users - Match exits: {(LT_exits['exit_type'] == 'match').sum()}")
    print(f"    LT users - Frustration exits: {(LT_exits['exit_type'] == 'frustration').sum()}")
    if len(LT_exits) > 0:
        print(f"    LT users - Mean exit time: {LT_exits['exit_time'].mean():.2f}")
    print()

print("✓ Simulations complete\n")

# ============================================================================
# FIGURE 1: LT User Survival by Selectivity
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for label, data in results.items():
    market = data['market']
    color = data['color']
    
    # LT user survival
    LT_users = market[market['goal'] == 'LT']
    exit_times = LT_users['exit_time'].dropna().values
    
    if len(exit_times) > 0:
        max_period = int(exit_times.max()) + 1
        periods = np.arange(0, max_period + 1)
        survival = [np.mean(exit_times >= t) for t in periods]
        ax.plot(periods, survival, label=label, 
               color=color, linewidth=3.5, linestyle='-')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Proportion of LT Users Still Active', fontsize=14)
ax.set_title('H3: More Selective LT Users Exit Faster', 
            fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H3_selectivity_effects.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: 'H3_selectivity_effects.png'")
plt.show()

# ============================================================================
# FIGURE 2: Frustration vs Match Exits
# ============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

labels = []
match_exits = []
frust_exits = []
colors = []

for label, data in results.items():
    market = data['market']
    LT_exits = market[(market['goal'] == 'LT') & (market['active'] == False)]
    
    labels.append(label)
    match_exits.append((LT_exits['exit_type'] == 'match').sum())
    frust_exits.append((LT_exits['exit_type'] == 'frustration').sum())
    colors.append(data['color'])

x = np.arange(len(labels))
width = 0.35

ax2.bar(x - width/2, match_exits, width, label='Match Exits', color='#2ca02c', alpha=0.8)
ax2.bar(x + width/2, frust_exits, width, label='Frustration Exits', color='#d62728', alpha=0.8)

ax2.set_xlabel('LT Selectivity', fontsize=14)
ax2.set_ylabel('Number of LT User Exits', fontsize=14)
ax2.set_title('H3: Narrow Selectivity Increases Frustration Exits', 
             fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=12)
ax2.legend(fontsize=12, framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('H3_exit_types.png', dpi=300, bbox_inches='tight')
print("✓ Figure saved: 'H3_exit_types.png'")
plt.show()