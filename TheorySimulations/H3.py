import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================================
# PARAMETERS (MATCHING COLLEAGUE'S APPROACH)
# ============================================================================

K = 25  # Colleague uses 25

def sigmoid(x):
    """Numerically stable sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))

def beta(r, base, slope):
    """Matchability function - colleague's approach"""
    return np.clip(base + slope * r, 0.0, 0.95)

# Acceptance regions (colleague's values)
theta_L_minus, theta_L_plus = 0.93, 1.05  
theta_S_minus, theta_S_plus = 0.25, 1.20  

# Beta parameters (colleague's values - MUCH higher)
beta_L_base, beta_L_slope = 0.02, 0.05  
beta_S_base, beta_S_slope = 0.01, 0.02  

# Frustration thresholds (colleague's values - MUCH higher)
p_bar_L = 0.07  # was 0.010
p_bar_S = 0.01  # was 0.006

# Sigmoid steepness (colleague's values)
k_L = 60.0  
k_S = 40.0  

# Exogenous churn
churn_L = 0.002
churn_S = 0.002

# Belief parameters (colleague uses VERY optimistic prior)
prior_mean = 0.80  # was 0.50 - much more optimistic!
prior_strength = 6.0  # was 10 - less sticky

# Market types
markets = {
    'LT-poor': 0.20,    # Colleague uses 0.20
    'Balanced': 0.55,
    'LT-rich': 0.85
}

clarity_levels = {'Low': 0.3, 'High': 0.7}

n_users = 500
n_periods = 60  # Colleague uses 60
np.random.seed(42)

# ============================================================================
# SIMULATION WITH SIGMOID EXIT (COLLEAGUE'S APPROACH)
# ============================================================================

def simulate_with_clarity_sigmoid(rho_m, psi_m, prior_mean, prior_strength):
    """Simulate with sigmoid exit probability (smooth, not hard threshold)"""
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None
    })
    
    # Initialize beliefs
    alpha_0 = prior_mean * prior_strength
    beta_0 = (1 - prior_mean) * prior_strength
    
    users['alpha'] = alpha_0
    users['beta_param'] = beta_0
    users['belief_rho'] = prior_mean
    
    initial_L = len(users[users['goal'] == 'L'])
    results = []
    
    for t in range(n_periods):
        active = users[users['active']]
        
        if len(active) < 10:
            break
        
        active_L = active[active['goal'] == 'L']
        active_S = active[active['goal'] == 'S']
        
        retention_L = len(active_L) / initial_L if initial_L > 0 else 0
        
        results.append({
            'period': t,
            'retention_L': retention_L
        })
        
        # Shuffle to avoid ordering artifacts
        active_indices = active.index.tolist()
        np.random.shuffle(active_indices)
        
        for idx in active_indices:
            if not users.loc[idx, 'active']:
                continue
                
            r_i = users.loc[idx, 'rating']
            goal = users.loc[idx, 'goal']
            
            if goal == 'L':
                # BELIEF UPDATING
                K_eff = np.random.binomial(K, psi_m)
                
                if K_eff > 0:
                    true_rho = len(active_L) / len(active)
                    K_L = np.random.binomial(K_eff, true_rho)
                    
                    users.loc[idx, 'alpha'] += K_L
                    users.loc[idx, 'beta_param'] += (K_eff - K_L)
                    
                    users.loc[idx, 'belief_rho'] = users.loc[idx, 'alpha'] / (
                        users.loc[idx, 'alpha'] + users.loc[idx, 'beta_param']
                    )
                
                # COMPUTE EXPECTED SUCCESS PROBABILITY
                belief_rho = users.loc[idx, 'belief_rho']
                
                # Fraction with acceptable rating
                rating_match_prob = np.mean(
                    (active['rating'].values >= theta_L_minus * r_i) & 
                    (active['rating'].values <= theta_L_plus * r_i)
                )
                
                # Expected acceptable fraction = belief_rho * rating_match_prob
                A_hat = belief_rho * rating_match_prob
                
                beta_i = beta(r_i, beta_L_base, beta_L_slope)
                p_hat = np.clip(K * beta_i * A_hat, 0.0, 1.0)
                
                # SIGMOID EXIT PROBABILITY (KEY DIFFERENCE!)
                exit_prob_frust = sigmoid(k_L * (p_bar_L - p_hat))
                exit_prob = 1.0 - (1.0 - exit_prob_frust) * (1.0 - churn_L)
                
            else:  # ST user
                # ST don't update beliefs, use actual pool
                rating_match_prob = np.mean(
                    (active['rating'].values >= theta_S_minus * r_i) & 
                    (active['rating'].values <= theta_S_plus * r_i)
                )
                
                A_hat = rating_match_prob
                
                beta_i = beta(r_i, beta_S_base, beta_S_slope)
                p_hat = np.clip(K * beta_i * A_hat, 0.0, 1.0)
                
                exit_prob_frust = sigmoid(k_S * (p_bar_S - p_hat))
                exit_prob = 1.0 - (1.0 - exit_prob_frust) * (1.0 - churn_S)
            
            # PROBABILISTIC EXIT
            if np.random.random() < exit_prob:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'frustration'
    
    return pd.DataFrame(results), users

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

print("Running H3 simulations (colleague's approach)...\n")

all_results = {}

for market_name, rho_m in markets.items():
    for clarity_name, psi_m in clarity_levels.items():
        key = f"{market_name}_{clarity_name}"
        print(f"Simulating {market_name} with {clarity_name} clarity (ψ={psi_m})...")
        
        results_df, users_df = simulate_with_clarity_sigmoid(rho_m, psi_m, prior_mean, prior_strength)
        all_results[key] = {
            'results': results_df,
            'users': users_df,
            'market': market_name,
            'clarity': clarity_name,
            'rho_m': rho_m,
            'psi_m': psi_m
        }

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("H3: INTENT CLARITY TRADE-OFF ANALYSIS")
print("="*80)

for market_name in markets.keys():
    print(f"\n{market_name} Market:")
    
    low_key = f"{market_name}_Low"
    high_key = f"{market_name}_High"
    
    low_retention = all_results[low_key]['results']['retention_L'].iloc[-1]
    high_retention = all_results[high_key]['results']['retention_L'].iloc[-1]
    
    print(f"  Low clarity (ψ=0.3): {100*low_retention:.1f}% final retention")
    print(f"  High clarity (ψ=0.7): {100*high_retention:.1f}% final retention")
    print(f"  Difference: {100*(high_retention - low_retention):+.1f} percentage points")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

colors = {'Low': '#1f77b4', 'High': '#ff7f0e'}
linestyles = {'Low': '--', 'High': '-'}

for idx, (market_name, rho_m) in enumerate(markets.items()):
    ax = axes[idx]
    
    for clarity_name in ['Low', 'High']:
        key = f"{market_name}_{clarity_name}"
        results = all_results[key]['results']
        psi = clarity_levels[clarity_name]
        
        ax.plot(results['period'], results['retention_L'],
               label=f'{clarity_name} clarity (ψ={psi})',
               color=colors[clarity_name],
               linestyle=linestyles[clarity_name],
               linewidth=2.5)
    
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('LT User Retention Rate', fontsize=11)
    ax.set_title(f'{market_name} (ρ={rho_m})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H3_clarity_tradeoff.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: H3_clarity_tradeoff.png")
plt.show()

# ============================================================================
# H3 HYPOTHESIS TEST
# ============================================================================

print("\n" + "="*80)
print("H3 HYPOTHESIS TEST")
print("="*80)

poor_effect = all_results['LT-poor_High']['results']['retention_L'].iloc[-1] - \
              all_results['LT-poor_Low']['results']['retention_L'].iloc[-1]
              
rich_effect = all_results['LT-rich_High']['results']['retention_L'].iloc[-1] - \
              all_results['LT-rich_Low']['results']['retention_L'].iloc[-1]

print(f"\nLT-poor: {100*poor_effect:+.1f} pp effect")
print(f"LT-rich: {100*rich_effect:+.1f} pp effect")

if poor_effect < -0.02 and rich_effect > 0.02:
    print("\n✓ H3 SUPPORTED: Non-monotone clarity effect")
    print("  High clarity hurts in LT-poor, helps in LT-rich")
elif abs(poor_effect) > 0.02 or abs(rich_effect) > 0.02:
    print("\n~ H3 PARTIALLY SUPPORTED: Clarity has differential effects")
else:
    print("\n✗ H3 NOT SUPPORTED")

print("="*80)