import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================================
# PARAMETERS
# ============================================================================

K = 30  

def beta(r):
    return 0.002 + 0.006 * r

theta_L_minus, theta_L_plus = 0.90, 1.10  
theta_S_minus, theta_S_plus = 0.60, 1.40  

p_bar_L = 0.010  
p_bar_S = 0.006  

markets = {
    'LT-poor': 0.25,
    'Balanced': 0.55,
    'LT-rich': 0.85
}

n_users = 500
n_periods = 40
ONBOARDING_PERIODS = 10
ONBOARDING_BOOST = 3.0  # 3x multiplier on success probability during onboarding

np.random.seed(42)

# ============================================================================
# SIMULATION WITH ONBOARDING BOOST
# ============================================================================

def simulate_with_onboarding(rho_m, curated=False):
    """
    Curated onboarding = temporarily boost success probabilities
    to simulate a better initial experience
    """
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None,
        'entry_period': 0
    })
    
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
        
        # Process exits
        for idx in active.index:
            r_i = users.loc[idx, 'rating']
            goal = users.loc[idx, 'goal']
            periods_active = t - users.loc[idx, 'entry_period']
            
            # Check if in onboarding period
            in_onboarding = periods_active < ONBOARDING_PERIODS
            
            # Compute success probability (STANDARD way)
            if goal == 'L':
                acceptable = active[
                    (active['goal'] == 'L') &
                    (active['rating'] >= theta_L_minus * r_i) &
                    (active['rating'] <= theta_L_plus * r_i)
                ]
            else:
                acceptable = active[
                    (active['rating'] >= theta_S_minus * r_i) &
                    (active['rating'] <= theta_S_plus * r_i)
                ]
            
            A = len(acceptable) / len(active) if len(active) > 0 else 0
            p_success = K * beta(r_i) * A
            
            # CURATED ONBOARDING = BOOST SUCCESS PROBABILITY
            if curated and goal == 'L' and in_onboarding:
                p_success = p_success * ONBOARDING_BOOST
            
            p_success = min(p_success, 1.0)
            
            # Check exit
            p_bar = p_bar_L if goal == 'L' else p_bar_S
            
            if p_success < p_bar:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'frustration'
            elif np.random.random() < p_success:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'match'
    
    return pd.DataFrame(results), users

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

print("Running H4 simulations (Onboarding Design)...\n")

all_results = {}

for market_name, rho_m in markets.items():
    for curated in [False, True]:
        condition = 'Curated' if curated else 'Random'
        key = f"{market_name}_{condition}"
        
        print(f"Simulating {market_name} with {condition} onboarding...")
        
        results_df, users_df = simulate_with_onboarding(rho_m, curated=curated)
        all_results[key] = {
            'results': results_df,
            'users': users_df,
            'market': market_name,
            'condition': condition
        }

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("H4: ONBOARDING DESIGN ANALYSIS")
print("="*80)

for market_name in markets.keys():
    print(f"\n{market_name} Market:")
    
    random_ret = all_results[f"{market_name}_Random"]['results']['retention_L'].iloc[-1]
    curated_ret = all_results[f"{market_name}_Curated"]['results']['retention_L'].iloc[-1]
    
    print(f"  Random onboarding: {100*random_ret:.1f}% final retention")
    print(f"  Curated onboarding: {100*curated_ret:.1f}% final retention")
    print(f"  Improvement: {100*(curated_ret - random_ret):+.1f} percentage points")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

colors = {'Random': '#d62728', 'Curated': '#2ca02c'}
linestyles = {'Random': '--', 'Curated': '-'}

for idx, (market_name, rho_m) in enumerate(markets.items()):
    ax = axes[idx]
    
    for condition in ['Random', 'Curated']:
        key = f"{market_name}_{condition}"
        results = all_results[key]['results']
        
        ax.plot(results['period'], results['retention_L'],
               label=f'{condition} onboarding',
               color=colors[condition],
               linestyle=linestyles[condition],
               linewidth=2.5)
    
    ax.axvspan(0, ONBOARDING_PERIODS, alpha=0.1, color='gray', label='Onboarding period')
    
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('LT User Retention Rate', fontsize=11)
    ax.set_title(f'{market_name} (ρ={rho_m})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H4_onboarding_design.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: H4_onboarding_design.png")
plt.show()

# ============================================================================
# H4 HYPOTHESIS TEST
# ============================================================================

print("\n" + "="*80)
print("H4 HYPOTHESIS TEST")
print("="*80)

poor_improvement = all_results['LT-poor_Curated']['results']['retention_L'].iloc[-1] - \
                   all_results['LT-poor_Random']['results']['retention_L'].iloc[-1]

balanced_improvement = all_results['Balanced_Curated']['results']['retention_L'].iloc[-1] - \
                       all_results['Balanced_Random']['results']['retention_L'].iloc[-1]

rich_improvement = all_results['LT-rich_Curated']['results']['retention_L'].iloc[-1] - \
                   all_results['LT-rich_Random']['results']['retention_L'].iloc[-1]

print(f"\nLT-poor: {100*poor_improvement:+.1f} pp")
print(f"Balanced: {100*balanced_improvement:+.1f} pp")
print(f"LT-rich: {100*rich_improvement:+.1f} pp")

if poor_improvement > 0.05:
    print("\n✓ H4 SUPPORTED: Curated onboarding significantly improves retention")
elif poor_improvement > 0.02:
    print("\n~ H4 PARTIALLY SUPPORTED: Modest improvement from curated onboarding")
else:
    print("\n✗ H4 NOT SUPPORTED: No clear benefit from curated onboarding")

print("="*80)