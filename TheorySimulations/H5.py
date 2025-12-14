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

# ST acceptance (constant across all conditions)
theta_S_minus, theta_S_plus = 0.60, 1.40  

p_bar_L = 0.010  
p_bar_S = 0.006  

# H5: Test different LT selectivity levels
selectivity_levels = {
    'Low selectivity': (0.85, 1.15),    # Wide window (±15%)
    'Medium selectivity': (0.90, 1.10),  # Baseline (±10%)
    'High selectivity': (0.95, 1.05)     # Narrow window (±5%)
}

# Focus on LT-poor and Balanced markets to show effect
markets = {
    'LT-poor': 0.25,
    'Balanced': 0.55
}

n_users = 500
n_periods = 40
np.random.seed(42)

# ============================================================================
# SIMULATION WITH VARIABLE SELECTIVITY
# ============================================================================

def simulate_with_selectivity(rho_m, theta_L_minus, theta_L_plus):
    """Simulate market with specified LT selectivity"""
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None
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
            
            # Compute success probability with current selectivity
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
            p_success = min(K * beta(r_i) * A, 1.0)
            
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

print("Running H5 simulations (Selectivity Effects)...\n")

all_results = {}

for market_name, rho_m in markets.items():
    for selectivity_name, (theta_minus, theta_plus) in selectivity_levels.items():
        key = f"{market_name}_{selectivity_name}"
        
        print(f"Simulating {market_name} with {selectivity_name} [{theta_minus:.2f}, {theta_plus:.2f}]...")
        
        results_df, users_df = simulate_with_selectivity(rho_m, theta_minus, theta_plus)
        all_results[key] = {
            'results': results_df,
            'users': users_df,
            'market': market_name,
            'selectivity': selectivity_name,
            'theta_minus': theta_minus,
            'theta_plus': theta_plus
        }

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("H5: SELECTIVITY EFFECTS ANALYSIS")
print("="*80)

for market_name in markets.keys():
    print(f"\n{market_name} Market:")
    
    for selectivity_name in ['Low selectivity', 'Medium selectivity', 'High selectivity']:
        key = f"{market_name}_{selectivity_name}"
        
        retention = all_results[key]['results']['retention_L'].iloc[-1]
        theta_minus = all_results[key]['theta_minus']
        theta_plus = all_results[key]['theta_plus']
        
        print(f"  {selectivity_name} [{theta_minus:.2f}, {theta_plus:.2f}]: {100*retention:.1f}% final retention")
        
        # Frustration rate
        users = all_results[key]['users']
        LT_users = users[users['goal'] == 'L']
        exited_LT = LT_users[LT_users['exit_type'].notna()]
        frust_LT = exited_LT[exited_LT['exit_type'] == 'frustration']
        
        frust_rate = len(frust_LT) / len(exited_LT) if len(exited_LT) > 0 else 0
        print(f"    Frustration rate: {100*frust_rate:.0f}%")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {
    'Low selectivity': '#2ca02c',
    'Medium selectivity': '#ff7f0e', 
    'High selectivity': '#d62728'
}

linestyles = {
    'Low selectivity': '-',
    'Medium selectivity': '--',
    'High selectivity': ':'
}

for idx, (market_name, rho_m) in enumerate(markets.items()):
    ax = axes[idx]
    
    for selectivity_name in ['Low selectivity', 'Medium selectivity', 'High selectivity']:
        key = f"{market_name}_{selectivity_name}"
        results = all_results[key]['results']
        theta_minus = all_results[key]['theta_minus']
        theta_plus = all_results[key]['theta_plus']
        
        label = f'{selectivity_name} [{theta_minus:.2f}, {theta_plus:.2f}]'
        
        ax.plot(results['period'], results['retention_L'],
               label=label,
               color=colors[selectivity_name],
               linestyle=linestyles[selectivity_name],
               linewidth=2.5)
    
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('LT User Retention Rate', fontsize=12)
    ax.set_title(f'{market_name} (ρ={rho_m})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('H5_selectivity_effects.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: H5_selectivity_effects.png")
plt.show()

# ============================================================================
# H5 HYPOTHESIS TEST
# ============================================================================

print("\n" + "="*80)
print("H5 HYPOTHESIS TEST")
print("="*80)

for market_name in markets.keys():
    print(f"\n{market_name}:")
    
    low_ret = all_results[f"{market_name}_Low selectivity"]['results']['retention_L'].iloc[-1]
    med_ret = all_results[f"{market_name}_Medium selectivity"]['results']['retention_L'].iloc[-1]
    high_ret = all_results[f"{market_name}_High selectivity"]['results']['retention_L'].iloc[-1]
    
    print(f"  Low selectivity: {100*low_ret:.1f}%")
    print(f"  Medium selectivity: {100*med_ret:.1f}%")
    print(f"  High selectivity: {100*high_ret:.1f}%")
    print(f"  Effect of high vs low: {100*(low_ret - high_ret):+.1f} pp")
    
    if low_ret > high_ret + 0.05:
        print(f"  ✓ Lower selectivity improves retention (H5 supported)")
    elif low_ret > high_ret + 0.02:
        print(f"  ~ Modest effect")
    else:
        print(f"  ✗ No clear selectivity effect")

print("\n" + "="*80)
print("CONCLUSION:")

# Focus on LT-poor market effect
poor_low = all_results['LT-poor_Low selectivity']['results']['retention_L'].iloc[-1]
poor_high = all_results['LT-poor_High selectivity']['results']['retention_L'].iloc[-1]

effect = poor_low - poor_high

if effect > 0.05:
    print("✓ H5 SUPPORTED: Narrower selectivity amplifies LT wash-out")
    print(f"  In LT-poor markets, high selectivity reduces retention by {100*effect:.1f} pp")
elif effect > 0.02:
    print("~ H5 PARTIALLY SUPPORTED: Modest selectivity effects")
else:
    print("✗ H5 NOT SUPPORTED: No clear selectivity effect")

print("="*80)