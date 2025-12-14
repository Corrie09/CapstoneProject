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

# LT-poor market only (where clarity matters most)
rho_m = 0.25

n_users = 500
n_periods = 40

# Prior parameters (from H2)
prior_rho = 0.50  # Optimistic prior
prior_strength = 10  # tau_i

np.random.seed(42)

# ============================================================================
# SIMULATION FUNCTION WITH BELIEF UPDATING
# ============================================================================

def simulate_with_clarity(rho_m, psi_m):
    """
    Simulate with varying intent clarity levels
    psi_m: probability that a profile reveals its goal clearly
    """
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None,
        'entry_period': 0,
        # Belief parameters (only for LT users, but initialize for all)
        'alpha': prior_strength * prior_rho,
        'beta_param': prior_strength * (1 - prior_rho),
        'belief_rho': prior_rho
    })
    
    initial_L = len(users[users['goal'] == 'L'])
    results = []
    
    for t in range(n_periods):
        active = users[users['active']]
        
        if len(active) < 10:
            break
        
        active_L = active[active['goal'] == 'L']
        retention_L = len(active_L) / initial_L if initial_L > 0 else 0
        
        # Average belief among active LT users
        avg_belief = active_L['belief_rho'].mean() if len(active_L) > 0 else np.nan
        
        results.append({
            'period': t,
            'retention_L': retention_L,
            'avg_belief': avg_belief
        })
        
        # Process each active user
        for idx in active.index:
            r_i = users.loc[idx, 'rating']
            goal = users.loc[idx, 'goal']
            
            # === BELIEF UPDATING (only for LT users) ===
            if goal == 'L':
                # Number of profiles that reveal goal
                K_eff = np.random.binomial(K, psi_m)
                
                # Among revealed profiles, count LT profiles
                # Sample from true active pool composition
                current_rho_active = len(active[active['goal'] == 'L']) / len(active)
                K_L = np.random.binomial(K_eff, current_rho_active)
                
                # Bayesian update
                users.loc[idx, 'alpha'] += K_L
                users.loc[idx, 'beta_param'] += (K_eff - K_L)
                users.loc[idx, 'belief_rho'] = (
                    users.loc[idx, 'alpha'] / 
                    (users.loc[idx, 'alpha'] + users.loc[idx, 'beta_param'])
                )
            
            # === COMPUTE SUCCESS PROBABILITY ===
            if goal == 'L':
                # Use BELIEF for acceptable pool calculation
                belief_rho = users.loc[idx, 'belief_rho']
                
                # Fraction of active users who are LT
                frac_LT = len(active[active['goal'] == 'L']) / len(active)
                
                # Fraction in acceptable rating range
                acceptable_ratings = active[
                    (active['rating'] >= theta_L_minus * r_i) &
                    (active['rating'] <= theta_L_plus * r_i)
                ]
                frac_rating = len(acceptable_ratings) / len(active)
                
                # Expected acceptable fraction based on BELIEF
                # (user thinks belief_rho fraction are LT, and knows frac_rating have right rating)
                A_expected = belief_rho * frac_rating
                
                p_success = K * beta(r_i) * A_expected
                
            else:  # ST user
                acceptable = active[
                    (active['rating'] >= theta_S_minus * r_i) &
                    (active['rating'] <= theta_S_plus * r_i)
                ]
                A = len(acceptable) / len(active)
                p_success = K * beta(r_i) * A
            
            p_success = min(p_success, 1.0)
            
            # === EXIT DECISIONS ===
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
# RUN SIMULATIONS FOR THREE CLARITY LEVELS
# ============================================================================

print("Running Intent Clarity Counterfactual...\n")

clarity_levels = {
    'No clarity (ψ=0)': 0.0,
    'Medium clarity (ψ=0.7, baseline)': 0.7,
    'Full clarity (ψ=1.0)': 1.0
}

all_results = {}

for name, psi in clarity_levels.items():
    print(f"Simulating: {name}")
    results_df, users_df = simulate_with_clarity(rho_m=rho_m, psi_m=psi)
    all_results[name] = {
        'results': results_df,
        'users': users_df
    }

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("INTENT CLARITY COUNTERFACTUAL (LT-poor market, ρ=0.25)")
print("="*80)

for name in clarity_levels.keys():
    retention = all_results[name]['results']['retention_L'].iloc[-1]
    
    users = all_results[name]['users']
    LT_users = users[users['goal'] == 'L']
    exited_LT = LT_users[LT_users['exit_type'].notna()]
    frust_LT = exited_LT[exited_LT['exit_type'] == 'frustration']
    
    frust_rate = len(frust_LT) / len(exited_LT) if len(exited_LT) > 0 else 0
    
    # Average tenure of frustrated exits
    avg_tenure = frust_LT['exit_period'].mean() if len(frust_LT) > 0 else np.nan
    
    print(f"\n{name}:")
    print(f"  Final retention: {100*retention:.1f}%")
    print(f"  Frustration rate: {100*frust_rate:.0f}%")
    print(f"  Avg tenure (frustrated): {avg_tenure:.1f} periods")

# ============================================================================
# VISUALIZATION - TWO PANELS
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {
    'No clarity (ψ=0)': '#2ca02c',
    'Medium clarity (ψ=0.7, baseline)': '#ff7f0e',
    'Full clarity (ψ=1.0)': '#d62728'
}

linestyles = {
    'No clarity (ψ=0)': '-',
    'Medium clarity (ψ=0.7, baseline)': '--',
    'Full clarity (ψ=1.0)': ':'
}

linewidths = {
    'No clarity (ψ=0)': 2.5,
    'Medium clarity (ψ=0.7, baseline)': 2.0,
    'Full clarity (ψ=1.0)': 2.5
}

# Panel A: Retention
for name in clarity_levels.keys():
    results = all_results[name]['results']
    
    ax1.plot(results['period'], results['retention_L'],
           label=name,
           color=colors[name],
           linestyle=linestyles[name],
           linewidth=linewidths[name])

ax1.set_xlabel('Period', fontsize=11)
ax1.set_ylabel('LT User Retention Rate', fontsize=11)
ax1.set_title('(a) Retention by Intent Clarity', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Panel B: Belief Evolution
for name in clarity_levels.keys():
    results = all_results[name]['results']
    
    # Only plot where we have belief data
    valid = results[results['avg_belief'].notna()]
    
    ax2.plot(valid['period'], valid['avg_belief'],
           label=name,
           color=colors[name],
           linestyle=linestyles[name],
           linewidth=linewidths[name])

# Add true rho line
ax2.axhline(y=rho_m, color='black', linestyle='-.', linewidth=1.5, 
           label=f'True ρ = {rho_m}', alpha=0.7)

ax2.set_xlabel('Period', fontsize=11)
ax2.set_ylabel('Average Belief about LT Share', fontsize=11)
ax2.set_title('(b) Belief Convergence by Clarity', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.2, 0.55])

plt.tight_layout()
plt.savefig('H5_intent_clarity.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: H5_intent_clarity.png")
plt.show()

# ============================================================================
# KEY INSIGHT SUMMARY
# ============================================================================

print("\n" + "="*80)
print("KEY INSIGHT: THE INFORMATION PARADOX")
print("="*80)

no_clarity_ret = all_results['No clarity (ψ=0)']['results']['retention_L'].iloc[-1]
full_clarity_ret = all_results['Full clarity (ψ=1.0)']['results']['retention_L'].iloc[-1]
baseline_ret = all_results['Medium clarity (ψ=0.7, baseline)']['results']['retention_L'].iloc[-1]

print(f"\nIn LT-poor markets (ρ = {rho_m}):")
print(f"  No clarity: {100*no_clarity_ret:.1f}% retention")
print(f"  Medium clarity: {100*baseline_ret:.1f}% retention")
print(f"  Full clarity: {100*full_clarity_ret:.1f}% retention")
print(f"\nEffect of full transparency: {100*(full_clarity_ret - no_clarity_ret):+.1f} pp")

if full_clarity_ret < no_clarity_ret:
    print("\n✓ PARADOX CONFIRMED: More information HURTS users in LT-poor markets")
    print("  → Users learn bad news faster → exit earlier")
else:
    print("\n→ More information helps users (no paradox)")

print("="*80)