import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)

# ============================================================================
# PARAMETERS
# ============================================================================

K = 30  

def beta(r):
    """Match probability - higher-rated users match faster"""
    return 0.002 + 0.006 * r

# Acceptance regions (from Section 2.1)
theta_L_minus, theta_L_plus = 0.90, 1.10  # LT: narrow
theta_S_minus, theta_S_plus = 0.60, 1.40  # ST: wide

# Frustration thresholds (from Section 2.4)
p_bar_L = 0.010  # LT users need 1% success rate
p_bar_S = 0.006  # ST users need 0.6% success rate

# Market types
markets = {
    'LT-poor': 0.25,
    'Balanced': 0.55,
    'LT-rich': 0.85
}

n_users = 500
n_periods = 40
np.random.seed(42)

# ============================================================================
# SIMULATION WITH RETENTION TRACKING
# ============================================================================

def simulate_retention(market_name, rho_m):
    """Simulate market and track RETENTION RATES"""
    
    # Initialize users
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None
    })
    
    # Track initial counts
    initial_L = len(users[users['goal'] == 'L'])
    initial_S = len(users[users['goal'] == 'S'])
    
    results = []
    
    for t in range(n_periods):
        active = users[users['active']]
        
        if len(active) < 10:
            break
        
        # Calculate RETENTION RATES (fraction of original users still active)
        active_L = active[active['goal'] == 'L']
        active_S = active[active['goal'] == 'S']
        
        retention_L = len(active_L) / initial_L if initial_L > 0 else 0
        retention_S = len(active_S) / initial_S if initial_S > 0 else 0
        
        results.append({
            'period': t,
            'retention_L': retention_L,
            'retention_S': retention_S,
            'n_active_L': len(active_L),
            'n_active_S': len(active_S)
        })
        
        # Process exits
        for idx in active.index:
            r_i = users.loc[idx, 'rating']
            goal = users.loc[idx, 'goal']
            
            # Compute success probability
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
            
            # Check frustration exit
            p_bar = p_bar_L if goal == 'L' else p_bar_S
            
            if p_success < p_bar:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'frustration'
            elif np.random.random() < p_success:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'match'
    
    return pd.DataFrame(results), users, initial_L, initial_S

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

all_results = {}
all_users = {}
all_initial = {}

print("Running simulations...\n")

for market_name, rho_m in markets.items():
    results_df, users_df, init_L, init_S = simulate_retention(market_name, rho_m)
    all_results[market_name] = results_df
    all_users[market_name] = users_df
    all_initial[market_name] = (init_L, init_S)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("="*80)
print("H1: MARKET WASH-OUT SUMMARY")
print("="*80)

for market_name, users_df in all_users.items():
    init_L, init_S = all_initial[market_name]
    exited = users_df[users_df['exit_type'].notna()]
    
    print(f"\n{market_name} Market (ρ0={markets[market_name]}):")
    print(f"  Initial users: {init_L} LT, {init_S} ST")
    
    for goal in ['L', 'S']:
        goal_name = 'LT' if goal == 'L' else 'ST'
        goal_users = users_df[users_df['goal'] == goal]
        goal_exited = exited[exited['goal'] == goal]
        
        frust = len(goal_exited[goal_exited['exit_type'] == 'frustration'])
        match = len(goal_exited[goal_exited['exit_type'] == 'match'])
        still_active = len(goal_users) - len(goal_exited)
        
        total = len(goal_users)
        final_retention = still_active / total if total > 0 else 0
        
        print(f"  {goal_name} users:")
        print(f"    Frustration exits: {frust}/{total} ({100*frust/total:.0f}%)")
        print(f"    Match exits: {match}/{total} ({100*match/total:.0f}%)")
        print(f"    Final retention: {100*final_retention:.0f}%")

# ============================================================================
# VISUALIZATION - TWO PANELS
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {'LT-poor': '#d62728', 'Balanced': '#ff7f0e', 'LT-rich': '#2ca02c'}

# Panel 1: LT User Retention
for market_name, results_df in all_results.items():
    ax1.plot(results_df['period'], results_df['retention_L'], 
            label=f"{market_name}", 
            linewidth=2.5,
            color=colors[market_name])

ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('LT User Retention Rate', fontsize=12)
ax1.set_title('(a) LT User Retention by Market Type', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# Panel 2: Frustration Rate by Market
frustration_rates = []
for market_name in ['LT-poor', 'Balanced', 'LT-rich']:
    users_df = all_users[market_name]
    exited_L = users_df[(users_df['goal'] == 'L') & (users_df['exit_type'].notna())]
    
    frust_L = len(exited_L[exited_L['exit_type'] == 'frustration'])
    total_exits_L = len(exited_L)
    
    frust_rate = frust_L / total_exits_L if total_exits_L > 0 else 0
    frustration_rates.append(frust_rate)

ax2.bar(['LT-poor', 'Balanced', 'LT-rich'], frustration_rates,
        color=[colors['LT-poor'], colors['Balanced'], colors['LT-rich']],
        alpha=0.8)
ax2.set_ylabel('Frustration Exit Rate (LT Users)', fontsize=12)
ax2.set_title('(b) LT User Exit Mechanism by Market', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.0])

# Add percentage labels on bars
for i, rate in enumerate(frustration_rates):
    ax2.text(i, rate + 0.02, f'{100*rate:.0f}%', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('H1_retention_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# H1 HYPOTHESIS TEST
# ============================================================================

print("\n" + "="*80)
print("H1 HYPOTHESIS: LT users in LT-poor markets exit faster than LT-rich")
print("="*80)

for market_name, results_df in all_results.items():
    if len(results_df) < 5:
        continue
    
    retention_final = results_df['retention_L'].iloc[-1]
    
    print(f"\n{market_name}:")
    print(f"  Final LT retention: {100*retention_final:.1f}%")

print("\n" + "="*80)
print("CONCLUSION:")

retention_poor = all_results['LT-poor']['retention_L'].iloc[-1]
retention_rich = all_results['LT-rich']['retention_L'].iloc[-1]

if retention_poor < retention_rich - 0.10:
    print("✓ H1 SUPPORTED: LT retention significantly lower in LT-poor markets")
else:
    print("✗ H1 NOT SUPPORTED: LT retention similar across market types")

print(f"\nLT-poor retention: {100*retention_poor:.1f}%")
print(f"LT-rich retention: {100*retention_rich:.1f}%")
print(f"Difference: {100*(retention_rich - retention_poor):.1f} percentage points")
print("="*80)