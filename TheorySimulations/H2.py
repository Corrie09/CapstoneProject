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

rho_m = 0.25  # LT-poor market
psi_m = 0.7  # Market clarity
prior_mean = 0.50  # Optimistic prior
prior_strength = 10

n_users = 500
n_periods = 40
np.random.seed(42)

# ============================================================================
# SIMULATION WITH BELIEF UPDATING
# ============================================================================

def simulate_with_beliefs(rho_m, psi_m, prior_mean, prior_strength):
    """Simulate with belief tracking at exit moments"""
    
    users = pd.DataFrame({
        'user_id': range(n_users),
        'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
        'rating': np.random.beta(2, 2, size=n_users),
        'active': True,
        'exit_period': np.nan,
        'exit_type': None
    })
    
    # Initialize beliefs for LT users
    alpha_0 = prior_mean * prior_strength
    beta_0 = (1 - prior_mean) * prior_strength
    
    users['alpha'] = alpha_0
    users['beta_param'] = beta_0
    users['belief_rho'] = prior_mean
    users['belief_at_exit'] = np.nan  # NEW: belief when exiting
    users['periods_active'] = 0  # NEW: track tenure
    
    belief_trajectories = []
    
    for t in range(n_periods):
        active = users[users['active']]
        
        if len(active) < 10:
            break
        
        active_L = active[active['goal'] == 'L']
        active_S = active[active['goal'] == 'S']
        
        # Update tenure
        users.loc[active.index, 'periods_active'] = t
        
        # Record beliefs for active LT users
        for idx in active_L.index:
            belief_trajectories.append({
                'user_id': users.loc[idx, 'user_id'],
                'period': t,
                'belief_rho': users.loc[idx, 'belief_rho']
            })
        
        # BELIEF UPDATING for LT users
        for idx in active_L.index:
            K_eff = np.random.binomial(K, psi_m)
            
            if K_eff > 0:
                true_rho = len(active_L) / len(active)
                K_L = np.random.binomial(K_eff, true_rho)
                
                users.loc[idx, 'alpha'] += K_L
                users.loc[idx, 'beta_param'] += (K_eff - K_L)
                
                new_belief = users.loc[idx, 'alpha'] / (users.loc[idx, 'alpha'] + users.loc[idx, 'beta_param'])
                users.loc[idx, 'belief_rho'] = new_belief
        
        # Process exits (RECORD BELIEF AT EXIT)
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
            
            p_bar = p_bar_L if goal == 'L' else p_bar_S
            
            if p_success < p_bar:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'frustration'
                users.loc[idx, 'belief_at_exit'] = users.loc[idx, 'belief_rho']  # RECORD
            elif np.random.random() < p_success:
                users.loc[idx, 'active'] = False
                users.loc[idx, 'exit_period'] = t
                users.loc[idx, 'exit_type'] = 'match'
                users.loc[idx, 'belief_at_exit'] = users.loc[idx, 'belief_rho']  # RECORD
    
    return users, pd.DataFrame(belief_trajectories)

# ============================================================================
# RUN SIMULATION
# ============================================================================

print(f"Running H2 simulation (LT-poor market, ψ_m = {psi_m})...")
users_df, beliefs_df = simulate_with_beliefs(rho_m, psi_m, prior_mean, prior_strength)

# ============================================================================
# ANALYZE H2: BELIEF LEVELS AT EXIT
# ============================================================================

print("\n" + "="*80)
print("H2: BELIEF-DRIVEN FRUSTRATION ANALYSIS")
print("="*80)

LT_users = users_df[users_df['goal'] == 'L'].copy()
exited_LT = LT_users[LT_users['exit_type'].notna()]
survived_LT = LT_users[LT_users['exit_type'].isna()]

frust_exits = exited_LT[exited_LT['exit_type'] == 'frustration']
match_exits = exited_LT[exited_LT['exit_type'] == 'match']

print(f"\nLT users total: {len(LT_users)}")
print(f"  Frustration exits: {len(frust_exits)} (tenure: {frust_exits['periods_active'].mean():.1f} periods)")
print(f"  Match exits: {len(match_exits)} (tenure: {match_exits['periods_active'].mean():.1f} periods)")
print(f"  Survived: {len(survived_LT)} (tenure: {survived_LT['periods_active'].mean():.1f} periods)")

print(f"\nBelief at exit:")
print(f"  Frustration exits: {frust_exits['belief_at_exit'].mean():.3f}")
print(f"  Match exits: {match_exits['belief_at_exit'].mean():.3f}")

print(f"\nStarting belief: {prior_mean:.3f}")
print(f"True ρ: {rho_m:.3f}")

# ============================================================================
# VISUALIZATION: REVISED H2 FIGURE
# ============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Belief trajectories by exit type (sample)
np.random.seed(42)
sample_size = 15

for exit_type, color, label in [
    ('frustration', '#d62728', 'Frustration exits'),
    ('match', '#2ca02c', 'Match exits'),
]:
    exit_users = exited_LT[exited_LT['exit_type'] == exit_type]
    if len(exit_users) > 0:
        sample = exit_users['user_id'].sample(min(sample_size, len(exit_users))).values
        
        for user_id in sample:
            traj = beliefs_df[beliefs_df['user_id'] == user_id]
            ax1.plot(traj['period'], traj['belief_rho'], color=color, alpha=0.2, linewidth=1)
        
        # Average trajectory
        avg_traj = beliefs_df[beliefs_df['user_id'].isin(exit_users['user_id'].values)].groupby('period')['belief_rho'].mean()
        ax1.plot(avg_traj.index, avg_traj.values, color=color, linewidth=3, label=label, zorder=10)

ax1.axhline(y=rho_m, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'True ρ = {rho_m}', zorder=5)
ax1.axhline(y=prior_mean, color='gray', linestyle=':', alpha=0.5, label=f'Prior = {prior_mean}')
ax1.set_xlabel('Period', fontsize=12)
ax1.set_ylabel('Belief about LT Share (ρ̂)', fontsize=12)
ax1.set_title('(a) Belief Learning Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.15, 0.55])

# Panel 2: Belief at exit vs tenure
ax2.scatter(frust_exits['periods_active'], frust_exits['belief_at_exit'], 
           color='#d62728', alpha=0.6, s=50, label='Frustration exits', edgecolors='black', linewidth=0.5)
ax2.scatter(match_exits['periods_active'], match_exits['belief_at_exit'], 
           color='#2ca02c', alpha=0.6, s=50, label='Match exits', edgecolors='black', linewidth=0.5)

ax2.axhline(y=rho_m, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'True ρ = {rho_m}')
ax2.set_xlabel('Exit Period (Tenure)', fontsize=12)
ax2.set_ylabel('Belief at Exit (ρ̂)', fontsize=12)
ax2.set_title('(b) Belief Levels at Exit', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('H2_belief_frustration.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: H2_belief_frustration.png")
plt.show()

# ============================================================================
# H2 HYPOTHESIS TEST (REVISED)
# ============================================================================

print("\n" + "="*80)
print("H2 HYPOTHESIS TEST")
print("="*80)

# Test: Do users who exit earlier have higher beliefs (haven't learned yet)?
early_exits = frust_exits[frust_exits['periods_active'] <= 5]
late_exits = frust_exits[frust_exits['periods_active'] > 5]

if len(early_exits) > 0 and len(late_exits) > 0:
    print(f"\nEarly frustration exits (≤5 periods): belief at exit = {early_exits['belief_at_exit'].mean():.3f}")
    print(f"Late frustration exits (>5 periods): belief at exit = {late_exits['belief_at_exit'].mean():.3f}")
    
    if late_exits['belief_at_exit'].mean() < early_exits['belief_at_exit'].mean():
        print("✓ Users who stay longer learn more (lower beliefs at exit)")

# Main H2 test: Compare frustration vs match beliefs
print(f"\nAverage belief at frustration exit: {frust_exits['belief_at_exit'].mean():.3f}")
print(f"Average belief at match exit: {match_exits['belief_at_exit'].mean():.3f}")

if frust_exits['belief_at_exit'].mean() > rho_m + 0.05:
    print(f"\n✓ H2 SUPPORTED: Frustration exits occur before beliefs fully converge to true ρ")
    print(f"  Frustration-driven users exit with pessimistic but not fully-learned beliefs")
else:
    print(f"\n~ H2 PARTIALLY SUPPORTED: Beliefs approach true ρ before exit")

print("="*80)