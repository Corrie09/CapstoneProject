"""
Dating Market Simulation: Teammate's Framework
Using OkCupid Data Parameters from JSON file

Run this locally - it will load simulation_parameters.json
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

np.random.seed(42)

print("="*80)
print("DATING MARKET SIMULATION")
print("Framework: Teammate's Model + OkCupid Parameters")
print("="*80)

# ============================================================================
# LOAD PARAMETERS FROM JSON
# ============================================================================

print("\n[STEP 1] Loading parameters from JSON file...")

with open('simulations/simulation_parameters.json', 'r') as f:
    params = json.load(f)

# Extract parameters
MEAN_RATING = params['mean_rating']
STD_RATING = params['std_rating']
ALPHA_RATING = params['alpha_rating']
BETA_RATING = params['beta_rating']
RHO_M_TRUE = params['ltr_share']
PSI_M = params['market_clarity']
K = params['K']
RHO_I0_LTR = params['rho_i0_ltr']
RHO_I0_CASUAL = params['rho_i0_casual']
TAU_I = params['tau_i_suggested']
P_BAR = 0.20  # Frustration threshold

# Simulation parameters
T_MAX = 400
N_USERS = 200

print(f"\n✓ Parameters loaded:")
print(f"  True LTR share: {RHO_M_TRUE:.1%}")
print(f"  Market clarity: {PSI_M:.3f}")
print(f"  Mean rating: {MEAN_RATING:.3f}")
print(f"  Rating Beta params: α={ALPHA_RATING:.2f}, β={BETA_RATING:.2f}")

# ============================================================================
# USER CLASS
# ============================================================================

@dataclass
class User:
    """Individual user with beliefs and behavior"""
    user_id: int
    rating: float
    goal: str
    rho_i0: float
    tau_i: float
    p_bar: float
    
    alpha: float = None
    beta: float = None
    exit_time: int = None
    belief_history: List[float] = None
    success_prob_history: List[float] = None
    
    def __post_init__(self):
        self.alpha = self.rho_i0 * self.tau_i
        self.beta = (1 - self.rho_i0) * self.tau_i
        self.belief_history = [self.rho_i0]
        p_hat_0 = self.rating * self.rho_i0
        self.success_prob_history = [p_hat_0]
    
    def get_rho_hat(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def get_p_hat(self) -> float:
        return self.rating * self.get_rho_hat()
    
    def observe_batch(self, rho_m_true: float, psi_m: float, K: int):
        K_eff = np.random.binomial(K, psi_m)
        K_L = np.random.binomial(K_eff, rho_m_true)
        
        self.alpha += K_L
        self.beta += (K_eff - K_L)
        
        self.belief_history.append(self.get_rho_hat())
        self.success_prob_history.append(self.get_p_hat())
    
    def is_frustrated(self) -> bool:
        return self.get_p_hat() < self.p_bar
    
    def simulate_until_exit(self, rho_m_true: float, psi_m: float, 
                           K: int, T_max: int) -> int:
        for t in range(1, T_max + 1):
            self.observe_batch(rho_m_true, psi_m, K)
            if self.is_frustrated():
                self.exit_time = t
                return t
        self.exit_time = T_max
        return T_max

# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def create_user(user_id: int, goal: str = None) -> User:
    rating = np.random.beta(ALPHA_RATING, BETA_RATING)
    rating = np.clip(rating, 0.1, 0.95)
    
    if goal is None:
        goal = 'ltr' if np.random.rand() < RHO_M_TRUE else 'casual'
    
    rho_i0 = RHO_I0_LTR if goal == 'ltr' else RHO_I0_CASUAL
    
    return User(
        user_id=user_id,
        rating=rating,
        goal=goal,
        rho_i0=rho_i0,
        tau_i=TAU_I,
        p_bar=P_BAR
    )

def run_scenario(rho_m_true: float, psi_m: float, n_users: int, 
                scenario_name: str) -> pd.DataFrame:
    
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  True LTR share: {rho_m_true:.1%}")
    print(f"  Market clarity: {psi_m:.3f}")
    print(f"{'='*80}")
    
    users = []
    results = []
    
    for i in range(n_users):
        user = create_user(i)
        exit_time = user.simulate_until_exit(rho_m_true, psi_m, K, T_MAX)
        
        results.append({
            'user_id': user.user_id,
            'rating': user.rating,
            'goal': user.goal,
            'rho_i0': user.rho_i0,
            'exit_time': exit_time,
            'exited': exit_time < T_MAX,
            'final_rho_hat': user.get_rho_hat(),
            'final_p_hat': user.get_p_hat()
        })
        
        users.append(user)
        
        if (i + 1) % 50 == 0:
            n_exited = sum(1 for r in results if r['exited'])
            avg_exit = np.mean([r['exit_time'] for r in results if r['exited']]) if n_exited > 0 else 0
            print(f"  Simulated {i+1:3d} users: {n_exited:3d} exited (avg time: {avg_exit:.1f})")
    
    df = pd.DataFrame(results)
    
    n_exited = df['exited'].sum()
    pct_exited = n_exited / len(df) * 100
    
    if n_exited > 0:
        avg_exit_time = df[df['exited']]['exit_time'].mean()
        ltr_exits = df[(df['goal'] == 'ltr') & (df['exited'])]
        casual_exits = df[(df['goal'] == 'casual') & (df['exited'])]
        
        print(f"\n📊 RESULTS:")
        print(f"  Total users: {len(df)}")
        print(f"  Exited: {n_exited} ({pct_exited:.1f}%)")
        print(f"  Average exit time: {avg_exit_time:.1f} periods")
        print(f"  LTR users exited: {len(ltr_exits)} / {(df['goal']=='ltr').sum()}")
        print(f"  Casual users exited: {len(casual_exits)} / {(df['goal']=='casual').sum()}")
    else:
        print(f"\n📊 RESULTS:")
        print(f"  Total users: {len(df)}")
        print(f"  Exited: 0 (0.0%) - Market too good!")
    
    return df, users

# ============================================================================
# RUN THREE SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SIMULATIONS")
print("="*80)

df_healthy, users_healthy = run_scenario(
    rho_m_true=0.55,
    psi_m=0.70,
    n_users=N_USERS,
    scenario_name="HEALTHY (55% LTR, High Clarity)"
)

df_poor, users_poor = run_scenario(
    rho_m_true=0.25,
    psi_m=0.50,
    n_users=N_USERS,
    scenario_name="POOR (25% LTR, Low Clarity)"
)

df_medium, users_medium = run_scenario(
    rho_m_true=0.40,
    psi_m=0.60,
    n_users=N_USERS,
    scenario_name="MEDIUM (40% LTR, Medium Clarity)"
)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors = {'healthy': '#2ecc71', 'poor': '#e74c3c', 'medium': '#f39c12'}

# 1. Exit time distribution
ax = axes[0, 0]
bins = np.arange(0, T_MAX + 20, 20)

for df, label, color in [
    (df_healthy, 'Healthy', colors['healthy']),
    (df_poor, 'Poor', colors['poor']),
    (df_medium, 'Medium', colors['medium'])
]:
    exited = df[df['exited']]
    if len(exited) > 0:
        ax.hist(exited['exit_time'], bins=bins, alpha=0.6, label=label, color=color)

ax.set_xlabel('Exit Time (periods)')
ax.set_ylabel('Number of Users')
ax.set_title('Exit Time Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 2. Exit rate by rating
ax = axes[0, 1]
rating_bins = np.linspace(0, 1, 11)

for df, label, color in [
    (df_healthy, 'Healthy', colors['healthy']),
    (df_poor, 'Poor', colors['poor']),
    (df_medium, 'Medium', colors['medium'])
]:
    df['rating_bin'] = pd.cut(df['rating'], rating_bins)
    exit_rate = df.groupby('rating_bin', observed=True)['exited'].mean()
    bin_centers = [(interval.left + interval.right) / 2 for interval in exit_rate.index]
    ax.plot(bin_centers, exit_rate * 100, marker='o', label=label, color=color, linewidth=2)

ax.set_xlabel('User Rating (r_i)')
ax.set_ylabel('Exit Rate (%)')
ax.set_title('Exit Rate by User Rating', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 3. Average exit time by rating
ax = axes[0, 2]

for df, label, color in [
    (df_healthy, 'Healthy', colors['healthy']),
    (df_poor, 'Poor', colors['poor']),
    (df_medium, 'Medium', colors['medium'])
]:
    exited = df[df['exited']].copy()
    if len(exited) > 5:
        exited['rating_bin'] = pd.cut(exited['rating'], rating_bins)
        avg_exit = exited.groupby('rating_bin', observed=True)['exit_time'].mean()
        bin_centers = [(interval.left + interval.right) / 2 for interval in avg_exit.index]
        ax.plot(bin_centers, avg_exit, marker='o', label=label, color=color, linewidth=2)

ax.set_xlabel('User Rating (r_i)')
ax.set_ylabel('Average Exit Time')
ax.set_title('Exit Time vs Rating (Among Exiters)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 4. Belief trajectory (Poor market)
ax = axes[1, 0]
poor_ltr_users = [u for u in users_poor if u.goal == 'ltr']
sample_users = np.random.choice(poor_ltr_users, min(5, len(poor_ltr_users)), replace=False)

for user in sample_users:
    periods = range(len(user.belief_history))
    alpha_val = 0.3 if user.exit_time < T_MAX else 1.0
    ax.plot(periods, user.belief_history, alpha=alpha_val, linewidth=1.5,
           label=f'r={user.rating:.2f}, exit={user.exit_time}')

ax.axhline(RHO_M_TRUE, linestyle='--', color='gray', label='True ρ_m', linewidth=2)
ax.set_xlabel('Period')
ax.set_ylabel('Belief ρ̂_m')
ax.set_title('Belief Evolution (Poor Market, LTR users)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# 5. Success probability trajectory
ax = axes[1, 1]

for user in sample_users:
    periods = range(len(user.success_prob_history))
    alpha_val = 0.3 if user.exit_time < T_MAX else 1.0
    ax.plot(periods, user.success_prob_history, alpha=alpha_val, linewidth=1.5)

ax.axhline(P_BAR, linestyle='--', color='red', label=f'Frustration threshold p̄={P_BAR}', linewidth=2)
ax.set_xlabel('Period')
ax.set_ylabel('Success Probability p̂')
ax.set_title('Success Probability Evolution (Poor Market)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# 6. Exit rate comparison
ax = axes[1, 2]

scenarios = ['Healthy\n(55% LTR)', 'Medium\n(40% LTR)', 'Poor\n(25% LTR)']
exit_rates = [
    df_healthy['exited'].mean() * 100,
    df_medium['exited'].mean() * 100,
    df_poor['exited'].mean() * 100
]

bars = ax.bar(scenarios, exit_rates, color=[colors['healthy'], colors['medium'], colors['poor']])
ax.set_ylabel('Exit Rate (%)')
ax.set_title('Exit Rate by Market Scenario', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, rate in zip(bars, exit_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('teammate_framework_simulation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: teammate_framework_simulation.png")

# Save results
df_healthy['scenario'] = 'healthy'
df_poor['scenario'] = 'poor'
df_medium['scenario'] = 'medium'

combined = pd.concat([df_healthy, df_poor, df_medium])
combined.to_csv('teammate_framework_results.csv', index=False)
print("✓ Saved: teammate_framework_results.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

summary_data = []
for name, df in [('Healthy (55% LTR)', df_healthy), 
                 ('Medium (40% LTR)', df_medium),
                 ('Poor (25% LTR)', df_poor)]:
    
    n_exit = df['exited'].sum()
    pct_exit = n_exit / len(df) * 100
    avg_time = df[df['exited']]['exit_time'].mean() if n_exit > 0 else np.nan
    
    summary_data.append({
        'Scenario': name,
        'N Users': len(df),
        'N Exited': n_exit,
        'Exit Rate': f'{pct_exit:.1f}%',
        'Avg Exit Time': f'{avg_time:.1f}' if not np.isnan(avg_time) else 'N/A'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)