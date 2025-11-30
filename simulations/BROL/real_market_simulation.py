"""
Dating Market Simulation: Using REAL Markets from OkCupid Data
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
print("Using THREE REAL MARKETS from OkCupid Data")
print("="*80)

# ============================================================================
# LOAD DATA AND EXTRACT REAL MARKETS
# ============================================================================

print("\n[STEP 1] Loading OkCupid data and extracting real markets...")

df = pd.read_csv('notebooks/outputs/Final/okcupid_final_analysis_ready.csv')

# Group by location and orientation
markets = df.groupby(['location', 'orientation']).agg({
    'is_ltr_oriented': ['mean', 'count'],
    'target_avg_effort': 'mean',
    'rating_index': 'mean'
}).round(3)

markets.columns = ['LTR_share', 'N_users', 'Market_clarity', 'Avg_rating']
markets = markets[markets['N_users'] >= 50]
markets = markets.sort_values('LTR_share')

# Select three real markets
LOW_MARKET = ('san francisco, california', 'bisexual')
MEDIUM_MARKET = ('el cerrito, california', 'straight')  
HIGH_MARKET = ('alameda, california', 'gay')

# Extract their parameters
low_params = markets.loc[LOW_MARKET]
medium_params = markets.loc[MEDIUM_MARKET]
high_params = markets.loc[HIGH_MARKET]

print(f"\n✓ Selected THREE REAL MARKETS from data:")
print(f"\n  LOW:    {LOW_MARKET[0]}, {LOW_MARKET[1]}")
print(f"          LTR: {low_params['LTR_share']:.1%}, Clarity: {low_params['Market_clarity']:.3f}, N={int(low_params['N_users'])}")
print(f"\n  MEDIUM: {MEDIUM_MARKET[0]}, {MEDIUM_MARKET[1]}")
print(f"          LTR: {medium_params['LTR_share']:.1%}, Clarity: {medium_params['Market_clarity']:.3f}, N={int(medium_params['N_users'])}")
print(f"\n  HIGH:   {HIGH_MARKET[0]}, {HIGH_MARKET[1]}")
print(f"          LTR: {high_params['LTR_share']:.1%}, Clarity: {high_params['Market_clarity']:.3f}, N={int(high_params['N_users'])}")

# ============================================================================
# LOAD OTHER PARAMETERS FROM JSON
# ============================================================================

with open('simulations/simulation_parameters.json', 'r') as f:
    params = json.load(f)

ALPHA_RATING = params['alpha_rating']
BETA_RATING = params['beta_rating']
K = params['K']
RHO_I0_LTR = params['rho_i0_ltr']
RHO_I0_CASUAL = params['rho_i0_casual']
TAU_I = params['tau_i_suggested']
P_BAR = 0.20

T_MAX = 400
N_USERS = 200

# ============================================================================
# USER CLASS (same as before)
# ============================================================================

@dataclass
class User:
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

def create_user(user_id: int, ltr_share: float) -> User:
    rating = np.random.beta(ALPHA_RATING, BETA_RATING)
    rating = np.clip(rating, 0.1, 0.95)
    
    goal = 'ltr' if np.random.rand() < ltr_share else 'casual'
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
                scenario_name: str, location: str, orientation: str) -> pd.DataFrame:
    
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"  Market: {location}, {orientation}")
    print(f"  True LTR share: {rho_m_true:.1%}")
    print(f"  Market clarity: {psi_m:.3f}")
    print(f"{'='*80}")
    
    users = []
    results = []
    
    for i in range(n_users):
        user = create_user(i, rho_m_true)
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
        print(f"  Exited: 0 (0.0%)")
    
    return df, users

# ============================================================================
# RUN THREE REAL MARKET SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("RUNNING SIMULATIONS FOR THREE REAL MARKETS")
print("="*80)

# LOW: San Francisco Bisexual (46.2% LTR)
df_low, users_low = run_scenario(
    rho_m_true=low_params['LTR_share'],
    psi_m=low_params['Market_clarity'],
    n_users=N_USERS,
    scenario_name="LOW LTR Market",
    location=LOW_MARKET[0],
    orientation=LOW_MARKET[1]
)

# MEDIUM: El Cerrito Straight (58.6% LTR)
df_medium, users_medium = run_scenario(
    rho_m_true=medium_params['LTR_share'],
    psi_m=medium_params['Market_clarity'],
    n_users=N_USERS,
    scenario_name="MEDIUM LTR Market",
    location=MEDIUM_MARKET[0],
    orientation=MEDIUM_MARKET[1]
)

# HIGH: Alameda Gay (70.2% LTR)
df_high, users_high = run_scenario(
    rho_m_true=high_params['LTR_share'],
    psi_m=high_params['Market_clarity'],
    n_users=N_USERS,
    scenario_name="HIGH LTR Market",
    location=HIGH_MARKET[0],
    orientation=HIGH_MARKET[1]
)

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors = {'low': '#e74c3c', 'medium': '#f39c12', 'high': '#2ecc71'}

# 1. Exit rate by rating
ax = axes[0]
rating_bins = np.linspace(0, 1, 11)

for df, label, color in [
    (df_low, 'Low (46% LTR)', colors['low']),
    (df_medium, 'Medium (59% LTR)', colors['medium']),
    (df_high, 'High (70% LTR)', colors['high'])
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

# 2. Average exit time by rating
ax = axes[1]

for df, label, color in [
    (df_low, 'Low', colors['low']),
    (df_medium, 'Medium', colors['medium']),
    (df_high, 'High', colors['high'])
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

# 3. Exit rate comparison
ax = axes[2]

scenarios = [
    f'Low\nSF Bisexual\n(46% LTR)', 
    f'Medium\nEl Cerrito Straight\n(59% LTR)', 
    f'High\nAlameda Gay\n(70% LTR)'
]
exit_rates = [
    df_low['exited'].mean() * 100,
    df_medium['exited'].mean() * 100,
    df_high['exited'].mean() * 100
]

bars = ax.bar(scenarios, exit_rates, color=[colors['low'], colors['medium'], colors['high']])
ax.set_ylabel('Exit Rate (%)')
ax.set_title('Exit Rate by Real Market', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

for bar, rate in zip(bars, exit_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('real_markets_simulation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: real_markets_simulation.png")

# Save results
df_low['scenario'] = 'low'
df_medium['scenario'] = 'medium'
df_high['scenario'] = 'high'

combined = pd.concat([df_low, df_medium, df_high])
combined.to_csv('real_markets_results.csv', index=False)
print("✓ Saved: real_markets_results.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY: Three Real Markets from OkCupid Data")
print("="*80)

summary_data = []
for name, location, orientation, df in [
    ('Low LTR', LOW_MARKET[0], LOW_MARKET[1], df_low),
    ('Medium LTR', MEDIUM_MARKET[0], MEDIUM_MARKET[1], df_medium),
    ('High LTR', HIGH_MARKET[0], HIGH_MARKET[1], df_high)
]:
    n_exit = df['exited'].sum()
    pct_exit = n_exit / len(df) * 100
    avg_time = df[df['exited']]['exit_time'].mean() if n_exit > 0 else np.nan
    
    summary_data.append({
        'Market': f'{name}\n{location}\n{orientation}',
        'N Users': len(df),
        'N Exited': n_exit,
        'Exit Rate': f'{pct_exit:.1f}%',
        'Avg Exit Time': f'{avg_time:.1f}' if not np.isnan(avg_time) else 'N/A'
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("✅ SIMULATION COMPLETE - USING REAL OKCUPID MARKETS!")
print("="*80)