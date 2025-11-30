"""
Dating Market Simulation: Pure Theory Demonstration
Using Parameters Calibrated from Real OkCupid Data

Purpose: Demonstrate Bayesian learning and frustration dynamics
Input: Parameters from YOUR OkCupid data analysis
Output: Simulated market evolution (no comparison to real data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict

np.random.seed(42)

print("="*80)
print("DATING MARKET SIMULATION")
print("Theory Demonstration Using Realistic Parameters")
print("="*80)

# ============================================================================
# STEP 1: EXTRACT PARAMETERS FROM YOUR REAL DATA
# ============================================================================

print("\n[1] Loading YOUR data to extract parameters...")

df = pd.read_csv('notebooks/outputs/Final/okcupid_final_analysis_ready.csv')

# Extract realistic parameters
INITIAL_MARKET_SIZE = 1000
INITIAL_LTR_SHARE = df['is_ltr_oriented'].mean()  # Your data: 54.7%
MEAN_RATING = df['rating_index'].mean()  # Your data: 0.611
STD_RATING = df['rating_index'].std()
MEAN_EFFORT = df['effort_index'].mean()  # Your data: 0.688

# Distribution parameters (fit Beta distribution to your ratings)
alpha_rating = MEAN_RATING * ((MEAN_RATING * (1 - MEAN_RATING) / (STD_RATING**2)) - 1)
beta_rating = (1 - MEAN_RATING) * ((MEAN_RATING * (1 - MEAN_RATING) / (STD_RATING**2)) - 1)

print(f"\n✓ Parameters extracted from YOUR OkCupid data:")
print(f"  Initial market size: {INITIAL_MARKET_SIZE:,}")
print(f"  Initial LTR share: {INITIAL_LTR_SHARE:.1%}")
print(f"  Mean rating: {MEAN_RATING:.3f} (SD: {STD_RATING:.3f})")
print(f"  Mean effort: {MEAN_EFFORT:.3f}")
print(f"  Rating distribution: Beta({alpha_rating:.2f}, {beta_rating:.2f})")

# ============================================================================
# STEP 2: MODEL PARAMETERS (from Proposal Section 2.2)
# ============================================================================

print("\n[2] Setting up Bayesian learning parameters...")

K = 20  # Profiles shown to newcomer
PRIOR_RHO = 0.5  # Prior belief about market LTR share
PRIOR_P = 0.3  # Prior belief about success probability

# Beta priors for Bayesian updating
ALPHA_RHO = 5
BETA_RHO = 5
ALPHA_P = 3
BETA_P = 7

# Frustration thresholds
FRUSTRATION_RHO_THRESHOLD = 0.45
FRUSTRATION_P_THRESHOLD = 0.30

print(f"  K (profiles shown): {K}")
print(f"  Prior beliefs: ρ_m = {PRIOR_RHO}, p_i = {PRIOR_P}")
print(f"  Frustration thresholds: ρ̂_m < {FRUSTRATION_RHO_THRESHOLD}, p̂_i < {FRUSTRATION_P_THRESHOLD}")

# ============================================================================
# STEP 3: CREATE SYNTHETIC USERS (using YOUR data parameters)
# ============================================================================

class SyntheticUser:
    """Synthetic user with realistic characteristics"""
    def __init__(self, user_id, rating, goal, effort):
        self.user_id = user_id
        self.rating = rating
        self.goal = goal  # 'ltr' or 'casual'
        self.effort = effort
        self.is_active = True
        self.rho_belief = None
        self.p_belief = None

def create_initial_market(size, ltr_share, rating_params):
    """Create initial synthetic market using YOUR data parameters"""
    users = []
    
    for i in range(size):
        # Draw rating from distribution fitted to YOUR data
        rating = np.random.beta(rating_params[0], rating_params[1])
        rating = np.clip(rating, 0.1, 0.95)  # Keep in reasonable range
        
        # Assign goal based on YOUR data's LTR share
        goal = 'ltr' if np.random.rand() < ltr_share else 'casual'
        
        # Initial effort: higher for LTR users (from YOUR data pattern)
        if goal == 'ltr':
            effort = np.random.uniform(0.6, 0.9)
        else:
            effort = np.random.uniform(0.5, 0.8)
        
        user = SyntheticUser(i, rating, goal, effort)
        users.append(user)
    
    return users

print(f"\n[3] Creating initial synthetic market...")

initial_users = create_initial_market(
    INITIAL_MARKET_SIZE,
    INITIAL_LTR_SHARE,
    (alpha_rating, beta_rating)
)

initial_ltr_count = sum(1 for u in initial_users if u.goal == 'ltr')
initial_avg_rating = np.mean([u.rating for u in initial_users])
initial_avg_effort = np.mean([u.effort for u in initial_users])

print(f"✓ Initial market created:")
print(f"  Size: {len(initial_users):,}")
print(f"  LTR users: {initial_ltr_count} ({initial_ltr_count/len(initial_users):.1%})")
print(f"  Mean rating: {initial_avg_rating:.3f}")
print(f"  Mean effort: {initial_avg_effort:.3f}")

# ============================================================================
# STEP 4: BAYESIAN LEARNING FUNCTIONS
# ============================================================================

def bayesian_update_rho(n_ltr_seen, n_total_seen):
    """Update belief about market LTR share"""
    alpha_post = ALPHA_RHO + n_ltr_seen
    beta_post = BETA_RHO + (n_total_seen - n_ltr_seen)
    return alpha_post / (alpha_post + beta_post)

def bayesian_update_p(n_success, n_attempts):
    """Update belief about success probability"""
    alpha_post = ALPHA_P + n_success
    beta_post = BETA_P + (n_attempts - n_success)
    return alpha_post / (alpha_post + beta_post)

def is_frustrated(rho_hat, p_hat):
    """Check if user is frustrated"""
    return (rho_hat < FRUSTRATION_RHO_THRESHOLD) and (p_hat < FRUSTRATION_P_THRESHOLD)

def choose_effort(rho_hat, p_hat, rating, goal):
    """Choose effort based on beliefs"""
    if goal == 'casual':
        return 0.7, False  # Casual users: moderate effort, don't exit
    
    # LTR users check frustration
    frustrated = is_frustrated(rho_hat, p_hat)
    
    if frustrated:
        # High-rated users more likely to exit
        exit_prob = 0.5 * rating
        if np.random.rand() < exit_prob:
            return 0.0, True  # EXIT
        else:
            return 0.5, False  # Low effort
    else:
        return 0.9, False  # High effort

# ============================================================================
# STEP 5: RUN SIMULATION
# ============================================================================

def run_simulation(initial_users, n_newcomers, scenario_name="Baseline"):
    """
    Run market simulation with Bayesian learning
    
    Returns: history of market states
    """
    
    print(f"\n{'='*80}")
    print(f"SIMULATING: {scenario_name}")
    print(f"{'='*80}")
    
    # Make copy of initial users
    users = [SyntheticUser(u.user_id, u.rating, u.goal, u.effort) for u in initial_users]
    next_id = len(users)
    
    history = []
    
    # Record initial state
    active_users = [u for u in users if u.is_active]
    ltr_count = sum(1 for u in active_users if u.goal == 'ltr')
    
    history.append({
        'timestep': 0,
        'market_size': len(active_users),
        'ltr_share': ltr_count / len(active_users),
        'avg_effort': np.mean([u.effort for u in active_users]),
        'avg_rating': np.mean([u.rating for u in active_users]),
        'ltr_avg_effort': np.mean([u.effort for u in active_users if u.goal == 'ltr']),
        'exits': 0,
        'frustrated': 0
    })
    
    # Simulate newcomers arriving
    for t in range(1, n_newcomers + 1):
        
        # Generate newcomer
        rating = np.random.beta(alpha_rating, beta_rating)
        rating = np.clip(rating, 0.1, 0.95)
        goal = 'ltr' if np.random.rand() < INITIAL_LTR_SHARE else 'casual'
        
        # Sample K profiles
        active_users = [u for u in users if u.is_active]
        if len(active_users) < K:
            sample = active_users
        else:
            sample = np.random.choice(active_users, K, replace=False)
        
        # Observe and update beliefs
        n_ltr_seen = sum(1 for u in sample if u.goal == 'ltr')
        rho_hat = bayesian_update_rho(n_ltr_seen, len(sample))
        
        # Simulate matching attempts
        n_attempts = 5
        success_prob = 0.15 + 0.25 * rating
        n_success = np.random.binomial(n_attempts, success_prob)
        p_hat = bayesian_update_p(n_success, n_attempts)
        
        # Choose effort
        effort, should_exit = choose_effort(rho_hat, p_hat, rating, goal)
        
        # Create newcomer
        newcomer = SyntheticUser(next_id, rating, goal, effort)
        newcomer.is_active = not should_exit
        newcomer.rho_belief = rho_hat
        newcomer.p_belief = p_hat
        next_id += 1
        
        users.append(newcomer)
        
        # Record state every 10 timesteps
        if t % 10 == 0:
            active_users = [u for u in users if u.is_active]
            ltr_users = [u for u in active_users if u.goal == 'ltr']
            
            if len(active_users) > 0 and len(ltr_users) > 0:
                history.append({
                    'timestep': t,
                    'market_size': len(active_users),
                    'ltr_share': len(ltr_users) / len(active_users),
                    'avg_effort': np.mean([u.effort for u in active_users]),
                    'avg_rating': np.mean([u.rating for u in active_users]),
                    'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
                    'exits': sum(1 for u in users[INITIAL_MARKET_SIZE:] if not u.is_active),
                    'frustrated': sum(1 for u in users[INITIAL_MARKET_SIZE:] 
                                    if hasattr(u, 'rho_belief') and is_frustrated(u.rho_belief, u.p_belief))
                })
        
        # Progress update
        if t % 100 == 0:
            state = history[-1]
            print(f"  t={t:4d}: Size={state['market_size']:4d}, "
                  f"LTR={state['ltr_share']:.1%}, "
                  f"Effort={state['avg_effort']:.3f}, "
                  f"Exits={state['exits']}")
    
    # Final state
    active_users = [u for u in users if u.is_active]
    ltr_users = [u for u in active_users if u.goal == 'ltr']
    
    if len(active_users) > 0 and len(ltr_users) > 0:
        history.append({
            'timestep': n_newcomers,
            'market_size': len(active_users),
            'ltr_share': len(ltr_users) / len(active_users),
            'avg_effort': np.mean([u.effort for u in active_users]),
            'avg_rating': np.mean([u.rating for u in active_users]),
            'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
            'exits': sum(1 for u in users[INITIAL_MARKET_SIZE:] if not u.is_active),
            'frustrated': sum(1 for u in users[INITIAL_MARKET_SIZE:] 
                            if hasattr(u, 'rho_belief') and is_frustrated(u.rho_belief, u.p_belief))
        })
    
    return pd.DataFrame(history), users

# ============================================================================
# SCENARIO 1: BASELINE (Random Sampling)
# ============================================================================

history_baseline, users_baseline = run_simulation(
    initial_users,
    n_newcomers=500,
    scenario_name="Baseline (Random Sampling)"
)

# ============================================================================
# SCENARIO 2: COUNTERFACTUAL (Platform Curates)
# ============================================================================

def run_simulation_curated(initial_users, n_newcomers):
    """
    Simulate with platform curation: show MORE LTR profiles to LTR seekers
    """
    
    print(f"\n{'='*80}")
    print(f"SIMULATING: Platform Curation (Counterfactual)")
    print(f"{'='*80}")
    
    users = [SyntheticUser(u.user_id, u.rating, u.goal, u.effort) for u in initial_users]
    next_id = len(users)
    
    history = []
    
    # Initial state
    active_users = [u for u in users if u.is_active]
    ltr_count = sum(1 for u in active_users if u.goal == 'ltr')
    
    history.append({
        'timestep': 0,
        'market_size': len(active_users),
        'ltr_share': ltr_count / len(active_users),
        'avg_effort': np.mean([u.effort for u in active_users]),
        'avg_rating': np.mean([u.rating for u in active_users]),
        'ltr_avg_effort': np.mean([u.effort for u in active_users if u.goal == 'ltr']),
        'exits': 0,
        'frustrated': 0
    })
    
    for t in range(1, n_newcomers + 1):
        
        rating = np.random.beta(alpha_rating, beta_rating)
        rating = np.clip(rating, 0.1, 0.95)
        goal = 'ltr' if np.random.rand() < INITIAL_LTR_SHARE else 'casual'
        
        active_users = [u for u in users if u.is_active]
        
        # COUNTERFACTUAL: If newcomer wants LTR, show them more LTR profiles
        if goal == 'ltr' and len(active_users) >= K:
            ltr_users = [u for u in active_users if u.goal == 'ltr']
            casual_users = [u for u in active_users if u.goal == 'casual']
            
            # Show 70% LTR instead of true market share
            n_ltr_show = min(int(K * 0.7), len(ltr_users))
            n_casual_show = K - n_ltr_show
            
            if len(ltr_users) >= n_ltr_show and len(casual_users) >= n_casual_show:
                sample = (list(np.random.choice(ltr_users, n_ltr_show, replace=False)) +
                         list(np.random.choice(casual_users, n_casual_show, replace=False)))
            else:
                sample = np.random.choice(active_users, min(K, len(active_users)), replace=False)
        else:
            # Random sampling
            sample = np.random.choice(active_users, min(K, len(active_users)), replace=False)
        
        # Rest is same as baseline
        n_ltr_seen = sum(1 for u in sample if u.goal == 'ltr')
        rho_hat = bayesian_update_rho(n_ltr_seen, len(sample))
        
        n_attempts = 5
        success_prob = 0.15 + 0.25 * rating
        n_success = np.random.binomial(n_attempts, success_prob)
        p_hat = bayesian_update_p(n_success, n_attempts)
        
        effort, should_exit = choose_effort(rho_hat, p_hat, rating, goal)
        
        newcomer = SyntheticUser(next_id, rating, goal, effort)
        newcomer.is_active = not should_exit
        newcomer.rho_belief = rho_hat
        newcomer.p_belief = p_hat
        next_id += 1
        
        users.append(newcomer)
        
        if t % 10 == 0:
            active_users = [u for u in users if u.is_active]
            ltr_users = [u for u in active_users if u.goal == 'ltr']
            
            if len(active_users) > 0 and len(ltr_users) > 0:
                history.append({
                    'timestep': t,
                    'market_size': len(active_users),
                    'ltr_share': len(ltr_users) / len(active_users),
                    'avg_effort': np.mean([u.effort for u in active_users]),
                    'avg_rating': np.mean([u.rating for u in active_users]),
                    'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
                    'exits': sum(1 for u in users[INITIAL_MARKET_SIZE:] if not u.is_active),
                    'frustrated': sum(1 for u in users[INITIAL_MARKET_SIZE:] 
                                    if hasattr(u, 'rho_belief') and is_frustrated(u.rho_belief, u.p_belief))
                })
        
        if t % 100 == 0:
            state = history[-1]
            print(f"  t={t:4d}: Size={state['market_size']:4d}, "
                  f"LTR={state['ltr_share']:.1%}, "
                  f"Effort={state['avg_effort']:.3f}, "
                  f"Exits={state['exits']}")
    
    # Final state
    active_users = [u for u in users if u.is_active]
    ltr_users = [u for u in active_users if u.goal == 'ltr']
    
    if len(active_users) > 0 and len(ltr_users) > 0:
        history.append({
            'timestep': n_newcomers,
            'market_size': len(active_users),
            'ltr_share': len(ltr_users) / len(active_users),
            'avg_effort': np.mean([u.effort for u in active_users]),
            'avg_rating': np.mean([u.rating for u in active_users]),
            'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
            'exits': sum(1 for u in users[INITIAL_MARKET_SIZE:] if not u.is_active),
            'frustrated': sum(1 for u in users[INITIAL_MARKET_SIZE:] 
                            if hasattr(u, 'rho_belief') and is_frustrated(u.rho_belief, u.p_belief))
        })
    
    return pd.DataFrame(history), users

history_curated, users_curated = run_simulation_curated(initial_users, 500)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SIMULATION RESULTS: Theory Demonstration")
print("="*80)

print("\n📊 BASELINE SCENARIO (Random Sampling):")
print(f"  Initial → Final")
print(f"  Market size:   {history_baseline.iloc[0]['market_size']:.0f} → {history_baseline.iloc[-1]['market_size']:.0f}")
print(f"  LTR share:     {history_baseline.iloc[0]['ltr_share']:.1%} → {history_baseline.iloc[-1]['ltr_share']:.1%}")
print(f"  Avg effort:    {history_baseline.iloc[0]['avg_effort']:.3f} → {history_baseline.iloc[-1]['avg_effort']:.3f}")
print(f"  LTR effort:    {history_baseline.iloc[0]['ltr_avg_effort']:.3f} → {history_baseline.iloc[-1]['ltr_avg_effort']:.3f}")
print(f"  Total exits:   {history_baseline.iloc[-1]['exits']:.0f}")
print(f"  Frustrated:    {history_baseline.iloc[-1]['frustrated']:.0f}")

print("\n📊 COUNTERFACTUAL (Platform Curation):")
print(f"  Initial → Final")
print(f"  Market size:   {history_curated.iloc[0]['market_size']:.0f} → {history_curated.iloc[-1]['market_size']:.0f}")
print(f"  LTR share:     {history_curated.iloc[0]['ltr_share']:.1%} → {history_curated.iloc[-1]['ltr_share']:.1%}")
print(f"  Avg effort:    {history_curated.iloc[0]['avg_effort']:.3f} → {history_curated.iloc[-1]['avg_effort']:.3f}")
print(f"  LTR effort:    {history_curated.iloc[0]['ltr_avg_effort']:.3f} → {history_curated.iloc[-1]['ltr_avg_effort']:.3f}")
print(f"  Total exits:   {history_curated.iloc[-1]['exits']:.0f}")
print(f"  Frustrated:    {history_curated.iloc[-1]['frustrated']:.0f}")

print("\n💡 KEY INSIGHTS:")
ltr_change_baseline = (history_baseline.iloc[-1]['ltr_share'] - history_baseline.iloc[0]['ltr_share']) * 100
ltr_change_curated = (history_curated.iloc[-1]['ltr_share'] - history_curated.iloc[0]['ltr_share']) * 100
curation_benefit = ltr_change_curated - ltr_change_baseline

print(f"  1. Baseline: LTR share changed by {ltr_change_baseline:+.1f} percentage points")
print(f"  2. Curated: LTR share changed by {ltr_change_curated:+.1f} percentage points")
print(f"  3. Curation benefit: {curation_benefit:+.1f} percentage points")

effort_change_baseline = history_baseline.iloc[-1]['avg_effort'] - history_baseline.iloc[0]['avg_effort']
effort_change_curated = history_curated.iloc[-1]['avg_effort'] - history_curated.iloc[0]['avg_effort']

print(f"  4. Baseline: Effort changed by {effort_change_baseline:+.3f}")
print(f"  5. Curated: Effort changed by {effort_change_curated:+.3f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n[6] Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. LTR Share Evolution
axes[0, 0].plot(history_baseline['timestep'], history_baseline['ltr_share'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[0, 0].plot(history_curated['timestep'], history_curated['ltr_share'], 
                label='Platform Curation', linewidth=2, color='green', alpha=0.7)
axes[0, 0].axhline(INITIAL_LTR_SHARE, linestyle='--', color='gray', 
                   label=f'Initial ({INITIAL_LTR_SHARE:.1%})')
axes[0, 0].set_xlabel('Timestep (newcomers arriving)')
axes[0, 0].set_ylabel('LTR Share')
axes[0, 0].set_title('Market LTR Share Over Time', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Average Effort Evolution
axes[0, 1].plot(history_baseline['timestep'], history_baseline['avg_effort'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[0, 1].plot(history_curated['timestep'], history_curated['avg_effort'], 
                label='Platform Curation', linewidth=2, color='green', alpha=0.7)
axes[0, 1].set_xlabel('Timestep (newcomers arriving)')
axes[0, 1].set_ylabel('Average Effort')
axes[0, 1].set_title('Average Effort Over Time', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. LTR User Effort
axes[1, 0].plot(history_baseline['timestep'], history_baseline['ltr_avg_effort'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[1, 0].plot(history_curated['timestep'], history_curated['ltr_avg_effort'], 
                label='Platform Curation', linewidth=2, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Timestep (newcomers arriving)')
axes[1, 0].set_ylabel('LTR User Effort')
axes[1, 0].set_title('LTR User Effort Over Time', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Market Size
axes[1, 1].plot(history_baseline['timestep'], history_baseline['market_size'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[1, 1].plot(history_curated['timestep'], history_curated['market_size'], 
                label='Platform Curation', linewidth=2, color='green', alpha=0.7)
axes[1, 1].axhline(INITIAL_MARKET_SIZE, linestyle='--', color='gray', 
                   label=f'Initial ({INITIAL_MARKET_SIZE})')
axes[1, 1].set_xlabel('Timestep (newcomers arriving)')
axes[1, 1].set_ylabel('Active Users')
axes[1, 1].set_title('Market Size Over Time', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_theory_demonstration.png', dpi=150, bbox_inches='tight')
print("✓ Saved: simulation_theory_demonstration.png")

# Save data
history_baseline['scenario'] = 'baseline'
history_curated['scenario'] = 'curated'
combined = pd.concat([history_baseline, history_curated])
combined.to_csv('simulation_results.csv', index=False)
print("✓ Saved: simulation_results.csv")

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print("\nThis demonstrates:")
print("  1. Bayesian learning dynamics with realistic parameters from YOUR data")
print("  2. How frustration can affect market composition")
print("  3. Potential impact of platform design (curation)")
print("\nNo comparison to real data - pure theory demonstration")
print("="*80)
