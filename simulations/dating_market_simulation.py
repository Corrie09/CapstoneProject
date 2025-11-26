"""
Dating Market Simulation with Bayesian Learning and Frustration
Based on Capstone Proposal Framework

This simulates:
1. Newcomers arriving with prior beliefs
2. Seeing K profiles → updating beliefs (Bayesian)
3. Choosing effort based on beliefs
4. Market composition evolving over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PARAMETERS (from your proposal)
# ============================================================================

K = 20  # Number of profiles shown to newcomer
PRIOR_RHO = 0.5  # Prior belief about LTR share
PRIOR_P = 0.3  # Prior belief about success probability

# Frustration threshold
FRUSTRATION_THRESHOLD_RHO = 0.4  # If belief about LTR < 40%
FRUSTRATION_THRESHOLD_P = 0.3    # If belief about success < 30%

# Effort levels
HIGH_EFFORT = 0.9
LOW_EFFORT = 0.5

# Beta prior parameters for Bayesian updating
ALPHA_RHO = 5  # Prior for market composition (mean = 5/(5+5) = 0.5)
BETA_RHO = 5

ALPHA_P = 3    # Prior for success probability (mean = 3/(3+7) = 0.3)
BETA_P = 7

print("="*80)
print("DATING MARKET SIMULATION")
print("="*80)
print(f"\nParameters:")
print(f"  K (profiles shown): {K}")
print(f"  Prior belief ρ_m: {PRIOR_RHO}")
print(f"  Prior belief p_i: {PRIOR_P}")
print(f"  Frustration threshold: ρ < {FRUSTRATION_THRESHOLD_RHO} AND p < {FRUSTRATION_THRESHOLD_P}")

# ============================================================================
# USER CLASS
# ============================================================================

@dataclass
class User:
    """Represents a user in the dating market"""
    user_id: int
    rating: float      # r_i ∈ [0,1]
    goal: str          # 'LTR' or 'casual'
    effort: float      # Effort level [0,1]
    is_active: bool    # Still in market or exited
    rho_belief: float  # Belief about market composition
    p_belief: float    # Belief about success probability
    
    def __repr__(self):
        return f"User(id={self.user_id}, r={self.rating:.2f}, goal={self.goal}, effort={self.effort:.2f})"

# ============================================================================
# BAYESIAN UPDATING FUNCTIONS
# ============================================================================

def bayesian_update_rho(n_ltr_seen: int, n_total_seen: int) -> float:
    """
    Update belief about market LTR share using Beta-Binomial conjugate prior
    
    EXPLANATION:
    - User starts with prior: ρ ~ Beta(5, 5) → mean = 0.5
    - Sees n_ltr_seen LTR profiles out of n_total_seen
    - Updates to posterior: ρ ~ Beta(5 + n_ltr_seen, 5 + n_total_seen - n_ltr_seen)
    - Returns posterior mean
    
    Example: If see 8 LTR out of 20:
      posterior mean = (5 + 8) / (5 + 5 + 20) = 13/30 = 0.433
    """
    alpha_post = ALPHA_RHO + n_ltr_seen
    beta_post = BETA_RHO + (n_total_seen - n_ltr_seen)
    
    rho_hat = alpha_post / (alpha_post + beta_post)
    
    return rho_hat


def bayesian_update_p(n_success: int, n_attempts: int) -> float:
    """
    Update belief about own success probability
    
    EXPLANATION:
    - User starts with prior: p ~ Beta(3, 7) → mean = 0.3
    - Attempts n_attempts matches, gets n_success acceptances
    - Updates to posterior
    - Returns posterior mean
    
    Example: If 1 success out of 5 attempts:
      posterior mean = (3 + 1) / (3 + 7 + 5) = 4/15 = 0.267 (lower than prior!)
    """
    alpha_post = ALPHA_P + n_success
    beta_post = BETA_P + (n_attempts - n_success)
    
    p_hat = alpha_post / (alpha_post + beta_post)
    
    return p_hat

# ============================================================================
# EFFORT CHOICE FUNCTION
# ============================================================================

def choose_effort(rho_hat: float, p_hat: float, user_goal: str, user_rating: float) -> Tuple[float, bool]:
    """
    User chooses effort based on updated beliefs
    
    EXPLANATION (implements the formula from proposal page 2):
    User maximizes: E[utility | beliefs] - Cost(effort)
    
    Decision rule:
    - If frustrated (both beliefs low) → LOW effort or EXIT
    - If optimistic → HIGH effort
    - Casual users always moderate effort
    
    Returns: (effort_level, should_exit)
    """
    
    # Casual users don't care as much about market composition
    if user_goal == 'casual':
        return 0.7, False
    
    # LTR users care about both market composition AND success probability
    is_frustrated = (rho_hat < FRUSTRATION_THRESHOLD_RHO) and (p_hat < FRUSTRATION_THRESHOLD_P)
    
    if is_frustrated:
        # Very frustrated → might EXIT
        # High-rated users more likely to exit (have better outside options)
        exit_probability = 0.3 * user_rating  # Higher rating → more likely to exit
        
        if np.random.rand() < exit_probability:
            return 0.0, True  # EXIT
        else:
            return LOW_EFFORT, False  # Stay but low effort
    
    elif rho_hat < FRUSTRATION_THRESHOLD_RHO or p_hat < FRUSTRATION_THRESHOLD_P:
        # Somewhat frustrated → medium effort
        return 0.7, False
    
    else:
        # Optimistic → high effort
        return HIGH_EFFORT, False

# ============================================================================
# MARKET CLASS
# ============================================================================

class DatingMarket:
    """Represents the dating market with existing users"""
    
    def __init__(self, initial_ltr_share: float, initial_size: int):
        """
        Initialize market with existing users
        
        EXPLANATION:
        - Start with some existing user pool
        - initial_ltr_share: fraction of LTR users (e.g., 0.6 = 60% LTR)
        - initial_size: number of initial users (e.g., 1000)
        """
        self.users: List[User] = []
        self.next_user_id = 0
        
        # Create initial population
        for i in range(initial_size):
            rating = np.random.beta(5, 3)  # Skewed toward higher ratings initially
            is_ltr = np.random.rand() < initial_ltr_share
            goal = 'LTR' if is_ltr else 'casual'
            
            # Initial users have randomly assigned effort
            effort = np.random.uniform(0.5, 0.9) if is_ltr else np.random.uniform(0.6, 0.8)
            
            user = User(
                user_id=self.next_user_id,
                rating=rating,
                goal=goal,
                effort=effort,
                is_active=True,
                rho_belief=initial_ltr_share,
                p_belief=0.3
            )
            
            self.users.append(user)
            self.next_user_id += 1
    
    def get_active_users(self) -> List[User]:
        """Return list of users still active in market"""
        return [u for u in self.users if u.is_active]
    
    def sample_profiles(self, k: int) -> List[User]:
        """
        Sample K profiles to show to newcomer
        
        EXPLANATION:
        - Randomly sample from active users
        - This is what newcomer SEES to form beliefs
        """
        active = self.get_active_users()
        if len(active) < k:
            return active
        return list(np.random.choice(active, size=k, replace=False))
    
    def get_market_composition(self) -> dict:
        """Calculate current market statistics"""
        active = self.get_active_users()
        
        if len(active) == 0:
            return {'ltr_share': 0, 'avg_effort': 0, 'avg_rating': 0, 'size': 0}
        
        ltr_users = [u for u in active if u.goal == 'LTR']
        
        return {
            'ltr_share': len(ltr_users) / len(active),
            'avg_effort': np.mean([u.effort for u in active]),
            'avg_rating': np.mean([u.rating for u in active]),
            'size': len(active),
            'ltr_avg_effort': np.mean([u.effort for u in ltr_users]) if ltr_users else 0
        }
    
    def simulate_matching_attempts(self, user_rating: float, n_attempts: int = 5) -> int:
        """
        Simulate user trying to match with n_attempts partners
        
        EXPLANATION:
        - User reaches out to n_attempts people
        - Success depends on user's rating (higher rating → higher success)
        - Returns number of successful matches
        """
        # Base success probability depends on rating
        # Higher rated users have better success
        base_p = 0.1 + 0.3 * user_rating  # Range: 0.1 to 0.4
        
        # Bernoulli trials
        successes = np.random.binomial(n_attempts, base_p)
        
        return successes
    
    def add_newcomer(self, newcomer: User):
        """Add newcomer to market if they didn't exit"""
        if newcomer.is_active:
            self.users.append(newcomer)

# ============================================================================
# SIMULATION FUNCTION
# ============================================================================

def run_simulation(
    initial_ltr_share: float,
    initial_size: int,
    n_newcomers: int,
    curated_sampling: bool = False,
    verbose: bool = True
) -> Tuple[DatingMarket, pd.DataFrame]:
    """
    Run full simulation of newcomers arriving over time
    
    EXPLANATION:
    - Start with initial market
    - Newcomers arrive one by one
    - Each sees K profiles → updates beliefs → chooses effort
    - Market composition evolves
    
    curated_sampling: If True, platform shows more LTR profiles (counterfactual)
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running simulation:")
        print(f"  Initial LTR share: {initial_ltr_share:.1%}")
        print(f"  Initial market size: {initial_size}")
        print(f"  Number of newcomers: {n_newcomers}")
        print(f"  Curated sampling: {curated_sampling}")
        print(f"{'='*80}\n")
    
    # Initialize market
    market = DatingMarket(initial_ltr_share, initial_size)
    
    # Track statistics over time
    history = []
    
    # Initial state
    comp = market.get_market_composition()
    history.append({
        'timestep': 0,
        'ltr_share': comp['ltr_share'],
        'avg_effort': comp['avg_effort'],
        'avg_rating': comp['avg_rating'],
        'market_size': comp['size'],
        'ltr_avg_effort': comp['ltr_avg_effort']
    })
    
    # Simulate newcomers arriving
    for t in range(1, n_newcomers + 1):
        
        # Generate newcomer characteristics
        rating = np.random.beta(4, 4)  # Centered around 0.5
        is_ltr = np.random.rand() < 0.55  # 55% want LTR (realistic)
        goal = 'LTR' if is_ltr else 'casual'
        
        # STEP 1: Sample K profiles for newcomer to see
        if curated_sampling and goal == 'LTR':
            # COUNTERFACTUAL: Platform shows more LTR profiles
            # Show 70% LTR instead of true market composition
            active = market.get_active_users()
            ltr_users = [u for u in active if u.goal == 'LTR']
            casual_users = [u for u in active if u.goal == 'casual']
            
            n_ltr_to_show = min(int(K * 0.7), len(ltr_users))
            n_casual_to_show = K - n_ltr_to_show
            
            if len(ltr_users) >= n_ltr_to_show and len(casual_users) >= n_casual_to_show:
                sample = (list(np.random.choice(ltr_users, n_ltr_to_show, replace=False)) +
                         list(np.random.choice(casual_users, n_casual_to_show, replace=False)))
            else:
                sample = market.sample_profiles(K)
        else:
            # BASELINE: Random sampling
            sample = market.sample_profiles(K)
        
        # STEP 2: Count LTR profiles in sample
        n_ltr_seen = sum(1 for u in sample if u.goal == 'LTR')
        
        # STEP 3: Bayesian update of market belief
        rho_hat = bayesian_update_rho(n_ltr_seen, len(sample))
        
        # STEP 4: Simulate matching attempts
        n_attempts = 5
        n_success = market.simulate_matching_attempts(rating, n_attempts)
        
        # STEP 5: Bayesian update of success belief
        p_hat = bayesian_update_p(n_success, n_attempts)
        
        # STEP 6: Choose effort based on beliefs
        effort, should_exit = choose_effort(rho_hat, p_hat, goal, rating)
        
        # STEP 7: Create user
        newcomer = User(
            user_id=market.next_user_id,
            rating=rating,
            goal=goal,
            effort=effort,
            is_active=not should_exit,
            rho_belief=rho_hat,
            p_belief=p_hat
        )
        market.next_user_id += 1
        
        # STEP 8: Add to market (if didn't exit)
        market.add_newcomer(newcomer)
        
        # Track statistics every 10 timesteps
        if t % 10 == 0:
            comp = market.get_market_composition()
            history.append({
                'timestep': t,
                'ltr_share': comp['ltr_share'],
                'avg_effort': comp['avg_effort'],
                'avg_rating': comp['avg_rating'],
                'market_size': comp['size'],
                'ltr_avg_effort': comp['ltr_avg_effort']
            })
            
            if verbose and t % 50 == 0:
                print(f"t={t:4d}: LTR={comp['ltr_share']:.1%}, "
                      f"Effort={comp['avg_effort']:.3f}, "
                      f"Size={comp['size']}")
    
    # Final state
    comp = market.get_market_composition()
    history.append({
        'timestep': n_newcomers,
        'ltr_share': comp['ltr_share'],
        'avg_effort': comp['avg_effort'],
        'avg_rating': comp['avg_rating'],
        'market_size': comp['size'],
        'ltr_avg_effort': comp['ltr_avg_effort']
    })
    
    if verbose:
        print(f"\nFinal state:")
        print(f"  LTR share: {comp['ltr_share']:.1%}")
        print(f"  Avg effort: {comp['avg_effort']:.3f}")
        print(f"  LTR avg effort: {comp['ltr_avg_effort']:.3f}")
        print(f"  Market size: {comp['size']}")
    
    return market, pd.DataFrame(history)

# ============================================================================
# RUN SIMULATIONS
# ============================================================================

print("\n" + "="*80)
print("SCENARIO 1: BASELINE (Random Sampling)")
print("="*80)

market_baseline, history_baseline = run_simulation(
    initial_ltr_share=0.60,  # Start with 60% LTR
    initial_size=1000,
    n_newcomers=500,
    curated_sampling=False,
    verbose=True
)

print("\n" + "="*80)
print("SCENARIO 2: COUNTERFACTUAL (Curated Sampling)")
print("="*80)

market_curated, history_curated = run_simulation(
    initial_ltr_share=0.60,  # Same starting point
    initial_size=1000,
    n_newcomers=500,
    curated_sampling=True,  # Platform shows more LTR profiles
    verbose=True
)

# ============================================================================
# COMPARE SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: Baseline vs Curated")
print("="*80)

final_baseline = history_baseline.iloc[-1]
final_curated = history_curated.iloc[-1]

print("\nFinal LTR Share:")
print(f"  Baseline:  {final_baseline['ltr_share']:.1%}")
print(f"  Curated:   {final_curated['ltr_share']:.1%}")
print(f"  Difference: {(final_curated['ltr_share'] - final_baseline['ltr_share'])*100:+.1f} percentage points")

print("\nFinal Average Effort:")
print(f"  Baseline:  {final_baseline['avg_effort']:.3f}")
print(f"  Curated:   {final_curated['avg_effort']:.3f}")
print(f"  Difference: {(final_curated['avg_effort'] - final_baseline['avg_effort']):.3f}")

print("\nFinal LTR User Effort:")
print(f"  Baseline:  {final_baseline['ltr_avg_effort']:.3f}")
print(f"  Curated:   {final_curated['ltr_avg_effort']:.3f}")
print(f"  Difference: {(final_curated['ltr_avg_effort'] - final_baseline['ltr_avg_effort']):.3f}")

print("\nFinal Market Size:")
print(f"  Baseline:  {final_baseline['market_size']:.0f}")
print(f"  Curated:   {final_curated['market_size']:.0f}")
print(f"  Difference: {(final_curated['market_size'] - final_baseline['market_size']):.0f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("Creating visualizations...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. LTR Share over time
axes[0, 0].plot(history_baseline['timestep'], history_baseline['ltr_share'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[0, 0].plot(history_curated['timestep'], history_curated['ltr_share'], 
                label='Curated', linewidth=2, color='green', alpha=0.7)
axes[0, 0].axhline(0.60, linestyle='--', color='gray', label='Initial (60%)')
axes[0, 0].set_xlabel('Timestep (newcomers)')
axes[0, 0].set_ylabel('LTR Share')
axes[0, 0].set_title('Market LTR Share Over Time', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Average Effort over time
axes[0, 1].plot(history_baseline['timestep'], history_baseline['avg_effort'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[0, 1].plot(history_curated['timestep'], history_curated['avg_effort'], 
                label='Curated', linewidth=2, color='green', alpha=0.7)
axes[0, 1].set_xlabel('Timestep (newcomers)')
axes[0, 1].set_ylabel('Average Effort')
axes[0, 1].set_title('Average Effort Over Time', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. LTR User Effort over time
axes[1, 0].plot(history_baseline['timestep'], history_baseline['ltr_avg_effort'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[1, 0].plot(history_curated['timestep'], history_curated['ltr_avg_effort'], 
                label='Curated', linewidth=2, color='green', alpha=0.7)
axes[1, 0].set_xlabel('Timestep (newcomers)')
axes[1, 0].set_ylabel('LTR User Effort')
axes[1, 0].set_title('LTR User Effort Over Time', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 4. Market Size over time
axes[1, 1].plot(history_baseline['timestep'], history_baseline['market_size'], 
                label='Baseline', linewidth=2, color='red', alpha=0.7)
axes[1, 1].plot(history_curated['timestep'], history_curated['market_size'], 
                label='Curated', linewidth=2, color='green', alpha=0.7)
axes[1, 1].axhline(1000, linestyle='--', color='gray', label='Initial (1000)')
axes[1, 1].set_xlabel('Timestep (newcomers)')
axes[1, 1].set_ylabel('Market Size')
axes[1, 1].set_title('Market Size Over Time', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('simulation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: simulation_results.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

history_baseline['scenario'] = 'baseline'
history_curated['scenario'] = 'curated'
combined_history = pd.concat([history_baseline, history_curated])
combined_history.to_csv('simulation_history.csv', index=False)
print("✓ Saved: simulation_history.csv")

print("\n" + "="*80)
print("SIMULATION COMPLETE!")
print("="*80)
print("\nKey Insights:")
print("1. Does frustration cause market unraveling? (Check if LTR share drops)")
print("2. Does curated sampling help? (Compare final states)")
print("3. What's the effect size? (Quantify differences)")
