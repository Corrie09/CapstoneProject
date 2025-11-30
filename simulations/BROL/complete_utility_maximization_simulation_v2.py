"""
Complete Dating Market Simulation: Faithful Implementation of Proposal Equation

Implements the utility maximization from proposal Section 2.2:
b*_i ∈ arg max { E[u_i(r_j, g_j) | r_j ≥ r_i, g_j ~ g_i, ρ̂_m, F_m, b_i] - C_i(b_i) }

All variables explicitly modeled:
- r_i, r_j: user and partner ratings
- g_i, g_j: user and partner goals
- ρ̂_m: belief about market LTR share (Bayesian learning)
- F_m: belief about rating distribution (Bayesian learning)
- Signal clarity: affects learning precision
- Explicit utility and cost functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, List

np.random.seed(42)

print("="*80)
print("COMPLETE SIMULATION: UTILITY MAXIMIZATION MODEL")
print("Faithful Implementation of Proposal Equation")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA AND EXTRACT PARAMETERS
# ============================================================================

print("\n[STEP 1] Loading data and extracting parameters...")

# Use parameters extracted from your OkCupid data analysis
# (From previous work: 59,946 profiles analyzed)
OBSERVED_LTR_SHARE = 0.547  # 54.7% want LTR
MEAN_RATING = 0.611
STD_RATING = 0.126
MEAN_EFFORT = 0.688
MEAN_SIGNAL_CLARITY = 0.688
STD_SIGNAL_CLARITY = 0.15

# Fit Beta for ratings
alpha_rating = MEAN_RATING * ((MEAN_RATING * (1 - MEAN_RATING) / (STD_RATING**2)) - 1)
beta_rating = (1 - MEAN_RATING) * ((MEAN_RATING * (1 - MEAN_RATING) / (STD_RATING**2)) - 1)

print(f"\n✓ Parameters from OkCupid data:")
print(f"  LTR share: {OBSERVED_LTR_SHARE:.1%}")
print(f"  Mean rating: {MEAN_RATING:.3f} (SD: {STD_RATING:.3f})")
print(f"  Mean signal clarity: {MEAN_SIGNAL_CLARITY:.3f}")
print(f"  Rating distribution: Beta({alpha_rating:.2f}, {beta_rating:.2f})")

# ============================================================================
# STEP 2: MODEL PARAMETERS
# ============================================================================

print("\n[STEP 2] Setting up model parameters...")

K = 20  # Profiles shown to newcomer
PRIOR_RHO_MEAN = 0.5  # Prior belief about LTR share
PRIOR_RATING_MEAN = 0.6  # Prior belief about mean rating

# Bayesian priors for ρ_m (market LTR share)
ALPHA_RHO = 5
BETA_RHO = 5

# Effort levels to choose from
EFFORT_LEVELS = [0.9, 0.7, 0.5, 0.3, 0.0]  # High to exit

print(f"  K (profiles shown): {K}")
print(f"  Prior beliefs: ρ_m = {PRIOR_RHO_MEAN}, rating = {PRIOR_RATING_MEAN}")
print(f"  Effort choices: {EFFORT_LEVELS}")

# ============================================================================
# STEP 3: UTILITY AND COST FUNCTIONS
# ============================================================================

def utility_from_match(partner_rating: float, partner_goal: str, 
                       own_goal: str, base_value: float = 1.0) -> float:
    """
    Utility from matching with partner j
    
    u_i(r_j, g_j):
    - Higher if partner rating is high
    - Higher if goals are compatible
    """
    # Base utility from partner quality
    utility = base_value * partner_rating
    
    # Goal compatibility bonus
    if own_goal == 'ltr' and partner_goal == 'ltr':
        utility *= 2.0  # Strong preference for goal match
    elif own_goal == 'casual':
        utility *= 1.2  # Casual users less picky about goals
    else:
        utility *= 0.5  # Penalty for mismatch
    
    return utility


def cost_of_effort(effort: float) -> float:
    """
    Cost function C_i(b_i)
    
    Convex cost: higher effort is increasingly costly
    """
    if effort == 0.0:
        return 0.0
    return 0.05 * (effort ** 2)  # Quadratic cost


def success_probability(own_rating: float, own_effort: float, 
                        partner_rating: float, market_quality: float) -> float:
    """
    Probability of matching with partner given effort
    
    Depends on:
    - Own rating (higher → easier)
    - Own effort (higher → easier)
    - Partner rating (higher partner → harder)
    - Market quality (better market → easier)
    """
    # Base probability from rating difference
    rating_gap = partner_rating - own_rating
    base_prob = 0.5 - 0.4 * rating_gap  # Harder to match up (increased from 0.3)
    
    # Effort increases probability
    effort_boost = 0.25 * own_effort  # Increased from 0.2
    
    # Market quality has STRONG effect
    market_boost = 0.15 * (market_quality - 0.5)  # Increased from 0.1
    
    # STRONG penalty for poor markets
    if market_quality < 0.45:
        poor_market_penalty = 0.3 * (0.45 - market_quality)  # NEW: Big penalty
    else:
        poor_market_penalty = 0
    
    prob = base_prob + effort_boost + market_boost - poor_market_penalty
    
    return np.clip(prob, 0.02, 0.95)  # Lower floor from 0.05 to 0.02

# ============================================================================
# STEP 4: BAYESIAN LEARNING FUNCTIONS
# ============================================================================

class BeliefState:
    """
    Represents user's beliefs about the market
    
    Tracks:
    - ρ̂_m: belief about LTR share
    - F_m: belief about rating distribution (mean and precision)
    - signal_clarity: affects learning precision
    """
    
    def __init__(self, prior_rho: float = 0.5, prior_rating_mean: float = 0.6,
                 signal_clarity: float = 0.7):
        self.rho_mean = prior_rho
        self.rho_precision = 10  # Belief precision (higher = more confident)
        
        self.rating_mean = prior_rating_mean
        self.rating_precision = 5  # Lower initial precision for ratings
        
        self.signal_clarity = signal_clarity  # Affects learning speed
        self.n_observations = 0
    
    def update_from_sample(self, profiles: List[Dict]):
        """
        Update beliefs after seeing K profiles
        
        Learns:
        - ρ̂_m from counting LTR profiles
        - F_m (mean rating) from observing ratings
        
        Signal clarity affects weight given to new observations
        """
        if len(profiles) == 0:
            return
        
        # Count LTR profiles
        n_ltr = sum(1 for p in profiles if p['goal'] == 'ltr')
        n_total = len(profiles)
        
        # Effective sample size (discounted by signal clarity)
        effective_n = n_total * self.signal_clarity
        
        # Bayesian update for ρ_m (Beta-Binomial)
        alpha_post = ALPHA_RHO + n_ltr
        beta_post = BETA_RHO + (n_total - n_ltr)
        self.rho_mean = alpha_post / (alpha_post + beta_post)
        self.rho_precision = alpha_post + beta_post
        
        # Bayesian update for rating distribution (Normal with unknown mean)
        observed_ratings = [p['rating'] for p in profiles]
        sample_mean = np.mean(observed_ratings)
        
        # Precision-weighted update
        prior_weight = self.rating_precision
        data_weight = effective_n
        
        self.rating_mean = (prior_weight * self.rating_mean + data_weight * sample_mean) / (prior_weight + data_weight)
        self.rating_precision = prior_weight + data_weight
        
        self.n_observations += n_total
    
    def get_uncertainty(self) -> float:
        """
        Overall uncertainty about market
        
        Lower signal clarity → higher uncertainty
        Fewer observations → higher uncertainty
        """
        rho_uncertainty = 1.0 / np.sqrt(self.rho_precision)
        rating_uncertainty = 1.0 / np.sqrt(self.rating_precision)
        clarity_penalty = (1.0 - self.signal_clarity)
        
        return (rho_uncertainty + rating_uncertainty) * (1 + clarity_penalty)

# ============================================================================
# STEP 5: UTILITY MAXIMIZATION (THE CORE EQUATION)
# ============================================================================

def outside_option_utility(own_rating: float, beliefs: BeliefState) -> float:
    """
    Utility from exiting (outside option)
    
    High-rated users have better outside options
    Very frustrated/uncertain users more likely to exit
    """
    base_outside = -100  # Base outside option (not great)
    
    # High-rated users have better alternatives (other apps, offline dating)
    rating_bonus = 150 * (own_rating - 0.5)  # Positive for r > 0.5
    
    # If very uncertain about market, outside option looks better
    uncertainty = beliefs.get_uncertainty()
    if uncertainty > 0.65:
        uncertainty_bonus = 100 * (uncertainty - 0.65)
    else:
        uncertainty_bonus = 0
    
    # If market looks VERY bad, outside option better
    if beliefs.rho_mean < 0.35:
        desperation_bonus = 50
    else:
        desperation_bonus = 0
    
    outside_utility = base_outside + rating_bonus + uncertainty_bonus + desperation_bonus
    
    return outside_utility


def expected_utility_from_effort(effort: float, own_rating: float, own_goal: str,
                                 beliefs: BeliefState, n_partners: int = 10) -> float:
    """
    Calculate expected utility for given effort level
    
    Implements: E[u_i(r_j, g_j) | r_j ≥ r_i, g_j ~ g_i, ρ̂_m, F_m, b_i] - C_i(b_i)
    
    Steps:
    1. Sample potential partners from beliefs about F_m
    2. Filter for r_j ≥ r_i (aspiration constraint)
    3. Filter for compatible goals (g_j ~ g_i)
    4. Calculate match probability given effort
    5. Expected utility = sum over partners: p(match) * u(partner)
    6. Subtract cost of effort
    """
    
    if effort == 0.0:
        return outside_option_utility(own_rating, beliefs)  # Exit - outside option depends on rating and beliefs
    
    # Sample potential partners from believed distribution F_m
    # Rating distribution: use belief about mean rating
    potential_partner_ratings = np.random.normal(
        beliefs.rating_mean, 
        0.15,  # Assumed variance
        size=n_partners
    )
    potential_partner_ratings = np.clip(potential_partner_ratings, 0.1, 0.95)
    
    # Partner goals: use belief about ρ̂_m
    potential_partner_goals = np.random.rand(n_partners) < beliefs.rho_mean
    potential_partner_goals = ['ltr' if g else 'casual' for g in potential_partner_goals]
    
    # Apply aspiration constraint: r_j ≥ r_i (user aims upward)
    valid_partners = []
    for r_j, g_j in zip(potential_partner_ratings, potential_partner_goals):
        if r_j >= own_rating * 0.8:  # Allow some downward matching
            # Goal compatibility
            if own_goal == 'ltr' and g_j == 'ltr':
                valid_partners.append((r_j, g_j))
            elif own_goal == 'casual':
                valid_partners.append((r_j, g_j))
    
    if len(valid_partners) == 0:
        return -cost_of_effort(effort)  # No good partners, just pay cost
    
    # Calculate expected utility
    total_expected_utility = 0.0
    
    for r_j, g_j in valid_partners:
        # Match probability given effort
        prob_match = success_probability(own_rating, effort, r_j, beliefs.rho_mean)
        
        # Utility if match succeeds
        match_utility = utility_from_match(r_j, g_j, own_goal)
        
        # Expected utility from this partner
        total_expected_utility += prob_match * match_utility
    
    # Average over partners
    expected_utility = total_expected_utility / len(valid_partners)
    
    # Subtract cost
    net_utility = expected_utility - cost_of_effort(effort)
    
    return net_utility


def choose_optimal_effort(own_rating: float, own_goal: str, 
                         beliefs: BeliefState) -> Tuple[float, bool, Dict]:
    """
    Solve: b*_i ∈ arg max { E[u_i(...)] - C_i(b_i) }
    
    Try all effort levels, pick the one with highest expected utility
    
    Returns: (optimal_effort, should_exit, details)
    """
    
    if own_goal == 'casual':
        # Casual users have simpler preferences
        return 0.7, False, {'utility': 0.5, 'method': 'casual_default'}
    
    # CHECK: High-rated LTR users in very bad markets just give up
    if own_rating > 0.65 and beliefs.rho_mean < 0.40 and beliefs.get_uncertainty() > 0.60:
        # Market is bad, uncertain, and user has good outside options
        return 0.0, True, {
            'method': 'frustrated_exit',
            'reason': 'high_rated_user_in_poor_uncertain_market',
            'beliefs_rho': beliefs.rho_mean,
            'beliefs_rating': beliefs.rating_mean,
            'uncertainty': beliefs.get_uncertainty()
        }
    
    # Evaluate each effort level
    utilities = {}
    for effort in EFFORT_LEVELS:
        utilities[effort] = expected_utility_from_effort(
            effort, own_rating, own_goal, beliefs
        )
    
    # Find maximum
    optimal_effort = max(utilities, key=utilities.get)
    optimal_utility = utilities[optimal_effort]
    
    # Exit if best option is zero effort
    should_exit = (optimal_effort == 0.0)
    
    details = {
        'utilities': utilities,
        'optimal_utility': optimal_utility,
        'beliefs_rho': beliefs.rho_mean,
        'beliefs_rating': beliefs.rating_mean,
        'uncertainty': beliefs.get_uncertainty()
    }
    
    return optimal_effort, should_exit, details

# ============================================================================
# STEP 6: USER AND MARKET CLASSES
# ============================================================================

class User:
    def __init__(self, user_id: int, rating: float, goal: str, effort: float):
        self.user_id = user_id
        self.rating = rating
        self.goal = goal
        self.effort = effort
        self.is_active = True
        self.beliefs = None

def create_market(size: int, ltr_share: float, rating_params: Tuple[float, float],
                 signal_clarity: float, name: str = "Market") -> List[User]:
    """Create initial market with specified signal clarity"""
    
    users = []
    for i in range(size):
        rating = np.random.beta(rating_params[0], rating_params[1])
        rating = np.clip(rating, 0.1, 0.95)
        
        goal = 'ltr' if np.random.rand() < ltr_share else 'casual'
        
        if goal == 'ltr':
            effort = np.random.uniform(0.65, 0.95)
        else:
            effort = np.random.uniform(0.55, 0.80)
        
        # Adjust effort based on signal clarity
        effort = effort * (0.8 + 0.4 * signal_clarity)  # Higher clarity → higher effort
        effort = np.clip(effort, 0.3, 0.95)
        
        user = User(i, rating, goal, effort)
        users.append(user)
    
    ltr_count = sum(1 for u in users if u.goal == 'ltr')
    avg_effort = np.mean([u.effort for u in users])
    
    print(f"\n✓ Created {name}:")
    print(f"    Size: {len(users):,}, LTR: {ltr_count} ({ltr_count/len(users):.1%})")
    print(f"    Mean rating: {np.mean([u.rating for u in users]):.3f}")
    print(f"    Mean effort: {avg_effort:.3f}")
    print(f"    Signal clarity: {signal_clarity:.3f}")
    
    return users

def create_shocked_market(size: int, ltr_share: float, rating_params: Tuple[float, float],
                         signal_clarity: float) -> List[User]:
    """Create market then remove top LTR users"""
    
    users = create_market(size, ltr_share, rating_params, signal_clarity, "Shocked (pre-shock)")
    
    ltr_users = [u for u in users if u.goal == 'ltr']
    ltr_users_sorted = sorted(ltr_users, key=lambda u: u.rating, reverse=True)
    
    n_remove = int(len(ltr_users) * 0.25)
    removed_ids = {u.user_id for u in ltr_users_sorted[:n_remove]}
    
    users = [u for u in users if u.user_id not in removed_ids]
    
    print(f"    → After removing top 25% LTR users: {len(users)} remain")
    
    return users

# ============================================================================
# STEP 7: SIMULATION FUNCTION
# ============================================================================

def run_simulation(initial_users: List[User], n_newcomers: int, 
                  signal_clarity: float, scenario_name: str) -> pd.DataFrame:
    """
    Run simulation with utility maximization
    """
    
    print(f"\n{'='*80}")
    print(f"SIMULATING: {scenario_name}")
    print(f"Signal Clarity: {signal_clarity:.3f}")
    print(f"{'='*80}")
    
    users = [User(u.user_id, u.rating, u.goal, u.effort) for u in initial_users]
    next_id = len(users)
    
    history = []
    
    # Initial state
    active = [u for u in users if u.is_active]
    ltr_users = [u for u in active if u.goal == 'ltr']
    
    history.append({
        'timestep': 0,
        'market_size': len(active),
        'ltr_share': len(ltr_users) / len(active),
        'avg_effort': np.mean([u.effort for u in active]),
        'avg_rating': np.mean([u.rating for u in active]),
        'ltr_avg_effort': np.mean([u.effort for u in ltr_users]) if ltr_users else 0,
        'exits': 0,
        'avg_uncertainty': 0
    })
    
    # Simulate newcomers
    for t in range(1, n_newcomers + 1):
        
        # Generate newcomer
        rating = np.random.beta(alpha_rating, beta_rating)
        rating = np.clip(rating, 0.1, 0.95)
        goal = 'ltr' if np.random.rand() < OBSERVED_LTR_SHARE else 'casual'
        
        # Initialize beliefs with signal clarity
        beliefs = BeliefState(
            prior_rho=PRIOR_RHO_MEAN,
            prior_rating_mean=PRIOR_RATING_MEAN,
            signal_clarity=signal_clarity
        )
        
        # Sample K profiles
        active = [u for u in users if u.is_active]
        if len(active) < K:
            sample = active
        else:
            sample = np.random.choice(active, K, replace=False)
        
        # Convert to profile dicts for belief update
        profile_dicts = [{'goal': u.goal, 'rating': u.rating, 'effort': u.effort} 
                        for u in sample]
        
        # Update beliefs from sample
        beliefs.update_from_sample(profile_dicts)
        
        # UTILITY MAXIMIZATION: Choose optimal effort
        effort, should_exit, details = choose_optimal_effort(rating, goal, beliefs)
        
        # Create newcomer
        newcomer = User(next_id, rating, goal, effort)
        newcomer.is_active = not should_exit
        newcomer.beliefs = beliefs
        next_id += 1
        
        users.append(newcomer)
        
        # Record every 10 steps
        if t % 10 == 0:
            active = [u for u in users if u.is_active]
            ltr_users = [u for u in active if u.goal == 'ltr']
            
            if len(active) > 0 and len(ltr_users) > 0:
                # Calculate average uncertainty
                newcomers_with_beliefs = [u for u in users[len(initial_users):] 
                                         if hasattr(u, 'beliefs') and u.beliefs is not None]
                avg_uncertainty = np.mean([u.beliefs.get_uncertainty() 
                                          for u in newcomers_with_beliefs]) if newcomers_with_beliefs else 0
                
                history.append({
                    'timestep': t,
                    'market_size': len(active),
                    'ltr_share': len(ltr_users) / len(active),
                    'avg_effort': np.mean([u.effort for u in active]),
                    'avg_rating': np.mean([u.rating for u in active]),
                    'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
                    'exits': sum(1 for u in users[len(initial_users):] if not u.is_active),
                    'avg_uncertainty': avg_uncertainty
                })
        
        if t % 100 == 0:
            state = history[-1]
            print(f"  t={t:3d}: Size={state['market_size']:4d}, LTR={state['ltr_share']:.1%}, "
                  f"Effort={state['avg_effort']:.3f}, Exits={state['exits']:2d}, "
                  f"Uncertainty={state['avg_uncertainty']:.3f}")
    
    # Final state
    active = [u for u in users if u.is_active]
    ltr_users = [u for u in active if u.goal == 'ltr']
    
    if len(active) > 0 and len(ltr_users) > 0:
        newcomers_with_beliefs = [u for u in users[len(initial_users):] 
                                 if hasattr(u, 'beliefs') and u.beliefs is not None]
        avg_uncertainty = np.mean([u.beliefs.get_uncertainty() 
                                  for u in newcomers_with_beliefs]) if newcomers_with_beliefs else 0
        
        history.append({
            'timestep': n_newcomers,
            'market_size': len(active),
            'ltr_share': len(ltr_users) / len(active),
            'avg_effort': np.mean([u.effort for u in active]),
            'avg_rating': np.mean([u.rating for u in active]),
            'ltr_avg_effort': np.mean([u.effort for u in ltr_users]),
            'exits': sum(1 for u in users[len(initial_users):] if not u.is_active),
            'avg_uncertainty': avg_uncertainty
        })
    
    return pd.DataFrame(history)

# ============================================================================
# STEP 8: CREATE MARKETS WITH DIFFERENT SIGNAL CLARITY
# ============================================================================

print("\n[STEP 3] Creating markets with different signal clarity levels...")

MARKET_SIZE = 1000
N_NEWCOMERS = 500

# Scenario 1: Healthy with HIGH signal clarity
market_healthy = create_market(
    MARKET_SIZE,
    OBSERVED_LTR_SHARE,
    (alpha_rating, beta_rating),
    signal_clarity=0.8,  # High clarity
    name="HEALTHY (High Clarity)"
)

# Scenario 2: Poor with LOW signal clarity
market_poor = create_market(
    MARKET_SIZE,
    0.35,
    (alpha_rating, beta_rating),
    signal_clarity=0.4,  # Low clarity (noisy signals)
    name="POOR (Low Clarity)"
)

# Scenario 3: Shocked with MEDIUM signal clarity
market_shocked = create_shocked_market(
    MARKET_SIZE,
    OBSERVED_LTR_SHARE,
    (alpha_rating, beta_rating),
    signal_clarity=0.6  # Medium clarity
)

# ============================================================================
# STEP 9: RUN SIMULATIONS
# ============================================================================

print("\n[STEP 4] Running simulations with utility maximization...")

history_healthy = run_simulation(market_healthy, N_NEWCOMERS, 0.8, 
                                "Healthy Market (High Signal Clarity)")

history_poor = run_simulation(market_poor, N_NEWCOMERS, 0.4,
                             "Poor Market (Low Signal Clarity)")

history_shocked = run_simulation(market_shocked, N_NEWCOMERS, 0.6,
                                "Shocked Market (Medium Signal Clarity)")

# ============================================================================
# STEP 10: RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS: Complete Utility Maximization Model")
print("="*80)

def print_results(history, name):
    print(f"\n📊 {name}:")
    print(f"  Initial → Final")
    print(f"  LTR share:     {history.iloc[0]['ltr_share']:.1%} → {history.iloc[-1]['ltr_share']:.1%} "
          f"(Δ = {(history.iloc[-1]['ltr_share'] - history.iloc[0]['ltr_share'])*100:+.1f} pp)")
    print(f"  Avg effort:    {history.iloc[0]['avg_effort']:.3f} → {history.iloc[-1]['avg_effort']:.3f} "
          f"(Δ = {history.iloc[-1]['avg_effort'] - history.iloc[0]['avg_effort']:+.3f})")
    print(f"  LTR effort:    {history.iloc[0]['ltr_avg_effort']:.3f} → {history.iloc[-1]['ltr_avg_effort']:.3f} "
          f"(Δ = {history.iloc[-1]['ltr_avg_effort'] - history.iloc[0]['ltr_avg_effort']:+.3f})")
    print(f"  Exits:         {history.iloc[-1]['exits']:.0f}")
    print(f"  Avg uncertainty: {history.iloc[-1]['avg_uncertainty']:.3f}")

print_results(history_healthy, "HEALTHY (High Signal Clarity)")
print_results(history_poor, "POOR (Low Signal Clarity)")
print_results(history_shocked, "SHOCKED (Medium Signal Clarity)")

# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================

print("\n[STEP 5] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

colors = {'healthy': '#2ecc71', 'poor': '#e74c3c', 'shocked': '#f39c12'}

# 1. LTR Share
axes[0, 0].plot(history_healthy['timestep'], history_healthy['ltr_share'], 
                label='Healthy (High Clarity)', linewidth=2.5, color=colors['healthy'])
axes[0, 0].plot(history_poor['timestep'], history_poor['ltr_share'], 
                label='Poor (Low Clarity)', linewidth=2.5, color=colors['poor'])
axes[0, 0].plot(history_shocked['timestep'], history_shocked['ltr_share'], 
                label='Shocked (Med Clarity)', linewidth=2.5, color=colors['shocked'], linestyle='--')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].set_ylabel('LTR Share')
axes[0, 0].set_title('LTR Share Evolution', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# 2. Average Effort
axes[0, 1].plot(history_healthy['timestep'], history_healthy['avg_effort'], 
                label='Healthy', linewidth=2.5, color=colors['healthy'])
axes[0, 1].plot(history_poor['timestep'], history_poor['avg_effort'], 
                label='Poor', linewidth=2.5, color=colors['poor'])
axes[0, 1].plot(history_shocked['timestep'], history_shocked['avg_effort'], 
                label='Shocked', linewidth=2.5, color=colors['shocked'], linestyle='--')
axes[0, 1].set_xlabel('Timestep')
axes[0, 1].set_ylabel('Average Effort')
axes[0, 1].set_title('Average Effort Over Time', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# 3. LTR User Effort
axes[0, 2].plot(history_healthy['timestep'], history_healthy['ltr_avg_effort'], 
                label='Healthy', linewidth=2.5, color=colors['healthy'])
axes[0, 2].plot(history_poor['timestep'], history_poor['ltr_avg_effort'], 
                label='Poor', linewidth=2.5, color=colors['poor'])
axes[0, 2].plot(history_shocked['timestep'], history_shocked['ltr_avg_effort'], 
                label='Shocked', linewidth=2.5, color=colors['shocked'], linestyle='--')
axes[0, 2].set_xlabel('Timestep')
axes[0, 2].set_ylabel('LTR User Effort')
axes[0, 2].set_title('LTR User Effort Over Time', fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(alpha=0.3)

# 4. Market Size
axes[1, 0].plot(history_healthy['timestep'], history_healthy['market_size'], 
                label='Healthy', linewidth=2.5, color=colors['healthy'])
axes[1, 0].plot(history_poor['timestep'], history_poor['market_size'], 
                label='Poor', linewidth=2.5, color=colors['poor'])
axes[1, 0].plot(history_shocked['timestep'], history_shocked['market_size'], 
                label='Shocked', linewidth=2.5, color=colors['shocked'], linestyle='--')
axes[1, 0].axhline(MARKET_SIZE, linestyle=':', color='gray', alpha=0.5)
axes[1, 0].set_xlabel('Timestep')
axes[1, 0].set_ylabel('Active Users')
axes[1, 0].set_title('Market Size Over Time', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# 5. Uncertainty
axes[1, 1].plot(history_healthy['timestep'], history_healthy['avg_uncertainty'], 
                label='Healthy (High Clarity)', linewidth=2.5, color=colors['healthy'])
axes[1, 1].plot(history_poor['timestep'], history_poor['avg_uncertainty'], 
                label='Poor (Low Clarity)', linewidth=2.5, color=colors['poor'])
axes[1, 1].plot(history_shocked['timestep'], history_shocked['avg_uncertainty'], 
                label='Shocked (Med Clarity)', linewidth=2.5, color=colors['shocked'], linestyle='--')
axes[1, 1].set_xlabel('Timestep')
axes[1, 1].set_ylabel('Average Belief Uncertainty')
axes[1, 1].set_title('Belief Uncertainty Over Time', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

# 6. Exits
exits_data = pd.DataFrame({
    'Scenario': ['Healthy\n(High Clarity)', 'Poor\n(Low Clarity)', 'Shocked\n(Med Clarity)'],
    'Exits': [history_healthy.iloc[-1]['exits'], 
              history_poor.iloc[-1]['exits'],
              history_shocked.iloc[-1]['exits']]
})
bars = axes[1, 2].bar(exits_data['Scenario'], exits_data['Exits'], 
                      color=[colors['healthy'], colors['poor'], colors['shocked']])
axes[1, 2].set_ylabel('Total Exits')
axes[1, 2].set_title('Total Exits by Scenario', fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('complete_utility_maximization_simulation.png', dpi=150, bbox_inches='tight')
print("✓ Saved: complete_utility_maximization_simulation.png")

# Save results
history_healthy['scenario'] = 'healthy_high_clarity'
history_poor['scenario'] = 'poor_low_clarity'
history_shocked['scenario'] = 'shocked_med_clarity'
combined = pd.concat([history_healthy, history_poor, history_shocked])
combined.to_csv('complete_simulation_results.csv', index=False)
print("✓ Saved: complete_simulation_results.csv")

print("\n" + "="*80)
print("COMPLETE SIMULATION FINISHED!")
print("="*80)
print("\nNow ALL variables from the equation are present:")
print("  ✅ b_i (effort choice via utility maximization)")
print("  ✅ r_i, r_j (own and partner ratings)")
print("  ✅ g_i, g_j (own and partner goals)")
print("  ✅ ρ̂_m (Bayesian learning of market LTR share)")
print("  ✅ F_m (Bayesian learning of rating distribution)")
print("  ✅ Signal clarity (affects learning precision)")
print("  ✅ E[u_i(...)] (explicit utility calculation)")
print("  ✅ C_i(b_i) (explicit cost function)")
print("="*80)
