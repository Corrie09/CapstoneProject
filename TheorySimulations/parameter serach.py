import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ============================================================================
# SIMPLIFIED PARAMETER SEARCH - RELAXED CRITERIA
# ============================================================================

def test_parameters(beta_base, beta_slope, pbar_L_base, pbar_L_slope):
    """Test a parameter combination - FASTER version"""
    
    def beta(r):
        return beta_base + beta_slope * r
    
    def p_bar(r, goal):
        if goal == 'L':
            return pbar_L_base + pbar_L_slope * r
        else:
            return 0.003 + 0.004 * r  # ST fixed
    
    K = 30
    LT_range, ST_range = 0.15, 0.35
    n_users, n_periods = 1000, 50
    np.random.seed(42)
    
    def simulate_market(rho_m):
        users = pd.DataFrame({
            'goal': np.random.choice(['L', 'S'], size=n_users, p=[rho_m, 1-rho_m]),
            'rating': np.random.beta(2, 2, size=n_users),
            'active': True,
            'exit_period': np.nan,
            'exit_type': None
        })
        
        rho_trajectory = []
        
        for t in range(n_periods):
            active = users[users['active']]
            if len(active) == 0:
                break
            
            rho_t = (active['goal'] == 'L').mean()
            rho_trajectory.append(rho_t)
            
            for idx in active.index:
                r_i, goal = users.loc[idx, ['rating', 'goal']]
                
                # Compute success prob
                if goal == 'L':
                    acceptable = active[
                        (active['goal'] == 'L') &
                        (active['rating'] >= r_i - LT_range) &
                        (active['rating'] <= r_i + LT_range)
                    ]
                else:
                    acceptable = active[
                        (active['rating'] >= r_i - ST_range) &
                        (active['rating'] <= r_i + ST_range)
                    ]
                
                A = len(acceptable) / len(active)
                p_success = min(K * beta(r_i) * A, 1.0)
                
                # Exit logic
                if p_success < p_bar(r_i, goal):
                    users.loc[idx, ['active', 'exit_period', 'exit_type']] = [False, t, 'frustration']
                elif np.random.random() < p_success:
                    users.loc[idx, ['active', 'exit_period', 'exit_type']] = [False, t, 'match']
        
        return rho_trajectory, users
    
    # Test markets
    results = {}
    for name, rho_m in [('LT-poor', 0.25), ('Balanced', 0.55), ('LT-rich', 0.85)]:
        traj, users = simulate_market(rho_m)
        
        if len(traj) < 10:  # Collapsed too fast
            results[name] = {'score': 0}
            continue
        
        decline = traj[0] - traj[-1]
        
        # RELAXED criteria for LT-poor: allow high frustration
        exited = users[users['exit_type'].notna()]
        LT_users = users[users['goal'] == 'L']
        LT_frust = exited[(exited['goal'] == 'L') & (exited['exit_type'] == 'frustration')]
        frust_rate = len(LT_frust) / len(LT_users) if len(LT_users) > 0 else 0
        
        # Good criteria:
        # - Decline > 0.1 for LT-poor
        # - Decline > 0.05 for others  
        # - Not too fast (>15 periods)
        # - Frustration rate 20-80% (realistic range)
        
        if name == 'LT-poor':
            good = decline > 0.10 and len(traj) > 15 and 0.3 < frust_rate < 0.85
            score = decline * 3 if good else 0
        elif name == 'Balanced':
            good = decline > 0.05 and len(traj) > 15 and 0.1 < frust_rate < 0.6
            score = decline * 2 if good else 0
        else:  # LT-rich
            good = 0 < decline < 0.4 and len(traj) > 15 and frust_rate < 0.4
            score = decline if good else 0
        
        results[name] = {
            'score': score,
            'decline': decline,
            'periods': len(traj),
            'frust_rate': frust_rate,
            'good': good
        }
    
    total_score = sum(r['score'] for r in results.values())
    return total_score, results

# ============================================================================
# GRID SEARCH - NARROWER RANGES
# ============================================================================

print("Searching for optimal parameters...")
best_score = 0
best_params = None
best_results = None

for beta_base in [0.002, 0.003, 0.004, 0.005]:
    for beta_slope in [0.010, 0.012, 0.015, 0.018]:
        for pbar_L_base in [0.020, 0.024, 0.028]:
            for pbar_L_slope in [0.020, 0.024, 0.028]:
                
                score, results = test_parameters(beta_base, beta_slope, pbar_L_base, pbar_L_slope)
                
                if score > best_score:
                    best_score = score
                    best_params = (beta_base, beta_slope, pbar_L_base, pbar_L_slope)
                    best_results = results
                    
                    print(f"\n🎯 Score: {score:.3f}")
                    print(f"   β={beta_base:.3f}+{beta_slope:.3f}r, p̄_L={pbar_L_base:.3f}+{pbar_L_slope:.3f}r")
                    for market, res in results.items():
                        if 'decline' in res:
                            print(f"   {market}: decline={res['decline']:+.2f}, "
                                  f"frust={res['frust_rate']:.1%}, periods={res['periods']}, "
                                  f"{'✓' if res['good'] else '✗'}")

print("\n" + "="*80)
print("BEST PARAMETERS:")
print("="*80)
if best_params:
    print(f"beta(r) = {best_params[0]:.4f} + {best_params[1]:.4f} * r")
    print(f"p_bar_L(r) = {best_params[2]:.4f} + {best_params[3]:.4f} * r")
    print(f"\nFinal score: {best_score:.3f}")
else:
    print("No valid parameters found!")