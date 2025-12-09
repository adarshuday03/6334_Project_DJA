"""
OE4 Sensitivity Analysis: Impact of f, s, c1 on VOSRC
Tests which parameter change gives largest improvement per unit change
"""

import numpy as np
import pandas as pd

# Baseline parameters
p = 5.00
c0 = 3.00
f_base = 0.5
s_base = 0.5
c1_base = 3.60

# Discrete demand states
D_H = 330
D_L = 190

# Signal probabilities
P_S_H = 0.45
P_S_L = 0.55

# Conditional probabilities
P_D_H_given_S_H = 0.8
P_D_L_given_S_H = 0.2
P_D_H_given_S_L = 0.4
P_D_L_given_S_L = 0.6

# Baseline profit from Part 2 (continuous uniform, single-stage)
baseline_profit_part2 = 370.00

def calculate_profit(Q0, Q1, D, f, s, c1):
    """Calculate profit for given policy and demand realization"""
    total_inventory = Q0 + Q1
    sales = min(total_inventory, D)
    leftover = max(total_inventory - D, 0)
    
    revenue = p * sales
    cost_stage1 = c0 * Q0
    cost_stage2 = c1 * Q1
    refund = f * c0 * leftover
    shipping = s * leftover
    
    profit = revenue - cost_stage1 - cost_stage2 + refund - shipping
    return profit

def calculate_critical_ratio(c, f, s, p):
    """Calculate critical ratio for given cost parameters"""
    Co = c * (1 - f) + s
    Cu = p - c
    CR = Cu / (Cu + Co)
    return CR

def calculate_inverse_critical_ratio(c1, f, s, p):
    """Calculate inverse critical ratio for expedited ordering"""
    Co = c1 * (1 - f) + s
    Cu = p - c1
    CR_inv = Co / (Co + Cu)
    return CR_inv

def find_optimal_targets(c1, f, s):
    """Find Q_H* and Q_L* using inverse critical ratio"""
    CR_inv = calculate_inverse_critical_ratio(c1, f, s, p)
    
    # High signal: D ~ U[240, 420]
    Q_H_star = 240 + (420 - 240) * CR_inv
    
    # Low signal: D ~ U[120, 260]
    Q_L_star = 120 + (260 - 120) * CR_inv
    
    return Q_H_star, Q_L_star

def optimize_two_stage(f, s, c1):
    """Optimize two-stage policy for given parameters"""
    Q_H_star, Q_L_star = find_optimal_targets(c1, f, s)
    
    # Round to nearest 10 for discrete optimization
    Q_H_star = round(Q_H_star / 10) * 10
    Q_L_star = round(Q_L_star / 10) * 10
    
    # Test Q0 values around key targets
    Q0_candidates = [170, 180, 190, 200, 210, 220, 330, 340]
    best_profit = -np.inf
    best_Q0 = None
    best_Q1_H = None
    best_Q1_L = None
    
    for Q0 in Q0_candidates:
        # Reactive policy
        Q1_H = max(0, Q_H_star - Q0)
        Q1_L = max(0, Q_L_star - Q0)
        
        # Expected profit given Signal High
        profit_H_DH = calculate_profit(Q0, Q1_H, D_H, f, s, c1)
        profit_H_DL = calculate_profit(Q0, Q1_H, D_L, f, s, c1)
        E_profit_given_H = P_D_H_given_S_H * profit_H_DH + P_D_L_given_S_H * profit_H_DL
        
        # Expected profit given Signal Low
        profit_L_DH = calculate_profit(Q0, Q1_L, D_H, f, s, c1)
        profit_L_DL = calculate_profit(Q0, Q1_L, D_L, f, s, c1)
        E_profit_given_L = P_D_H_given_S_L * profit_L_DH + P_D_L_given_S_L * profit_L_DL
        
        # Total expected profit
        total_profit = P_S_H * E_profit_given_H + P_S_L * E_profit_given_L
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_Q0 = Q0
            best_Q1_H = Q1_H
            best_Q1_L = Q1_L
    
    return best_Q0, best_Q1_H, best_Q1_L, best_profit

# Baseline calculation
print("=" * 70)
print("BASELINE (f=0.5, s=0.5, c1=3.60)")
print("=" * 70)
Q0_base, Q1_H_base, Q1_L_base, profit_base = optimize_two_stage(f_base, s_base, c1_base)
VOSRC_base = profit_base - baseline_profit_part2
print(f"Optimal: Q0={Q0_base}, Q1(H)={Q1_H_base}, Q1(L)={Q1_L_base}")
print(f"Expected Profit: ${profit_base:.2f}")
print(f"VOSRC: ${VOSRC_base:.2f}")
print()

# Test 1: Refund fraction sensitivity (±10%)
print("=" * 70)
print("TEST 1: REFUND FRACTION (f)")
print("=" * 70)
f_values = [0.45, 0.50, 0.55]
results_f = []

for f in f_values:
    Q0, Q1_H, Q1_L, profit = optimize_two_stage(f, s_base, c1_base)
    VOSRC = profit - baseline_profit_part2
    results_f.append({
        'f': f,
        'Q0': Q0,
        'Q1(H)': Q1_H,
        'Q1(L)': Q1_L,
        'Profit': profit,
        'VOSRC': VOSRC
    })
    print(f"f={f:.2f}: Q0={Q0}, Profit=${profit:.2f}, VOSRC=${VOSRC:.2f}")

df_f = pd.DataFrame(results_f)
delta_VOSRC_f = (df_f.iloc[2]['VOSRC'] - df_f.iloc[0]['VOSRC']) / 0.10
print(f"\nMarginal impact: ΔVOSRC/Δf = ${delta_VOSRC_f:.2f} per 0.01 change")
print()

# Test 2: Shipping cost sensitivity (±10%)
print("=" * 70)
print("TEST 2: SHIPPING COST (s)")
print("=" * 70)
s_values = [0.45, 0.50, 0.55]
results_s = []

for s in s_values:
    Q0, Q1_H, Q1_L, profit = optimize_two_stage(f_base, s, c1_base)
    VOSRC = profit - baseline_profit_part2
    results_s.append({
        's': s,
        'Q0': Q0,
        'Q1(H)': Q1_H,
        'Q1(L)': Q1_L,
        'Profit': profit,
        'VOSRC': VOSRC
    })
    print(f"s={s:.2f}: Q0={Q0}, Profit=${profit:.2f}, VOSRC=${VOSRC:.2f}")

df_s = pd.DataFrame(results_s)
delta_VOSRC_s = (df_s.iloc[0]['VOSRC'] - df_s.iloc[2]['VOSRC']) / 0.10
print(f"\nMarginal impact: ΔVOSRC/Δs = ${delta_VOSRC_s:.2f} per 0.01 change (reduction)")
print()

# Test 3: Expedited cost sensitivity (±10%)
print("=" * 70)
print("TEST 3: EXPEDITED COST (c1)")
print("=" * 70)
c1_values = [3.24, 3.60, 3.96]  # ±10%
results_c1 = []

for c1 in c1_values:
    Q0, Q1_H, Q1_L, profit = optimize_two_stage(f_base, s_base, c1)
    VOSRC = profit - baseline_profit_part2
    results_c1.append({
        'c1': c1,
        'Q0': Q0,
        'Q1(H)': Q1_H,
        'Q1(L)': Q1_L,
        'Profit': profit,
        'VOSRC': VOSRC
    })
    print(f"c1={c1:.2f}: Q0={Q0}, Profit=${profit:.2f}, VOSRC=${VOSRC:.2f}")

df_c1 = pd.DataFrame(results_c1)
delta_VOSRC_c1 = (df_c1.iloc[0]['VOSRC'] - df_c1.iloc[2]['VOSRC']) / 0.72
print(f"\nMarginal impact: ΔVOSRC/Δc1 = ${delta_VOSRC_c1:.2f} per $0.01 change (reduction)")
print()

# Summary comparison
print("=" * 70)
print("SUMMARY: MARGINAL IMPROVEMENT PER UNIT CHANGE")
print("=" * 70)
print(f"Refund fraction (f):  ${abs(delta_VOSRC_f):.2f} per 0.01 increase")
print(f"Shipping cost (s):    ${abs(delta_VOSRC_s):.2f} per 0.01 decrease")
print(f"Expedited cost (c1):  ${abs(delta_VOSRC_c1):.2f} per $0.01 decrease")
print()

# Determine winner
sensitivities = {
    'f (0.01 increase)': abs(delta_VOSRC_f),
    's (0.01 decrease)': abs(delta_VOSRC_s),
    'c1 ($0.01 decrease)': abs(delta_VOSRC_c1)
}

winner = max(sensitivities, key=sensitivities.get)
print(f"WINNER: {winner} with ${sensitivities[winner]:.2f} improvement per unit change")
print()

# Create detailed table for report
print("=" * 70)
print("TABLE FOR REPORT")
print("=" * 70)
print("\nParameter | Base | Test (-10%) | Test (+10%) | ΔVOSRC | Per Unit")
print("-" * 70)
print(f"f         | 0.50 | {df_f.iloc[0]['VOSRC']:6.2f} | {df_f.iloc[2]['VOSRC']:6.2f} | {df_f.iloc[2]['VOSRC']-df_f.iloc[0]['VOSRC']:+6.2f} | {abs(delta_VOSRC_f):5.2f}")
print(f"s         | 0.50 | {df_s.iloc[0]['VOSRC']:6.2f} | {df_s.iloc[2]['VOSRC']:6.2f} | {df_s.iloc[0]['VOSRC']-df_s.iloc[2]['VOSRC']:+6.2f} | {abs(delta_VOSRC_s):5.2f}")
print(f"c1        | 3.60 | {df_c1.iloc[0]['VOSRC']:6.2f} | {df_c1.iloc[2]['VOSRC']:6.2f} | {df_c1.iloc[0]['VOSRC']-df_c1.iloc[2]['VOSRC']:+6.2f} | {abs(delta_VOSRC_c1):5.2f}")
