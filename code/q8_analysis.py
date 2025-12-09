"""
ISYE 6334 - Task 8 Analysis
Comparing Dom's discrete approximation vs continuous distribution approach
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 5.00
c0 = 3.00
c1 = 3.60
f = 0.5
s = 0.5
K = 20

P_SH = 0.45
P_SL = 0.55

print("="*70)
print("TASK 8: TWO-STAGE ORDERING WITH DEMAND SIGNAL")
print("="*70)

# DOM'S APPROACH: Discrete demand states

print("\n" + "="*70)
print("DOM'S DISCRETE APPROXIMATION")
print("="*70)

# Dom uses two demand states: 330 and 190 (means of the distributions)
D_high = 330
D_low = 190

# Dom's assumed conditional probabilities
P_DH_given_SH = 0.8
P_DL_given_SH = 0.2
P_DH_given_SL = 0.2
P_DL_given_SL = 0.8

print(f"\nDemand States:")
print(f"  D_high = {D_high} (mean of U[240,420])")
print(f"  D_low = {D_low} (mean of U[120,260])")

print(f"\nConditional Probabilities:")
print(f"  P(D={D_high} | S=High) = {P_DH_given_SH}")
print(f"  P(D={D_low} | S=High) = {P_DL_given_SH}")
print(f"  P(D={D_high} | S=Low) = {P_DH_given_SL}")
print(f"  P(D={D_low} | S=Low) = {P_DL_given_SL}")

def calc_profit_discrete(Q0, Q1, D, c0_used, c1_used):
    """Calculate profit for discrete demand realization"""
    Q_total = Q0 + Q1
    sales = min(D, Q_total)
    leftover = max(0, Q_total - D)
    
    revenue = sales * p
    var_cost = Q0 * c0_used + Q1 * c1_used
    salvage = leftover * c0_used * f  # Use c0 for all refunds
    shipping = leftover * s
    
    profit = revenue - var_cost + salvage - shipping - K
    return profit

# Dom's optimal solution from Excel
Q0_dom = 190
Q1_dom_H = 140
Q1_dom_L = 0

print(f"\nDom's Optimal Policy:")
print(f"  Q0* = {Q0_dom}")
print(f"  Q1*(High) = {Q1_dom_H} → Total when High = {Q0_dom + Q1_dom_H}")
print(f"  Q1*(Low) = {Q1_dom_L} → Total when Low = {Q0_dom + Q1_dom_L}")

# Calculate expected profits for Dom's policy
# When S=High
profit_H_DH = calc_profit_discrete(Q0_dom, Q1_dom_H, D_high, c0, c1)
profit_H_DL = calc_profit_discrete(Q0_dom, Q1_dom_H, D_low, c0, c1)
exp_profit_H = P_DH_given_SH * profit_H_DH + P_DL_given_SH * profit_H_DL

# When S=Low  
profit_L_DH = calc_profit_discrete(Q0_dom, Q1_dom_L, D_high, c0, c1)
profit_L_DL = calc_profit_discrete(Q0_dom, Q1_dom_L, D_low, c0, c1)
exp_profit_L = P_DH_given_SL * profit_L_DH + P_DL_given_SL * profit_L_DL

# Overall expected profit
exp_profit_dom = P_SH * exp_profit_H + P_SL * exp_profit_L

print(f"\nExpected Profit Breakdown:")
print(f"  E[Π | S=High] = {exp_profit_H:.2f}")
print(f"  E[Π | S=Low] = {exp_profit_L:.2f}")
print(f"  E[Π total] = {exp_profit_dom:.2f}")
print(f"  (Dom reports: $415.6)")

# THEORETICALLY CORRECT APPROACH: Continuous distributions

print("\n" + "="*70)
print("CONTINUOUS DISTRIBUTION APPROACH (Theoretically Correct)")
print("="*70)

def expected_sales_uniform(Q, a, b):
    """Expected sales for Uniform(a,b) demand"""
    if Q <= a:
        return Q
    elif Q >= b:
        return (a + b) / 2
    else:
        return (Q**2 - a**2) / (2 * (b - a)) + Q * (b - Q) / (b - a)

def calc_profit_continuous(Q0, Q1, a, b, c0_used, c1_used):
    """Calculate expected profit for continuous uniform demand"""
    Q_total = Q0 + Q1
    exp_sales = expected_sales_uniform(Q_total, a, b)
    exp_leftover = Q_total - exp_sales
    
    revenue = exp_sales * p
    var_cost = Q0 * c0_used + Q1 * c1_used
    salvage = exp_leftover * c0_used * f
    shipping = exp_leftover * s
    
    profit = revenue - var_cost + salvage - shipping - K
    return profit

# Conditional distributions from problem
a_H, b_H = 240, 420  # D | S=High ~ U[240, 420]
a_L, b_L = 120, 260  # D | S=Low ~ U[120, 260]

print(f"\nConditional Demand Distributions:")
print(f"  D | S=High ~ U[{a_H}, {b_H}], mean = {(a_H + b_H)/2}")
print(f"  D | S=Low ~ U[{a_L}, {b_L}], mean = {(a_L + b_L)/2}")

# Step 1: Find optimal Q*_S for each signal (newsvendor at stage 2 cost)
Co = c1 * (1 - f) + s
Cu = p - c1
CR = Cu / (Cu + Co)

Q_star_H = a_H + (b_H - a_H) * CR
Q_star_L = a_L + (b_L - a_L) * CR

print(f"\nStage 2 Newsvendor Optimums (at c = ${c1}):")
print(f"  Critical Ratio = {CR:.4f}")
print(f"  Q*_H (target when S=High) = {Q_star_H:.1f}")
print(f"  Q*_L (target when S=Low) = {Q_star_L:.1f}")

# Step 2: Optimize Q0 by testing range
print(f"\nOptimizing Q0...")

best_Q0 = None
best_exp_profit = -np.inf
results = []

for Q0_test in range(150, 230, 5):
    # Policy: Q1(H) = max(0, Q*_H - Q0), Q1(L) = max(0, Q*_L - Q0)
    Q1_H = max(0, Q_star_H - Q0_test)
    Q1_L = max(0, Q_star_L - Q0_test)
    
    # Expected profit when S=High
    exp_profit_H = calc_profit_continuous(Q0_test, Q1_H, a_H, b_H, c0, c1)
    
    # Expected profit when S=Low
    exp_profit_L = calc_profit_continuous(Q0_test, Q1_L, a_L, b_L, c0, c1)
    
    # Overall expected profit
    exp_profit = P_SH * exp_profit_H + P_SL * exp_profit_L
    
    results.append({
        'Q0': Q0_test,
        'Q1_H': Q1_H,
        'Q1_L': Q1_L,
        'E[Π|H]': exp_profit_H,
        'E[Π|L]': exp_profit_L,
        'E[Π]': exp_profit
    })
    
    if exp_profit > best_exp_profit:
        best_exp_profit = exp_profit
        best_Q0 = Q0_test
        best_Q1_H = Q1_H
        best_Q1_L = Q1_L

print(f"\nOptimal Policy (Continuous):")
print(f"  Q0* = {best_Q0}")
print(f"  Q1*(High) = {best_Q1_H:.1f} → Total = {best_Q0 + best_Q1_H:.1f}")
print(f"  Q1*(Low) = {best_Q1_L:.1f} → Total = {best_Q0 + best_Q1_L:.1f}")
print(f"  E[Π total] = ${best_exp_profit:.2f}")

# COMPARISON

print("\n" + "="*70)
print("COMPARISON: DOM'S DISCRETE vs CONTINUOUS APPROACH")
print("="*70)

print(f"\nDom's Discrete Approximation:")
print(f"  Q0* = {Q0_dom}, Q1*(H) = {Q1_dom_H}, Q1*(L) = {Q1_dom_L}")
print(f"  E[Π] = ${exp_profit_dom:.2f}")

print(f"\nContinuous (Theoretically Correct):")
print(f"  Q0* = {best_Q0}, Q1*(H) = {best_Q1_H:.1f}, Q1*(L) = {best_Q1_L:.1f}")
print(f"  E[Π] = ${best_exp_profit:.2f}")

print(f"\nDifference:")
print(f"  ΔQ0 = {abs(Q0_dom - best_Q0)}")
print(f"  ΔE[Π] = ${abs(exp_profit_dom - best_exp_profit):.2f}")
print(f"  Relative error = {abs(exp_profit_dom - best_exp_profit) / best_exp_profit * 100:.2f}%")

# VOSRC CALCULATION

print("\n" + "="*70)
print("VALUE OF SIGNAL AND REACTIVE CAPACITY (VOSRC)")
print("="*70)

# Baseline profit from Task 2
baseline_Q = 270
baseline_profit = 370.00

print(f"\nBaseline (Task 2, no signal):")
print(f"  Q* = {baseline_Q}, E[Π] = ${baseline_profit:.2f}")

print(f"\nWith Signal (Dom's approach):")
print(f"  E[Π] = ${exp_profit_dom:.2f}")
print(f"  VOSRC = ${exp_profit_dom - baseline_profit:.2f}")

print(f"\nWith Signal (Continuous approach):")
print(f"  E[Π] = ${best_exp_profit:.2f}")
print(f"  VOSRC = ${best_exp_profit - baseline_profit:.2f}")

# ASSESSMENT

print("\n" + "="*70)
print("ASSESSMENT OF DOM'S APPROACH")
print("="*70)

print(f"\n✓ STRENGTHS:")
print(f"  1. Computationally simple - uses discrete states")
print(f"  2. Captures key insight: High signal → stock more, Low signal → stock less")
print(f"  3. Uses demand means (330, 190) as representative values")
print(f"  4. Explicitly models uncertainty via conditional probabilities")

print(f"\n⚠ LIMITATIONS:")
print(f"  1. Approximates continuous U[a,b] with discrete states")
print(f"  2. Requires assumption of P(D=330|S) and P(D=190|S) values")
print(f"  3. Ignores within-distribution variance")
print(f"  4. May not be globally optimal")

profit_gap = abs(exp_profit_dom - best_exp_profit)
if profit_gap < 10:
    print(f"\n✅ VERDICT: Dom's approach is ACCEPTABLE")
    print(f"   Profit difference < $10 ({profit_gap:.2f}), pedagogically clear")
else:
    print(f"\n⚠️ VERDICT: Consider continuous approach")
    print(f"   Profit difference ${profit_gap:.2f} may be material")

print("="*70)
