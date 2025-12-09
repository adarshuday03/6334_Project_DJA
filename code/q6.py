# team dja - dominic, jatin, adarsh
# isye 6334 group project fall 2025
# task 6

import os
import matplotlib.pyplot as plt
import numpy as np

# setup directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

PRIMARY_BLUE = '#003366'
ACCENT_RED = '#990000'

# model parameters

# wholesale cost
c = 3.00
# selling price
p = 5
# refund fraction
f = 0.50
# return shipping cost
s = 0.50
# admin fee
K = 20

# min demand
a = 120
# max demand
b = 420

# corvette prize value
prize = 40000

# helper functions

def expected_sales(Q, a, b):
    """calculate expected sales for order quantity q with uniform(a,b) demand"""
    if Q <= a:
        return Q
    elif Q >= b:
        return (a + b) / 2
    else:
        part1 = (Q**2 - a**2) / (2 * (b - a))
        part2 = Q * (b - Q) / (b - a)
        return part1 + part2

def calculate_expected_profit_with_prize(Q, p, c, f, s, K, a, b, prize):
    """
    calculate expected profit including prize component
    
    prize structure (multiple qualifying thresholds):
    - 5% chance of winning if sales >= 400 units
    - 3% chance of winning if sales >= 380 units  
    - these are treated as separate qualifying categories (can win in multiple)
    
    formula (based on consolidated approach):
    e[prize] = prize × [0.05 × p(d≥400) + 0.03 × p(380≤d<400)]
    
    where p(d≥400) = probability demand meets or exceeds 400
          p(380≤d<400) = probability demand falls between 380 and 400
    """
    exp_sales = expected_sales(Q, a, b)
    exp_leftover = Q - exp_sales
    
    # base profit components
    revenue = exp_sales * p
    salvage = exp_leftover * c * f
    shipping_cost = exp_leftover * s
    ordering_cost = K
    variable_cost = Q * c
    
    base_profit = revenue + salvage - shipping_cost - ordering_cost - variable_cost
    
    # prize component - multiple thresholds
    # component 1: 5% chance at 400 threshold
    # p(d >= 400) for uniform(120, 420) = (420 - 400) / 300
    p_d_ge_400 = max(0, min(1, (b - 400) / (b - a)))
    prize_400 = prize * 0.05 * p_d_ge_400 if Q >= 400 else 0
    
    # component 2: 3% chance at 380 threshold  
    # p(380 <= d < 400) = p(d >= 380) - p(d >= 400)
    p_d_ge_380 = max(0, min(1, (b - 380) / (b - a)))
    p_380_to_400 = p_d_ge_380 - p_d_ge_400
    prize_380 = prize * 0.03 * p_380_to_400 if Q >= 380 else 0
    
    # total expected prize
    expected_prize = prize_400 + prize_380
    
    total_expected_profit = base_profit + expected_prize
    
    return {
        'Q': Q,
        'expected_sales': exp_sales,
        'expected_leftover': exp_leftover,
        'base_profit': base_profit,
        'prize_400': prize_400,
        'prize_380': prize_380,
        'expected_prize': expected_prize,
        'total_expected_profit': total_expected_profit
    }

# part 6: find q** that maximizes expected profit with prize

print("=" * 70)
print("PART 6: Behavioral Incentive (Prize)")
print("=" * 70)
print()
print(f"Prize: ${prize:,}")
print()
print("Prize Structure (Multiple Qualifying Thresholds):")
print("  - 5% chance of winning if sales >= 400 units")
print("  - 3% chance of winning if sales >= 380 units")
print("  - Treated as separate categories (can qualify for both)")
print()

# search for optimal q**
best_Q = None
best_profit = float('-inf')
results = []

for Q in range(a, b + 50):
    result = calculate_expected_profit_with_prize(Q, p, c, f, s, K, a, b, prize)
    results.append(result)
    
    if result['total_expected_profit'] > best_profit:
        best_profit = result['total_expected_profit']
        best_Q = Q

Q_star_star = best_Q
Q_star = 270  # from part 2

print(f"Optimal Q** (with prize) = {Q_star_star}")
print(f"Original Q* (without prize) = {Q_star}")
print()

# get detailed results for both
result_without = calculate_expected_profit_with_prize(Q_star, p, c, f, s, K, a, b, prize)
result_with = calculate_expected_profit_with_prize(Q_star_star, p, c, f, s, K, a, b, prize)

print("=" * 70)
print("COMPARISON: Q* vs Q**")
print("=" * 70)
print()
print(f"{'Metric':<35} {'Q*='+str(Q_star):<20} {'Q**='+str(Q_star_star):<20}")
print("-" * 70)
print(f"{'Expected Sales':<35} {result_without['expected_sales']:<20.2f} {result_with['expected_sales']:<20.2f}")
print(f"{'Expected Leftover':<35} {result_without['expected_leftover']:<20.2f} {result_with['expected_leftover']:<20.2f}")
print(f"{'Base Profit':<35} ${result_without['base_profit']:<19.2f} ${result_with['base_profit']:<19.2f}")
print(f"{'E[Prize from 400 threshold]':<35} ${result_without['prize_400']:<19.2f} ${result_with['prize_400']:<19.2f}")
print(f"{'E[Prize from 380 threshold]':<35} ${result_without['prize_380']:<19.2f} ${result_with['prize_380']:<19.2f}")
print(f"{'Total Expected Prize':<35} ${result_without['expected_prize']:<19.2f} ${result_with['expected_prize']:<19.2f}")
print(f"{'Total Expected Profit':<35} ${result_without['total_expected_profit']:<19.2f} ${result_with['total_expected_profit']:<19.2f}")
print()

print("=" * 70)
print("ANALYSIS")
print("=" * 70)
print()
print(f"The prize incentive causes Q** ({Q_star_star}) > Q* ({Q_star})")
print()
print("This incentive scheme encourages RISK-SEEKING behavior:")
print(f"  - SparkFire orders {Q_star_star - Q_star} more units to chase the prize")
print(f"  - This increases base profit loss by ${result_without['base_profit'] - result_with['base_profit']:.2f}")
print(f"  - But expected prize value of ${result_with['expected_prize']:.2f} more than compensates")
print()
print("Prize Breakdown at Q**:")
print(f"  - Prize from 400 threshold (5%): ${result_with['prize_400']:.2f}")
print(f"  - Prize from 380 threshold (3%): ${result_with['prize_380']:.2f}")
print(f"  - Total expected prize: ${result_with['expected_prize']:.2f}")
print()
print("Behavioral Note:")
print(f"  The expected prize value (${result_with['expected_prize']:.0f}) is relatively small compared to the")
print("  emotional appeal of potentially winning a $40,000 Corvette.")
print("  Real decision-makers might over-order beyond Q** due to this psychological effect.")

# show key quantities at different q values
print()
print("=" * 70)
print("KEY QUANTITIES AT DIFFERENT ORDER LEVELS")
print("=" * 70)
print()

key_quantities = [270, 380, 400, 420]
print(f"{'Q':<10} {'E[Sales]':<15} {'Base Profit':<15} {'E[Prize]':<15} {'Total Profit':<15}")
print("-" * 70)

for Q_val in key_quantities:
    result = calculate_expected_profit_with_prize(Q_val, p, c, f, s, K, a, b, prize)
    print(f"{Q_val:<10} {result['expected_sales']:<15.2f} ${result['base_profit']:<14.2f} ${result['expected_prize']:<14.2f} ${result['total_expected_profit']:<14.2f}")

# save to csv

csv_file = os.path.join(CSV_DIR, 'q6_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 6 - Behavioral Incentive (Prize)\n")
    file.write("\n")
    file.write("Assumptions\n")
    file.write(f"Prize Value,${prize}\n")
    file.write("Prize Structure,Multiple qualifying thresholds\n")
    file.write("Threshold 1,5% chance if sales >= 400\n")
    file.write("Threshold 2,3% chance if sales >= 380\n")
    file.write("\n")
    
    file.write("Results\n")
    file.write(f"Q* (without prize),{Q_star}\n")
    file.write(f"Q** (with prize),{Q_star_star}\n")
    file.write(f"Base Profit at Q*,${result_without['base_profit']:.2f}\n")
    file.write(f"Base Profit at Q**,${result_with['base_profit']:.2f}\n")
    file.write(f"E[Prize from 400] at Q*,${result_without['prize_400']:.2f}\n")
    file.write(f"E[Prize from 400] at Q**,${result_with['prize_400']:.2f}\n")
    file.write(f"E[Prize from 380] at Q*,${result_without['prize_380']:.2f}\n")
    file.write(f"E[Prize from 380] at Q**,${result_with['prize_380']:.2f}\n")
    file.write(f"Total E[Prize] at Q*,${result_without['expected_prize']:.2f}\n")
    file.write(f"Total E[Prize] at Q**,${result_with['expected_prize']:.2f}\n")
    file.write(f"Total E[Profit] at Q*,${result_without['total_expected_profit']:.2f}\n")
    file.write(f"Total E[Profit] at Q**,${result_with['total_expected_profit']:.2f}\n")
    file.write("\n")
    
    file.write("Profit by Order Quantity\n")
    file.write("Q,Expected Sales,Base Profit,Prize 400,Prize 380,Total Prize,Total E[Profit]\n")
    for r in results:
        file.write(f"{r['Q']},{r['expected_sales']:.2f},{r['base_profit']:.2f},"
                   f"{r['prize_400']:.2f},{r['prize_380']:.2f},{r['expected_prize']:.2f},{r['total_expected_profit']:.2f}\n")

# generate plots

# extract data for plotting
Q_vals = [r['Q'] for r in results]
base_profits = [r['base_profit'] for r in results]
total_profits = [r['total_expected_profit'] for r in results]
prize_EVs = [r['expected_prize'] for r in results]

# plot 1: expected profit with vs without prize
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Q_vals, base_profits, color=PRIMARY_BLUE, linewidth=2, label='Without Prize')
ax.plot(Q_vals, total_profits, color=ACCENT_RED, linewidth=2, label='With Prize')

ax.axvline(x=Q_star, color=PRIMARY_BLUE, linestyle='--', alpha=0.7)
ax.axvline(x=Q_star_star, color=ACCENT_RED, linestyle='--', alpha=0.7)

ax.scatter([Q_star], [result_without['base_profit']], color=PRIMARY_BLUE, s=100, zorder=5)
ax.scatter([Q_star_star], [result_with['total_expected_profit']], color=ACCENT_RED, s=100, zorder=5)

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 6: Expected Profit With vs Without Prize', fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate(f'Q*={Q_star}\n${result_without["base_profit"]:.0f}', 
            xy=(Q_star, result_without['base_profit']), 
            xytext=(Q_star-30, result_without['base_profit']+30),
            fontsize=10, color=PRIMARY_BLUE)
ax.annotate(f'Q**={Q_star_star}\n${result_with["total_expected_profit"]:.0f}', 
            xy=(Q_star_star, result_with['total_expected_profit']), 
            xytext=(Q_star_star+10, result_with['total_expected_profit']-30),
            fontsize=10, color=ACCENT_RED)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q6_profit_comparison.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# plot 2: profit breakdown at q**
fig, ax = plt.subplots(figsize=(8, 5))
components = ['Base\nProfit', 'Prize\n(400)', 'Prize\n(380)', 'Total\nProfit']
values = [result_with['base_profit'], result_with['prize_400'], 
          result_with['prize_380'], result_with['total_expected_profit']]
colors = [PRIMARY_BLUE, 'gold', 'orange', ACCENT_RED]

bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Amount ($)', fontsize=12)
ax.set_title(f'Part 6: Profit Breakdown at Q** = {Q_star_star}', fontsize=14, fontweight='bold')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'${val:.0f}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(values) * 1.2)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q6_breakdown.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")

# sensitivity analysis

print()
print("=" * 70)
print("SENSITIVITY ANALYSIS")
print("=" * 70)
print()

# scenario analysis: different prize rules
print("SCENARIO COMPARISON: Different Prize Rules")
print("-" * 70)

scenarios = [
    {'name': 'A: 5% @ 400 only', 'prob_400': 0.05, 'prob_380': 0.00, 'prob_420': 0.00},
    {'name': 'B: Multi-threshold (5%@400 + 3%@380)', 'prob_400': 0.05, 'prob_380': 0.03, 'prob_420': 0.00},
    {'name': 'C: 7% @ 420 only', 'prob_400': 0.00, 'prob_380': 0.00, 'prob_420': 0.07},
]

scenario_results = []

for scenario in scenarios:
    best_Q_scenario = None
    best_profit_scenario = float('-inf')
    
    for Q in range(a, b + 50):
        exp_sales = expected_sales(Q, a, b)
        exp_leftover = Q - exp_sales
        
        base_profit = (p * exp_sales + f * c * exp_leftover - 
                      s * exp_leftover - K - c * Q)
        
        # Calculate prize for this scenario
        prize_ev = 0
        if scenario['prob_400'] > 0 and Q >= 400:
            prob_D_ge_400 = max(0, (b - 400) / (b - a))
            prize_ev += prize * scenario['prob_400'] * prob_D_ge_400
        
        if scenario['prob_380'] > 0 and Q >= 380:
            prob_D_ge_380 = max(0, (b - 380) / (b - a))
            prize_ev += prize * scenario['prob_380'] * prob_D_ge_380
        
        if scenario['prob_420'] > 0 and Q >= 420:
            prob_D_ge_420 = max(0, (b - 420) / (b - a))
            prize_ev += prize * scenario['prob_420'] * prob_D_ge_420
        
        total_profit = base_profit + prize_ev
        
        if total_profit > best_profit_scenario:
            best_profit_scenario = total_profit
            best_Q_scenario = Q
    
    scenario_results.append({
        'name': scenario['name'],
        'Q_optimal': best_Q_scenario,
        'total_profit': best_profit_scenario
    })
    
    print(f"{scenario['name']:<45} Q**={best_Q_scenario:<5} E[Profit]=${best_profit_scenario:.2f}")

print()

# probability sensitivity (fix threshold at 400, vary probability)
print("PROBABILITY SENSITIVITY: P(win | sales ≥ 400)")
print("-" * 70)

prob_sensitivities = [0.01, 0.03, 0.05, 0.07, 0.10]
prob_results = []

for prob_val in prob_sensitivities:
    best_Q_prob = None
    best_profit_prob = float('-inf')
    
    for Q in range(a, b + 50):
        exp_sales = expected_sales(Q, a, b)
        exp_leftover = Q - exp_sales
        
        base_profit = (p * exp_sales + f * c * exp_leftover - 
                      s * exp_leftover - K - c * Q)
        
        prize_ev = 0
        if Q >= 400:
            prob_D_ge_400 = max(0, (b - 400) / (b - a))
            prize_ev = prize * prob_val * prob_D_ge_400
        
        total_profit = base_profit + prize_ev
        
        if total_profit > best_profit_prob:
            best_profit_prob = total_profit
            best_Q_prob = Q
    
    prob_results.append({
        'prob': prob_val,
        'Q_optimal': best_Q_prob,
        'total_profit': best_profit_prob,
        'increase_pct': (best_profit_prob - 370) / 370 * 100
    })
    
    print(f"P(win)={prob_val:4.0%}   Q**={best_Q_prob:<5} E[Profit]=${best_profit_prob:6.2f}   (+{(best_profit_prob - 370) / 370 * 100:4.1f}% vs baseline)")

print()

# prize amount sensitivity (fix 5% @ 400, vary prize)
print("PRIZE AMOUNT SENSITIVITY: P(win)=5% at sales ≥ 400")
print("-" * 70)

prize_sensitivities = [20000, 30000, 40000, 60000, 80000]
prize_amount_results = []

for prize_val in prize_sensitivities:
    best_Q_prize = None
    best_profit_prize = float('-inf')
    
    for Q in range(a, b + 50):
        exp_sales = expected_sales(Q, a, b)
        exp_leftover = Q - exp_sales
        
        base_profit = (p * exp_sales + f * c * exp_leftover - 
                      s * exp_leftover - K - c * Q)
        
        prize_ev = 0
        if Q >= 400:
            prob_D_ge_400 = max(0, (b - 400) / (b - a))
            prize_ev = prize_val * 0.05 * prob_D_ge_400
        
        total_profit = base_profit + prize_ev
        
        if total_profit > best_profit_prize:
            best_profit_prize = total_profit
            best_Q_prize = Q
    
    prize_amount_results.append({
        'prize': prize_val,
        'Q_optimal': best_Q_prize,
        'total_profit': best_profit_prize
    })
    
    print(f"Prize=${prize_val/1000:3.0f}k   Q**={best_Q_prize:<5} E[Profit]=${best_profit_prize:6.2f}")

# save sensitivity results
csv_sensitivity = os.path.join(CSV_DIR, 'q6_sensitivity.csv')

with open(csv_sensitivity, 'w') as file:
    file.write("Part 6 - Sensitivity Analysis\n\n")
    
    file.write("Scenario Analysis\n")
    file.write("Scenario,Q**,Total E[Profit]\n")
    for r in scenario_results:
        file.write(f"{r['name']},{r['Q_optimal']},${r['total_profit']:.2f}\n")
    file.write("\n")
    
    file.write("Probability Sensitivity (threshold=400)\n")
    file.write("P(win),Q**,Total E[Profit],% Increase vs Baseline\n")
    for r in prob_results:
        file.write(f"{r['prob']:.2f},{r['Q_optimal']},${r['total_profit']:.2f},{r['increase_pct']:.1f}%\n")
    file.write("\n")
    
    file.write("Prize Amount Sensitivity (P=5% @ 400)\n")
    file.write("Prize Amount,Q**,Total E[Profit]\n")
    for r in prize_amount_results:
        file.write(f"${r['prize']},{r['Q_optimal']},${r['total_profit']:.2f}\n")

print()
print(f"Sensitivity analysis saved to: {csv_sensitivity}")

# plot sensitivity analyses
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot: Probability sensitivity
probs = [r['prob'] for r in prob_results]
Q_probs = [r['Q_optimal'] for r in prob_results]
ax1.plot(probs, Q_probs, 'o-', color=PRIMARY_BLUE, linewidth=2, markersize=8)
ax1.set_xlabel('P(win | sales ≥ 400)', fontsize=11)
ax1.set_ylabel('Optimal Q**', fontsize=11)
ax1.set_title('Probability Sensitivity', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
for prob, Q in zip(probs, Q_probs):
    ax1.annotate(f'{Q}', (prob, Q), textcoords="offset points", 
                xytext=(0,8), ha='center', fontsize=9, fontweight='bold')

# Plot: Prize amount sensitivity
prizes = [r['prize']/1000 for r in prize_amount_results]
Q_prizes = [r['Q_optimal'] for r in prize_amount_results]
ax2.plot(prizes, Q_prizes, 's-', color=ACCENT_RED, linewidth=2, markersize=8)
ax2.set_xlabel('Prize Amount ($1000s)', fontsize=11)
ax2.set_ylabel('Optimal Q**', fontsize=11)
ax2.set_title('Prize Amount Sensitivity', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
for prize_k, Q in zip(prizes, Q_prizes):
    ax2.annotate(f'{Q}', (prize_k, Q), textcoords="offset points", 
                xytext=(0,8), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plot_sensitivity = os.path.join(PLOTS_DIR, 'q6_sensitivity.png')
plt.savefig(plot_sensitivity, dpi=150)
plt.close()
print(f"Plot saved to: {plot_sensitivity}")
