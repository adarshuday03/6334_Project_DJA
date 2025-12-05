"""
ISYE 6334 - Group Project (Fall 2025)
Part 6: Behavioral Incentive (Prize)

Leisure Limited awards $40,000 to highest sales stand.
Modify profit model and find Q** that maximizes expected returns.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Path Configuration (robust for reproducibility)
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot colors
PRIMARY_BLUE = '#003366'
ACCENT_RED = '#990000'

# =============================================================================
# Parameters
# =============================================================================

c = 3.00        # wholesale cost
p = 5           # selling price  
f = 0.50        # refund fraction
s = 0.50        # return shipping cost
K = 20          # admin fee

a = 120         # min demand
b = 420         # max demand

prize = 40000   # Corvette prize value

# Prize thresholds and probabilities (from project description)
# We'll use the 5% @ 400 rule as our primary assumption
prize_threshold = 400
prize_prob = 0.05

# =============================================================================
# Helper Functions
# =============================================================================

def expected_sales(Q, a, b):
    """Calculate expected sales for order quantity Q with Uniform(a,b) demand"""
    if Q <= a:
        return Q
    elif Q >= b:
        return (a + b) / 2
    else:
        part1 = (Q**2 - a**2) / (2 * (b - a))
        part2 = Q * (b - Q) / (b - a)
        return part1 + part2

def prob_sales_ge_threshold(Q, threshold, a, b):
    """
    Calculate probability that sales >= threshold
    Sales = min(D, Q), so Sales >= threshold requires:
      - D >= threshold (demand high enough)
      - Q >= threshold (ordered enough to sell that many)
    
    If Q < threshold: P(Sales >= threshold) = 0
    If Q >= threshold: P(Sales >= threshold) = P(D >= threshold)
    """
    if Q < threshold:
        return 0.0
    
    # P(D >= threshold) for Uniform(a, b)
    if threshold <= a:
        return 1.0
    elif threshold >= b:
        return 0.0
    else:
        return (b - threshold) / (b - a)

def calculate_expected_profit_with_prize(Q, p, c, f, s, K, a, b, prize, threshold, prize_prob):
    """
    Calculate expected profit including prize component
    
    Prize logic: If sales >= threshold, you have prize_prob chance of winning
    """
    exp_sales = expected_sales(Q, a, b)
    exp_leftover = Q - exp_sales
    
    # Base profit components
    revenue = exp_sales * p
    salvage = exp_leftover * c * f
    shipping_cost = exp_leftover * s
    ordering_cost = K
    variable_cost = Q * c
    
    base_profit = revenue + salvage - shipping_cost - ordering_cost - variable_cost
    
    # Prize component
    # P(win) = P(sales >= threshold) * prize_prob
    p_sales_ge = prob_sales_ge_threshold(Q, threshold, a, b)
    expected_prize = p_sales_ge * prize_prob * prize
    
    total_expected_profit = base_profit + expected_prize
    
    return {
        'Q': Q,
        'expected_sales': exp_sales,
        'expected_leftover': exp_leftover,
        'base_profit': base_profit,
        'p_sales_ge_threshold': p_sales_ge,
        'expected_prize': expected_prize,
        'total_expected_profit': total_expected_profit
    }

# =============================================================================
# Part 6: Find Q** that maximizes expected profit with prize
# =============================================================================

print("=" * 70)
print("PART 6: Behavioral Incentive (Prize)")
print("=" * 70)
print()
print(f"Prize: ${prize:,}")
print(f"Assumption: {prize_prob*100:.0f}% chance of winning if sales >= {prize_threshold}")
print(f"Expected prize value if eligible: ${prize * prize_prob:,.0f}")
print()

# Search for optimal Q** by evaluating all integer values
best_Q = None
best_profit = float('-inf')
results = []

for Q in range(a, b + 50):  # search beyond max demand just in case
    result = calculate_expected_profit_with_prize(Q, p, c, f, s, K, a, b, prize, prize_threshold, prize_prob)
    results.append(result)
    
    if result['total_expected_profit'] > best_profit:
        best_profit = result['total_expected_profit']
        best_Q = Q

Q_star_star = best_Q
Q_star = 270  # from Part 2

print(f"Optimal Q** (with prize) = {Q_star_star}")
print(f"Original Q* (without prize) = {Q_star}")
print()

# Get detailed results for both
result_without = calculate_expected_profit_with_prize(Q_star, p, c, f, s, K, a, b, prize, prize_threshold, prize_prob)
result_with = calculate_expected_profit_with_prize(Q_star_star, p, c, f, s, K, a, b, prize, prize_threshold, prize_prob)

print("=" * 70)
print("COMPARISON: Q* vs Q**")
print("=" * 70)
print()
print(f"{'Metric':<30} {'Q*='+str(Q_star):<20} {'Q**='+str(Q_star_star):<20}")
print("-" * 70)
print(f"{'Expected Sales':<30} {result_without['expected_sales']:<20.2f} {result_with['expected_sales']:<20.2f}")
print(f"{'Expected Leftover':<30} {result_without['expected_leftover']:<20.2f} {result_with['expected_leftover']:<20.2f}")
print(f"{'Base Profit':<30} ${result_without['base_profit']:<19.2f} ${result_with['base_profit']:<19.2f}")
print(f"{'P(Sales >= {threshold})':<30} {result_without['p_sales_ge_threshold']:<20.4f} {result_with['p_sales_ge_threshold']:<20.4f}".format(threshold=prize_threshold))
print(f"{'Expected Prize Value':<30} ${result_without['expected_prize']:<19.2f} ${result_with['expected_prize']:<19.2f}")
print(f"{'Total Expected Profit':<30} ${result_without['total_expected_profit']:<19.2f} ${result_with['total_expected_profit']:<19.2f}")
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
print("Behavioral Note:")
print("  The expected prize value (${:.0f}) is relatively small compared to the".format(result_with['expected_prize']))
print("  emotional appeal of potentially winning a $40,000 Corvette.")
print("  Real decision-makers might over-order beyond Q** due to this psychological effect.")

# =============================================================================
# Also analyze other prize thresholds mentioned in problem
# =============================================================================

print()
print("=" * 70)
print("SENSITIVITY TO PRIZE RULES")
print("=" * 70)
print()

prize_scenarios = [
    (380, 0.03, "3% @ 380"),
    (400, 0.05, "5% @ 400 (our assumption)"),
    (420, 0.07, "7% @ 420")
]

for threshold, prob, desc in prize_scenarios:
    # Find optimal Q for this scenario
    best_Q_scenario = None
    best_profit_scenario = float('-inf')
    
    for Q in range(a, b + 50):
        result = calculate_expected_profit_with_prize(Q, p, c, f, s, K, a, b, prize, threshold, prob)
        if result['total_expected_profit'] > best_profit_scenario:
            best_profit_scenario = result['total_expected_profit']
            best_Q_scenario = Q
    
    result = calculate_expected_profit_with_prize(best_Q_scenario, p, c, f, s, K, a, b, prize, threshold, prob)
    print(f"{desc}: Q** = {best_Q_scenario}, E[Prize] = ${result['expected_prize']:.2f}, "
          f"Total E[Profit] = ${result['total_expected_profit']:.2f}")

# =============================================================================
# Save to CSV
# =============================================================================

csv_file = os.path.join(CSV_DIR, 'q6_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 6 - Behavioral Incentive (Prize)\n")
    file.write("\n")
    file.write("Assumptions\n")
    file.write(f"Prize Value,${prize}\n")
    file.write(f"Prize Threshold,{prize_threshold}\n")
    file.write(f"Prize Probability,{prize_prob}\n")
    file.write("\n")
    
    file.write("Results\n")
    file.write(f"Q* (without prize),{Q_star}\n")
    file.write(f"Q** (with prize),{Q_star_star}\n")
    file.write(f"Base Profit at Q*,${result_without['base_profit']:.2f}\n")
    file.write(f"Base Profit at Q**,${result_with['base_profit']:.2f}\n")
    file.write(f"Expected Prize at Q*,${result_without['expected_prize']:.2f}\n")
    file.write(f"Expected Prize at Q**,${result_with['expected_prize']:.2f}\n")
    file.write(f"Total E[Profit] at Q*,${result_without['total_expected_profit']:.2f}\n")
    file.write(f"Total E[Profit] at Q**,${result_with['total_expected_profit']:.2f}\n")
    file.write("\n")
    
    file.write("Profit by Order Quantity\n")
    file.write("Q,Expected Sales,Base Profit,P(Sales>=Threshold),Expected Prize,Total E[Profit]\n")
    for r in results:
        file.write(f"{r['Q']},{r['expected_sales']:.2f},{r['base_profit']:.2f},"
                   f"{r['p_sales_ge_threshold']:.4f},{r['expected_prize']:.2f},{r['total_expected_profit']:.2f}\n")

print(f"\nCSV saved to: {csv_file}")

# =============================================================================
# Generate Plots
# =============================================================================

# Extract data for plotting
Q_vals = [r['Q'] for r in results]
base_profits = [r['base_profit'] for r in results]
total_profits = [r['total_expected_profit'] for r in results]
prize_EVs = [r['expected_prize'] for r in results]

# Plot 1: Expected Profit With vs Without Prize
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

# Plot 2: Profit Breakdown at Q**
fig, ax = plt.subplots(figsize=(7, 5))
components = ['Base Profit', 'Expected Prize', 'Total']
values = [result_with['base_profit'], result_with['expected_prize'], result_with['total_expected_profit']]
colors = [PRIMARY_BLUE, 'gold', ACCENT_RED]

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

# Plot 3: Expected Prize vs Q (threshold effect)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Q_vals, prize_EVs, color='gold', linewidth=2, label='E[Prize]')
ax.axvline(x=prize_threshold, color=ACCENT_RED, linestyle='--', linewidth=2, 
           label=f'Threshold = {prize_threshold}')

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Expected Prize Value ($)', fontsize=12)
ax.set_title('Part 6: Expected Prize Value by Order Quantity', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

ax.annotate('Q < 400:\nNo prize eligibility', xy=(300, 10), fontsize=10, color=ACCENT_RED)
ax.annotate('Q >= 400:\nPrize possible', xy=(420, 50), fontsize=10, color='green')

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'q6_prize_threshold.png')
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot3_path}")
