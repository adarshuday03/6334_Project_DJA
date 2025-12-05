"""
ISYE 6334 - Group Project (Fall 2025)
Part 3: Refund Sensitivity Analysis

Recompute Q* for refund rates f = 0.25 and f = 0.75
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
s = 0.50        # return shipping cost
K = 20          # admin fee

a = 120         # min demand
b = 420         # max demand

# Refund scenarios to analyze
refund_rates = [0.25, 0.50, 0.75]

# =============================================================================
# Helper Functions (same as q1_q2)
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

def expected_leftover(Q, a, b):
    """Calculate expected leftover inventory"""
    return Q - expected_sales(Q, a, b)

def calculate_profit(Q, p, c, f, s, K, a, b):
    """Calculate expected profit components"""
    exp_sales = expected_sales(Q, a, b)
    exp_leftover = expected_leftover(Q, a, b)
    
    revenue = exp_sales * p
    salvage = exp_leftover * c * f
    shipping_cost = exp_leftover * s
    ordering_cost = K
    variable_cost = Q * c
    
    profit = revenue + salvage - shipping_cost - ordering_cost - variable_cost
    
    return {
        'Q': Q,
        'expected_sales': exp_sales,
        'expected_leftover': exp_leftover,
        'revenue': revenue,
        'salvage': salvage,
        'shipping_cost': shipping_cost,
        'ordering_cost': ordering_cost,
        'variable_cost': variable_cost,
        'expected_profit': profit
    }

# =============================================================================
# Part 3: Sensitivity Analysis
# =============================================================================

print("=" * 70)
print("PART 3: Refund Sensitivity Analysis")
print("=" * 70)
print(f"\nBase parameters: p=${p}, c=${c}, s=${s}, K=${K}")
print(f"Demand: Uniform({a}, {b})")
print()

results = []

for f in refund_rates:
    # Calculate costs
    Co = c * (1 - f) + s
    Cu = p - c
    critical_ratio = Cu / (Cu + Co)
    
    # Optimal order quantity
    Q_star = a + (b - a) * critical_ratio
    
    # Calculate profit
    profit_data = calculate_profit(Q_star, p, c, f, s, K, a, b)
    
    # Store results
    results.append({
        'f': f,
        'Co': Co,
        'Cu': Cu,
        'critical_ratio': critical_ratio,
        'Q_star': Q_star,
        'expected_sales': profit_data['expected_sales'],
        'expected_leftover': profit_data['expected_leftover'],
        'expected_profit': profit_data['expected_profit']
    })
    
    print(f"--- Refund Rate f = {f*100:.0f}% ---")
    print(f"  Co = c*(1-f) + s = {c}*(1-{f}) + {s} = ${Co:.2f}")
    print(f"  Cu = p - c = ${Cu:.2f}")
    print(f"  Critical Ratio = {critical_ratio:.4f}")
    print(f"  Q* = {Q_star:.2f}")
    print(f"  Expected Profit = ${profit_data['expected_profit']:.2f}")
    print()

# =============================================================================
# Analysis
# =============================================================================

print("=" * 70)
print("SUMMARY COMPARISON")
print("=" * 70)
print()
print(f"{'Refund Rate':<15} {'Co':<10} {'Crit Ratio':<12} {'Q*':<10} {'E[Profit]':<12}")
print("-" * 60)

for r in results:
    print(f"{r['f']*100:.0f}%{'':<12} ${r['Co']:.2f}{'':<5} {r['critical_ratio']:.4f}{'':<6} "
          f"{r['Q_star']:.1f}{'':<5} ${r['expected_profit']:.2f}")

print()
print("Key Observations:")
print("  1. Higher refund rate (f) -> Lower Co (less penalty for overage)")
print("  2. Lower Co -> Higher critical ratio -> Higher Q*")
print("  3. Higher refund generosity encourages ordering more inventory")

# =============================================================================
# Save to CSV
# =============================================================================

csv_file = os.path.join(CSV_DIR, 'q3_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 3 - Refund Sensitivity Analysis\n")
    file.write("\n")
    file.write("Refund Rate (f),Co,Cu,Critical Ratio,Q*,Expected Sales,Expected Leftover,Expected Profit\n")
    
    for r in results:
        file.write(f"{r['f']:.2f},${r['Co']:.2f},${r['Cu']:.2f},{r['critical_ratio']:.4f},"
                   f"{r['Q_star']:.2f},{r['expected_sales']:.2f},{r['expected_leftover']:.2f},"
                   f"${r['expected_profit']:.2f}\n")

print(f"\nCSV saved to: {csv_file}")

# =============================================================================
# Generate Plots
# =============================================================================

# Extract data for plots
f_vals = [r['f'] for r in results]
Q_vals = [r['Q_star'] for r in results]
profit_vals = [r['expected_profit'] for r in results]
Co_vals = [r['Co'] for r in results]

# Plot 1: Q* vs Refund Rate
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f_vals, Q_vals, 'o-', color=PRIMARY_BLUE, linewidth=2, markersize=10, label='Q*')
ax.set_xlabel('Refund Rate (f)', fontsize=12)
ax.set_ylabel('Optimal Order Quantity (Q*)', fontsize=12)
ax.set_title('Part 3: Q* vs Refund Rate', fontsize=14, fontweight='bold')
ax.set_xticks(f_vals)
ax.set_xticklabels([f'{f:.0%}' for f in f_vals])
ax.grid(True, alpha=0.3)

for f, Q in zip(f_vals, Q_vals):
    ax.annotate(f'{Q:.0f}', (f, Q), textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=11, fontweight='bold', color=PRIMARY_BLUE)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q3_Qstar_vs_refund.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# Plot 2: Expected Profit vs Refund Rate
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(f_vals, profit_vals, 's-', color=ACCENT_RED, linewidth=2, markersize=10, label='E[Profit]')
ax.set_xlabel('Refund Rate (f)', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 3: Expected Profit vs Refund Rate', fontsize=14, fontweight='bold')
ax.set_xticks(f_vals)
ax.set_xticklabels([f'{f:.0%}' for f in f_vals])
ax.grid(True, alpha=0.3)

for f, profit in zip(f_vals, profit_vals):
    ax.annotate(f'${profit:.0f}', (f, profit), textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=11, fontweight='bold', color=ACCENT_RED)

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q3_profit_vs_refund.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")

# Plot 3: Overage Cost vs Refund Rate
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(f_vals, Co_vals, width=0.15, color=PRIMARY_BLUE, edgecolor='black')
ax.set_xlabel('Refund Rate (f)', fontsize=12)
ax.set_ylabel('Overage Cost Co ($)', fontsize=12)
ax.set_title('Part 3: Overage Cost vs Refund Rate', fontsize=14, fontweight='bold')
ax.set_xticks(f_vals)
ax.set_xticklabels([f'{f:.0%}' for f in f_vals])

for f, Co in zip(f_vals, Co_vals):
    ax.text(f, Co + 0.05, f'${Co:.2f}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(Co_vals) * 1.3)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'q3_Co_vs_refund.png')
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot3_path}")
