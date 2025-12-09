"""
ISYE 6334 - Group Project (Fall 2025)
Part 4: Pricing Decision

Compare p=$5 vs p=$6, both with f=0.5
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Setup directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

PRIMARY_BLUE = '#003366'
ACCENT_RED = '#990000'

# Model parameters

c = 3.00        # wholesale cost
f = 0.50        # refund fraction
s = 0.50        # return shipping cost
K = 20          # admin fee

a = 120         # min demand
b = 420         # max demand

prices = [5, 6]

# Helper functions

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

# Part 4: Pricing Comparison

print("=" * 70)
print("PART 4: Pricing Decision")
print("=" * 70)
print(f"\nBase parameters: c=${c}, f={f}, s=${s}, K=${K}")
print(f"Demand: Uniform({a}, {b}) - assumed independent of price")
print()

results = []

for p in prices:
    # Calculate costs
    Co = c * (1 - f) + s
    Cu = p - c
    critical_ratio = Cu / (Cu + Co)
    
    # Optimal order quantity
    Q_star = a + (b - a) * critical_ratio
    
    # Calculate profit
    profit_data = calculate_profit(Q_star, p, c, f, s, K, a, b)
    
    results.append({
        'p': p,
        'Co': Co,
        'Cu': Cu,
        'critical_ratio': critical_ratio,
        'Q_star': Q_star,
        'expected_sales': profit_data['expected_sales'],
        'expected_leftover': profit_data['expected_leftover'],
        'revenue': profit_data['revenue'],
        'expected_profit': profit_data['expected_profit']
    })
    
    print(f"--- Selling Price p = ${p} ---")
    print(f"  Co = ${Co:.2f}")
    print(f"  Cu = p - c = {p} - {c} = ${Cu:.2f}")
    print(f"  Critical Ratio = {Cu}/{Cu}+{Co} = {critical_ratio:.4f}")
    print(f"  Q* = {Q_star:.2f}")
    print(f"  Expected Sales = {profit_data['expected_sales']:.2f}")
    print(f"  Expected Leftover = {profit_data['expected_leftover']:.2f}")
    print(f"  Revenue = ${profit_data['revenue']:.2f}")
    print(f"  Expected Profit = ${profit_data['expected_profit']:.2f}")
    print()

# Comparison and Recommendation

print("=" * 70)
print("COMPARISON")
print("=" * 70)
print()
print(f"{'Price':<10} {'Q*':<10} {'E[Sales]':<12} {'E[Leftover]':<14} {'E[Profit]':<12}")
print("-" * 60)

for r in results:
    print(f"${r['p']:<9} {r['Q_star']:<10.1f} {r['expected_sales']:<12.2f} "
          f"{r['expected_leftover']:<14.2f} ${r['expected_profit']:.2f}")

profit_diff = results[1]['expected_profit'] - results[0]['expected_profit']

print()
print("=" * 70)
print("RECOMMENDATION")
print("=" * 70)
print()
print(f"Setting p = $6 increases expected profit by ${profit_diff:.2f}")
print(f"(from ${results[0]['expected_profit']:.2f} to ${results[1]['expected_profit']:.2f})")
print()
print("Since demand is assumed independent of price, the $6 price is clearly better.")
print("We order more units (300 vs 270) and capture higher revenue per sale.")
print()
print("REFLECTION: If raising price decreased demand, then:")
print("  - The advantage of $6 would shrink or possibly reverse")
print("  - We would need to model price elasticity to find optimal price")
print("  - The demand distribution would shift left with higher prices")

# Save to CSV

csv_file = os.path.join(CSV_DIR, 'q4_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 4 - Pricing Decision\n")
    file.write("\n")
    file.write("Price (p),Co,Cu,Critical Ratio,Q*,Expected Sales,Expected Leftover,Revenue,Expected Profit\n")
    
    for r in results:
        file.write(f"${r['p']:.2f},${r['Co']:.2f},${r['Cu']:.2f},{r['critical_ratio']:.4f},"
                   f"{r['Q_star']:.2f},{r['expected_sales']:.2f},{r['expected_leftover']:.2f},"
                   f"${r['revenue']:.2f},${r['expected_profit']:.2f}\n")
    
    file.write("\n")
    file.write(f"Profit Improvement (p=$6 vs p=$5),${profit_diff:.2f}\n")
    file.write("Recommendation,Set price to $6\n")

print(f"\nCSV saved to: {csv_file}")

# Generate Plots

Q_range = np.linspace(a, b, 300)

fig, ax = plt.subplots(figsize=(10, 6))

for r in results:
    price = r['p']
    Co = r['Co']
    profits = []
    for Q in Q_range:
        profit_data = calculate_profit(Q, price, c, f, s, K, a, b)
        profits.append(profit_data['expected_profit'])
    
    color = PRIMARY_BLUE if price == 5 else ACCENT_RED
    ax.plot(Q_range, profits, color=color, linewidth=2, label=f'p = ${price}')
    ax.axvline(x=r['Q_star'], color=color, linestyle='--', alpha=0.7)
    ax.scatter([r['Q_star']], [r['expected_profit']], color=color, s=100, zorder=5)

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 4: Expected Profit Curves by Price', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(a, b)

# Annotate optimal points
ax.annotate(f'Q*={results[0]["Q_star"]:.0f}\n${results[0]["expected_profit"]:.0f}', 
            xy=(results[0]['Q_star'], results[0]['expected_profit']),
            xytext=(results[0]['Q_star']-40, results[0]['expected_profit']+50),
            fontsize=10, color=PRIMARY_BLUE)
ax.annotate(f'Q*={results[1]["Q_star"]:.0f}\n${results[1]["expected_profit"]:.0f}', 
            xy=(results[1]['Q_star'], results[1]['expected_profit']),
            xytext=(results[1]['Q_star']+10, results[1]['expected_profit']-30),
            fontsize=10, color=ACCENT_RED)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q4_profit_curves.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# Plot 2: Critical Ratio Comparison
fig, ax = plt.subplots(figsize=(6, 5))
prices_labels = [f'p=${r["p"]}' for r in results]
crit_ratios = [r['critical_ratio'] for r in results]
colors = [PRIMARY_BLUE, ACCENT_RED]

bars = ax.bar(prices_labels, crit_ratios, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Critical Ratio', fontsize=12)
ax.set_title('Part 4: Critical Ratio by Price', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)

for bar, ratio in zip(bars, crit_ratios):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{ratio:.2f}', ha='center', fontsize=12, fontweight='bold')

ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Mean demand threshold')
ax.legend()

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q4_critical_ratio.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")
