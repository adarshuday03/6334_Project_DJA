# team dja - dominic, jatin, adarsh
# isye 6334 group project fall 2025
# task 1 & 2

import os
import matplotlib.pyplot as plt
import numpy as np

# setup directories for output
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
CSV_DIR = os.path.join(OUTPUT_DIR, 'csv')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# model parameters

c = 3.00        # wholesale cost per unit
p = 5           # selling price per unit
f = 0.50        # refund fraction (50% of cost refunded)
s = 0.50        # return shipping cost per unit
K = 20          # fixed administrative fee per order

# demand distribution: uniform(a, b)
a = 120         # minimum demand
b = 420         # maximum demand
mean_demand = (a + b) / 2   # = 270

# part 1: conceptual analysis

# cost of overage (co): what we lose per unsold unit
# when we don't sell a unit:
#   - we paid c for it
#   - we get back f*c as refund
#   - we pay s to ship it back
# net loss = c - f*c + s = c*(1-f) + s

Co = c * (1 - f) + s

# cost of underage (cu): opportunity cost per unit of unmet demand
# when we miss a sale:
#   - we lose the profit margin we could have made
# lost profit = p - c

Cu = p - c

# the critical ratio tells us where to set q* in the demand distribution
critical_ratio = Cu / (Cu + Co)

print("=" * 60)
print("PART 1: Conceptual Analysis")
print("=" * 60)
print(f"\nCost of Overage (Co) = c*(1-f) + s = {c}*(1-{f}) + {s} = ${Co:.2f}")
print(f"Cost of Underage (Cu) = p - c = {p} - {c} = ${Cu:.2f}")
print(f"\nCritical Ratio = Cu/(Cu+Co) = {Cu}/({Cu}+{Co}) = {critical_ratio:.4f}")

if critical_ratio > 0.5:
    print(f"\nSince critical ratio ({critical_ratio:.2f}) > 0.5, Q* should be ABOVE mean demand ({mean_demand})")
elif critical_ratio < 0.5:
    print(f"\nSince critical ratio ({critical_ratio:.2f}) < 0.5, Q* should be BELOW mean demand ({mean_demand})")
else:
    print(f"\nSince critical ratio ({critical_ratio:.2f}) = 0.5, Q* should EQUAL mean demand ({mean_demand})")

print("\nExplanation:")
print(f"  - Every unsold unit costs us ${Co:.2f}")
print(f"  - Every missed sale costs us ${Cu:.2f}")
if Co == Cu:
    print("  - Since these are equal, we balance exactly at the median (= mean for uniform)")

# part 2: optimal order quantity

# for uniform(a, b), the optimal q* satisfies:
# f(q*) = (q* - a) / (b - a) = critical_ratio
# therefore: q* = a + (b - a) * critical_ratio

Q_star = a + (b - a) * critical_ratio

print("\n" + "=" * 60)
print("PART 2: Optimal Order Quantity")
print("=" * 60)
print(f"\nFor Uniform({a}, {b}) distribution:")
print(f"Q* = a + (b-a) * critical_ratio")
print(f"Q* = {a} + ({b}-{a}) * {critical_ratio:.4f}")
print(f"Q* = {Q_star:.2f} units")

# expected profit calculation

# for uniform distribution, expected sales when ordering q:
# e[sales] = integral from a to q of (x * 1/(b-a)) dx + integral from q to b of (q * 1/(b-a)) dx
#          = (q^2 - a^2) / (2*(b-a)) + q*(b-q) / (b-a)

def expected_sales(Q, a, b):
    """calculate expected sales for order quantity q with uniform(a,b) demand"""
    # sell everything ordered
    if Q <= a:
        return Q
    # expected demand
    elif Q >= b:
        return (a + b) / 2
    else:
        # integral calculation for uniform
        # when demand < q
        part1 = (Q**2 - a**2) / (2 * (b - a))
        # when demand >= q
        part2 = Q * (b - Q) / (b - a)
        return part1 + part2

def expected_leftover(Q, a, b):
    """calculate expected leftover inventory"""
    return Q - expected_sales(Q, a, b)

def calculate_profit(Q, p, c, f, s, K, a, b):
    """
    calculate expected profit for given order quantity q
    
    returns: dictionary with all components
    """
    exp_sales = expected_sales(Q, a, b)
    exp_leftover = expected_leftover(Q, a, b)
    
    revenue = exp_sales * p
    salvage = exp_leftover * c * f          # refund on unsold units
    shipping_cost = exp_leftover * s        # cost to ship back unsold
    ordering_cost = K                       # fixed cost
    variable_cost = Q * c                   # cost of all units ordered
    
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

# calculate for q*
result = calculate_profit(Q_star, p, c, f, s, K, a, b)

print(f"\n--- Expected Profit Breakdown at Q* = {Q_star:.0f} ---")
print(f"Expected Sales:      {result['expected_sales']:.2f} units")
print(f"Expected Leftover:   {result['expected_leftover']:.2f} units")
print(f"")
print(f"a. Revenue:          ${result['revenue']:.2f}")
print(f"b. Salvage Value:    ${result['salvage']:.2f}")
print(f"c. Shipping Cost:    (${result['shipping_cost']:.2f})")
print(f"d. Ordering Cost:    (${result['ordering_cost']:.2f})")
print(f"e. Variable Cost:    (${result['variable_cost']:.2f})")
print(f"")
print(f"Expected Profit:     ${result['expected_profit']:.2f}")

# save results to csv

csv_file = os.path.join(CSV_DIR, 'q1_q2_results.csv')

with open(csv_file, 'w') as file:
    file.write("SparkFire Newsvendor Model - Parts 1 & 2\n")
    file.write("\n")
    
    file.write("Parameters\n")
    file.write(f"Wholesale cost (c),${c:.2f}\n")
    file.write(f"Selling price (p),${p:.2f}\n")
    file.write(f"Refund fraction (f),{f}\n")
    file.write(f"Return shipping (s),${s:.2f}\n")
    file.write(f"Admin fee (K),${K:.2f}\n")
    file.write(f"Demand min (a),{a}\n")
    file.write(f"Demand max (b),{b}\n")
    file.write("\n")
    
    file.write("Part 1 - Conceptual Analysis\n")
    file.write(f"Cost of Overage (Co),${Co:.2f}\n")
    file.write(f"Cost of Underage (Cu),${Cu:.2f}\n")
    file.write(f"Critical Ratio,{critical_ratio:.4f}\n")
    file.write(f"Mean Demand,{mean_demand}\n")
    file.write(f"Q* relative to mean,Equal (since ratio = 0.5)\n")
    file.write("\n")
    
    file.write("Part 2 - Optimal Order Quantity\n")
    file.write(f"Optimal Q*,{Q_star:.2f}\n")
    file.write(f"Expected Sales,{result['expected_sales']:.2f}\n")
    file.write(f"Expected Leftover,{result['expected_leftover']:.2f}\n")
    file.write(f"Revenue,${result['revenue']:.2f}\n")
    file.write(f"Salvage Value,${result['salvage']:.2f}\n")
    file.write(f"Shipping Cost,${result['shipping_cost']:.2f}\n")
    file.write(f"Ordering Cost,${result['ordering_cost']:.2f}\n")
    file.write(f"Variable Cost,${result['variable_cost']:.2f}\n")
    file.write(f"Expected Profit,${result['expected_profit']:.2f}\n")

print(f"\nCSV saved to: {csv_file}")

# generate plots

PRIMARY_BLUE = '#003366'
ACCENT_RED = '#990000'

# plot 1: expected profit vs order quantity
Q_range = np.linspace(a, b, 300)
profits = [calculate_profit(Q, p, c, f, s, K, a, b)['expected_profit'] for Q in Q_range]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Q_range, profits, color=PRIMARY_BLUE, linewidth=2, label='E[Profit]')
ax.axvline(x=Q_star, color=ACCENT_RED, linestyle='--', linewidth=1.5, label=f'Q* = {Q_star:.0f}')
ax.axhline(y=result['expected_profit'], color='gray', linestyle=':', alpha=0.7)

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Parts 1-2: Expected Profit vs Order Quantity', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(a, b)

# mark the optimal point
ax.scatter([Q_star], [result['expected_profit']], color=ACCENT_RED, s=100, zorder=5)
ax.annotate(f"  Q* = {Q_star:.0f}\n  E[Profit] = ${result['expected_profit']:.0f}", 
            xy=(Q_star, result['expected_profit']), fontsize=10, color=ACCENT_RED)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q1_q2_profit_curve.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# plot 2: cost structure (cu vs co)
fig, ax = plt.subplots(figsize=(6, 5))
costs = [Cu, Co]
labels = ['Underage Cost\n(Cu)', 'Overage Cost\n(Co)']
colors = [ACCENT_RED, PRIMARY_BLUE]

bars = ax.bar(labels, costs, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Cost per Unit ($)', fontsize=12)
ax.set_title('Cost Structure: Cu vs Co', fontsize=14, fontweight='bold')

for bar, cost in zip(bars, costs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
            f'${cost:.2f}', ha='center', fontsize=12, fontweight='bold')

ax.set_ylim(0, max(costs) * 1.3)
ax.axhline(y=Cu, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q1_q2_cost_structure.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")
