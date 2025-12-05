"""
ISYE 6334 - Group Project (Fall 2025)
Part 7: Quantity Discounts

All-units discount structure:
  1-199 units:   $3.00 per unit
  200-399 units: $2.85 per unit
  400+ units:    $2.70 per unit
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
TIER_COLORS = ['#003366', '#990000', '#006600']

# =============================================================================
# Parameters
# =============================================================================

p = 5           # selling price
f = 0.50        # refund fraction
s = 0.50        # return shipping cost
K = 20          # admin fee

a = 120         # min demand
b = 420         # max demand

# Discount tiers: (min_qty, max_qty, unit_cost)
discount_tiers = [
    (1, 199, 3.00),
    (200, 399, 2.85),
    (400, float('inf'), 2.70)
]

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

def calculate_profit(Q, p, c, f, s, K, a, b):
    """Calculate expected profit for given Q and cost c"""
    exp_sales = expected_sales(Q, a, b)
    exp_leftover = Q - exp_sales
    
    revenue = exp_sales * p
    salvage = exp_leftover * c * f
    shipping_cost = exp_leftover * s
    ordering_cost = K
    variable_cost = Q * c
    
    profit = revenue + salvage - shipping_cost - ordering_cost - variable_cost
    
    return {
        'Q': Q,
        'c': c,
        'expected_sales': exp_sales,
        'expected_leftover': exp_leftover,
        'revenue': revenue,
        'salvage': salvage,
        'shipping_cost': shipping_cost,
        'variable_cost': variable_cost,
        'expected_profit': profit
    }

def find_optimal_Q_for_cost(c, p, f, s, a, b):
    """Find newsvendor optimal Q* for a given unit cost c"""
    Co = c * (1 - f) + s
    Cu = p - c
    critical_ratio = Cu / (Cu + Co)
    Q_star = a + (b - a) * critical_ratio
    return Q_star, critical_ratio, Co, Cu

# =============================================================================
# Part 7: Quantity Discount Analysis
# =============================================================================

print("=" * 70)
print("PART 7: Quantity Discounts")
print("=" * 70)
print()
print("Discount Structure:")
print("  1-199 units:   $3.00/unit")
print("  200-399 units: $2.85/unit")
print("  400+ units:    $2.70/unit")
print()

# Step 1: Find optimal Q* for each cost level
print("=" * 70)
print("STEP 1: Optimal Q* at each cost level (ignoring quantity constraints)")
print("=" * 70)
print()

tier_analysis = []

for min_qty, max_qty, c in discount_tiers:
    Q_star, crit_ratio, Co, Cu = find_optimal_Q_for_cost(c, p, f, s, a, b)
    
    # Check if Q* falls within this tier's range
    if max_qty == float('inf'):
        max_display = "inf"
        in_range = Q_star >= min_qty
    else:
        max_display = str(max_qty)
        in_range = min_qty <= Q_star <= max_qty
    
    tier_analysis.append({
        'min_qty': min_qty,
        'max_qty': max_qty,
        'c': c,
        'Q_star': Q_star,
        'crit_ratio': crit_ratio,
        'Co': Co,
        'Cu': Cu,
        'in_range': in_range
    })
    
    print(f"Tier {min_qty}-{max_display} @ ${c:.2f}:")
    print(f"  Co = {c}*(1-{f}) + {s} = ${Co:.2f}")
    print(f"  Cu = {p} - {c} = ${Cu:.2f}")
    print(f"  Critical Ratio = {crit_ratio:.4f}")
    print(f"  Unconstrained Q* = {Q_star:.2f}")
    print(f"  Falls in tier range? {'YES' if in_range else 'NO'}")
    print()

# Step 2: Identify candidate solutions
print("=" * 70)
print("STEP 2: Candidate Solutions")
print("=" * 70)
print()

candidates = []

for tier in tier_analysis:
    min_qty = tier['min_qty']
    max_qty = tier['max_qty']
    c = tier['c']
    Q_star = tier['Q_star']
    
    if tier['in_range']:
        # Q* is valid for this tier
        Q_candidate = Q_star
        candidates.append({'Q': Q_candidate, 'c': c, 'reason': f'Q* within tier'})
        print(f"Q = {Q_candidate:.1f} @ ${c:.2f}: Q* falls within range [{min_qty}, {max_qty if max_qty != float('inf') else 'inf'}]")
    else:
        # Q* is outside range - check breakpoints
        if Q_star < min_qty and min_qty > 1:
            # Q* is below this tier - consider minimum of tier
            Q_candidate = min_qty
            candidates.append({'Q': Q_candidate, 'c': c, 'reason': f'Tier minimum (breakpoint)'})
            print(f"Q = {Q_candidate} @ ${c:.2f}: Breakpoint (Q*={Q_star:.1f} < min={min_qty})")
        elif Q_star > max_qty and max_qty != float('inf'):
            # Q* is above this tier - consider maximum
            Q_candidate = max_qty
            candidates.append({'Q': Q_candidate, 'c': c, 'reason': f'Tier maximum'})
            print(f"Q = {Q_candidate} @ ${c:.2f}: Tier maximum (Q*={Q_star:.1f} > max={max_qty})")

print()

# Step 3: Calculate profit for each candidate
print("=" * 70)
print("STEP 3: Expected Profit for Each Candidate")
print("=" * 70)
print()

best_candidate = None
best_profit = float('-inf')

print(f"{'Q':<10} {'Cost':<10} {'E[Sales]':<12} {'E[Leftover]':<14} {'E[Profit]':<12} {'Reason'}")
print("-" * 75)

for cand in candidates:
    result = calculate_profit(cand['Q'], p, cand['c'], f, s, K, a, b)
    cand['result'] = result
    
    print(f"{cand['Q']:<10.1f} ${cand['c']:<9.2f} {result['expected_sales']:<12.2f} "
          f"{result['expected_leftover']:<14.2f} ${result['expected_profit']:<11.2f} {cand['reason']}")
    
    if result['expected_profit'] > best_profit:
        best_profit = result['expected_profit']
        best_candidate = cand

print()

# Step 4: Report optimal
Q_star_d = best_candidate['Q']
c_star_d = best_candidate['c']
profit_star_d = best_candidate['result']['expected_profit']

# Compare to Part 2 baseline
Q_star_base = 270
c_base = 3.00
result_base = calculate_profit(Q_star_base, p, c_base, f, s, K, a, b)
profit_base = result_base['expected_profit']

print("=" * 70)
print("RESULTS")
print("=" * 70)
print()
print(f"Optimal with discounts: Q*_d = {Q_star_d:.0f} @ ${c_star_d:.2f}/unit")
print(f"Expected Profit: ${profit_star_d:.2f}")
print()
print(f"Baseline (Part 2): Q* = {Q_star_base} @ ${c_base:.2f}/unit")  
print(f"Expected Profit: ${profit_base:.2f}")
print()
print(f"Improvement from quantity discount: ${profit_star_d - profit_base:.2f}")
print()

print("=" * 70)
print("SUPPLY CHAIN COORDINATION")
print("=" * 70)
print()
print("Quantity discounts help coordinate the supply chain by:")
print("  1. Encouraging larger orders, reducing wholesaler's per-order costs")
print("  2. Shifting inventory risk to the retailer (SparkFire)")
print("  3. Aligning retailer incentives with supply chain efficiency")
print("  4. Reducing total transactions and logistics complexity")

# =============================================================================
# Save to CSV
# =============================================================================

csv_file = os.path.join(CSV_DIR, 'q7_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 7 - Quantity Discounts\n")
    file.write("\n")
    file.write("Discount Tiers\n")
    file.write("Min Qty,Max Qty,Unit Cost\n")
    for min_q, max_q, c in discount_tiers:
        file.write(f"{min_q},{max_q if max_q != float('inf') else 'inf'},${c:.2f}\n")
    file.write("\n")
    
    file.write("Tier Analysis\n")
    file.write("Tier,Unit Cost,Co,Cu,Crit Ratio,Unconstrained Q*,In Range\n")
    for t in tier_analysis:
        max_display = str(int(t['max_qty'])) if t['max_qty'] != float('inf') else 'inf'
        file.write(f"{t['min_qty']}-{max_display},${t['c']:.2f},${t['Co']:.2f},${t['Cu']:.2f},"
                   f"{t['crit_ratio']:.4f},{t['Q_star']:.2f},{'Yes' if t['in_range'] else 'No'}\n")
    file.write("\n")
    
    file.write("Candidate Solutions\n")
    file.write("Q,Unit Cost,Expected Sales,Expected Leftover,Expected Profit,Reason\n")
    for cand in candidates:
        r = cand['result']
        file.write(f"{cand['Q']:.1f},${cand['c']:.2f},{r['expected_sales']:.2f},"
                   f"{r['expected_leftover']:.2f},${r['expected_profit']:.2f},{cand['reason']}\n")
    file.write("\n")
    
    file.write("Optimal Solution\n")
    file.write(f"Q*_d,{Q_star_d:.0f}\n")
    file.write(f"Unit Cost,${c_star_d:.2f}\n")
    file.write(f"Expected Profit,${profit_star_d:.2f}\n")
    file.write(f"Improvement over baseline,${profit_star_d - profit_base:.2f}\n")

print(f"\nCSV saved to: {csv_file}")

# =============================================================================
# Generate Plots
# =============================================================================

# Plot 1: Cost Function (step function)
fig, ax = plt.subplots(figsize=(10, 5))

Q_plot = np.arange(1, 500)
costs = []
for Q in Q_plot:
    if Q < 200:
        costs.append(3.00)
    elif Q < 400:
        costs.append(2.85)
    else:
        costs.append(2.70)

ax.step(Q_plot, costs, where='post', color=PRIMARY_BLUE, linewidth=2)
ax.axvline(x=200, color='gray', linestyle=':', alpha=0.7)
ax.axvline(x=400, color='gray', linestyle=':', alpha=0.7)

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Unit Cost ($)', fontsize=12)
ax.set_title('Part 7: All-Units Quantity Discount Structure', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 500)
ax.set_ylim(2.5, 3.2)

# Annotate tiers
ax.annotate('$3.00\n(1-199)', xy=(100, 3.05), ha='center', fontsize=10)
ax.annotate('$2.85\n(200-399)', xy=(300, 2.90), ha='center', fontsize=10)
ax.annotate('$2.70\n(400+)', xy=(450, 2.75), ha='center', fontsize=10)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q7_cost_function.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# Plot 2: Expected Profit by Tier (separate curves)
fig, ax = plt.subplots(figsize=(10, 6))

for i, (min_q, max_q, cost) in enumerate(discount_tiers):
    max_display = int(max_q) if max_q != float('inf') else b + 50
    Q_range = np.arange(min_q, min(max_display, b + 50) + 1)
    profits = [calculate_profit(Q, p, cost, f, s, K, a, b)['expected_profit'] for Q in Q_range]
    
    label = f'${cost:.2f} ({min_q}-{max_display if max_q != float("inf") else "inf"})'
    ax.plot(Q_range, profits, color=TIER_COLORS[i], linewidth=2, label=label)

# Mark optimal
ax.scatter([Q_star_d], [profit_star_d], color=ACCENT_RED, s=150, zorder=5, marker='*')
ax.annotate(f'Q*_d = {Q_star_d:.0f}\n${profit_star_d:.0f}', 
            xy=(Q_star_d, profit_star_d), xytext=(Q_star_d + 20, profit_star_d + 10),
            fontsize=10, fontweight='bold', color=ACCENT_RED)

ax.axvline(x=200, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=400, color='gray', linestyle=':', alpha=0.5)

ax.set_xlabel('Order Quantity (Q)', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 7: Expected Profit by Discount Tier', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(a, b + 50)

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q7_profit_by_tier.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")

# Plot 3: Candidate Comparison Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))

cand_labels = [f"Q={int(c['Q'])}\n(${c['c']:.2f})" for c in candidates]
cand_profits = [c['result']['expected_profit'] for c in candidates]
colors = [ACCENT_RED if c['Q'] == Q_star_d else PRIMARY_BLUE for c in candidates]

bars = ax.bar(cand_labels, cand_profits, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 7: Candidate Solutions Comparison', fontsize=14, fontweight='bold')

for bar, profit in zip(bars, cand_profits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
            f'${profit:.0f}', ha='center', fontsize=10, fontweight='bold')

ax.axhline(y=profit_base, color='gray', linestyle='--', alpha=0.7, label=f'Baseline (no discount) = ${profit_base:.0f}')
ax.legend()
ax.set_ylim(0, max(cand_profits) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'q7_candidates.png')
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot3_path}")
