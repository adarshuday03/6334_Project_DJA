"""
ISYE 6334 - Group Project (Fall 2025)
Part 8: Context Signal & Deferred Purchasing

Two-stage ordering with market signal:
- Pre-season: Order Q0 at cost c0 = $3.00
- In-season: After signal S, order Q1(S) at cost c1 = $3.60

Signal: S in {High, Low}
P(High) = 0.45, P(Low) = 0.55
D|High ~ Uniform(240, 420)
D|Low ~ Uniform(120, 260)
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

p = 5           # selling price
f = 0.50        # refund fraction
s = 0.50        # return shipping cost
K = 20          # admin fee (per order)

c0 = 3.00       # pre-season cost
c1 = 3.60       # in-season expedited cost

# Signal probabilities
P_high = 0.45
P_low = 0.55

# Demand distributions by signal
# High signal: Uniform(a_H, b_H)
a_H, b_H = 240, 420

# Low signal: Uniform(a_L, b_L)
a_L, b_L = 120, 260

# Baseline profit from Part 2 (for VOSRC calculation)
baseline_profit = 370.00

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

def find_optimal_Q(c, p, f, s, a, b):
    """Find newsvendor optimal Q* for given cost c and demand Uniform(a,b)"""
    Co = c * (1 - f) + s
    Cu = p - c
    if Cu + Co == 0:
        return a
    critical_ratio = Cu / (Cu + Co)
    Q_star = a + (b - a) * critical_ratio
    return Q_star

def calculate_expected_profit_twostage(Q0, Q1_H, Q1_L, p, c0, c1, f, s, K, 
                                        a_H, b_H, a_L, b_L, P_high, P_low):
    """
    Calculate expected profit for two-stage policy
    
    Q0: pre-season order (at cost c0)
    Q1_H: top-up order if signal is High (at cost c1)
    Q1_L: top-up order if signal is Low (at cost c1)
    """
    
    # High signal scenario
    total_Q_H = Q0 + Q1_H
    exp_sales_H = expected_sales(total_Q_H, a_H, b_H)
    exp_leftover_H = total_Q_H - exp_sales_H
    
    revenue_H = exp_sales_H * p
    # Salvage: leftover from pre-season at f*c0, leftover from in-season at f*c1
    # Simplification: assume proportional allocation
    if total_Q_H > 0:
        frac_c0 = Q0 / total_Q_H
        frac_c1 = Q1_H / total_Q_H
        avg_salvage_rate = frac_c0 * c0 * f + frac_c1 * c1 * f
    else:
        avg_salvage_rate = 0
    salvage_H = exp_leftover_H * avg_salvage_rate
    shipping_H = exp_leftover_H * s
    cost_H = Q0 * c0 + Q1_H * c1
    order_cost_H = K if Q1_H > 0 else 0  # extra order cost if we order in-season
    
    profit_H = revenue_H + salvage_H - shipping_H - cost_H - K - order_cost_H
    
    # Low signal scenario
    total_Q_L = Q0 + Q1_L
    exp_sales_L = expected_sales(total_Q_L, a_L, b_L)
    exp_leftover_L = total_Q_L - exp_sales_L
    
    revenue_L = exp_sales_L * p
    if total_Q_L > 0:
        frac_c0_L = Q0 / total_Q_L
        frac_c1_L = Q1_L / total_Q_L
        avg_salvage_rate_L = frac_c0_L * c0 * f + frac_c1_L * c1 * f
    else:
        avg_salvage_rate_L = 0
    salvage_L = exp_leftover_L * avg_salvage_rate_L
    shipping_L = exp_leftover_L * s
    cost_L = Q0 * c0 + Q1_L * c1
    order_cost_L = K if Q1_L > 0 else 0
    
    profit_L = revenue_L + salvage_L - shipping_L - cost_L - K - order_cost_L
    
    # Expected profit across both scenarios
    expected_profit = P_high * profit_H + P_low * profit_L
    
    return {
        'Q0': Q0,
        'Q1_H': Q1_H,
        'Q1_L': Q1_L,
        'total_Q_H': total_Q_H,
        'total_Q_L': total_Q_L,
        'exp_sales_H': exp_sales_H,
        'exp_sales_L': exp_sales_L,
        'exp_leftover_H': exp_leftover_H,
        'exp_leftover_L': exp_leftover_L,
        'profit_H': profit_H,
        'profit_L': profit_L,
        'expected_profit': expected_profit
    }

# =============================================================================
# Part 8(a): Formulate profit function - done above
# =============================================================================

print("=" * 70)
print("PART 8: Context Signal & Deferred Purchasing")
print("=" * 70)
print()
print("Two-Stage Model:")
print(f"  Pre-season cost c0 = ${c0:.2f}")
print(f"  In-season cost c1 = ${c1:.2f}")
print(f"  P(High) = {P_high}, P(Low) = {P_low}")
print(f"  D|High ~ Uniform({a_H}, {b_H})")
print(f"  D|Low ~ Uniform({a_L}, {b_L})")
print()

# =============================================================================
# Part 8(b): Find optimal Q*_H and Q*_L using second-stage cost
# =============================================================================

print("=" * 70)
print("PART 8(b): Optimal Total Inventory by Signal (using c1=${:.2f})".format(c1))
print("=" * 70)
print()

# These are the optimal TOTAL quantities if all inventory were at c1
Q_star_H = find_optimal_Q(c1, p, f, s, a_H, b_H)
Q_star_L = find_optimal_Q(c1, p, f, s, a_L, b_L)

# Calculate Co and Cu for explanation
Co_c1 = c1 * (1 - f) + s
Cu_c1 = p - c1
crit_ratio_c1 = Cu_c1 / (Cu_c1 + Co_c1)

print(f"Using expedited cost c1 = ${c1:.2f}:")
print(f"  Co = {c1}*(1-{f}) + {s} = ${Co_c1:.2f}")
print(f"  Cu = {p} - {c1} = ${Cu_c1:.2f}")
print(f"  Critical Ratio = {crit_ratio_c1:.4f}")
print()
print(f"Q*_H (High signal): {a_H} + ({b_H}-{a_H})*{crit_ratio_c1:.4f} = {Q_star_H:.2f}")
print(f"Q*_L (Low signal): {a_L} + ({b_L}-{a_L})*{crit_ratio_c1:.4f} = {Q_star_L:.2f}")
print()

# =============================================================================
# Part 8(c): Optimize Q0
# =============================================================================

print("=" * 70)
print("PART 8(c): Optimize Initial Order Q0")
print("=" * 70)
print()

# Search over possible Q0 values
best_Q0 = None
best_profit = float('-inf')
all_results = []

# Q0 should be between 0 and min(Q*_H, Q*_L) approximately
# But we'll search broadly to be safe
for Q0 in range(0, 400):
    # Given Q0, the top-up orders are:
    # Q1(H) = max(0, Q*_H - Q0)
    # Q1(L) = max(0, Q*_L - Q0)
    Q1_H = max(0, Q_star_H - Q0)
    Q1_L = max(0, Q_star_L - Q0)
    
    result = calculate_expected_profit_twostage(
        Q0, Q1_H, Q1_L, p, c0, c1, f, s, K,
        a_H, b_H, a_L, b_L, P_high, P_low
    )
    all_results.append(result)
    
    if result['expected_profit'] > best_profit:
        best_profit = result['expected_profit']
        best_Q0 = Q0
        best_result = result

# Get the optimal policy
Q0_star = best_Q0
Q1_star_H = max(0, Q_star_H - Q0_star)
Q1_star_L = max(0, Q_star_L - Q0_star)

print(f"Optimal Policy:")
print(f"  Q0* = {Q0_star} (pre-season order)")
print(f"  Q1*(High) = {Q1_star_H:.2f} (top-up if High signal)")
print(f"  Q1*(Low) = {Q1_star_L:.2f} (top-up if Low signal)")
print()
print(f"Total inventory:")
print(f"  If High: {Q0_star} + {Q1_star_H:.2f} = {Q0_star + Q1_star_H:.2f}")
print(f"  If Low: {Q0_star} + {Q1_star_L:.2f} = {Q0_star + Q1_star_L:.2f}")
print()
print(f"Expected Profit (Full Model): ${best_result['expected_profit']:.2f}")
print()

# Also show breakdown
print("Profit Breakdown:")
print(f"  Profit if High signal: ${best_result['profit_H']:.2f}")
print(f"  Profit if Low signal: ${best_result['profit_L']:.2f}")
print(f"  Expected Profit: {P_high}*{best_result['profit_H']:.2f} + {P_low}*{best_result['profit_L']:.2f} = ${best_result['expected_profit']:.2f}")
print()

# =============================================================================
# Part 8(d): VOSRC Calculation
# =============================================================================

print("=" * 70)
print("PART 8(d): Value of Signal and Reactive Capacity (VOSRC)")
print("=" * 70)
print()

VOSRC = best_result['expected_profit'] - baseline_profit

print(f"Full Model Profit: ${best_result['expected_profit']:.2f}")
print(f"Baseline Profit (Part 2): ${baseline_profit:.2f}")
print(f"VOSRC = {best_result['expected_profit']:.2f} - {baseline_profit:.2f} = ${VOSRC:.2f}")
print()

if VOSRC > 0:
    print("The signal and reactive capacity ADD value.")
    print(f"SparkFire should pay up to ${VOSRC:.2f} for access to the signal and expedited ordering.")
else:
    print("The signal and reactive capacity do NOT add value (VOSRC < 0).")
    print("The expedited cost premium ($3.60 vs $3.00) is too high to justify.")
    print("SparkFire is better off with the single-stage optimal from Part 2.")

print()
print("Justification:")
if VOSRC < 0:
    print("  The 20% cost premium on expedited orders ($3.60 vs $3.00) erodes the")
    print("  value of having demand information. The signal quality is not high enough")
    print("  to overcome this premium. It would be rational to pay for expedited")
    print("  capacity only if c1 were lower or the signal were more informative.")
else:
    print("  The signal provides enough information about demand to profitably adjust")
    print("  inventory levels. The value exceeds the expedited cost premium.")

# =============================================================================
# Save to CSV
# =============================================================================

csv_file = os.path.join(CSV_DIR, 'q8_results.csv')

with open(csv_file, 'w') as file:
    file.write("Part 8 - Context Signal & Deferred Purchasing\n")
    file.write("\n")
    file.write("Parameters\n")
    file.write(f"Pre-season cost (c0),${c0:.2f}\n")
    file.write(f"In-season cost (c1),${c1:.2f}\n")
    file.write(f"P(High),{P_high}\n")
    file.write(f"P(Low),{P_low}\n")
    file.write(f"D|High,Uniform({a_H},{b_H})\n")
    file.write(f"D|Low,Uniform({a_L},{b_L})\n")
    file.write("\n")
    
    file.write("Part 8(b) - Optimal Total by Signal\n")
    file.write(f"Q*_H (at c1),{Q_star_H:.2f}\n")
    file.write(f"Q*_L (at c1),{Q_star_L:.2f}\n")
    file.write("\n")
    
    file.write("Part 8(c) - Optimal Policy\n")
    file.write(f"Q0*,{Q0_star}\n")
    file.write(f"Q1*(High),{Q1_star_H:.2f}\n")
    file.write(f"Q1*(Low),{Q1_star_L:.2f}\n")
    file.write(f"Total Q if High,{Q0_star + Q1_star_H:.2f}\n")
    file.write(f"Total Q if Low,{Q0_star + Q1_star_L:.2f}\n")
    file.write(f"Profit if High,${best_result['profit_H']:.2f}\n")
    file.write(f"Profit if Low,${best_result['profit_L']:.2f}\n")
    file.write(f"Expected Profit,${best_result['expected_profit']:.2f}\n")
    file.write("\n")
    
    file.write("Part 8(d) - VOSRC\n")
    file.write(f"Full Model Profit,${best_result['expected_profit']:.2f}\n")
    file.write(f"Baseline Profit,${baseline_profit:.2f}\n")
    file.write(f"VOSRC,${VOSRC:.2f}\n")
    file.write("\n")
    
    file.write("Sensitivity - Expected Profit by Q0\n")
    file.write("Q0,Q1_H,Q1_L,Total_Q_H,Total_Q_L,Profit_H,Profit_L,E[Profit]\n")
    for r in all_results:
        file.write(f"{r['Q0']},{r['Q1_H']:.2f},{r['Q1_L']:.2f},{r['total_Q_H']:.2f},"
                   f"{r['total_Q_L']:.2f},{r['profit_H']:.2f},{r['profit_L']:.2f},"
                   f"{r['expected_profit']:.2f}\n")

print(f"\nCSV saved to: {csv_file}")

# =============================================================================
# Generate Plots
# =============================================================================

# Plot 1: Conditional Demand Distributions
fig, ax = plt.subplots(figsize=(10, 5))

# Unconditional (mixture)
x_uncond = np.linspace(120, 420, 500)
y_uncond = np.zeros_like(x_uncond)
for i, x in enumerate(x_uncond):
    if a_L <= x <= b_L:
        y_uncond[i] += P_low / (b_L - a_L)
    if a_H <= x <= b_H:
        y_uncond[i] += P_high / (b_H - a_H)

# Conditional distributions
x_high = np.linspace(a_H, b_H, 200)
y_high = np.ones_like(x_high) / (b_H - a_H)

x_low = np.linspace(a_L, b_L, 200)
y_low = np.ones_like(x_low) / (b_L - a_L)

ax.fill_between(x_high, y_high, alpha=0.3, color=ACCENT_RED, label=f'D|High ~ U[{a_H},{b_H}]')
ax.fill_between(x_low, y_low, alpha=0.3, color=PRIMARY_BLUE, label=f'D|Low ~ U[{a_L},{b_L}]')
ax.plot(x_uncond, y_uncond, color='black', linewidth=2, linestyle='--', label='Unconditional (mixture)')

ax.set_xlabel('Demand (cases)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Part 8: Conditional Demand Distributions by Signal', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(100, 440)

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q8_demand_distributions.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# Plot 2: Expected Profit vs Q0
Q0_vals = [r['Q0'] for r in all_results]
exp_profits = [r['expected_profit'] for r in all_results]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Q0_vals, exp_profits, color=PRIMARY_BLUE, linewidth=2, label='Two-Stage E[Profit]')
ax.axhline(y=baseline_profit, color=ACCENT_RED, linestyle='--', linewidth=2, label=f'Baseline = ${baseline_profit:.0f}')

ax.scatter([Q0_star], [best_result['expected_profit']], color=ACCENT_RED, s=100, zorder=5)
ax.annotate(f'Q0* = {Q0_star}\n${best_result["expected_profit"]:.0f}', 
            xy=(Q0_star, best_result['expected_profit']),
            xytext=(Q0_star + 20, best_result['expected_profit'] + 5),
            fontsize=10, fontweight='bold', color=ACCENT_RED)

ax.set_xlabel('Pre-season Order Q0', fontsize=12)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 8: Expected Profit vs Pre-season Order Quantity', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q8_profit_vs_Q0.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")

# Plot 3: Two-Stage vs Baseline Comparison
fig, ax = plt.subplots(figsize=(7, 5))
strategies = ['Two-Stage\n(with signal)', 'Baseline\n(no signal)']
profits = [best_result['expected_profit'], baseline_profit]
colors = [PRIMARY_BLUE, 'gray']

bars = ax.bar(strategies, profits, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Expected Profit ($)', fontsize=12)
ax.set_title('Part 8: VOSRC Comparison', fontsize=14, fontweight='bold')

for bar, profit in zip(bars, profits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
            f'${profit:.2f}', ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(profits) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

# Add VOSRC annotation
ax.annotate(f'VOSRC = ${VOSRC:.2f}', xy=(0.5, min(profits) - 20), 
            xycoords=('axes fraction', 'data'),
            ha='center', fontsize=12, fontweight='bold', color=ACCENT_RED)

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'q8_vosrc_comparison.png')
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot3_path}")

# Plot 4: Expedited Order Q1 vs Pre-season Q0
Q1_H_vals = [r['Q1_H'] for r in all_results]
Q1_L_vals = [r['Q1_L'] for r in all_results]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Q0_vals, Q1_H_vals, color=ACCENT_RED, linewidth=2, label='Q1*(High signal)')
ax.plot(Q0_vals, Q1_L_vals, color=PRIMARY_BLUE, linewidth=2, label='Q1*(Low signal)')

ax.axvline(x=Q0_star, color='gray', linestyle='--', alpha=0.7)
ax.scatter([Q0_star], [Q1_star_H], color=ACCENT_RED, s=80, zorder=5)
ax.scatter([Q0_star], [Q1_star_L], color=PRIMARY_BLUE, s=80, zorder=5)

ax.set_xlabel('Pre-season Order Q0', fontsize=12)
ax.set_ylabel('Expedited Order Q1', fontsize=12)
ax.set_title('Part 8: Optimal Expedited Order by Signal', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

ax.annotate(f'Q0*={Q0_star}', xy=(Q0_star, max(Q1_H_vals)/2), fontsize=10)

plt.tight_layout()
plot4_path = os.path.join(PLOTS_DIR, 'q8_Q1_vs_Q0.png')
plt.savefig(plot4_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot4_path}")
