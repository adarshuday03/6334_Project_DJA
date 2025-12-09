"""
ISYE 6334 - Group Project (Fall 2025)
Part 5: Risk & Simulation

Simulate 500+ trials using optimal policy from Part 2 (p=$5, Q*=270)
Report mean profit, std dev, and probability of loss
"""

import os
import random
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
p = 5           # selling price
f = 0.50        # refund fraction
s = 0.50        # return shipping cost
K = 20          # admin fee

a = 120         # min demand
b = 420         # max demand

Q_star = 270    # optimal order quantity from Part 2

num_trials = 500
random_seeds = [6334, 1234, 5678]

# Helper functions

def simulate_single_trial(Q, demand, p, c, f, s, K):
    """
    Simulate profit for a single demand realization
    
    Parameters:
        Q: order quantity
        demand: realized demand
        p, c, f, s, K: model parameters
    
    Returns:
        dictionary with profit breakdown
    """
    # Actual sales = min(demand, Q)
    sales = min(demand, Q)
    
    # Leftover = Q - sales
    leftover = Q - sales
    
    # Calculate profit components
    revenue = sales * p
    salvage = leftover * c * f          # refund for unsold
    shipping_cost = leftover * s        # shipping to return unsold
    ordering_cost = K
    variable_cost = Q * c
    
    profit = revenue + salvage - shipping_cost - ordering_cost - variable_cost
    
    return {
        'demand': demand,
        'sales': sales,
        'leftover': leftover,
        'revenue': revenue,
        'salvage': salvage,
        'shipping_cost': shipping_cost,
        'ordering_cost': ordering_cost,
        'variable_cost': variable_cost,
        'profit': profit
    }

# Run simulation
random.seed(random_seed)

trials = []
for i in range(num_trials):
    # Generate random demand from Uniform(a, b) as integer
    demand = random.randint(a, b)
    
    result = simulate_single_trial(Q_star, demand, p, c, f, s, K)
    result['trial'] = i + 1
    trials.append(result)

# Calculate Statistics

profits = [t['profit'] for t in trials]

mean_profit = sum(profits) / len(profits)
variance = sum((x - mean_profit) ** 2 for x in profits) / len(profits)
std_dev = variance ** 0.5

num_losses = sum(1 for profit in profits if profit < 0)
prob_loss = num_losses / len(profits)

# Also calculate min and max for context
min_profit = min(profits)
max_profit = max(profits)

# Print Results

print("=" * 70)
print("PART 5: Risk & Simulation")
print("=" * 70)
print()
print(f"Simulation Parameters:")
print(f"  Order Quantity Q* = {Q_star}")
print(f"  Price p = ${p}")
print(f"  Number of trials = {num_trials}")
print(f"  Random seed = {random_seed}")
print(f"  Demand ~ Uniform({a}, {b})")
print()

print("=" * 70)
print("SIMULATION RESULTS")
print("=" * 70)
print()
print(f"Mean Profit:              ${mean_profit:.2f}")
print(f"Standard Deviation:       ${std_dev:.2f}")
print(f"Minimum Profit:           ${min_profit:.2f}")
print(f"Maximum Profit:           ${max_profit:.2f}")
print()
print(f"Number of Loss Trials:    {num_losses} out of {num_trials}")
print(f"Probability of Loss:      {prob_loss:.2%} ({prob_loss:.4f})")
print()

# Comparison with analytical
analytical_profit = 370  # from Part 2
print(f"Analytical E[Profit]:     ${analytical_profit:.2f}")
print(f"Simulated Mean Profit:    ${mean_profit:.2f}")
print(f"Difference:               ${mean_profit - analytical_profit:.2f}")
print()

print("=" * 70)
print("RISK ANALYSIS")
print("=" * 70)
print()
print("The simulation shows significant profit variability:")
print(f"  - When demand > Q* ({Q_star}): profit = ${max_profit:.2f} (maximum)")
print(f"  - When demand << Q*: profit can be as low as ${min_profit:.2f}")
print()
print(f"About {prob_loss:.0%} of scenarios result in a loss (profit < 0).")
print("This occurs when demand is low and we have excess inventory.")
print()

print("RISK MITIGATION STRATEGY:")
print("  Consider ordering slightly less than Q* to reduce downside risk,")
print("  especially if SparkFire is risk-averse or has limited capital.")
print("  Alternatively, negotiate a higher refund rate (f) with Leisure Limited")
print("  to reduce the cost of overage situations.")

# Save to CSV

csv_file = os.path.join(CSV_DIR, 'q5_simulation.csv')

with open(csv_file, 'w') as file:
    # Summary section
    file.write("Part 5 - Risk & Simulation Results\n")
    file.write("\n")
    file.write("Summary Statistics\n")
    file.write(f"Order Quantity (Q*),{Q_star}\n")
    file.write(f"Number of Trials,{num_trials}\n")
    file.write(f"Random Seed,{random_seed}\n")
    file.write(f"Mean Profit,${mean_profit:.2f}\n")
    file.write(f"Standard Deviation,${std_dev:.2f}\n")
    file.write(f"Minimum Profit,${min_profit:.2f}\n")
    file.write(f"Maximum Profit,${max_profit:.2f}\n")
    file.write(f"Number of Losses,{num_losses}\n")
    file.write(f"Probability of Loss,{prob_loss:.4f}\n")
    file.write("\n")
    
    # Detailed trial data
    file.write("Trial Data\n")
    file.write("Trial,Demand,Sales,Leftover,Revenue,Salvage,Shipping,Ordering,Variable Cost,Profit,Loss\n")
    
    for t in trials:
        is_loss = 1 if t['profit'] < 0 else 0
        file.write(f"{t['trial']},{t['demand']:.2f},{t['sales']:.2f},{t['leftover']:.2f},"
                   f"{t['revenue']:.2f},{t['salvage']:.2f},{t['shipping_cost']:.2f},"
                   f"{t['ordering_cost']:.2f},{t['variable_cost']:.2f},{t['profit']:.2f},{is_loss}\n")

print(f"\nCSV saved to: {csv_file}")

# Generate Plots

# Extract data
trial_nums = [t['trial'] for t in trials]
demands = [t['demand'] for t in trials]
profits_list = [t['profit'] for t in trials]

# Theoretical expected profit (from Part 2)
theoretical_profit = 370.0

# Plot 1: Profit Histogram
fig, ax = plt.subplots(figsize=(10, 6))
n, bins, patches = ax.hist(profits_list, bins=30, color=PRIMARY_BLUE, edgecolor='black', alpha=0.7)
ax.axvline(x=mean_profit, color=ACCENT_RED, linestyle='-', linewidth=2, label=f'Mean = ${mean_profit:.0f}')
ax.axvline(x=theoretical_profit, color='green', linestyle='--', linewidth=2, label=f'Theoretical = ${theoretical_profit:.0f}')
ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, label='Break-even')

ax.set_xlabel('Profit ($)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Part 5: Profit Distribution (500 Simulations)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot1_path = os.path.join(PLOTS_DIR, 'q5_profit_histogram.png')
plt.savefig(plot1_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot1_path}")

# Plot 2: Cumulative Average Profit (convergence)
cumulative_avg = np.cumsum(profits_list) / np.arange(1, num_trials + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trial_nums, cumulative_avg, color=PRIMARY_BLUE, linewidth=1.5, label='Running Average')
ax.axhline(y=theoretical_profit, color=ACCENT_RED, linestyle='--', linewidth=2, label=f'Theoretical E[Profit] = ${theoretical_profit:.0f}')

ax.set_xlabel('Number of Trials', fontsize=12)
ax.set_ylabel('Cumulative Average Profit ($)', fontsize=12)
ax.set_title('Part 5: Convergence of Sample Mean to Theoretical Mean', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, num_trials)

plt.tight_layout()
plot2_path = os.path.join(PLOTS_DIR, 'q5_convergence.png')
plt.savefig(plot2_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot2_path}")

# Plot 3: Demand vs Trial Number
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(trial_nums, demands, color=PRIMARY_BLUE, alpha=0.5, s=15)
ax.axhline(y=(a + b) / 2, color=ACCENT_RED, linestyle='--', linewidth=2, label=f'Mean Demand = {(a+b)/2:.0f}')
ax.axhline(y=Q_star, color='green', linestyle=':', linewidth=2, label=f'Q* = {Q_star}')

ax.set_xlabel('Trial Number', fontsize=12)
ax.set_ylabel('Demand (cases)', fontsize=12)
ax.set_title('Part 5: Demand Realizations Across Trials', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, num_trials)
ax.set_ylim(a - 20, b + 20)

plt.tight_layout()
plot3_path = os.path.join(PLOTS_DIR, 'q5_demand_scatter.png')
plt.savefig(plot3_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot3_path}")

# Plot 4: Profit vs Demand (piecewise relationship)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(demands, profits_list, color=PRIMARY_BLUE, alpha=0.6, s=20)
ax.axvline(x=Q_star, color=ACCENT_RED, linestyle='--', linewidth=2, label=f'Q* = {Q_star}')
ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

ax.set_xlabel('Demand (cases)', fontsize=12)
ax.set_ylabel('Profit ($)', fontsize=12)
ax.set_title('Part 5: Profit vs Demand Relationship', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Add annotation
ax.annotate('Demand < Q*\n(overage)', xy=(200, -100), fontsize=10, color=ACCENT_RED)
ax.annotate('Demand >= Q*\n(sellout)', xy=(350, 600), fontsize=10, color='green')

plt.tight_layout()
plot4_path = os.path.join(PLOTS_DIR, 'q5_profit_vs_demand.png')
plt.savefig(plot4_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot4_path}")

# Plot 5: Profit/Loss Bar (summary)
fig, ax = plt.subplots(figsize=(6, 5))
categories = ['Profit', 'Loss']
counts = [num_trials - num_losses, num_losses]
colors = ['green', ACCENT_RED]

bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
ax.set_ylabel('Number of Trials', fontsize=12)
ax.set_title('Part 5: Profit vs Loss Outcomes', fontsize=14, fontweight='bold')

for bar, count in zip(bars, counts):
    pct = count / num_trials * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
            f'{count}\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')

ax.set_ylim(0, max(counts) * 1.2)

plt.tight_layout()
plot5_path = os.path.join(PLOTS_DIR, 'q5_profit_loss_bar.png')
plt.savefig(plot5_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot5_path}")

# Plot 6: 95% Confidence Interval Convergence
ci_widths = []
for n in range(1, num_trials + 1):
    sample = profits_list[:n]
    if n > 1:
        std_sample = np.std(sample, ddof=1)
        ci_width = 2 * 1.96 * std_sample / np.sqrt(n)
    else:
        ci_width = np.nan
    ci_widths.append(ci_width)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(trial_nums, ci_widths, color=PRIMARY_BLUE, linewidth=1.5)
ax.set_xlabel('Number of Trials', fontsize=12)
ax.set_ylabel('95% CI Width ($)', fontsize=12)
ax.set_title('Part 5: Confidence Interval Width Convergence', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, num_trials)

plt.tight_layout()
plot6_path = os.path.join(PLOTS_DIR, 'q5_ci_convergence.png')
plt.savefig(plot6_path, dpi=150)
plt.close()
print(f"Plot saved to: {plot6_path}")
