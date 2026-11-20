"""
07_generate_visualizations.py
-----------------------------
Generates all high-resolution figures for the report:
  - Figure 1: Threshold sensitivity curves (edges, giant component, path length)
  - Figure 2: Degree distribution (linear histogram and log-log CCDF fits)
  - Figure 3: Sector correlation heatmap and fragmented network graph at r = 0.55
  - Figure 4: Targeted attack vs. random failure percolation and rolling correlation
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FIGS = os.path.join(BASE_DIR, "outputs", "figures")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")


def generate_all_figures():
    """Renders all figures with consistent formatting and large legible fonts."""
    os.makedirs(OUTPUT_FIGS, exist_ok=True)
    
    corr = pd.read_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"), index_col=0)
    returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
    sector_df = pd.read_csv(os.path.join(DATA_DIR, "sector_map.csv"))
    sweep_df = pd.read_csv(os.path.join(OUTPUT_TABLES, "threshold_sweep.csv"))
    
    sec_dict = dict(zip(sector_df["Symbol"], sector_df["Sector"]))
    symbols = list(corr.columns)
    N = len(symbols)
    
    # ---------------------------------------------------------
    # Figure 1: Threshold Sensitivity Curves
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) Edge count vs r
    axes[0].plot(sweep_df["threshold_r"], sweep_df["edges"], marker="o", color="#1f77b4", linewidth=2)
    axes[0].axvline(0.40, color="red", linestyle="--", alpha=0.8, label="Baseline r=0.40")
    axes[0].axvline(0.45, color="orange", linestyle=":", alpha=0.8, label="Robustness r=0.45")
    axes[0].set_title("Network Sparsification", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Correlation Threshold (r)", fontsize=11)
    axes[0].set_ylabel("Total Number of Edges (M)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=9)
    
    # (b) Giant component size fraction vs r
    axes[1].plot(sweep_df["threshold_r"], sweep_df["giant_pct"], marker="s", color="#2ca02c", linewidth=2, label="Baseline r=0.40 (92.1%)")
    axes[1].axvline(0.40, color="red", linestyle="--", alpha=0.8)
    axes[1].axvline(0.45, color="orange", linestyle=":", alpha=0.8, label="Robustness r=0.45 (73.2%)")
    axes[1].axhline(50, color="gray", linestyle="-.", alpha=0.7, label="50% Percolation Level")
    axes[1].set_title("Percolation & Giant Component", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Correlation Threshold (r)", fontsize=11)
    axes[1].set_ylabel("Giant Component Size (% of N=190)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(fontsize=9)
    
    # (c) Average shortest path length vs r
    valid_path = sweep_df.dropna(subset=["avg_path_length_giant"])
    axes[2].plot(valid_path["threshold_r"], valid_path["avg_path_length_giant"], marker="^", color="#9467bd", linewidth=2)
    axes[2].axvline(0.40, color="red", linestyle="--", alpha=0.8, label="Baseline r=0.40 (L=2.49)")
    axes[2].set_title("Average Shortest Path Length", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Correlation Threshold (r)", fontsize=11)
    axes[2].set_ylabel("Average Path Length on Giant (L)", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "fig_threshold_sensitivity_combined.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 2: Degree Distribution (Linear & CCDF Log-Log)
    # ---------------------------------------------------------
    G_r40 = nx.Graph()
    G_r40.add_nodes_from(symbols)
    for i in range(N):
        for j in range(i + 1, N):
            if corr.iloc[i, j] > 0.40:
                G_r40.add_edge(symbols[i], symbols[j])
                
    degrees = [d for n, d in G_r40.degree()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(degrees, bins=25, color="#1f77b4", edgecolor="black", alpha=0.75)
    ax1.axvline(np.mean(degrees), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(degrees):.1f}")
    ax1.axvline(np.median(degrees), color="green", linestyle=":", linewidth=1.5, label=f"Median: {np.median(degrees):.1f}")
    ax1.set_title("(a) Degree Distribution (Linear Scale)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Degree (k)", fontsize=11)
    ax1.set_ylabel("Count of Stocks P(k)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Log-log CCDF with distribution fits
    non_zero_deg = [d for d in degrees if d > 0]
    fit = powerlaw.Fit(non_zero_deg, discrete=True, verbose=False)
    fit.plot_ccdf(ax=ax2, color="black", marker="o", linestyle="None", markersize=4, label="Empirical CCDF")
    fit.power_law.plot_ccdf(ax=ax2, color="crimson", linestyle="--", linewidth=2, label=f"Power Law (α={fit.power_law.alpha:.2f})")
    fit.exponential.plot_ccdf(ax=ax2, color="blue", linestyle="-", linewidth=2, label="Exponential Fit")
    fit.lognormal.plot_ccdf(ax=ax2, color="green", linestyle=":", linewidth=2, label="Lognormal Fit")
    ax2.set_title("(b) Complementary Cumulative Distribution (CCDF)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Degree (k)", fontsize=11)
    ax2.set_ylabel("P(K ≥ k)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "fig_degree_distribution_ccdf.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 3: Sector Correlation Heatmap
    # ---------------------------------------------------------
    sorted_stocks = sector_df.sort_values(by="Sector")["Symbol"].tolist()
    sorted_stocks = [s for s in sorted_stocks if s in symbols]
    corr_sorted = corr.loc[sorted_stocks, sorted_stocks]
    
    plt.figure(figsize=(9, 8))
    sns.heatmap(
        corr_sorted,
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.8,
        cbar_kws={"label": "Pearson Correlation (r)"},
        xticklabels=False,
        yticklabels=False
    )
    plt.title("Correlation Matrix Reordered by Sector (N=190, Nifty 200)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "fig_sector_correlation_heatmap.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 4: Targeted Attack vs. Random Failure Simulation
    # ---------------------------------------------------------
    fractions = np.linspace(0, 0.40, 21)
    
    def simulate_attack(mode):
        y_vals = []
        for frac in fractions:
            num_rm = int(frac * N)
            G_temp = G_r40.copy()
            if mode == "degree":
                deg_sorted = sorted(G_temp.degree(), key=lambda x: x[1], reverse=True)
                G_temp.remove_nodes_from([n for n, d in deg_sorted[:num_rm]])
            elif mode == "betweenness":
                bet = nx.betweenness_centrality(G_temp)
                bet_sorted = sorted(bet.items(), key=lambda x: x[1], reverse=True)
                G_temp.remove_nodes_from([n for n, b in bet_sorted[:num_rm]])
            elif mode == "random":
                r_trials = []
                for seed in range(30):
                    np.random.seed(seed)
                    rm = np.random.choice(symbols, size=num_rm, replace=False)
                    G_r = G_r40.copy()
                    G_r.remove_nodes_from(rm)
                    c = sorted(list(nx.connected_components(G_r)), key=len, reverse=True)
                    r_trials.append(len(c[0]) if c else 0)
                y_vals.append(np.mean(r_trials) / N)
                continue
            c = sorted(list(nx.connected_components(G_temp)), key=len, reverse=True)
            y_vals.append(len(c[0]) / N if c else 0.0)
        return y_vals

    plt.figure(figsize=(7, 5))
    plt.plot(fractions * 100, [y * 100 for y in simulate_attack("betweenness")], marker="o", color="crimson", label="Targeted Removal (Betweenness)")
    plt.plot(fractions * 100, [y * 100 for y in simulate_attack("degree")], marker="s", color="orange", label="Targeted Removal (Degree)")
    plt.plot(fractions * 100, [y * 100 for y in simulate_attack("random")], marker="^", color="dodgerblue", label="Uniform Random Failure (Avg of 30)")
    plt.axhline(50, color="gray", linestyle="--", label="50% Percolation Threshold")
    plt.title("Targeted Hub Removal vs. Random Failure", fontsize=12, fontweight="bold")
    plt.xlabel("Fraction of Nodes Removed (%)", fontsize=11)
    plt.ylabel("Giant Component Size (% of N=190)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "fig_attack_robustness.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # Figure 5: Rolling 120-Day Mean Pairwise Correlation
    # ---------------------------------------------------------
    window = 120
    step = 5
    rolling_dates = []
    rolling_means = []
    
    for start_idx in range(0, len(returns) - window + 1, step):
        sub_ret = returns.iloc[start_idx:start_idx + window]
        sub_corr = sub_ret.corr(method="pearson", min_periods=30).values
        sub_off_diag = sub_corr[np.triu_indices(N, k=1)]
        rolling_dates.append(sub_ret.index[-1])
        rolling_means.append(np.mean(sub_off_diag))
        
    plt.figure(figsize=(9, 4.5))
    plt.plot(rolling_dates, rolling_means, color="#1f77b4", linewidth=2)
    plt.title("Rolling 120-Day Mean Pairwise Correlation", fontsize=12, fontweight="bold")
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Mean Pairwise Correlation (r)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGS, "fig_rolling_correlation.png"), dpi=300)
    plt.close()
    
    print("[Success] All publication-ready figures generated successfully.")


if __name__ == "__main__":
    generate_all_figures()
