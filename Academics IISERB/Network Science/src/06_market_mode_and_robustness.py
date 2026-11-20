"""
06_market_mode_and_robustness.py
--------------------------------
Isolates the collective market mode via PCA and linear residual regression,
simulates network robustness against targeted hub removal vs. random failures,
and examines temporal stability across annual observation halves.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")
OUTPUT_EXT = os.path.join(BASE_DIR, "outputs", "extensions")


def run_market_mode_and_robustness():
    """Executes market-mode subtraction, percolation breakdown, and stability checks."""
    returns = pd.read_csv(os.path.join(DATA_DIR, "returns.csv"), index_col=0, parse_dates=True)
    corr = pd.read_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"), index_col=0)
    sector_df = pd.read_csv(os.path.join(DATA_DIR, "sector_map.csv"))
    
    sec_dict = dict(zip(sector_df["Symbol"], sector_df["Sector"]))
    symbols = list(returns.columns)
    N = len(symbols)
    T = len(returns)
    total_possible_edges = N * (N - 1) / 2
    
    # ---------------------------------------------------------
    # 1. Principal Component Analysis (Eigenvalue Spectrum)
    # ---------------------------------------------------------
    eigenvalues, _ = np.linalg.eigh(corr.values)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    lambda_1 = float(eigenvalues[0])
    variance_explained = float(lambda_1 / N * 100)
    marchenko_pastur_upper = (1 + np.sqrt(N / T)) ** 2
    
    # ---------------------------------------------------------
    # 2. Market-Mode Removal (OLS Residual Correlation)
    # ---------------------------------------------------------
    # Equal-weighted market index return
    m = returns.mean(axis=1)
    residuals = pd.DataFrame(index=returns.index, columns=symbols, dtype=float)
    betas = {}
    
    for sym in symbols:
        y = returns[sym].dropna()
        x = m.loc[y.index]
        cov_xy = np.cov(y, x)[0, 1]
        var_x = np.var(x)
        beta = cov_xy / var_x
        betas[sym] = beta
        residuals[sym] = returns[sym] - beta * m
        
    corr_resid = residuals.corr(method="pearson", min_periods=30)
    
    # Construct residual network at exactly matched baseline density (14.34%, M = 2,574)
    off_diag_resid = corr_resid.values[np.triu_indices(N, k=1)]
    r_resid_cutoff = np.percentile(off_diag_resid, 100 * (1 - 2574 / total_possible_edges))
    
    G_raw = nx.Graph()
    G_resid = nx.Graph()
    G_raw.add_nodes_from(symbols)
    G_resid.add_nodes_from(symbols)
    
    for i in range(N):
        for j in range(i + 1, N):
            if corr.iloc[i, j] > 0.40:
                G_raw.add_edge(symbols[i], symbols[j])
            if corr_resid.iloc[i, j] > r_resid_cutoff:
                G_resid.add_edge(symbols[i], symbols[j])
                
    nx.set_node_attributes(G_raw, sec_dict, "sector")
    nx.set_node_attributes(G_resid, sec_dict, "sector")
    
    # Louvain community modularity using NetworkX native algorithms
    comms_raw = nx.community.louvain_communities(G_raw, seed=42)
    comms_resid = nx.community.louvain_communities(G_resid, seed=42)
    Q_raw = nx.community.modularity(G_raw, comms_raw)
    Q_resid = nx.community.modularity(G_resid, comms_resid)
    
    assort_raw = nx.attribute_assortativity_coefficient(G_raw, "sector")
    assort_resid = nx.attribute_assortativity_coefficient(G_resid, "sector")
    
    # ---------------------------------------------------------
    # 3. Targeted Attack vs. Random Failure Simulation
    # ---------------------------------------------------------
    def simulate_removal(G_init, mode="degree"):
        G = G_init.copy()
        fractions = np.linspace(0, 0.40, 21)
        giant_fractions = []
        
        for frac in fractions:
            num_remove = int(frac * N)
            G_sub = G_init.copy()
            
            if mode == "degree":
                nodes_sorted = sorted(G_sub.degree(), key=lambda x: x[1], reverse=True)
                nodes_to_remove = [n for n, d in nodes_sorted[:num_remove]]
            elif mode == "betweenness":
                bet = nx.betweenness_centrality(G_sub)
                nodes_sorted = sorted(bet.items(), key=lambda x: x[1], reverse=True)
                nodes_to_remove = [n for n, b in nodes_sorted[:num_remove]]
            elif mode == "random":
                # Average over 50 random trials
                r_giants = []
                for seed in range(50):
                    np.random.seed(seed)
                    to_remove = np.random.choice(symbols, size=num_remove, replace=False)
                    G_temp = G_init.copy()
                    G_temp.remove_nodes_from(to_remove)
                    comps = sorted(list(nx.connected_components(G_temp)), key=len, reverse=True)
                    r_giants.append(len(comps[0]) if comps else 0)
                giant_fractions.append(np.mean(r_giants) / N)
                continue
                
            G_sub.remove_nodes_from(nodes_to_remove)
            comps = sorted(list(nx.connected_components(G_sub)), key=len, reverse=True)
            giant_fractions.append(len(comps[0]) / N if comps else 0.0)
            
        return fractions, giant_fractions

    # ---------------------------------------------------------
    # 4. Temporal Stability Across Observation Halves
    # ---------------------------------------------------------
    T_half = T // 2
    returns_h1 = returns.iloc[:T_half]
    returns_h2 = returns.iloc[T_half:]
    
    corr_h1 = returns_h1.corr(method="pearson", min_periods=30)
    corr_h2 = returns_h2.corr(method="pearson", min_periods=30)
    
    edges_h1 = set()
    edges_h2 = set()
    for i in range(N):
        for j in range(i + 1, N):
            if corr_h1.iloc[i, j] > 0.40:
                edges_h1.add((symbols[i], symbols[j]))
            if corr_h2.iloc[i, j] > 0.40:
                edges_h2.add((symbols[i], symbols[j]))
                
    shared_edges = edges_h1.intersection(edges_h2)
    union_edges = edges_h1.union(edges_h2)
    jaccard_overlap = len(shared_edges) / len(union_edges) if union_edges else 0.0
    
    stability_summary = {
        "lambda_1": round(lambda_1, 2),
        "variance_explained_pct": round(variance_explained, 2),
        "marchenko_pastur_bound": round(marchenko_pastur_upper, 2),
        "raw_assortativity": round(assort_raw, 4),
        "residual_assortativity": round(assort_resid, 4),
        "raw_modularity_Q": round(Q_raw, 4),
        "residual_modularity_Q": round(Q_resid, 4),
        "annual_half_jaccard_overlap": round(jaccard_overlap * 100, 2)
    }
    
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    os.makedirs(OUTPUT_EXT, exist_ok=True)
    with open(os.path.join(OUTPUT_TABLES, "halves_stability_detailed.json"), "w") as f:
        json.dump(stability_summary, f, indent=4)
        
    print("[Success] Market mode removal, percolation attacks, and stability analysis complete.")
    print(f"[Market Mode] lambda_1 = {lambda_1:.2f} ({variance_explained:.1f}% variance, vs MP bound {marchenko_pastur_upper:.2f}).")
    print(f"[Residual Transformation] Assortativity: {assort_raw:.3f} -> {assort_resid:.3f}, Modularity Q: {Q_raw:.3f} -> {Q_resid:.3f}.")
    print(f"[Annual Stability] Edge Jaccard similarity across halves = {jaccard_overlap:.2%}.")
    return stability_summary


if __name__ == "__main__":
    run_market_mode_and_robustness()
