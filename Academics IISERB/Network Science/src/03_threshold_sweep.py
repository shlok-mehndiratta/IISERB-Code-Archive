"""
03_threshold_sweep.py
---------------------
Computes the pairwise-complete Pearson correlation matrix, performs a systematic
threshold sweep across r in [0.20, 0.80], enforces strict node-conservation checks,
and identifies the baseline correlation network threshold.
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx

# Resolve paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_TABLES = os.path.join(BASE_DIR, "outputs", "tables")


def run_threshold_sweep():
    """
    Computes pairwise Pearson correlations and evaluates network topology
    across correlation thresholds r in [0.20, 0.80].
    """
    returns_path = os.path.join(DATA_DIR, "returns.csv")
    returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    symbols = list(returns.columns)
    N = len(symbols)
    assert N == 190, f"Expected N=190, found {N}"
    
    # ---------------------------------------------------------
    # 1. Pearson Correlation Matrix (Pairwise Complete)
    # ---------------------------------------------------------
    corr = returns.corr(method="pearson", min_periods=30)
    
    # Mathematical sanity assertions
    assert corr.shape == (N, N), f"Unexpected shape {corr.shape}"
    assert not corr.isnull().values.any(), "Correlation matrix contains NaNs!"
    assert np.allclose(corr.values, corr.values.T), "Correlation matrix is asymmetric!"
    assert np.allclose(np.diag(corr.values), 1.0), "Diagonal elements must equal 1.0!"
    
    corr.to_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"))
    
    # Summary statistics of off-diagonal correlations
    triu_idx = np.triu_indices(N, k=1)
    off_diag_corrs = corr.values[triu_idx]
    
    corr_stats = {
        "num_stocks": N,
        "num_pairs": len(off_diag_corrs),
        "mean_correlation": round(float(np.mean(off_diag_corrs)), 4),
        "median_correlation": round(float(np.median(off_diag_corrs)), 4),
        "std_correlation": round(float(np.std(off_diag_corrs)), 4),
        "min_correlation": round(float(np.min(off_diag_corrs)), 4),
        "max_correlation": round(float(np.max(off_diag_corrs)), 4),
        "pct_positive_pairs": round(float((off_diag_corrs > 0).mean() * 100), 2)
    }
    
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    with open(os.path.join(OUTPUT_TABLES, "correlation_stats.json"), "w") as f:
        json.dump(corr_stats, f, indent=4)
        
    # ---------------------------------------------------------
    # 2. Threshold Sweep & Strict Graph Topology Validation
    # ---------------------------------------------------------
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    total_possible_edges = N * (N - 1) / 2
    
    sweep_records = []
    
    for r in thresholds:
        # Build unweighted graph: add all nodes first to preserve universe size
        G = nx.Graph()
        G.add_nodes_from(symbols)
        
        # Add undirected edges where correlation exceeds threshold r
        for i in range(N):
            for j in range(i + 1, N):
                if corr.iloc[i, j] > r:
                    G.add_edge(symbols[i], symbols[j])
                    
        # Node conservation assertions
        assert G.number_of_nodes() == N, f"Node count mismatch at r={r}"
        
        components = sorted(list(nx.connected_components(G)), key=len, reverse=True)
        comp_sizes = [len(c) for c in components]
        assert sum(comp_sizes) == N, f"Component size sum {sum(comp_sizes)} != N ({N}) at r={r}"
        
        isolated_nodes_list = [n for n, deg in G.degree() if deg == 0]
        isolated_count = len(isolated_nodes_list)
        
        non_trivial_comps = [c for c in components if len(c) > 1]
        giant_size = comp_sizes[0] if len(comp_sizes) > 0 else 0
        second_size = comp_sizes[1] if len(comp_sizes) > 1 and comp_sizes[1] > 1 else (1 if len(comp_sizes) > 1 else 0)
        
        # Shortest path length on the giant component
        if giant_size > 1:
            G_giant = G.subgraph(components[0]).copy()
            avg_path_length = round(nx.average_shortest_path_length(G_giant), 4)
        else:
            avg_path_length = np.nan
            
        avg_clustering = round(nx.average_clustering(G), 4)
        num_edges = G.number_of_edges()
        density = num_edges / total_possible_edges
        
        sweep_records.append({
            "threshold_r": r,
            "edges": num_edges,
            "density": round(density, 4),
            "total_components": len(components),
            "non_trivial_components": len(non_trivial_comps),
            "giant_size": giant_size,
            "giant_pct": round((giant_size / N) * 100, 2),
            "second_size": comp_sizes[1] if len(comp_sizes) > 1 else 0,
            "isolated_nodes": isolated_count,
            "avg_path_length_giant": avg_path_length,
            "avg_clustering": avg_clustering
        })
        
    sweep_df = pd.DataFrame(sweep_records)
    sweep_df.to_csv(os.path.join(OUTPUT_TABLES, "threshold_sweep.csv"), index=False)
    
    print("[Success] Threshold sweep validated across 13 correlation levels.")
    print(f"[Baseline r=0.40] M = {sweep_df.loc[sweep_df['threshold_r']==0.40, 'edges'].values[0]} edges, "
          f"Giant Component = {sweep_df.loc[sweep_df['threshold_r']==0.40, 'giant_size'].values[0]} stocks "
          f"({sweep_df.loc[sweep_df['threshold_r']==0.40, 'giant_pct'].values[0]}%), "
          f"Density = {sweep_df.loc[sweep_df['threshold_r']==0.40, 'density'].values[0]:.2%}.")
    return corr, sweep_df


if __name__ == "__main__":
    run_threshold_sweep()
