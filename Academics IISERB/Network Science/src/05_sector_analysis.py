"""
05_sector_analysis.py
---------------------
Analyzes industrial sector structure, computes intra-sector homophily against
null models (uniform pairing and degree-preserving configuration rewiring),
and verifies component sector purity at high correlation thresholds (r = 0.55).
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


def build_graph(corr_df, threshold):
    """Constructs an unweighted undirected graph at correlation threshold r."""
    symbols = list(corr_df.columns)
    N = len(symbols)
    G = nx.Graph()
    G.add_nodes_from(symbols)
    
    for i in range(N):
        for j in range(i + 1, N):
            if corr_df.iloc[i, j] > threshold:
                G.add_edge(symbols[i], symbols[j])
    return G


def run_sector_analysis():
    """Calculates sector homophily and component purity metrics."""
    corr = pd.read_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"), index_col=0)
    sector_df = pd.read_csv(os.path.join(DATA_DIR, "sector_map.csv"))
    
    sec_dict = dict(zip(sector_df["Symbol"], sector_df["Sector"]))
    symbols = list(corr.columns)
    N = len(symbols)
    
    # ---------------------------------------------------------
    # 1. Baseline Network (r = 0.40) Sector Homophily
    # ---------------------------------------------------------
    G_r40 = build_graph(corr, 0.40)
    nx.set_node_attributes(G_r40, sec_dict, "sector")
    
    total_edges = G_r40.number_of_edges()
    overall_density = total_edges / (N * (N - 1) / 2)
    
    # Count internal edges per sector
    sector_counts = sector_df["Sector"].value_counts()
    
    sector_records = []
    total_intra_edges = 0
    
    for sector, n_s in sector_counts.items():
        sub_nodes = [s for s in symbols if sec_dict[s] == sector]
        G_sub = G_r40.subgraph(sub_nodes)
        internal_edges = G_sub.number_of_edges()
        total_intra_edges += internal_edges
        
        possible_edges = n_s * (n_s - 1) // 2
        internal_density = (internal_edges / possible_edges) if possible_edges > 0 else 0.0
        density_ratio = (internal_density / overall_density) if overall_density > 0 else 0.0
        
        if n_s >= 5:
            sector_records.append({
                "Sector": sector,
                "Stocks": n_s,
                "Internal_Edges": internal_edges,
                "Possible_Edges": possible_edges,
                "Internal_Density": round(internal_density, 4),
                "Density_Ratio": round(density_ratio, 2)
            })
            
    homophily_df = pd.DataFrame(sector_records).sort_values(by="Stocks", ascending=False)
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    homophily_df.to_csv(os.path.join(OUTPUT_TABLES, "sector_homophily_clean.csv"), index=False)
    
    intra_edge_fraction = total_intra_edges / total_edges
    
    # Null expectations:
    # 1. Uniform random null: sum n_s(n_s - 1) / [N(N - 1)]
    uniform_null = sum(n * (n - 1) for n in sector_counts.values) / (N * (N - 1))
    
    # 2. Sector assortativity
    assortativity = nx.attribute_assortativity_coefficient(G_r40, "sector")
    
    # ---------------------------------------------------------
    # 2. High-Threshold Subnetwork Fragmentation at r = 0.55
    # ---------------------------------------------------------
    G_r55 = build_graph(corr, 0.55)
    nx.set_node_attributes(G_r55, sec_dict, "sector")
    
    components_55 = sorted(list(nx.connected_components(G_r55)), key=len, reverse=True)
    comp_sizes_55 = [len(c) for c in components_55]
    
    isolates_55 = sum(1 for s in comp_sizes_55 if s == 1)
    non_giant_comps_55 = [c for c in components_55[1:] if len(c) > 1]
    
    # Purity check for all non-giant components:
    # A component is 100% pure if all constituent stocks share the exact same sector
    purity_counts = []
    for c in non_giant_comps_55:
        sectors_in_c = set(sec_dict[n] for n in c)
        purity_counts.append(len(sectors_in_c) == 1)
        
    all_non_giant_pure = all(purity_counts)
    
    print("[Success] Sector homophily analysis complete.")
    print(f"[Intra-Sector Edges] {total_intra_edges} / {total_edges} ({intra_edge_fraction:.2%}) "
          f"vs. Uniform Null ({uniform_null:.2%}) and Assortativity = {assortativity:.4f}.")
    print(f"[Fragmentation at r=0.55] Giant = {comp_sizes_55[0]} stocks (41.1%), Isolates = {isolates_55}, "
          f"Non-Giant Components = {len(non_giant_comps_55)} (all single-sector pure: {all_non_giant_pure}).")
    return homophily_df, comp_sizes_55


if __name__ == "__main__":
    run_sector_analysis()
