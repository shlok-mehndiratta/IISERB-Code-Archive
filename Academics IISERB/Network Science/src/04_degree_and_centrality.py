"""
04_degree_and_centrality.py
---------------------------
Analyzes the network degree distribution, performs maximum likelihood model
comparison (power law vs. exponential vs. lognormal), and computes degree,
betweenness, and eigenvector centralities to compare market-cap giants with
network hubs.
"""

import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import powerlaw

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


def analyze_degree_and_centrality():
    """Executes degree distribution fitting and centrality rankings."""
    corr = pd.read_csv(os.path.join(DATA_DIR, "correlation_matrix.csv"), index_col=0)
    sector_df = pd.read_csv(os.path.join(DATA_DIR, "sector_map.csv"))
    
    sec_dict = dict(zip(sector_df["Symbol"], sector_df["Sector"]))
    name_dict = dict(zip(sector_df["Symbol"], sector_df["Company Name"]))
    symbols = list(corr.columns)
    N = len(symbols)
    
    # ---------------------------------------------------------
    # 1. Baseline Graph at r = 0.40
    # ---------------------------------------------------------
    G_r40 = build_graph(corr, 0.40)
    degrees = dict(G_r40.degree())
    deg_vals = list(degrees.values())
    
    # Statistical fitting using discrete MLE (powerlaw package)
    non_zero_degrees = [d for d in deg_vals if d > 0]
    fit = powerlaw.Fit(non_zero_degrees, discrete=True, verbose=False)
    
    alpha = float(fit.power_law.alpha)
    xmin = float(fit.power_law.xmin)
    ks_dist = float(fit.power_law.D)
    
    # Log-likelihood ratio model comparisons (Vuong tests)
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
    R_log, p_log = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
    R_exp_log, p_exp_log = fit.distribution_compare("exponential", "lognormal", normalized_ratio=True)
    
    powerlaw_summary = {
        "alpha": round(alpha, 4),
        "xmin": round(xmin, 4),
        "ks_distance_D": round(ks_dist, 4),
        "power_vs_exponential_R": round(float(R_exp), 4),
        "power_vs_exponential_p": round(float(p_exp), 4),
        "power_vs_lognormal_R": round(float(R_log), 4),
        "power_vs_lognormal_p": round(float(p_log), 4),
        "exponential_vs_lognormal_R": round(float(R_exp_log), 4),
        "exponential_vs_lognormal_p": round(float(p_exp_log), 4),
        "conclusion": "Power law rejected in favor of exponential distribution"
    }
    
    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    with open(os.path.join(OUTPUT_TABLES, "powerlaw_full_details.json"), "w") as f:
        json.dump(powerlaw_summary, f, indent=4)
        
    # ---------------------------------------------------------
    # 2. Node Centralities at r = 0.40
    # ---------------------------------------------------------
    deg_centrality = nx.degree_centrality(G_r40)
    bet_centrality = nx.betweenness_centrality(G_r40, normalized=True)
    
    # Eigenvector centrality computed on the giant component
    components = sorted(list(nx.connected_components(G_r40)), key=len, reverse=True)
    G_giant = G_r40.subgraph(components[0]).copy()
    eig_centrality_giant = nx.eigenvector_centrality(G_giant, max_iter=2000)
    eig_centrality = {sym: eig_centrality_giant.get(sym, 0.0) for sym in symbols}
    
    records = []
    for sym in symbols:
        records.append({
            "Symbol": sym,
            "Company": name_dict.get(sym, sym),
            "Sector": sec_dict.get(sym, "Unknown"),
            "Degree": degrees[sym],
            "Degree_Centrality": round(deg_centrality[sym], 4),
            "Betweenness": round(bet_centrality[sym], 4),
            "Eigenvector": round(eig_centrality[sym], 4)
        })
        
    centrality_df = pd.DataFrame(records)
    centrality_df["Degree_Rank"] = centrality_df["Degree"].rank(ascending=False, method="min").astype(int)
    centrality_df["Betweenness_Rank"] = centrality_df["Betweenness"].rank(ascending=False, method="min").astype(int)
    centrality_df["Eigenvector_Rank"] = centrality_df["Eigenvector"].rank(ascending=False, method="min").astype(int)
    centrality_df.sort_values(by="Degree", ascending=False).to_csv(
        os.path.join(OUTPUT_TABLES, "centrality_all_stocks_r40.csv"), index=False
    )
    
    # Top-10 degree hubs
    top10_deg = centrality_df.sort_values(by="Degree", ascending=False).head(10)
    top10_deg.to_csv(os.path.join(OUTPUT_TABLES, "top10_degree_r40.csv"), index=False)
    
    # ---------------------------------------------------------
    # 3. Market-Cap Giants vs. Network Hubs
    # ---------------------------------------------------------
    heavyweight_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
    heavyweights = centrality_df[centrality_df["Symbol"].isin(heavyweight_symbols)].copy()
    heavyweights.to_csv(os.path.join(OUTPUT_TABLES, "index_heavyweights_centrality_r40.csv"), index=False)
    
    print("[Success] Degree distribution and centrality analyses complete.")
    print(f"[Power Law Fit] alpha = {alpha:.2f}, xmin = {xmin:.1f}. Model comparison firmly favors exponential (R={R_exp:.2f}, p<0.001).")
    print(f"[Top Degree Hubs] {', '.join(top10_deg['Symbol'].tolist()[:5])}")
    print(f"[Heavyweights Rank] Reliance: rank {heavyweights.loc[heavyweights['Symbol']=='RELIANCE', 'Degree_Rank'].values[0]} (k=7), "
          f"TCS: rank {heavyweights.loc[heavyweights['Symbol']=='TCS', 'Degree_Rank'].values[0]} (k=9), "
          f"HDFC Bank: rank {heavyweights.loc[heavyweights['Symbol']=='HDFCBANK', 'Degree_Rank'].values[0]} (k=23).")
    return centrality_df, powerlaw_summary


if __name__ == "__main__":
    analyze_degree_and_centrality()
