# Network Analysis of the Indian Stock Market (Nifty 200)

This repository provides an empirical study of the topological and structural properties of the Indian equity market using complex network theory. Financial correlation networks are constructed from daily returns of the National Stock Exchange (NSE) Nifty 200 index constituents over a two-year observation period (22 November 2024 to 20 November 2026, comprising $N = 190$ stocks and $T = 495$ trading days).

The complete research paper detailing all mathematical formulations, derivations, and discussion is available in [`report.pdf`](./report.pdf).

---

## Key Findings

1. **Exponential Rather than Scale-Free Architecture**:
   - Discrete maximum likelihood estimation and Vuong likelihood-ratio tests firmly reject the scale-free power-law hypothesis in favor of an exponential degree distribution ($R = -5.20, p < 0.001$).
   - The fitted power law has an exponent of $\alpha = 1.55$ ($x_{\min} = 4.0$), falling well outside the valid scale-free range ($2 < \alpha < 3$). Broad market-wide synchronization imposes a natural connectivity ceiling that truncates heavy degree tails.

2. **Decoupling of Index Capitalization and Network Centrality**:
   - The largest constituents by market capitalization do not occupy central hub positions in the graph:
     - **Reliance Industries**: degree $k = 7$ (rank 125)
     - **TCS**: degree $k = 9$ (rank 115)
     - **HDFC Bank**: degree $k = 23$ (rank 78)
     - **Infosys**: degree $k = 10$ (rank 107)
     - **ICICI Bank**: degree $k = 18$ (rank 92)
   - Network centrality is led instead by domestic credit infrastructure lenders and real estate developers: **Jio Financial Services** ($k = 103$), **DLF** ($k = 101$), **REC Ltd** ($k = 91$), **IRFC** ($k = 85$), and **PFC** ($k = 83$).
   - Node degree correlates strongly with market beta ($\rho = +0.596, p < 0.001$). High market sensitivity drives cross-market co-movement, while mega-caps have lower correlation with the equal-weighted domestic cross-section.

3. **High Sector Homophily & Component Purity**:
   - Intra-sector connections make up $28.05\%$ of all active edges at baseline ($r = 0.40$), vastly exceeding the uniform random pairing null ($9.90\%$) and the degree-preserving configuration null ($16.22\%$).
   - At a higher threshold of $r = 0.55$, cross-sector shortcuts are eliminated and the graph fragments into 74 isolated stocks, a giant component of 78 stocks ($41.1\%$), and 13 non-giant components ($38$ stocks). All 13 non-giant components ($100\%$) are single-sector pure (IT size 9, Auto size 6, Financials size 3, and 10 components of size 2).

4. **Market-Mode Subtraction (Residual Correlation Network)**:
   - Principal component analysis shows that the leading eigenvalue ($\lambda_1 = 57.25$) accounts for $30.13\%$ of system variance, well above the Marchenko-Pastur theoretical random noise bound ($\lambda_+ = 2.62$).
   - Removing the market factor via OLS residual regression ($e_i(t) = R_i(t) - \beta_i m(t)$) doubles sector modularity ($Q = 0.22 \to 0.40$), uncouples degree from market beta ($\rho = +0.596 \to -0.195$), and reshuffles central hubs toward core industrial manufacturers (Maruti $k=56$, Eicher Motors $k=52$, Pidilite $k=52$).

5. **Structural Fragility vs. Random Resilience**:
   - The network displays pronounced vulnerability to targeted hub removal compared to random failures: the giant component falls below $50\%$ after removing $24.2\%$ of nodes by betweenness (46 stocks) or $29.5\%$ by degree (56 stocks), whereas random failures require over $40\%$ node removal to reach the same fragmentation level.

6. **Dynamic Turnover Beneath Steady Aggregate Statistics**:
   - Across two consecutive annual halves ($T_1 = 247$ vs. $T_2 = 248$), aggregate statistical metrics remain steady (mean correlation $\approx 0.28$, clustering $\approx 0.65$).
   - However, individual pairwise edges exhibit substantial turnover: Jaccard edge overlap is only $26.67\%$ ($24.05\%$ when density-matched), significantly lower than an interleaved odd/even baseline ($41.04\%$), reflecting genuine sector rotation over time.

---

## Repository Structure

```
├── README.md                      # Overview, findings, and execution instructions
├── report.pdf                     # Complete 10-page research paper
├── requirements.txt               # Python package dependencies
├── data/
│   ├── ind_nifty200list.csv       # Official Nifty 200 constituent metadata from NSE
│   ├── prices_raw.csv             # Downloaded daily adjusted close prices (Yahoo Finance)
│   ├── returns.csv                # Cleaned daily logarithmic return series (N=190, T=495)
│   ├── correlation_matrix.csv     # Pairwise-complete Pearson correlation matrix
│   └── sector_map.csv             # Sector classification for the 190 retained stocks
├── src/
│   ├── 01_download_data.py        # Ingests daily price series from Yahoo Finance
│   ├── 02_clean_and_preprocess.py # Filters IPOs, audits corporate actions, computes returns
│   ├── 03_threshold_sweep.py      # Sweeps thresholds r in [0.20, 0.80] with node conservation
│   ├── 04_degree_and_centrality.py# Degree distribution MLE fits and node centrality rankings
│   ├── 05_sector_analysis.py      # Sector homophily, clustering, and component purity tests
│   ├── 06_market_mode_and_robustness.py # PCA, OLS residual network, and targeted attacks
│   ├── 07_generate_visualizations.py    # Publication-ready figures (CCDF, heatmap, attacks)
│   └── run_pipeline.py            # Master runner executing the entire workflow end-to-end
└── outputs/
    ├── figures/                   # Rendered high-resolution plots (.png)
    │   ├── fig_threshold_sensitivity_combined.png
    │   ├── fig_degree_distribution_ccdf.png
    │   ├── fig_sector_correlation_heatmap.png
    │   ├── fig_network_r55_readable.png
    │   ├── fig_attack_robustness.png
    │   └── fig_rolling_correlation.png
    └── tables/                    # Summary data tables and metrics (.csv, .json)
        ├── threshold_sweep.csv
        ├── centrality_all_stocks_r40.csv
        ├── top10_degree_r40.csv
        ├── index_heavyweights_centrality_r40.csv
        ├── sector_homophily_clean.csv
        ├── dropped_stocks.csv
        └── extreme_returns_flagged.csv
```

---

## Data Cleaning & Quality Control

- **Sample Selection**: From the 200 index constituents, 10 stocks were systematically excluded because they were not listed for the full two-year window (nine recent IPOs listed across 2025–2026 and one late-listing stock, VMM). Retained stocks ($N=190$) have complete histories spanning 495 daily return sessions ($T=495$).
- **Exchange Halts**: Exactly 10 isolated single-day trading halts across the 94,240 price matrix cells were forward-filled (maximum 2 consecutive days).
- **Corporate Action Audits**: Single-day price movements exceeding $25\%$ were audited:
  - Three unadjusted corporate actions (Tata Motors Passenger Vehicles demerger, Trent bonus issue, Vedanta 5-way demerger spinoff) were identified as data artefacts and set to `NaN` on the event session to compute pairwise-complete correlations.
  - Confirmed economic crashes (IndusInd Bank $-27.2\%$ on 11 March 2025; Policybazaar $-36.0\%$ on 24 September 2026) were retained as genuine economic variance.

---

## Getting Started

### 1. Prerequisites
Python 3.10+ is recommended. Install the required dependencies:

```bash
git clone https://github.com/shlok-mehndiratta/IISERB-Code-Archive.git
cd "IISERB-Code-Archive/Academics IISERB/Network Science"
pip install -r requirements.txt
```

### 2. Running the Complete Analysis
To run the full end-to-end pipeline from data ingestion to figure generation:

```bash
cd src
python run_pipeline.py
```

### 3. Running Individual Modules
Individual steps can also be run independently:

```bash
# 1. Download adjusted price data
python src/01_download_data.py

# 2. Clean prices and calculate log returns
python src/02_clean_and_preprocess.py

# 3. Correlation matrix & threshold sweep
python src/03_threshold_sweep.py

# 4. Degree distribution fitting & centralities
python src/04_degree_and_centrality.py

# 5. Sector homophily & component purity
python src/05_sector_analysis.py

# 6. Market mode removal & attack simulations
python src/06_market_mode_and_robustness.py

# 7. Render high-resolution figures
python src/07_generate_visualizations.py
```

All generated plots will be saved in `outputs/figures/`, and all structured summary tables will be placed in `outputs/tables/`.
