import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_explore_data(filepath):
    """Load data and perform initial exploration"""

    df = pd.read_csv(filepath)

    print("\n1. DATASET SHAPE:")
    print(f"   - Number of instances: {df.shape[0]}")
    print(f"   - Number of features: {df.shape[1] - 1}") 
    print(f"   - Total columns: {df.shape[1]}")

    print("\n2. COLUMN INFORMATION:")
    print(f"   Columns: {list(df.columns)}")
    print("\n   Data types:")
    for col in df.columns:
        print(f"   - {col}: {df[col].dtype}")
    
    # Check for missing values
    print("\n3. MISSING VALUES CHECK:")
    missing = df.isnull().sum()
    print(f"   {missing.sum()} missing values in total")
    if missing.sum() == 0:
        print("   No missing values")
    else:
        print("   Missing values per column:")
        print(missing)
    
    # Statistical Summary
    print("\n4. FEATURE STATISTICS:")
    print(df.describe().round(4))
    
    # 1.5 Target Class Distribution
    print("\n5. TARGET CLASS DISTRIBUTION:")
    class_counts = df['Product Quality'].value_counts().sort_index()
    print(f"   {dict(class_counts)}")
    
    class_pct = (df['Product Quality'].value_counts(normalize=True) * 100).sort_index()
    print("\n   Class Distribution (%):")
    for cls, pct in class_pct.items():
        print(f"   - Class {cls}: {pct:.2f}%")

    # Check for imbalance
    min_class_count = class_counts.min()
    max_class_count = class_counts.max()
    imbalance_ratio = max_class_count / min_class_count
    print(f"\n   Imbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 2:
        print("   WARNING: Dataset is imbalanced - consider stratified splitting")
    else:
        print("   Dataset is reasonably balanced")

    
    # 1.6 Feature Analysis
    print("\n6. FEATURE RANGES:")
    feature_cols = [col for col in df.columns if col != 'Product Quality']
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"   {col}:")
        print(f"      Range: [{min_val:.4f}, {max_val:.4f}]")
        print(f"      Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    
    # 1.7 Check for duplicates
    print("\n7. DUPLICATE CHECK:")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates}")
    if duplicates == 0:
        print("    No duplicates found")
    
    # 1.8 Correlation Analysis
    print("\n8. FEATURE CORRELATION:")
    correlation = df.corr()
    print(correlation.round(4))
    
    return df, feature_cols





def visualize_data(csv_filepath):
    """
    Visualize 3D data points with 3D and 2D projections
    """
    
    # Load data
    df = pd.read_csv(csv_filepath)
    
    feature_cols = ['Temperature', 'Vibration', 'Stress index']
    X = df[feature_cols].values
    y = df['Product Quality'].values
    
    print(f"Dataset loaded: {df.shape[0]} instances, {len(feature_cols)} features")
    print(f"Classes: {sorted(np.unique(y))}")
    print(f"Class distribution: {dict(pd.Series(y).value_counts().sort_index())}")
    
    # Define colors for classes
    colors = {1: '#FF6B6B', 2: '#4ECDC4'}
    
    # ===== FIGURE 1: 3D Visualization =====
    fig1 = plt.figure(figsize=(16, 6))
    fig1.suptitle('3D Data Visualization - Product Quality Classification', 
                 fontsize=16, fontweight='bold')
    
    # 3D Plot - View 1
    ax1 = fig1.add_subplot(1, 2, 1, projection='3d')
    for cls in sorted(np.unique(y)):
        mask = y == cls
        ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                   c=colors[cls], label=f'Class {cls}', 
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Temperature', fontweight='bold', labelpad=15)
    ax1.set_ylabel('Vibration', fontweight='bold', labelpad=15)
    ax1.set_zlabel('Stress Index', fontweight='bold', labelpad=15)
    ax1.set_title('3D View - Angle 1', fontweight='bold', pad=20)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.view_init(elev=20, azim=45)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    
    # 3D Plot - View 2
    ax2 = fig1.add_subplot(1, 2, 2, projection='3d')
    for cls in sorted(np.unique(y)):
        mask = y == cls
        ax2.scatter(X[mask, 0], X[mask, 1], X[mask, 2], 
                   c=colors[cls], label=f'Class {cls}', 
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Temperature', fontweight='bold', labelpad=15)
    ax2.set_ylabel('Vibration', fontweight='bold', labelpad=15)
    ax2.set_zlabel('Stress Index', fontweight='bold', labelpad=15)
    ax2.set_title('3D View - Angle 2', fontweight='bold', pad=20)
    ax2.legend(fontsize=11, loc='upper right')
    ax2.view_init(elev=10, azim=120)
    ax2.xaxis.pane.fill = False
    ax2.yaxis.pane.fill = False
    ax2.zaxis.pane.fill = False
    
    plt.tight_layout()
    plt.savefig('data_plots.png', dpi=300, bbox_inches='tight')
    print("\n  3D visualization saved: data_plots.png")
    plt.show()