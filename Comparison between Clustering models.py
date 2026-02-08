# ---------- import packages ----------

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.datasets import make_moons

import warnings
warnings.filterwarnings('ignore')

# --------- Generate synthetic moon-shaped dataset ---------
# Moon dataset is challenging for traditional centroid-based algorithms like KMeans
# It helps demonstrate the strengths of density-based algorithms like DBSCAN
X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')
print(f'True class distribution: {np.bincount(y)}')


# --------- Normalize the features ---------
# Standardization is crucial for distance-based algorithms
# It ensures all features contribute equally to distance calculations
std = StandardScaler()
X_std = std.fit_transform(X)

# --------- Define clustering algorithms with optimal parameters ---------
# Each algorithm has different strengths for moon-shaped data

clustering_algorithms = {
    'KMeans': {
        'model': KMeans(n_clusters=2, random_state=42, n_init=10),
        'color': 'red',
        'description': 'Centroid-based, assumes spherical clusters'
    },
    'DBSCAN': {
        'model': DBSCAN(eps=0.3, min_samples=5),  # Adjusted for moon dataset
        'color': 'blue',
        'description': 'Density-based, finds arbitrary shaped clusters'
    },
    'Agglomerative': {
        'model': AgglomerativeClustering(n_clusters=2, linkage='ward'),
        'color': 'green',
        'description': 'Hierarchical, builds nested clusters'
    }
}

# --------- Initialize results storage ---------
results = pd.DataFrame(columns=['Algorithm',
                                'Silhouette',
                                'ARI',
                                'Davies-Bouldin',
                                'Calinski-Harabasz',
                                'n_clusters',
                                'noise_points'
                                ]
                       )

# --------- 1. VISUALIZATION: Plot the original data and clustering results ---------
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Clustering Algorithms Comparison on Moon Dataset', fontsize=16, fontweight='bold')

# Plot 1: True labels (ground truth)
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8, edgecolors='k')
axes[0, 0].set_title('Ground Truth (True Labels)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Feature 1', fontsize=12)
axes[0, 0].set_ylabel('Feature 2', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# Initialize counter for subplot positions
plot_row, plot_col = 0, 1

# --------- 2. CLUSTERING & EVALUATION: Apply and evaluate each algorithm ---------
print("\n" + "="*80)
print("CLUSTERING ALGORITHMS COMPARISON")
print("="*80)

for idx, (algo_name, algo_info) in enumerate(clustering_algorithms.items()):
    print(f'{'='*50}')
    print(f'Algorithm: {algo_name}')
    print(f'Description: {algo_info['description']}')
    print('='*50)
    
    # fit the cls algorithm
    model = algo_info['model']
    
    # Special handling for DBSCAN (may produce noise points labeled as -1)
    if algo_info == 'DBSCAN':
        labels = model.fit_predict(X_std)
        n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
        noise_count = np.sum(labels == -1)
    else:
        labels = model.fit_predict(X_std)
        n_clusters = len(np.unique(labels))
        noise_count = 0

    # --------- 3. METRICS CALCULATION: Compute all evaluation metrics ---------

    # Metric 1: Silhouette Score (Internal - measures cohesion & separation)
    # Range: [-1, 1], Higher is better!
    if n_clusters > 1 and n_clusters < len(X_std):
        silhouette = silhouette_score(X_std, labels)

    # Metric 2: Adjusted Rand Index (External - compares with true labels)
    # Range: [-1, 1], Higher is better, 1 = perfect match
    # only vlid when we have ground truth
    ari = adjusted_rand_score(y, labels)
    
    # Metric 3: Davies-Bouldin Index (Internal - compactness vs separation)
    # Range: [0, ∞), Lower is better
    if n_clusters > 1 :
        dbi = davies_bouldin_score(X_std, labels)
    else:
        dbi = np.nan
    
    # Metric 4: Calinski-Harabasz Index (Variance Ratio Criterion)
    # Range: [0, ∞), Higher is better, measures between-cluster dispersion
    if n_clusters > 1:
        chi = calinski_harabasz_score(X_std, labels)
    else:
        chi = np.nan
        
    # score result
    results.loc[idx] = [algo_name, silhouette, ari, dbi, chi, n_clusters, noise_count]
    
    # Print detailed results
    print(f"Number of clusters found: {n_clusters}")
    if noise_count > 0:
        print(f"Noise/outlier points: {noise_count}")
    print(f"Silhouette Score: {silhouette:.4f} (↑ better)")
    print(f"Adjusted Rand Index: {ari:.4f} (↑ better, 1=perfect)")
    print(f"Davies-Bouldin Index: {dbi:.4f} (↓ better)")
    print(f"Calinski-Harabasz Index: {chi:.2f} (↑ better)")
    
    # --------- 4. VISUALIZATION: Plot clustering results ---------
    ax = axes[plot_row, plot_col]
    
    # Create a colormap that handles noise points (for DBSCAN)
    if -1 in labels and algo_name == 'DBSCAN':
        # Create a colormap where -1 (noise) is black
        from matplotlib.colors import ListedColormap
        cmap = plt.cm.tab10
        colors = cmap(np.arange(max(0, n_clusters)))
        colors = np.vstack([np.array([0,0,0,1]), colors]) # add black for noise
        cmap_with_noise = ListedColormap(colors)
        
        # plot with noise as black
        scatter = ax.scatter(X_std[:, 0],
                             X_std[:, 1],
                             c=labels +1, # shift labes to make -1 -> 0
                             cmap=cmap_with_noise,
                             s=50,
                             alpha=0.8,
                             edgecolors='k'
                             )
    else:
        scatter = ax.scatter(X_std[:, 0],
                             X_std[:, 1],
                             c=labels, 
                             cmap='tab10',
                             s=50,
                             alpha=0.8,
                             edgecolors='k'
                             )        
    ax.set_title(f'{algo_name} \n(silhouette: {silhouette:.3f}, ARI: {ari:.3f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # add text with key metrics
    metrics_text = f"Clusters: {n_clusters}\n"
    if noise_count > 0:
        metrics_text += f'Noise: {noise_count}\n'
    metrics_text += f'DBI: {dbi:.3f}\nCHI: {chi:.3f}'
    
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8
                )
            )
    # update subplot position
    plot_col += 1
    if plot_col > 1:
        plot_row = 1
        plot_col = 0
plt.tight_layout()
plt.show()

# --------- 5. RESULTS SUMMARY: Create comprehensive comparison table ---------
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON RESULTS")
print("="*80)

# sort by ARI (primary metric since we have ground truth)
result_sorted = results.sort_values('ARI', ascending=False)

# Display formatted results table
print("\n📊 Ranked by Adjusted Rand Index (higher is better):")
print("-" * 90)
print(f"{'Algorithm':<12} {'Clusters':<10} {'Silhouette':<12} {'ARI':<10} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<15} {'Noise':<8}")
print("-" * 90)

for _, row in result_sorted.iterrows():
    print(f"{row['Algorithm']:<12} {row['n_clusters']:<10} {row['Silhouette']:<12.4f} "
          f"{row['ARI']:<10.4f} {row['Davies-Bouldin']:<15.4f} "
          f"{row['Calinski-Harabasz']:<15.1f} {row['noise_points']:<8}")

print("-" * 90)

# --------- 6. METRICS VISUALIZATION: Radar chart for multi-metric comparison ---------

fig = plt.figure(figsize=(14, 2))

# subplot 1: Bar chart comparison
ax1 = fig.add_subplot(121)

# Normalize metrics for fair comprison
metrics_to_plot = ['Silhouette', 'ARI', 'Calinski-Harabasz']
# for DBI, lower is better, so we take inverse
results['DBI_inverse'] = 1 / (results['Davies-Bouldin'] + 1e-10) # Add small epsilon
metrics_to_plot.append('DBI_inverse')

x = np.arange(len(results))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    if metric == 'DBI_inverse':
        values = results['DBI_inverse'].values
        label = '1/DBI (↑ better)'
    else:
        values = results[metric].values
        label = metric

    offset = width * (i - len(metrics_to_plot) /2 + 0.5 )
    bars = ax1.bar(x + offset, values, width, label=label, alpha=0.8)
    
ax1.set_xlabel('Clustering Algorithm', fontsize=12)
ax1.set_ylabel('Metric Values (normalized)' , fontsize=12)
ax1.set_title('Multi-Metric Algorithm Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
ax1.grid(True, alpha=0.3, axis='y')

# subplot 2: Reader chart
ax2 = fig.add_subplot(122, polar=True)

# Prepare data for radar chart
metrics_radar = ['Silhouette', 'ARI', 'Calinski-Harabasz', 'DBI_inverse']
labels_radar = ['Silhouette\n(↑)', 'ARI\n(↑)', 'Calinski-Harabasz\n(↑)', '1/DBI\n(↑)']

# Normalize each metric to [0,1] range for radar chart
normalized_data = []
for metric in metrics_radar:
    if metric == 'DBI_inverse':
        values = results['DBI_inverse'].values
    else:
        values = results[metric].values
    
    # Handle NaN values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) > 0:
        min_val = valid_values.min()
        max_val = valid_values.max()
        if max_val - min_val > 0:
            normalized = (values - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(values)
    else:
        normalized = np.zeros_like(values)
    
    normalized_data.append(normalized)
    
normalized_data = np.array(normalized_data)    

# Number of Variables
categories = len(labels_radar)
angles = np.linspace(0, 2 *np.pi, categories, endpoint=False).tolist()
angles += angles[:1] # Close the polygon

# plot each Algorithm
for idx, algo_name in enumerate(results['Algorithm']):
    values = normalized_data[:, idx].tolist()
    values += values[:1] # Close the polygon
    
    ax2.plot(angles, values, 'o-', linewidth=2, label=algo_name,
             color=clustering_algorithms[algo_name]['color'], alpha=0.7)
    ax2.fill(angles, values, alpha=0.1, color=clustering_algorithms[algo_name]['color'])
    
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels_radar, fontsize=10)
ax2.set_ylim([0, 1])
ax2.set_title('Radar chart: Normalized Metrics Comparson',
              fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()

# --------- 7. ALGORITHM RECOMMENDATION: Based on analysis ---------
print("\n" + "="*80)
print("ALGORITHM RECOMMENDATIONS FOR MOON DATASET")
print("="*80)

print("\n🔍 ANALYSIS SUMMARY:")
print("Moon dataset has non-spherical, interleaving clusters which is challenging for centroid-based methods.")

print("\n🏆 WINNER ALGORITHM:")
best_algo = result_sorted.iloc[0]
print(f"  → {best_algo['Algorithm']}: ARI = {best_algo['ARI']:.4f}")
print("    Reason: Density-based algorithms excel at finding arbitrary-shaped clusters.")

print("\n📈 PERFORMANCE INSIGHTS:")

for _, row in result_sorted.iterrows():
    algo = row['Algorithm']
    ari = row['ARI']

    if algo == 'DBSCAN':
        print(f"  • DBSCAN: Excellent (ARI={ari:.4f}) - correctly identifies moon shapes")
    elif algo == 'Agglomerative':
        print(f"  • Agglomerative: Good (ARI={ari:.4f}) - works well with 'ward' linkage")
    elif algo == 'KMeans':
        print(f"  • KMeans: Poor (ARI={ari:.4f}) - assumes spherical clusters, fails on moon shapes")

print("\n🎯 PRACTICAL RECOMMENDATIONS:")
print("1. For arbitrary-shaped data: Use DBSCAN or HDBSCAN")
print("2. For hierarchical relationships: Use Agglomerative Clustering")
print("3. For spherical, well-separated clusters: Use KMeans")
print("4. Always use multiple metrics (Silhouette + ARI + DBI) for comprehensive evaluation")
print("5. Visualize results to validate metric findings")


