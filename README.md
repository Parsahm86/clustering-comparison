# Clustering Algorithms Comparison - Moon Dataset Analysis


## 📊 Project Overview
A practical comparison of KMeans, DBSCAN, and Agglomerative clustering algorithms on moon-shaped datasets. This project explores how different clustering approaches handle non-linear, non-spherical data.

## 🎯 Key Results

### Performance Summary
| Algorithm | Clusters | Silhouette | ARI | Davies-Bouldin | Calinski-Harabasz |
|-----------|----------|------------|-----|----------------|-------------------|
| **DBSCAN** | 2 | 0.3860 | **1.0000** | 1.0211 | 259.6 |
| **Agglomerative** | 2 | 0.4487 | 0.5363 | 0.8405 | 326.9 |
| **KMeans** | 2 | 0.4954 | 0.4697 | 0.8067 | 418.4 |

### 🏆 Algorithm Ranking (by ARI Score)
1. **DBSCAN** - Perfect score (ARI = 1.0000)
2. **Agglomerative** - Moderate performance (ARI = 0.5363) 
3. **KMeans** - Lowest performance (ARI = 0.4697)

## 🔍 Key Insights

### 🚀 DBSCAN Excellence
- **Perfect clustering** on moon data (ARI = 1.0000)
- **Density-based approach** successfully captures complex moon shapes
- **No parameter tuning needed** for this specific dataset

### 📊 Metric Analysis
- **ARI (Adjusted Rand Index)**: DBSCAN shows perfect alignment with ground truth
- **Silhouette Score**: KMeans has highest (0.4954) but misleading for this dataset
- **Davies-Bouldin**: All algorithms show reasonable compactness/separation balance
- **Calinski-Harabasz**: KMeans scores highest (418.4) showing good between-cluster variance

## 💡 What I Learned

### Technical Insights:
1. **Algorithm selection matters** - Different data shapes require different approaches
2. **Metrics can be misleading** - High silhouette score doesn't always mean good clustering
3. **DBSCAN excels** with arbitrary-shaped clusters like moon data
4. **KMeans limitations** - Assumes spherical clusters, fails on complex geometries
5. **Multiple evaluation metrics** are essential for comprehensive understanding

### Practical Skills:
- Implemented and compared three major clustering algorithms
- Calculated and interpreted four different evaluation metrics
- Learned when to use which algorithm based on data characteristics
- Understood the importance of visualizing clustering results

## 📈 Results Interpretation

### Why DBSCAN Performed Best:
- Moon dataset has **interleaving, non-spherical clusters**
- DBSCAN's density-based approach can detect **arbitrary shapes**
- **eps=0.3** and **min_samples=5** parameters were optimal for this dataset

### Why KMeans Struggled:
- Assumes **spherical, similarly-sized clusters**
- Cannot handle **crescent-shaped, interleaving data**
- Centroids cannot accurately represent moon shapes

### Agglomerative Performance:
- **Ward linkage** helps but still limited by hierarchical approach
- Better than KMeans but not as good as density-based methods
- Shows the value of hierarchical relationships in some cases


## 🚀 Getting Started

### Quick Run:
```bash
# Clone repository
git clone https://github.com/Parsahm86/clustering-analysis.git
cd clustering-analysis

# Install dependencies
pip install -r requirements.txt

# Run analysis
python clustering_comparison.py

numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0

