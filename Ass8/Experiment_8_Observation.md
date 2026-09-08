# Experiment 8: Clustering Human Activity Recognition Data using K-Means, DBSCAN, and Hierarchical Clustering

## Aim
To implement and analyze K-Means, DBSCAN, and Hierarchical Agglomerative Clustering (HAC) on the UCI Human Activity Recognition (HAR) Using Smartphones dataset, select k for K-Means via the elbow method, tune DBSCAN's eps/min_samples, compare HAC linkage criteria, visualize clusters via PCA/t-SNE, and evaluate all three with internal and external metrics.

## Dataset
- UCI Human Activity Recognition Using Smartphones Dataset
- 30 volunteers, 6 activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
- 10,299 samples, 561 time/frequency-domain features per window
- Features standardized then PCA-reduced to 65 components (90% variance retained) for clustering

## Table 1: K-Means Elbow Method Results
| Number of Clusters (k) | WCSS (Inertia) | Silhouette Score |
|---|---|---|
| 2 | 2,697,926.76 | 0.4375 |
| 3 | 2,346,425.10 | 0.3513 |
| 4 | 2,207,133.31 | 0.1809 |
| 5 | 2,081,104.64 | 0.1595 |
| 6 | 2,003,581.42 | 0.1403 |
| 7 | 1,947,312.31 | 0.1176 |
| 8 | 1,892,509.02 | 0.0998 |

**Best k = 2** (highest silhouette score, 0.4375).

DBSCAN was tuned to eps=16, min_samples=15 (2 clusters, 5.27% noise). For Hierarchical Clustering, Ward linkage was selected over single/complete/average after comparison, since Ward gave the best external agreement (ARI) while single/average collapsed into one dominant cluster (chaining).

## Evaluation Metrics and Results
| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI | NMI |
|---|---|---|---|---|---|
| K-Means (k=2) | 0.4375 | 0.9611 | 9560.02 | 0.3296 | 0.5455 |
| DBSCAN (eps=16, min_samples=15) | 0.3613 | 0.8423 | 146.56 | 0.0095 | 0.0476 |
| HAC (Ward linkage) | 0.1698 | 1.8134 | 449.59 | 0.3487 | 0.5182 |

## Observation Questions

**Which algorithm produced the most meaningful clusters? Why?**
HAC (Ward linkage) produced the most meaningful clusters overall by external agreement with the true activity labels (ARI=0.3487, NMI=0.5182), closely followed by K-Means (ARI=0.3296, NMI=0.5455). K-Means scored highest on Silhouette alone (0.4375), showing that internal compactness and external agreement with ground truth do not always pick the same winner.

**How sensitive was K-Means to the choice of k?**
K-Means is clearly sensitive to k here: the silhouette score peaks sharply at k=2 (0.4375) and falls at every larger k tested (down to 0.0998 at k=8), because the six HAR activities separate mainly into a coarse static/dynamic split in PCA space rather than six equally well-separated blobs.

**Did DBSCAN detect noise or small clusters effectively?**
With eps=16, min_samples=15, DBSCAN labelled 5.27% of points as noise and found 2 clusters, correctly isolating borderline transition-window samples as noise rather than forcing them into a cluster. However, its external agreement (ARI=0.0095) was much weaker than K-Means or HAC.

**How does linkage choice (single/complete/ward) affect hierarchical clustering?**
Single and average linkage concentrate over 90% of points into a single dominant cluster (chaining), which inflates their Silhouette score without recovering real activity structure (ARI near zero). Ward linkage instead produces balanced cluster sizes and the best external agreement (ARI=0.3487).

**Which internal metric best matched your visual intuition of cluster quality?**
Calinski-Harabasz matched the PCA/t-SNE scatter plots most closely: it favored K-Means' visually well-separated clusters and penalized DBSCAN's noise-heavy groups. Silhouette alone was misleading for single/average-linkage HAC, where it rewarded a chaining-driven single dominant cluster.

## Conclusion
HAC (Ward) best recovered the true activity structure externally (ARI=0.3487), narrowly ahead of K-Means (ARI=0.3296), while K-Means scored highest on internal Silhouette alone (0.4375) — a reminder that internal metrics must be cross-checked against external metrics before being trusted. All three algorithms separate static activities (SITTING, STANDING, LAYING) from dynamic ones (WALKING variants) more cleanly than they separate the six individual activities.

## Learning Outcomes
- Learned to select k for K-Means using both the elbow curve and the silhouette score.
- Understood that DBSCAN's eps must be chosen relative to the actual distance scale of the feature space, not guessed arbitrarily.
- Learned that Silhouette score alone can be misleading for hierarchical clustering when linkage causes chaining.
- Practiced comparing clustering algorithms using both internal and external metrics against ground-truth labels.
