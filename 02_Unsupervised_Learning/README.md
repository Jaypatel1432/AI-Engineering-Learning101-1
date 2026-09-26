# 📙 Topic 02: Unsupervised Learning & Recommendation Systems

Unsupervised Learning works with unlabelled data where no target variable ($y$) is provided. The objective is to discover underlying patterns, groupings, hidden structures, or relationships directly from input feature matrix ($X$).

---

## 💡 Core Concepts

### 1. Clustering
Clustering groups data points such that items in the same cluster are more similar to each other than to items in other clusters.

* **K-Means Clustering**: Partitioning method that splits data into $K$ pre-defined clusters.
  * *Centroid Update*: Iteratively calculates mean position of points assigned to centroids until convergence.
  * *Elbow Method*: Plots **Inertia (Within-Cluster Sum of Squares)** against $K$ to select the optimal number of clusters at the "elbow" point.
* **Hierarchical Clustering**:
  * *Agglomerative (Bottom-Up)*: Starts with each point as an individual cluster and iteratively merges closest pairs.
  * *Dendrogram*: Tree visualizer used to determine optimal cluster cutoffs.
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
  * Groups points based on spatial density ($\epsilon$ radius and `min_samples`).
  * Handles non-spherical clusters and automatically isolates noise/outliers without requiring a preset $K$.

### 2. Dimensionality Reduction & Anomaly Detection
* **Principal Component Analysis (PCA)**: Linear technique that transforms high-dimensional feature spaces into orthogonal principal components while maximizing variance retention.
* **Anomaly Detection (Isolation Forest)**: Identifies rare observations by isolating outliers through random feature splits (outliers require fewer splits to isolate).

---

## 🛍️ Recommendation Systems

Recommendation engines analyze user preferences and item properties to provide personalized content or product suggestions.

| Approach | Description | Key Metric / Method |
| :--- | :--- | :--- |
| **Content-Based Filtering** | Compares item features with historical preferences of the active user. | **Cosine Similarity** |
| **Collaborative Filtering** | Recommends items based on similarities between users or item rating patterns. | **User-Item Matrix, SVD / Matrix Factorization** |
| **Hybrid Systems** | Combines Content-Based and Collaborative methods to resolve cold-start issues. | Deep Learning Embeddings + Matrix Factorization |

---

## 💻 Python Implementation Example

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------
# 1. CLUSTERING EXAMPLE: Customer Segmentation (K-Means)
# -------------------------------------------------------------
# Features: [Annual Income ($k), Spending Score (1-100)]
X_customers = np.array([
    [15, 39], [15, 81], [16, 6], [16, 77], [17, 40], # Segment 1
    [55, 42], [58, 60], [60, 49], [62, 53], [64, 42], # Segment 2
    [87, 88], [88, 91], [92, 72], [95, 89], [99, 97]  # Segment 3
])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_customers)

print("--- CUSTOMER SEGMENTATION RESULTS ---")
for idx, label in enumerate(cluster_labels):
    print(f"Customer {idx+1} (Income: {X_customers[idx][0]}k, Score: {X_customers[idx][1]}) -> Cluster {label}")

# -------------------------------------------------------------
# 2. RECOMMENDATION EXAMPLE: Content-Based Movie Recommender
# -------------------------------------------------------------
# Movie Features Matrix: [Action, Comedy, Romance, Sci-Fi]
movies_data = {
    'Avengers': [1.0, 0.2, 0.0, 0.9],
    'Interstellar': [0.8, 0.0, 0.1, 1.0],
    'The Hangover': [0.1, 1.0, 0.2, 0.0],
    'La La Land': [0.0, 0.3, 1.0, 0.0]
}

df_movies = pd.DataFrame(movies_data, index=['Action', 'Comedy', 'Romance', 'Sci-Fi']).T

# Calculate Cosine Similarity
sim_matrix = cosine_similarity(df_movies)
df_sim = pd.DataFrame(sim_matrix, index=df_movies.index, columns=df_movies.index)

print("\n--- MOVIE RECOMMENDATION SYSTEM ---")
target_movie = 'Avengers'
recommended_movie = df_sim[target_movie].drop(target_movie).idxmax()
similarity_score = df_sim[target_movie].drop(target_movie).max()

print(f"Target Movie: '{target_movie}' | Top Recommendation: '{recommended_movie}' (Cosine Similarity: {similarity_score:.2f})")
```

---

## 🎯 Practical Exercise Assignment

### Task Title: "The E-Commerce Customer & Product Intelligence Engine"

#### Task 1 (Clustering)
* **Dataset:** Mall Customer Segmentation Dataset (Kaggle).
* **Goal:** Apply K-Means clustering, use the **Elbow Method** to find optimal $K$, and visualize the resulting customer segments.

#### Task 2 (Recommendation System)
* **Dataset:** MovieLens 100K Dataset.
* **Goal:** Build a **Collaborative Filtering** recommendation model using a User-Item Matrix and Cosine Similarity to output Top-5 movie recommendations for a given User ID.
