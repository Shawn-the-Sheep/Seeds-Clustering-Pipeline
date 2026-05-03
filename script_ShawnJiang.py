import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

candidates = ["seed_dataset.txt", "seeds_dataset.txt", "/mnt/data/seeds_dataset.txt"]
data_path = None
for p in candidates:
    if os.path.exists(p):
        data_path = p
        break

if data_path is None:
    raise FileNotFoundError("Could not find dataset file.")

print("Using data file:", data_path)

colnames = [
    "area", "perimeter", "compactness",
    "length_kernel", "width_kernel",
    "asymmetry_coef", "groove_length", "class"
]
df = pd.read_csv(data_path, sep=r"\s+", header=None, names=colnames, engine="python")
X = df.iloc[:, :-1].values
n_samples, n_features = X.shape
print(f"Loaded {n_samples} samples with {n_features} features.")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

ks = [1, 2, 3, 4, 5, 7, 10]
k_distances = {}

for k in ks:
    neigh = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(Xs)
    distances, indices = neigh.kneighbors(Xs)
    kth_dist = distances[:, -1]
    kth_sorted = np.sort(kth_dist)
    k_distances[k] = kth_sorted
    print(f"k={k:2d}: mean={kth_dist.mean():.4f}, max={kth_dist.max():.4f}")

out_dir = os.path.abspath(".")
overlay_path = os.path.join(out_dir, "k_distance_plots_overlay.png")

plt.figure(figsize=(10, 6))
for k in ks:
    plt.plot(k_distances[k], label=f'k={k}')
plt.xlabel('Points sorted by k-th NN distance')
plt.ylabel('k-th nearest neighbour distance')
plt.title('k-distance curves (overlay)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(overlay_path, dpi=200)
print("Saved overlay plot to:", overlay_path)
plt.close()

def compute_within_sse(X_scaled, labels):
    sse = 0.0
    unique = np.unique(labels)
    for lab in unique:
        if lab == -1:
            continue
        pts = X_scaled[labels == lab]
        if pts.shape[0] == 0:
            continue
        centroid = pts.mean(axis=0)
        diffs = pts - centroid
        sse += np.sum(diffs ** 2)
    return float(sse)

param_sets = [
    {"min_samples": 2, "eps": 1.1},
    {"min_samples": 3, "eps": 1.12},
    {"min_samples": 7, "eps": 1.325}
]

print("\nRunning DBSCAN on scaled data for requested parameter sets...\n")
results = []
for params in param_sets:
    eps = params["eps"]
    min_s = params["min_samples"]
    db = DBSCAN(eps=eps, min_samples=min_s, metric='euclidean')
    labels = db.fit_predict(Xs)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    sse = compute_within_sse(Xs, labels)
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(map(int, unique), map(int, counts)))
    results.append({
        "eps": eps,
        "min_samples": min_s,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "sse": sse,
        "label_counts": label_counts
    })

for r in results:
    print(f"DBSCAN params: eps = {r['eps']}, min_samples = {r['min_samples']}")
    print(f"  clusters found (excluding noise): {r['n_clusters']}")
    print(f"  noise points: {r['n_noise']}")
    print(f"  within-cluster SSE (scaled space): {r['sse']:.6f}")
    print(f"  label counts (label:count)  -- label -1 = noise:")
    for lab, cnt in r['label_counts'].items():
        print(f"    {lab} : {cnt}")
    print("-" * 60)

def run_bisecting_kmeans(data, k_total):
    clusters = [np.arange(len(data))]
    
    while len(clusters) < k_total:
        max_sse = -1.0
        index_to_split = -1
        
        for i, idxs in enumerate(clusters):
            points = data[idxs]
            if len(points) < 2:
                current_sse = 0.0
            else:
                centroid = points.mean(axis=0)
                current_sse = np.sum((points - centroid)**2)
            
            if current_sse > max_sse:
                max_sse = current_sse
                index_to_split = i
        
        if index_to_split == -1:
            break
            
        idxs_to_split = clusters.pop(index_to_split)
        points_to_split = data[idxs_to_split]
        
        kmeans_sub = KMeans(n_clusters=2, n_init=10)
        sub_labels = kmeans_sub.fit_predict(points_to_split)
        
        c1 = idxs_to_split[sub_labels == 0]
        c2 = idxs_to_split[sub_labels == 1]
        
        clusters.append(c1)
        clusters.append(c2)
        
    total_sse = 0.0
    for idxs in clusters:
        points = data[idxs]
        if len(points) > 0:
            centroid = points.mean(axis=0)
            total_sse += np.sum((points - centroid)**2)
            
    return total_sse, clusters

print("\nDetermining optimal k using the Elbow Method...")
max_eval_k = 10
k_values = list(range(1, max_eval_k + 1))
sse_values = []
clusters_dict = {}

for k in k_values:
    sse, clusters = run_bisecting_kmeans(Xs, k_total=k)
    sse_values.append(sse)
    clusters_dict[k] = clusters

k_norm = [(k - min(k_values)) / (max(k_values) - min(k_values)) for k in k_values]
sse_norm = [(sse - min(sse_values)) / (max(sse_values) - min(sse_values)) for sse in sse_values]

x1, y1 = k_norm[0], sse_norm[0]
x2, y2 = k_norm[-1], sse_norm[-1]

max_distance = -1
optimal_k = 1

for i in range(len(k_values)):
    x0, y0 = k_norm[i], sse_norm[i]
    numerator = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    distance = numerator / denominator
    
    if distance > max_distance:
        max_distance = distance
        optimal_k = k_values[i]

print(f"--> The elbow method determined the optimal number of clusters is: k = {optimal_k}")

elbow_path = os.path.join(out_dir, "elbow_curve.png")
plt.figure(figsize=(8, 5))
plt.plot(k_values, sse_values, marker='o', linestyle='-', color='b', label='SSE')
plt.axvline(optimal_k, color='green', linestyle='--', label=f'Optimal k={optimal_k}')
plt.title('Elbow Method for Bisecting K-Means')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(elbow_path, dpi=200)
print("Saved elbow curve plot to:", elbow_path)
plt.close()

real_sse = sse_values[optimal_k - 1]
real_clusters = clusters_dict[optimal_k]
print(f"\nReal Data Bisecting K-Means (k={optimal_k}) SSE: {real_sse:.4f}")

print("Running 300 iterations of randomization test...")

n_iterations = 300
random_sses = []
feat_mins = Xs.min(axis=0)
feat_maxs = Xs.max(axis=0)

for i in range(n_iterations):
    X_rand = np.random.uniform(low=feat_mins, high=feat_maxs, size=Xs.shape)
    rand_sse, _ = run_bisecting_kmeans(X_rand, k_total=optimal_k)
    random_sses.append(rand_sse)

random_sses = np.array(random_sses)
p_value = np.sum(random_sses <= real_sse) / n_iterations

print(f"Mean Random SSE: {random_sses.mean():.4f}")
print(f"Std Random SSE: {random_sses.std():.4f}")
print(f"P-value (Prob Random SSE <= Real SSE): {p_value:.6f}")

hist_path = os.path.join(out_dir, "sse_randomization_hist.png")
plt.figure(figsize=(10, 6))
plt.hist(random_sses, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Random SSEs')
plt.axvline(real_sse, color='red', linestyle='dashed', linewidth=2, label=f'Real SSE ({real_sse:.2f})')
plt.title('Distribution of SSE from Random Data vs Real Data SSE')
plt.xlabel('SSE')
plt.ylabel('Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(hist_path, dpi=200)
print("Saved histogram to:", hist_path)
plt.close()

print(f"\nExporting {optimal_k} cluster contents to text files and calculating misclassification...")

for i, indices in enumerate(real_clusters):
    cluster_num = i + 1
    filename = f"cluster{cluster_num}.txt"
    file_path = os.path.join(out_dir, filename)
    
    cluster_data = df.iloc[indices]
    cluster_data.to_csv(file_path, sep='\t', index=False, header=True)
    
    class_values = cluster_data['class'].values
    unique_classes, counts = np.unique(class_values, return_counts=True)
    
    if len(counts) > 0:
        majority_idx = np.argmax(counts)
        majority_class = unique_classes[majority_idx]
        majority_count = counts[majority_idx]
    else:
        majority_class = -1
        majority_count = 0
        
    total_in_cluster = len(cluster_data)
    misclassified_count = total_in_cluster - majority_count
    misclassified_pct = (misclassified_count / total_in_cluster) * 100 if total_in_cluster > 0 else 0.0
    
    print(f"Cluster {cluster_num}: Saved to {filename}")
    print(f"  Total items: {total_in_cluster}")
    print(f"  Majority Class: {majority_class}")
    print(f"  Misclassified items: {misclassified_count}")
    print(f"  Misclassification Percentage: {misclassified_pct:.2f}%")
    print("-" * 40)

print("\nScript processing complete.")