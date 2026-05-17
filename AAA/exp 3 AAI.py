import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# ---------- 1. Generate Sample 2D Data ----------
np.random.seed(42)

X1 = np.random.multivariate_normal(mean=[2, 2],   cov=[[0.5, 0.1], [0.1, 0.5]], size=150)
X2 = np.random.multivariate_normal(mean=[-2, -2], cov=[[0.6, -0.2], [-0.2, 0.6]], size=150)
X3 = np.random.multivariate_normal(mean=[2, -2],  cov=[[0.4, 0.0], [0.0, 0.4]], size=150)

X = np.vstack((X1, X2, X3))

# True labels (for ARI evaluation only)
true_labels = np.array([0]*150 + [1]*150 + [2]*150)

# ---------- 2. Fit Gaussian Mixture Model ----------
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X)

# ---------- 3. Visualize Cluster Results ----------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=40)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='X', s=200, label='Cluster Means')
plt.title("Gaussian Mixture Model Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

ari = adjusted_rand_score(true_labels, labels)
print("Adjusted Rand Index (ARI):", ari)

posterior_probs = gmm.predict_proba(X[:5])
print("\nPosterior Probabilities for First 5 Samples:")
for i, probs in enumerate(posterior_probs):
    print(f"Sample {i + 1}: {probs}")

print("\nCluster Means:")
print(gmm.means_)
 
print("\nCluster Covariances:")
for i, cov in enumerate(gmm.covariances_):
    print(f"\nGaussian Component {i}:")
    print(cov)