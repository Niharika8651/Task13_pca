import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

print("Original shape:", X.shape)

# 2. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA with different components
components = [2, 10, 30, 50]
explained_variance = []

for n in components:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

# 4. Plot explained variance
plt.figure()
plt.plot(components, explained_variance, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.show()

# 5. Train-test split (original data)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Logistic Regression on original data
model_original = LogisticRegression(max_iter=5000)
model_original.fit(X_train, y_train)
y_pred_original = model_original.predict(X_test)
acc_original = accuracy_score(y_test, y_pred_original)

# 7. PCA with 30 components
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

# 8. Logistic Regression on PCA data
model_pca = LogisticRegression(max_iter=5000)
model_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = model_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test_pca, y_pred_pca)

print("Accuracy without PCA:", acc_original)
print("Accuracy with PCA (30 components):", acc_pca)

# 9. 2D PCA Visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure()
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar(scatter)
plt.title("2D PCA Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
