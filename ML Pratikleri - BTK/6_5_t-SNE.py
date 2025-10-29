from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

mist = fetch_openml('mnist_784', version=1)

X=mist.data
y=mist.target.astype("int")

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

plt.figure()
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap = "tab10", alpha=0.6)
plt.title("t-SNE of Iris Dataset")
plt.xlabel("T-SNE 1")
plt.ylabel("T-SNE 2")
plt.show()