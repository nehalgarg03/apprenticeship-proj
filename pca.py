from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def run_pca(data, n_components=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    return reduced, pca


def plot_3d_clusters(pca_data, labels, highlight_label=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        pca_data[:, 2],
        c=labels,
        cmap="tab10",
        alpha=0.6
    )

    if highlight_label is not None:
        idx = labels == highlight_label
        ax.scatter(
            pca_data[idx, 0],
            pca_data[idx, 1],
            pca_data[idx, 2],
            c="red",
            label="Target Cluster"
        )

    ax.set_title("PCA-Reduced County Clusters")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.legend()
    plt.show()
