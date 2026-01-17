import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def standardize_features(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data


def run_elbow_method(data, cluster_options):
    inertias = []

    for k in cluster_options:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        model.fit(data)
        inertias.append(model.inertia_)

    return inertias


def plot_elbow(cluster_options, inertias):
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_options, inertias, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans Clustering")
    plt.show()


def train_kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = model.fit_predict(data)
    return labels, model
