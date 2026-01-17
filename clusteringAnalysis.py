from src.preprocessing import load_and_clean_geographic_data
from src.clustering import (
    standardize_features,
    run_elbow_method,
    plot_elbow,
    train_kmeans
)

# Load placeholder data
df = load_and_clean_geographic_data(
    "synthetic_data/example_county_data.csv",
    drop_cols=["GEO_ID", "NAME"]
)

scaled_data = standardize_features(df)

cluster_options = [10, 25, 50, 100, 150, 200]
inertias = run_elbow_method(scaled_data, cluster_options)

plot_elbow(cluster_options, inertias)

labels, model = train_kmeans(scaled_data, n_clusters=150)
df["cluster_id"] = labels
df.head()
