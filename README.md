# apprenticeship-proj
# Data-Driven County Similarity Analysis for Nonprofit Expansion

## Project Overview
This project was developed as part of a data-driven initiative to support a nonprofit organization in identifying optimal U.S. counties for program expansion. The goal was to leverage demographic, education, and career outcomes data to surface regions with similar characteristics to an existing successful location.

Due to organizational and data privacy restrictions, the original repository and datasets are private. This public repository demonstrates the modeling, preprocessing, and evaluation approach using generalized code and synthetic data placeholders.

---

## Problem Statement
The nonprofit sought to expand its programming to new regions while maintaining strong impact. The challenge was identifying counties that shared meaningful similarities with an existing target region across education attainment, employment metrics, and demographic features.

This required working with large, multi-source datasets, selecting relevant features, and applying unsupervised learning techniques to identify comparable regions.

---

## My Contributions
I was responsible for the following components of the project:

- Cleaning and preprocessing census-style datasets  
- Selecting and transforming numeric features  
- Aggregating metrics at city, county, and state levels  
- Standardizing features prior to modeling  
- Applying KMeans clustering and evaluating cluster quality  
- Using PCA for dimensionality reduction and visualization  
- Interpreting cluster results to generate expansion recommendations  

---

## Modeling Approach

### Data Preparation
- Removed non-numeric and identifier columns
- Converted features to numeric values and handled missing data
- Aggregated metrics by geographic level

### Feature Scaling
- Applied StandardScaler to normalize features prior to clustering

### Clustering
- Used KMeans for unsupervised grouping
- Evaluated multiple cluster sizes using the elbow method
- Selected an optimal cluster size based on inertia trends

### Dimensionality Reduction
- Applied PCA to visualize high-dimensional data in lower dimensions
- Compared clusters in 2D and 3D space for interpretability

---

## Results
The model identified clusters of counties with strong similarity to the target region. These clusters were used to narrow down candidate expansion locations based on shared educational attainment and career outcome metrics. The results provided actionable guidance under real-world time constraints.

---

## Limitations and Future Work
- Relied on static census-style data
- Future work could incorporate time-series or economic indicators
- Exploring alternative clustering methods or ensemble approaches could improve robustness

---

## Ethical Considerations
All analysis was conducted with care to avoid exposing sensitive or identifiable data. This public repository contains generalized code and does not include private datasets.

