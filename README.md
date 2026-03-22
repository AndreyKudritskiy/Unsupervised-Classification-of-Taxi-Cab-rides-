# Unsupervised-Classification-of-Taxi-Cab-rides
*Note: this is a group project, in the ipynb file subheaders labeled with [Andrey Kudritskiy] were written by me, I won't take credit for the work*.

## 1. EDA & Feature Engineering (*Not my work*)
Raw data includes trip ID, call type, day type, timestamp, and GPS polyline coordinates. Key engineered features include geo_len (number of GPS points as a distance proxy), trip_duration_min (estimated from 15s intervals), hour_of_day, and starting lat/lon coordinates extracted via regex from the polyline. Exploratory plots include histograms, boxplots, and a correlation heatmap.

## 2. Preprocessing Pipeline (*Other Group Member + Andrey (me)*)
A sklearn pipeline applies the custom TripFeatureEngineer transformer, followed by a ColumnTransformer that standardizes numeric features and one-hot encodes categoricals. Three train/test splits (10%, 25%, 30%) are generated and processed.

## 3. Model 1 — K-Means (*Andrey (me)*)
Grid search over n_clusters (3, 5, 7, 9), n_init, and max_iter across all splits. Results are evaluated via inertia, silhouette, Calinski-Harabasz, and Davies-Bouldin scores. An elbow curve analysis suggests k=5 or k=7 as optimal. Clusters are also visualized geographically on a Folium map. Reasoned for why lon/lat coordinates must be projected onto a 2d plane to make location data be applicable to calculations of euclidean distance. WCSS for instance.

## 4. Model 2 — Agglomerative (Hierarchical) Clustering (*Andrey(me)*)
Grid search over linkage types and cluster counts. Despite strong metric scores, inspection revealed the model was funneling nearly all data into a single cluster — flagged as a limitation. A dendrogram is plotted for the best configuration.

## 5. Model 3 — Gaussian Mixture Model (GMM) (*Not my work*)
Grid search on GMM parameters. Evaluated using BIC and AIC (lower = better), with scatter plots showing probability ellipses for cluster membership.

## 6. Model Comparison & Selection (*Andrey(me) + Other Group Member*)
All models' metrics (silhouette, Calinski-Harabasz, Davies-Bouldin) are standardized and combined into a single composite score. The top model is selected, refitted on the best train split, and evaluated on the held-out test set. Cluster profiles (mean feature values per cluster) are computed. The final model is saved via pickle.
