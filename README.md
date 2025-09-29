# Project Summary

This project addresses the gap in the literature related to two emerging trends: sustainable portfolio construction and research on the technology sector. It aims to integrate ESG factors with financial performance using machine learning techniques applied to NASDAQ-100 technology stocks to create sustainable portfolios.

A three-stage framework is developed: First, the stocks are clustered using four different combinations of variables, where the approach with the best silhouette score is chosen. Then, for each cluster, machine learning models are fitted to the data to forecast return and volatility, selecting the one with the lowest RMSE. Finally, with the forecasted values, different ESG-aware portfolios are obtained and compared with portfolios created without considering the ESG score, by applying a Monte Carlo simulation and obtaining the portfolio metrics under uncertainty.

The results showed that a simple approach, using historical returns, volatility, and volume values, generates the most homogeneous and interpretable clusters, with two groups. In forecasting, the SVM model is the best in predicting both returns, while XGBoost and Random Forest excel in forecasting volatility. Finally, the ESG-aware portfolios can sacrifice a small portion of return and volatility to improve the ESG score by a greater amount.

These findings propose a new framework to create ESG-aware portfolios based on large market indexes using clustering, machine learning forecasting, and optimization, offering alternatives to investors where a sustainable portfolio can be obtained without sacrificing financial performance.

### 📂 File Structure

* The full project documentation, including methodology and detailed results, is available in [`Project.pdf`](report_clustering_forecasting_and_optimization_of_nasdaq_100_stocks_with_esg_integration.pdf).
* The code is divided into nine scripts, organized sequentially:

1. `1- Data processing.py` – Data collection and preprocessing.
2. `2- Data description.py` – Exploratory analysis of the dataset.
3. `3- Clustering.py` – Clustering NASDAQ 100 stocks based on fundamentals
4. `4a- Cluster 0 return forecasting.py` – ML models for predicting **returns** of Cluster 0.
5. `4b- Cluster 0 volatility forecasting.py` – ML models for predicting **volatility** of Cluster 0.
6. `4c- Cluster 1 return forecasting.py` – ML models for predicting **returns** of Cluster 1.
7. `4d- Cluster 1 volatility forecasting.py` – ML models for predicting **volatility** of Cluster 1.
8. `5a- Portfolio optimization with ESG score.py` – Portfolio optimization including ESG variables.
9. `5b- Portfolio optimization without ESG score.py` – Portfolio optimization without ESG variables.

### Execution Order

To reproduce the full workflow, run the scripts in the following order:
`1 → 2 → 3 → 4a → 4b → 4c → 4d → 5a → 5b`

Each step generates intermediate outputs (e.g., cleaned datasets, cluster assignments, trained models), which are then used in the subsequent scripts.
