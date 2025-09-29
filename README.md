# Project Summary

This document addresses the gap in the literature related to two emerging trends: sustainable portfolio construction and research on the technology sector. It aims to integrate ESG factors with financial performance using machine learning techniques applied to NASDAQ-100 technology stocks to create sustainable portfolios.

A three-stage framework is developed: First, the stocks are clustered using four different combinations of variables, where the approach with the best silhouette score is chosen. Then, for each cluster, machine learning models are fitted to the data to forecast return and volatility, selecting the one with the lowest RMSE. Finally, with the forecasted values, different ESG-aware portfolios are obtained and compared with portfolios created without considering the ESG score, by applying a Monte Carlo simulation and obtaining the portfolio metrics under uncertainty.

The results showed that a simple approach, using historical returns, volatility, and volume values, generates the most homogeneous and interpretable clusters, with two groups. In forecasting, the SVM model is the best in predicting both returns, while XGBoost and Random Forest excel in forecasting volatility. Finally, the ESG-aware portfolios can sacrifice a small portion of return and volatility to improve the ESG score by a greater amount.

These findings propose a new framework to create ESG-aware portfolios based on large market indexes using clustering, machine learning forecasting, and optimization, offering alternatives to investors where a sustainable portfolio can be obtained without sacrificing financial performance.
