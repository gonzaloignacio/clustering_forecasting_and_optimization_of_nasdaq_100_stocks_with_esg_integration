# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv and filter it for the columns of interest
cluster_input_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_input.csv",
                               index_col = [0])
cluster_input_df = cluster_input_df.iloc[:, 1:]

# Get a dataframe with the description
var_description = cluster_input_df.describe()

# Create correlation matrix
cluster_corr = cluster_input_df.corr().round(2)

# Plot correlation matrix
sns.heatmap(cluster_corr, annot= True)
plt.title("Correlation matrix for Clustering variables")
plt.show()

# Order by ROA and plot
roa_order_input = cluster_input_df.sort_values("ROA", 
                                               ascending = False).iloc[0:10, :]
sns.barplot(data = roa_order_input, x = "ROA", y = roa_order_input.index)
plt.xlabel("Return on Assets")
plt.ylabel("Company")
plt.title("Top 10 companies by ROA")
plt.show()

# Order by ROE and plot
roe_order_input = cluster_input_df.sort_values("ROE", 
                                               ascending = False).iloc[0:10, :]
sns.barplot(data = roe_order_input, x = "ROE", y = roe_order_input.index)
plt.xlabel("Return on Equity")
plt.ylabel("Company")
plt.title("Top 10 companies by ROE")
plt.show()

# Order by EBITDA margin and plot
ebitda_order_input = cluster_input_df.sort_values("EBITDA margin", 
                                                  ascending = False).iloc[0:10, :]
sns.barplot(data = ebitda_order_input, x = "EBITDA margin", y = ebitda_order_input.index)
plt.xlabel("EBITDA margin")
plt.ylabel("Company")
plt.title("Top 10 companies by EBITDA margin")
plt.show()

# Order by Operating margin and plot
operating_order_input = cluster_input_df.sort_values("Operating margin", 
                                                     ascending = False).iloc[0:10, :]
sns.barplot(data = operating_order_input, x = "Operating margin", y = operating_order_input.index)
plt.xlabel("Operating margin")
plt.ylabel("Company")
plt.title("Top 10 companies by Operating margin")
plt.show()

# Order by Current ratio and plot
current_order_input = cluster_input_df.sort_values("Current Ratio", 
                                                     ascending = False).iloc[0:10, :]
sns.barplot(data = current_order_input, x = "Current Ratio", y = current_order_input.index)
plt.xlabel("Current ratio")
plt.ylabel("Company")
plt.title("Top 10 companies by Current ratio")
plt.show()

# Order by Quick ratio and plot
quick_order_input = cluster_input_df.sort_values("Quick Ratio", 
                                                     ascending = False).iloc[0:10, :]
sns.barplot(data = quick_order_input, x = "Quick Ratio", y = quick_order_input.index)
plt.xlabel("Quick ratio")
plt.ylabel("Company")
plt.title("Top 10 companies by Quick ratio")
plt.show()

# Order by Debt to equity ratio and plot
debt_order_input = cluster_input_df.sort_values("Debt-to-Equity", 
                                                     ascending = True).iloc[0:10, :]
sns.barplot(data = debt_order_input, x = "Debt-to-Equity", y = debt_order_input.index)
plt.xlabel("Debt-to-Equity ratio")
plt.ylabel("Company")
plt.title("Top 10 companies by Debt-to-Equity ratio")
plt.show()

# Order by Return and plot
return_order_input = cluster_input_df.sort_values("Historic Return", 
                                                     ascending = False).iloc[0:10, :]
sns.barplot(data = return_order_input, x = "Historic Return", y = return_order_input.index)
plt.xlabel("Historic Return")
plt.ylabel("Company")
plt.title("Top 10 companies by Historic Return")
plt.xticks([0, 0.0005, 0.001, 0.0015, 0.002])
plt.show()

# Order by Volatility and plot
volatility_order_input = cluster_input_df.sort_values("Historic Volatility", 
                                                     ascending = True).iloc[0:10, :]
sns.barplot(data = volatility_order_input, x = "Historic Volatility", y = volatility_order_input.index)
plt.xlabel("Historic Volatility")
plt.ylabel("Company")
plt.title("Top 10 companies by Historic Volatility")
plt.show()

# Order by Volume and plot
volume_order_input = cluster_input_df.sort_values("Historic Volume", 
                                                     ascending = False).iloc[0:10, :]
sns.barplot(data = volume_order_input, x = "Historic Volume", y = volume_order_input.index)
plt.xlabel("Historic Volume")
plt.ylabel("Company")
plt.title("Top 10 companies by Historic Volume")
plt.show()



###############################################################################



# Read csv and select useful columns
forecasting_input_df = pd.read_csv(r"C:\Users\gonza\Documents\forecasting_input.csv",
                                   parse_dates = [1],
                                   index_col = [1])
forecasting_input_df = forecasting_input_df.iloc[ :,  1 : 14]

# Group by ticker and compute the mean
forecasting_grouped_df = forecasting_input_df.groupby("Ticker").mean()

# Save the description as a dataframe
forecasting_description = forecasting_grouped_df.describe()

# Get the correlation matrix
forecasting_corr = forecasting_input_df.iloc[:, 1:].corr(method = "spearman").round(2)

# Plot the correlation matrix
sns.heatmap(forecasting_corr, annot = True, annot_kws = {"size": 8})
plt.title("Correlation matrix for Forecasting variables")
plt.show()

# Get the 3 tickers with higher average return
return_high_forecasting_index = forecasting_grouped_df.sort_values("Rolling_Return", ascending = False)\
    .iloc[0 : 6, :].index

# Filter the original dataset and plot
return_high_forecasting_df = forecasting_input_df[forecasting_input_df["Ticker"].isin(return_high_forecasting_index)]
g_r_high = sns.FacetGrid(return_high_forecasting_df,
                  col="Ticker",
                  col_wrap=3,
                  sharey=True)
g_r_high.map_dataframe(sns.lineplot,
                x = return_high_forecasting_df.index.name,
                y = "Rolling_Return")
g_r_high.set_axis_labels("Date", "Return")
g_r_high.fig.suptitle("Stocks with Higher Return by Date")
g_r_high.fig.tight_layout()
plt.show()

# Get the 3 tickers with lower average return
return_low_forecasting_index = forecasting_grouped_df.sort_values("Rolling_Return", ascending = True)\
    .iloc[0 : 6,:].index

# Filter the original dataset and plot
return_low_forecasting_df = forecasting_input_df[forecasting_input_df["Ticker"].isin(return_low_forecasting_index)]
g_r_low = sns.FacetGrid(return_low_forecasting_df,
                  col="Ticker",
                  col_wrap=3,
                  sharey=True)
g_r_low.map_dataframe(sns.lineplot,
                x = return_low_forecasting_df.index.name,
                y = "Rolling_Return")
g_r_low.set_axis_labels("Date", "Return")
g_r_low.fig.suptitle("Stocks with Lower Return by Date")
g_r_low.fig.tight_layout()
plt.show()

# Get the 3 tickers with higher average volatility
volatility_high_forecasting_index = forecasting_grouped_df.sort_values("Rolling_Volatility", ascending = False)\
    .iloc[0 : 6, :].index

# Filter the original dataset and plot
volatility_high_forecasting_df = forecasting_input_df[forecasting_input_df["Ticker"].isin(volatility_high_forecasting_index)]
g_v_high = sns.FacetGrid(volatility_high_forecasting_df,
                  col="Ticker",
                  col_wrap=3,
                  sharey=True)
g_v_high.map_dataframe(sns.lineplot,
                x = volatility_high_forecasting_df.index.name,
                y = "Rolling_Volatility")
g_v_high.set_axis_labels("Date", "Volatility")
g_v_high.fig.suptitle("Stocks with Higher Volatility by Date")
g_v_high.fig.tight_layout()
plt.show()

# Get the 3 tickers with lower average volatility
volatility_low_forecasting_index = forecasting_grouped_df.sort_values("Rolling_Volatility", ascending = True)\
    .iloc[0 : 6, :].index

# Filter the original dataset and plot
volatility_low_forecasting_df = forecasting_input_df[forecasting_input_df["Ticker"].isin(volatility_low_forecasting_index)]
g_v_low = sns.FacetGrid(volatility_low_forecasting_df,
                  col="Ticker",
                  col_wrap=3,
                  sharey=True)
g_v_low.map_dataframe(sns.lineplot,
                x = volatility_low_forecasting_df.index.name,
                y = "Rolling_Volatility")
g_v_low.set_axis_labels("Date", "Volatility")
g_v_low.fig.suptitle("Stocks with Lower Volatility by Date")
g_v_low.fig.tight_layout()
plt.show()



###############################################################################



# Read the input csv
esg_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_input.csv",
                               index_col = [0])
sp500 = pd.read_csv(r"C:\Users\gonza\Documents\sp500.csv",
                    parse_dates = ["Date"],
                    index_col = ["Date"])
stocks_corr_df = pd.read_csv(r"C:\Users\gonza\Documents\corr_matrix.csv",
                      index_col = [0])


# For the ESG score, filter the desired column and obtain statistics
esg_df = esg_df[["ESG score"]]
esg_description = esg_df.describe()

# Sort and Filter by lowest ESG scores and plot
esg_low_df = esg_df.sort_values("ESG score", ascending = True).iloc[0:10, ]
sns.barplot(data = esg_low_df, x = "ESG score", y = esg_low_df.index)
plt.xlabel("ESG score")
plt.ylabel("Company")
plt.title("10 Companies with the lowest ESG scores")
plt.show()

# Sort and Filter by highest ESG scores and plot
esg_high_df = esg_df.sort_values("ESG score", ascending = False).iloc[0:10, ]
sns.barplot(data = esg_high_df, x = "ESG score", y = esg_high_df.index)
plt.xlabel("ESG score")
plt.ylabel("Company")
plt.title("10 Companies with the highest ESG scores")
plt.show()

# Obtain summary statistics about sp500
sp500_description = sp500.describe()

# For the correlation plot, obtain only the high correlations
mask = np.triu(np.ones_like(stocks_corr_df, dtype=bool), k=1)
flat_corr = stocks_corr_df.where(mask).stack().reset_index()
flat_corr.columns = ["Stock1", "Stock2", "Correlation"]
filtered = flat_corr[np.abs(flat_corr["Correlation"]) >= 0.8]
stocks = pd.unique(filtered[['Stock1', 'Stock2']].values.ravel())
filtered_corr = stocks_corr_df.loc[stocks, stocks]
filtered_corr = filtered_corr.round(2)

# Plot the correlations
sns.heatmap(filtered_corr, annot = True, annot_kws = {"size": 7})
plt.title("Stocks with an absolute correlation higher than 0.8")
plt.show()




