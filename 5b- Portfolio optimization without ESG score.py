# Import libraries
import pandas as pd
import numpy as np
from amplpy import AMPL, ampl_notebook

# Read the inputs csv, sp500 for benchmark, predictions, correlations and esg
sp500 = pd.read_csv(r"C:\Users\gonza\Documents\sp500.csv",
                    parse_dates = ["Date"],
                    index_col = ["Date"])
predictions_return_c0_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_0_forecast_r.csv",
                                       index_col = [1])
predictions_return_c1_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_1_forecast_r.csv",
                                       index_col = [1])
predictions_volatility_c0_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_0_forecast_v.csv",
                                           index_col = [1])
predictions_volatility_c1_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_1_forecast_v.csv",
                                           index_col = [1])
predictions_return_df = pd.concat([predictions_return_c0_df,
                                   predictions_return_c1_df],
                                  ignore_index = False)
predictions_volatility_df = pd.concat([predictions_volatility_c0_df,
                                       predictions_volatility_c1_df],
                                      ignore_index = False)
corr_df = pd.read_csv(r"C:\Users\gonza\Documents\corr_matrix.csv",
                      index_col = [0])
esg_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_input.csv",
                     index_col = [0])
esg_df = esg_df["ESG score"]



# Store the companies
companies = predictions_return_df.index

# Get the covariance matrix as the correlation multiplied by the forecasted v
diag = np.diag(predictions_volatility_df["forecast"])
cov_matrix = pd.DataFrame(diag @ corr_df.values @ diag,
                          index = corr_df.index,
                          columns = corr_df.columns)

# Store returns, volatilities and ESG scores in dictionaries
mean_dict = dict(zip(companies, predictions_return_df["forecast"]))
cov_dict = {(i, j): cov_matrix.loc[i, j] for i in companies for j in companies}
esg_dict = dict(zip(companies, esg_df))

# Define the threshold values for the optimizations, based on the sp500 values
r_threshold = sp500.mean().values[0] * 1.5
v_threshold = (sp500.std().values[0] * 1.5) ** 2

# Define the risk-free asset return
r_f_year = 0.038
r_f = ((1 + r_f_year) ** (1/252)) - 1



###############################################################################


# Define the minimum variance optimization
min_v_model = ampl_notebook(
    modules=["coin"],
    license_uuid="f145edbd-5947-47fa-8d37-e5324bdd62a9")

# Define the set
min_v_model.eval("set companies;") # set of tickers
min_v_model.set["companies"] = companies

#Define the parameters
min_v_model.eval("param r{companies};") # stocks return
min_v_model.eval("param v{companies, companies};") # covariance matrix
min_v_model.eval("param esg{companies};") # ESG score by company
min_v_model.eval("param r_threshold;") # return threshold
min_v_model.eval("param r_f;") # risk free asset return

# Define the variables
min_v_model.eval("var x{companies} >= 0;") # weights
min_v_model.eval("var portfolio_v;") # portfolio volatility
min_v_model.eval("var portfolio_r;") # portfolio return
min_v_model.eval("var portfolio_esg;") # portfolio average ESG score

# Define the objective function
min_v_model.eval("""minimize portfolio_volatility: 
                     portfolio_v;""")

# Define portfolio volatility as the multiplication of weights and cov matrix
min_v_model.eval("""subject to set_portfolio_v:
                     sum{i in companies, j in companies}
                         x[i] * v[i, j] * x[j] = portfolio_v;""")

# Define portfolio return as the multiplication of weights and returns
min_v_model.eval("""subject to set_portfolio_r:
                     sum{i in companies}
                         x[i] * r[i] = portfolio_r;""")

# Define portfolio ESG score as the multiplication of weights and scores                         
min_v_model.eval("""subject to set_portfolio_esg:
                     sum{i in companies}
                         x[i] * esg[i] = portfolio_esg;""")

# The sum of all weights must be equal to 1
min_v_model.eval("""subject to weights_constraint:
                     sum{i in companies}
                         x[i] = 1;""")

# Portfolio return must be greater or equal to the threshold                         
min_v_model.eval("""subject to minimum_return:
                     sum{i in companies}
                         x[i] * r[i] >= r_threshold;""")
                         
# Assign dictionary variables to parameters
min_v_model.param["r"] = mean_dict
min_v_model.param["v"] = cov_dict
min_v_model.param["esg"] = esg_dict
min_v_model.param["r_threshold"] = r_threshold
min_v_model.param["r_f"] = r_f

# Select a solver for non-linear programming and solve                    
min_v_model.set_option("solver", "bonmin")
min_v_model.solve()

# Store portfolio volatility, return, ESG score and Sharpe ratio
min_v_volatility = np.sqrt(min_v_model.get_variable("portfolio_v")[()].value())
min_v_return= min_v_model.get_variable("portfolio_r")[()].value()
min_v_esg = min_v_model.get_variable("portfolio_esg")[()].value()
min_v_sharpe = (min_v_return - r_f) / min_v_volatility
min_v_result = pd.DataFrame({"model":"min_v_portfolio",
                             "return": [min_v_return],
                             "volatility": [min_v_volatility],
                             "sharpe": [min_v_sharpe],
                             "esg": [min_v_esg]})
min_v_result = min_v_result.set_index("model")

# Store the weights
x = min_v_model.get_variable("x")
min_v_weights = {c: x[c].value() for c in min_v_model.get_set("companies")}
min_v_weights_df = pd.DataFrame([min_v_weights])

# Round and normalize the weights to avoid too many decimals
min_v_weights_df = min_v_weights_df.round(3)
min_v_weights_df = min_v_weights_df.div(min_v_weights_df.sum(axis = 1), axis = 0)

# Store the approach name
min_v_weights_df["model"] = "min_v_portfolio"
min_v_weights_df = min_v_weights_df.set_index("model")

# Get all together
min_v_portfolio = pd.concat([min_v_result, min_v_weights_df],
                            axis = 1)   



###############################################################################


# Define the maximum return optimization
max_r_model = ampl_notebook(
    modules=["coin"],
    license_uuid="f145edbd-5947-47fa-8d37-e5324bdd62a9")

# Define and assign sets
max_r_model.eval("set companies;") # tickers
max_r_model.set["companies"] = companies

# Define parameters
max_r_model.eval("param r{companies};") # stock returns
max_r_model.eval("param v{companies, companies};") # covariance matrix
max_r_model.eval("param esg{companies};") # ESG score
max_r_model.eval("param v_threshold;") # volatility threshold
max_r_model.eval("param r_f;") # Risk free asset return

# Define variables
max_r_model.eval("var x{companies} >= 0;") # weights
max_r_model.eval("var portfolio_v;") # portfolio volatility
max_r_model.eval("var portfolio_r;") # portfolio return
max_r_model.eval("var portfolio_esg;") # portfolio ESG score

# Define objective function
max_r_model.eval("""maximize portfolio_return: 
                     portfolio_r;""")

# Set portfolio volatility
max_r_model.eval("""subject to set_portfolio_v:
                     sum{i in companies, j in companies}
                         x[i] * v[i, j] * x[j] = portfolio_v;""")

# Set portfolio return
max_r_model.eval("""subject to set_portfolio_r:
                     sum{i in companies}
                         x[i] * r[i] = portfolio_r;""")
 
# Set portfolio ESG score
max_r_model.eval("""subject to set_portfolio_esg:
                     sum{i in companies}
                         x[i] * esg[i] = portfolio_esg;""")
 
# The sum of all weights must be equal to 1
max_r_model.eval("""subject to weights_constraint:
                     sum{i in companies}
                         x[i] = 1;""")

# Portfolio volatility must be less or equal to the threshold value                        
max_r_model.eval("""subject to maximum_volatility:
                     sum{i in companies, j in companies}
                         x[i] * v[i,j] * x[j] <= v_threshold;""")
    
# Assign the dictionaries to the parameters
max_r_model.param["r"] = mean_dict
max_r_model.param["v"] = cov_dict
max_r_model.param["esg"] = esg_dict
max_r_model.param["v_threshold"] = v_threshold
max_r_model.param["r_f"] = r_f

# Solve the model with a non-linear solver                    
max_r_model.set_option("solver", "bonmin")
max_r_model.solve()

# Store portfolio return, volatility, Sharpe ratio and ESG score
max_r_volatility = np.sqrt(max_r_model.get_variable("portfolio_v")[()].value())
max_r_return= max_r_model.get_variable("portfolio_r")[()].value()
max_r_esg = max_r_model.get_variable("portfolio_esg")[()].value()
max_r_sharpe = (max_r_return - r_f) / max_r_volatility
max_r_result = pd.DataFrame({"model": "max_r_portfolio",
                             "return": [max_r_return],
                             "volatility": [max_r_volatility],
                             "sharpe": [max_r_sharpe],
                             "esg": [max_r_esg]})
max_r_result = max_r_result.set_index("model")

# Store portfolio weights
x = max_r_model.get_variable("x")
max_r_weights = {c: x[c].value() for c in max_r_model.get_set("companies")}
max_r_weights_df = pd.DataFrame([max_r_weights])
max_r_weights_df = max_r_weights_df.round(3)
max_r_weights_df = max_r_weights_df.div(max_r_weights_df.sum(axis = 1), axis = 0)
max_r_weights_df["model"] = "max_r_portfolio"
max_r_weights_df = max_r_weights_df.set_index("model")

# Get indexes and weights together
max_r_portfolio = pd.concat([max_r_result, max_r_weights_df],
                            axis = 1)



###############################################################################



# Define the maximum Sharpe ratio portfolio
max_sharpe_model = ampl_notebook(
    modules=["coin"],
    license_uuid="f145edbd-5947-47fa-8d37-e5324bdd62a9")

# Define sets
max_sharpe_model.eval("set companies;") # Tickers
max_sharpe_model.set["companies"] = companies

# Define parameters
max_sharpe_model.eval("param r{companies};") # Stocks return
max_sharpe_model.eval("param v{companies, companies};") # Covariance matrix
max_sharpe_model.eval("param esg{companies};") # ESG scores
max_sharpe_model.eval("param r_threshold;") # Threshold return
max_sharpe_model.eval("param v_threshold;") # Threshold volatility
max_sharpe_model.eval("param r_f;") # Risk free asset return

# Define variables
max_sharpe_model.eval("var x{companies} >= 0;") # Weights
max_sharpe_model.eval("var portfolio_v >= 1e-14;") # Portfolio volatility
max_sharpe_model.eval("var portfolio_r;") # Portfolio return
max_sharpe_model.eval("var portfolio_esg;") # Portfolio ESG score
max_sharpe_model.eval("var portfolio_sharpe;") # Portfolio Sharpe ratio

# Define objective function
max_sharpe_model.eval("""maximize portfolio_sharperatio: 
                     portfolio_sharpe;""")

# Define portfolio volatility
max_sharpe_model.eval("""subject to set_portfolio_v:
                     sum{i in companies, j in companies}
                         x[i] * v[i, j] * x[j] = portfolio_v;""")

# Define portfolio return
max_sharpe_model.eval("""subject to set_portfolio_r:
                     sum{i in companies}
                         x[i] * r[i] = portfolio_r;""")

# Define portfolio ESG score                      
max_sharpe_model.eval("""subject to set_portfolio_esg:
                     sum{i in companies}
                         x[i] * esg[i] = portfolio_esg;""")

# Define portfolio Sharpe ratio
max_sharpe_model.eval("""subject to set_portfolio_sharpe:
                          (portfolio_r - r_f) = portfolio_sharpe * sqrt(portfolio_v);""")

# The sum of weights must be equal to 1                        
max_sharpe_model.eval("""subject to weights_constraint:
                     sum{i in companies}
                         x[i] = 1;""")

# Portfolio volatility should be less or equal to the threshold                         
max_sharpe_model.eval("""subject to maximum_volatility:
                     sum{i in companies, j in companies}
                         x[i] * v[i,j] * x[j] <= v_threshold;""")

# Portfolio return should be greater or equal toi the threshold                         
max_sharpe_model.eval("""subject to minimum_return:
                     sum{i in companies}
                         x[i] * r[i] >= r_threshold;""")
    
# Assign dictionaries to parameters
max_sharpe_model.param["r"] = mean_dict
max_sharpe_model.param["v"] = cov_dict
max_sharpe_model.param["esg"] = esg_dict
max_sharpe_model.param["r_threshold"] = r_threshold
max_sharpe_model.param["v_threshold"] = v_threshold
max_sharpe_model.param["r_f"] = r_f
  
# Solve the problem with a non-linear solver                  
max_sharpe_model.set_option("solver", "bonmin")
max_sharpe_model.solve()

# Store portfolio indexes
max_sharpe_volatility = np.sqrt(max_sharpe_model.get_variable("portfolio_v")[()].value())
max_sharpe_return= max_sharpe_model.get_variable("portfolio_r")[()].value()
max_sharpe_esg = max_sharpe_model.get_variable("portfolio_esg")[()].value()
max_sharpe_sharpe = max_sharpe_model.get_variable("portfolio_sharpe")[()].value()
max_sharpe_result = pd.DataFrame({"model": "max_sharpe_portfolio",
                                  "return": [max_sharpe_return],
                                  "volatility": [max_sharpe_volatility],
                                  "sharpe": [max_sharpe_sharpe],
                                  "esg": [max_sharpe_esg]})
max_sharpe_result = max_sharpe_result.set_index("model")

# Store portfolio weights
x = max_sharpe_model.get_variable("x")
max_sharpe_weights = {c: x[c].value() for c in max_sharpe_model.get_set("companies")}
max_sharpe_weights_df = pd.DataFrame([max_sharpe_weights])
max_sharpe_weights_df = max_sharpe_weights_df.round(3)
max_sharpe_weights_df = max_sharpe_weights_df.div(max_sharpe_weights_df.sum(axis = 1), axis = 0)
max_sharpe_weights_df["model"] = "max_sharpe_portfolio"
max_sharpe_weights_df = max_sharpe_weights_df.set_index("model")

# Store everything together
max_sharpe_portfolio = pd.concat([max_sharpe_result, max_sharpe_weights_df],
                                 axis = 1)



###############################################################################



# Save all the stats in one dataframe
opt_portfolios = pd.concat([max_r_portfolio,
                            min_v_portfolio,
                            max_sharpe_portfolio],
                           axis = 0)

opt_portfolios.to_csv(r"C:\Users\gonza\Documents\opt_portfolios.csv")

# Save all the weights in a dataframe
best_weights = pd.concat([max_r_weights_df,
                          min_v_weights_df,
                          max_sharpe_weights_df],
                         axis = 0)

# Start a simulation with 10000 cases
n_simulations = 10000
seed = 0
np.random.seed(seed)

# Returns distribute multivariate normal
rand_returns = np.random.multivariate_normal(predictions_return_df["forecast"],
                                             cov_matrix, 
                                             size = n_simulations)

# Store portfolio returns
portfolios_sim = (best_weights @ rand_returns.T).T

# Import ploting libraries
import matplotlib.pyplot as plt
import seaborn as sns

sns.kdeplot(data=portfolios_sim, x="max_r_portfolio", label="Maximum Return")
sns.kdeplot(data=portfolios_sim, x="min_v_portfolio", label="Minimum Volatility")
sns.kdeplot(data=portfolios_sim, x="max_sharpe_portfolio", label="Maximum Sharpe ratio")
plt.legend()
plt.title("Simulated Portfolio Return Distributions")
plt.show()

# Store descriptive statistics
description = portfolios_sim.describe()
IQR = (portfolios_sim.quantile(0.75) - portfolios_sim.quantile(0.25))
VaR = portfolios_sim.quantile(0.05)
skewness = portfolios_sim.skew()
kurtosis = portfolios_sim.kurt()
prob_loss = (portfolios_sim < 0).mean()
description.loc["IQR"] = IQR
description.loc["VaR"] = VaR
description.loc["ESG"] = opt_portfolios["esg"]
description.loc["prob_loss"] = prob_loss
description.loc["skewness"] = skewness
description.loc["kurtosis"] = kurtosis
description.loc["sharpe"] = (description.loc["mean"] - r_f) / description.loc["std"]