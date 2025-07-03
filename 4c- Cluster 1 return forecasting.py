# Import libraries
import pandas as pd
import numpy as np

# Read csv for forecasting variables and cluster labels
forecasting_input_df = pd.read_csv(r"C:\Users\gonza\Documents\forecasting_input.csv")
cluster_labels_df = pd.read_csv(r"C:\Users\gonza\Documents\cluster_labels.csv",
                                index_col = [0])

# Filter to get only cluster 0 in the cluster csv and store the tickers
cluster_labels_df = cluster_labels_df[cluster_labels_df["Cluster"] == 1]
ticker_c0 = cluster_labels_df.index

# Filter the variables csv with the desired tickers
forecasting_input_df = forecasting_input_df[forecasting_input_df["Ticker"].isin(ticker_c0)]
forecasting_input_df = forecasting_input_df.iloc[: , 1 :]

# Import scaler for preprocessing
from sklearn.preprocessing import StandardScaler

# Initialize a scaler dictionary to fill it with each variable and ticker
scalers = {}

# Initialize a list to store the scaled variables
train_scaled = []
test_scaled = []

# loop over the tickers and its data
for ticker, group in forecasting_input_df.groupby("Ticker"):
    group = group.sort_values("Date").reset_index(drop=True)
# Split the data using 65% for training
    split_idx = int(len(group) * 0.65)
    train = group.iloc[:split_idx].copy()
    test = group.iloc[split_idx:].copy()
# Get the feature columns
    features = group.columns.difference(["Date", "Ticker"])
# Scale the feature columns for train and test    
    scaler = StandardScaler()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])

# Save scaler per stock
    scalers[ticker] = scaler
# Add the scaled features to the list
    train_scaled.append(train)
    test_scaled.append(test)

# Combine the items on the list to get the dataframes
ticker_train_df = pd.concat(train_scaled).reset_index(drop=True)
ticker_test_df = pd.concat(test_scaled).reset_index(drop=True)

# Drop the ticker column
train_df = ticker_train_df.drop(columns = ["Ticker"])
test_df = ticker_test_df.drop(columns = ["Ticker"])

# Aggregate by date
train_df = train_df.groupby("Date").mean()
test_df = test_df.groupby("Date").mean().iloc[:-1]

# Import the ts split
from sklearn.model_selection import TimeSeriesSplit

# Set the seed and the split
seed = 0
np.random.seed(seed)
ts = TimeSeriesSplit(n_splits = 5)

# Set the X and y for train and test
X_train = train_df.iloc[:, 0: 10]
X_test = test_df.iloc[:, 0: 10]
y_train = train_df["Rolling_Return"]
y_test = test_df["Rolling_Return"]

# Import rf model, accuracy metric and searchcv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid
rfr_parameters = {"n_estimators": range(10, 100, 10),
                  "max_depth": range(3, 10),
                  "max_features": range(3, 15, 3),
                  "min_samples_leaf": range(1, 10)}

# Run the grid search with the RMSE as the score
rfr_cv = RandomizedSearchCV(estimator = RandomForestRegressor(random_state = seed),
                            param_distributions = rfr_parameters,
                            cv = ts,
                            n_iter = 100,
                            random_state = seed,
                            scoring = "neg_mean_absolute_percentage_error",
                            n_jobs = -1)
rfr_cv.fit(X_train, y_train)

# Store the results
rfr_best_score = - rfr_cv.best_score_
rfr_best_estimators = rfr_cv.best_estimator_.n_estimators
rfr_best_features = rfr_cv.best_estimator_.max_features
rfr_best_depth = rfr_cv.best_estimator_.max_depth
rfr_best_samples = rfr_cv.best_estimator_.min_samples_leaf

# Run the model with the best parameters
rfr_best = RandomForestRegressor(n_estimators = rfr_best_estimators,
                                 max_depth = rfr_best_depth,
                                 max_features = rfr_best_features,
                                 min_samples_leaf = rfr_best_samples,
                                 random_state = seed)
rfr_best.fit(X_train, y_train)

# Compute the rmse for train and test
rfr_pred_train = rfr_best.predict(X_train)
rfr_pred_test = rfr_best.predict(X_test)
rfr_rmse_train = np.sqrt(mean_squared_error(y_train, rfr_pred_train))
rfr_rmse_test = np.sqrt(mean_squared_error(y_test, rfr_pred_test))



###############################################################################



# Import neural nets model
from sklearn.neural_network import MLPRegressor

# Define the grid
nnr_parameters = {"hidden_layer_sizes": [(6),                
                                         (3), 
                                         (4),
                                         (4, 3),
                                         (6, 3),
                                         (5, 2), 
                                         (7, 2),            
                                         (7, 5, 3),        
                                         (8, 5, 2),           
                                         (5, 4, 3),        
                                         (4, 3, 2)],  
                  "learning_rate": ["constant", "invscaling", "adaptative"],
                  "learning_rate_init": [0.01, 0.001, 0.0001],
                  "activation": ["identity", "logistic", "tanh", "relu"],
                  "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}

# Run the grid search
nnr_cv = RandomizedSearchCV(MLPRegressor(shuffle = False,
                                         random_state = seed,
                                         early_stopping = True),
                            param_distributions = nnr_parameters,
                            cv = ts,
                            n_iter = 100,
                            random_state = seed,
                            scoring = "neg_mean_absolute_percentage_error",
                            n_jobs = -1)
nnr_cv.fit(X_train, y_train)

# Store the best values
nnr_best_score = - nnr_cv.best_score_
nnr_best_sizes = nnr_cv.best_estimator_.hidden_layer_sizes
nnr_best_lr = nnr_cv.best_estimator_.learning_rate
nnr_best_lri = nnr_cv.best_estimator_.learning_rate_init
nnr_best_activation = nnr_cv.best_estimator_.activation
nnr_best_alpha = nnr_cv.best_estimator_.alpha

# Run the model with the best values
nnr_best = MLPRegressor(hidden_layer_sizes = nnr_best_sizes,
                        learning_rate = nnr_best_lr,
                        learning_rate_init = nnr_best_lri,
                        activation = nnr_best_activation,
                        alpha = nnr_best_alpha,
                        shuffle = False,
                        random_state = seed,
                        early_stopping = True)
nnr_best.fit(X_train, y_train)

# Compute the rmse for train and validation
nnr_pred_train = nnr_best.predict(X_train)
nnr_pred_test = nnr_best.predict(X_test)
nnr_rmse_train = np.sqrt(mean_squared_error(y_train, nnr_pred_train))
nnr_rmse_test = np.sqrt(mean_squared_error(y_test, nnr_pred_test))



###############################################################################



# Import svm
from sklearn.svm import SVR

# Define the grid
svr_parameters = {"kernel": ["linear", "rbf", "poly"],
                  "C": [0.1, 1, 10, 100],
                  "gamma": ["scale", "auto", 0.01, 0.1],
                  "epsilon": [0.01, 0.1, 1]}

# Run the grid search
svr_cv = RandomizedSearchCV(estimator = SVR(),
                            param_distributions = svr_parameters,
                            cv = ts,
                            n_iter = 100,
                            random_state = seed,
                            scoring = "neg_mean_absolute_percentage_error",
                            n_jobs = -1)
svr_cv.fit(X_train, y_train)

# Store the best parameters
svr_best_score = - svr_cv.best_score_
svr_best_kernel = svr_cv.best_estimator_.kernel
svr_best_C = svr_cv.best_estimator_.C
svr_best_gamma = svr_cv.best_estimator_.gamma
svr_best_epsilon = svr_cv.best_estimator_.epsilon

# Run the model with the best parameters
svr_best = SVR(kernel = svr_best_kernel,
               C = svr_best_C,
               gamma = svr_best_gamma,
               epsilon = svr_best_epsilon)
svr_best.fit(X_train, y_train)

# Compute rmse for train and validation
svr_pred_train = svr_best.predict(X_train)
svr_pred_test = svr_best.predict(X_test)
svr_rmse_train = np.sqrt(mean_squared_error(y_train, svr_pred_train))
svr_rmse_test = np.sqrt(mean_squared_error(y_test, svr_pred_test))


###############################################################################



# Import xgboost
from xgboost import XGBRegressor

# Define the grid
xgr_parameters = {"n_estimators": range(100, 1000, 50),
                  "max_depth": range(1, 10),
                  "learning_rate": np.arange(0.01, 0.1, 0.01),
                  "reg_alpha": [0, 0.1, 1, 10],
                  "reg_lambda": [1, 5, 10, 50],
                  "min_child_weight": [1, 5, 10]}

# Run the grid search
xgr_cv = RandomizedSearchCV(estimator = XGBRegressor(random_state = seed),
                            param_distributions = xgr_parameters,
                            cv = ts,
                            n_iter = 100,
                            random_state = seed,
                            scoring = "neg_mean_absolute_percentage_error",
                            n_jobs = -1)
xgr_cv.fit(X_train, y_train)

# Store the best parameters
xgr_best_score = - xgr_cv.best_score_
xgr_best_estimators = xgr_cv.best_estimator_.n_estimators
xgr_best_depth = xgr_cv.best_estimator_.max_depth
xgr_best_lr = xgr_cv.best_estimator_.learning_rate
xgr_best_alpha = xgr_cv.best_estimator_.reg_alpha
xgr_best_lambda = xgr_cv.best_estimator_.reg_lambda
xgr_best_weight = xgr_cv.best_estimator_.min_child_weight

# Run the model with the best parameters
xgr_best = XGBRegressor(n_estimators = xgr_best_estimators,
                        max_depth = xgr_best_depth,
                        learning_rate = xgr_best_lr,
                        reg_alpha = xgr_best_alpha,
                        reg_lambda = xgr_best_lambda,
                        min_child_weight = xgr_best_weight,
                        random_state = seed)
xgr_best.fit(X_train, y_train)

#  Compute rmse for train and test
xgr_pred_train = xgr_best.predict(X_train)
xgr_pred_test = xgr_best.predict(X_test)
xgr_rmse_train = np.sqrt(mean_squared_error(y_train, xgr_pred_train))
xgr_rmse_test = np.sqrt(mean_squared_error(y_test, xgr_pred_test))



###############################################################################



# Import naive for benchmark
from sklearn.dummy import DummyRegressor

# Fit the model
dr_model = DummyRegressor(strategy = "mean")
dr_model.fit(X_train, y_train)

# Comoute the rmse on train and test
dr_pred_train = dr_model.predict(X_train)
dr_pred_test = dr_model.predict(X_test)
dr_rmse_train = np.sqrt(mean_squared_error(y_train, dr_pred_train))
dr_rmse_test = np.sqrt(mean_squared_error(y_test, dr_pred_test))

# Store all the rmse values to compare and choose the model with the lowest
rmse_df = pd.DataFrame({
    "model": ["dr", "rfr", "xgr", "nnr", "svr"],
    "rmse_train": [dr_rmse_train, rfr_rmse_train, xgr_rmse_train, nnr_rmse_train, svr_rmse_train],
    "rmse_test": [dr_rmse_test, rfr_rmse_test, xgr_rmse_test, nnr_rmse_test, svr_rmse_test]
})

# Merge train and test datasets
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])

# Train the selected model in the complete dataset
xgr_selected = XGBRegressor(n_estimators = xgr_best_estimators,
                            max_depth = xgr_best_depth,
                            learning_rate = xgr_best_lr,
                            reg_alpha = xgr_best_alpha,
                            reg_lambda = xgr_best_lambda,
                            min_child_weight = xgr_best_weight,
                            random_state = seed)
xgr_selected.fit(X_full, y_full)

# To predict tickers return, create empty list for ticker and forecast
tickers = []
forecast = []

# Run a cycle to forecast each stock in the cluster
for ticker in ticker_c0:
    train_split = ticker_train_df[ticker_train_df["Ticker"] == ticker]
    test_split = ticker_test_df[ticker_test_df["Ticker"] == ticker]
    full_data = pd.concat([train_split, test_split])
    full_data.set_index("Date", inplace = True)
    X_ticker = full_data.iloc[-1, 1 : 11]
    y_pred_scaled = xgr_selected.predict(pd.DataFrame([X_ticker]))
    scaler = scalers[ticker]
    target_idx = list(scaler.feature_names_in_).index("Rolling_Return")
    mean = scaler.mean_[target_idx]
    std = scaler.scale_[target_idx]
    y_pred = y_pred_scaled * std + mean
    tickers.append(ticker)
    forecast.append(y_pred[0])
 
# Save the results in a dataframe
cluster_1_forecast_r = pd.DataFrame({"company": tickers,
                                   "forecast": forecast})
 
# Save it as a csv   
cluster_1_forecast_r.to_csv(r"C:\Users\gonza\Documents\cluster_1_forecast_r.csv")