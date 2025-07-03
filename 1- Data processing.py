# Import libraries
import pandas as pd
import numpy as np
import yfinance as yf

# Create list of tickers
tickers = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", 
    "BKNG", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", 
    "CPRT", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", 
    "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", 
    "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", 
    "LIN", "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST", 
    "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", 
    "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM", "REGN", 
    "ROP", "ROST", "SBUX", "SHOP", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", 
    "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS"
]

# Create list of variables
esg_score = []
companies = []
roa = []
roe = []
ebitda_margin = []
operating_margin = []
quick_ratio = []
current_ratio = []
debt_equity_ratio = []


# Fill the lists with ticker information
for i in tickers:
    ticker = yf.Ticker(i)
    try:
       if ticker.sustainability is not None:
           financial_data = ticker.get_info()
           esg_data = ticker.sustainability
           esg_score.append(esg_data.loc["totalEsg"][0])
           companies.append(i)
           roa.append(financial_data.get("returnOnAssets", None))
           roe.append(financial_data.get("returnOnEquity", None))
           ebitda_margin.append(financial_data.get("ebitdaMargins", None))
           operating_margin.append(financial_data.get("operatingMargins", None))
           quick_ratio.append(financial_data.get("quickRatio", None))
           current_ratio.append(financial_data.get("currentRatio", None))
           debt_equity_ratio.append(financial_data.get("debtToEquity", None))
    except:
        continue
 
# Create a dataframe with the ticker information
fund_df = pd.DataFrame({"company": companies,
                       "ESG score": esg_score,
                       "ROA": roa,
                       "ROE": roe,
                       "EBITDA margin": ebitda_margin,
                       "Operating margin": operating_margin,
                       "Quick Ratio": quick_ratio,
                       "Current Ratio": current_ratio,
                       "Debt-to-Equity": debt_equity_ratio})

# Drop na and set the company ticker as index
fund_df.dropna(inplace = True)
fund_df.set_index("company", inplace=True)

# Get historical prices and volumes to add them as variables
prices_df = yf.download(list(fund_df.index), start = "2015-01-01", end = "2025-05-01")

# Use only the tickers with 10 years of data
prices_df.dropna(axis = 1, inplace = True)
# Convert it to logarithmic returns
log_prices_df = np.log(prices_df["Close"])
return_df = log_prices_df - log_prices_df.shift(1)
return_df.dropna(inplace = True)

# Get the volume
volume_df = prices_df["Volume"]

# Obtain the mean returns, volatility and volume, and set the series name
historic_return = return_df.mean()
historic_volatility = return_df.std()
historic_volume = volume_df.mean()
historic_return.name = "Historic Return"
historic_volatility.name = "Historic Volatility"
historic_volume.name = "Historic Volume"

# Concatenate all together
cluster_input_df = pd.concat([fund_df,
                             historic_return,
                             historic_volatility,
                             historic_volume],
                             axis = 1)

# Use only the tickers with 10 years of data
cluster_input_df.dropna(inplace = True)

cluster_input_df.to_csv(r"C:\Users\gonza\Documents\cluster_input.csv")



###############################################################################



# Import library for technical analysis
import ta

# Initialize the 
results = []

# Get the forecasting variables per ticker
for ticker in cluster_input_df.index:
    try:
        df = prices_df.xs(key=ticker, level=1, axis=1).copy()
        df = df.reset_index()
        df.dropna(inplace=True)

        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Rolling_Return"] = df["Log_Return"].rolling(21).mean().shift(-20)
        df["Rolling_Volatility"] = df["Log_Return"].rolling(21).std().shift(-20)
        df["Rolling_Volume"] = df["Volume"].rolling(21).mean().shift(-20)
        df["Lagged_Return"] = df["Rolling_Return"].shift(21)
        df["Lagged_Volatility"] = df["Rolling_Volatility"].shift(21)
        df["Lagged_Volume"] = df["Rolling_Volume"].shift(21)
        
        df["SMA"] = ta.trend.SMAIndicator(df["Close"], window = 21).sma_indicator()
        df["EMA"] = ta.trend.EMAIndicator(df["Close"], window = 21).ema_indicator()
        bb = ta.volatility.BollingerBands(df["Close"], window = 21, window_dev = 21)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_lower"] = bb.bollinger_lband()
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window = 21).rsi()
        stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"], window = 21, smooth_window=3)
        df["Stoch_K"] = stoch.stoch()
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()

        indicators = ["SMA", "EMA", "BB_upper", "BB_lower", "RSI", "Stoch_K", "OBV"]
        for col in indicators:
            df[f"{col}_lag"] = df[col].shift(1)

        final_columns = [f"{col}_lag" for col in indicators] + ["Lagged_Return", "Lagged_Volatility", "Lagged_Volume", "Rolling_Return", "Rolling_Volatility", "Rolling_Volume"]
        df["Ticker"] = ticker
        df = df[["Date", "Ticker"] + final_columns].dropna()
        results.append(df)

    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Concatenate the data for each ticker
final_df = pd.concat(results).reset_index(drop=True)

# Save it as a CSV
final_df.to_csv(r"C:\Users\gonza\Documents\forecasting_input.csv")



###############################################################################



# Get SP500 info for portfolio benchmark
sp500 = yf.download('^GSPC', start='2015-01-01', end='2025-05-01')["Close"]
sp500 = np.log(sp500)
sp500 = sp500 - sp500.shift(1)
sp500.dropna(inplace = True)
sp500.to_csv(r"C:\Users\gonza\Documents\sp500.csv")


rolling_returns_df = return_df.rolling(21).mean().shift(-20).dropna().iloc[:-1, ]
corr_df = rolling_returns_df.corr()

corr_df.to_csv(r"C:\Users\gonza\Documents\corr_matrix.csv")
