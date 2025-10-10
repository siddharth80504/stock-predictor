import pandas as pd
import numpy as np
import requests
from io import StringIO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import dateparser
try:
    _ = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download('vader_lexicon', quiet=True)
def fetch_stooq(symbol):
    """
    Fetch daily 'Close' & 'Volume' from Stooq for the given symbol.
    """
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df[['Close','Volume']]
def fetch_sentiment_vader(symbol):
    """
    Use yfinance to get the ticker's recent news headlines, compute VADER compound scores,
    then average per date to produce a daily 'Sentiment' series.
    """
    sia = SentimentIntensityAnalyzer()
    ticker = yf.Ticker(symbol)
    news_items = ticker.news  
    records = []
    for item in news_items:
        publish_ts = item.get("providerPublishTime", None)
        if publish_ts is None:
            continue
        date = datetime.utcfromtimestamp(publish_ts).date()
        text = item.get("title", "") + " " + item.get("summary", "")
        if not text.strip():
            continue
        score = sia.polarity_scores(text)["compound"]
        records.append((date, score))
    if not records:
        return pd.DataFrame(columns=["Date","Sentiment"]).set_index("Date")
    df = pd.DataFrame(records, columns=["Date","Sentiment"])
    daily = df.groupby("Date").mean().sort_index()
    all_dates = pd.date_range(daily.index.min(), daily.index.max())
    daily = daily.reindex(all_dates, method="ffill").fillna(0)
    daily.index.name = "Date"
    daily = (
        daily.rename_axis(None)
             .reset_index()
             .rename(columns={"index":"Date"})
             .set_index("Date")
    )
    return daily
def create_features(df, sentiment_df):
    """
    Given a DataFrame df with 'Close' and 'Volume', add technicals and merge in daily 'Sentiment'.
    """
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Lag1"] = df["Close"].shift(1)
    df["Volume_Change"] = df["Volume"].pct_change()
    ma20 = df["Close"].rolling(20).mean()
    std20 = df["Close"].rolling(20).std()
    df["BB_upper"] = ma20 + 2 * std20
    df["BB_lower"] = ma20 - 2 * std20
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()
    sent = sentiment_df.copy()
    sent.index = pd.to_datetime(sent.index).date
    df_dates = df.copy()
    df_dates["Date_only"] = df_dates.index.date
    df_dates = df_dates.reset_index().set_index("Date_only")
    merged = df_dates.join(sent, how="left")
    merged["Sentiment"] = merged["Sentiment"].ffill().fillna(0).infer_objects(copy=False)
    merged.index = merged["Date"]
    merged = merged.drop(columns=["Date"])
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged.dropna()
while True:
    symbol = input("\nEnter stock symbol (e.g., AAPL), or 'exit' to quit: ").strip().upper()
    if symbol.lower() == 'exit':
        print("Goodbye!")
        break
    predict_str = input("Enter prediction date (e.g., 2025-06-10 or 'next Friday'): ")
    predict_date_parsed = dateparser.parse(predict_str)
    if not predict_date_parsed:
        print("Invalid date format.")
        continue
    predict_date = predict_date_parsed.date()
    try:
        history = fetch_stooq(symbol)
    except Exception:
        print(f"Could not fetch data for {symbol}.")
        continue
    if history.empty:
        print(f"No historical data found for {symbol}.")
        continue
    print(f"Fetched {len(history)} rows of history for {symbol}.")
    sentiment = fetch_sentiment_vader(symbol)
    df_feat = create_features(history, sentiment)
    if df_feat.empty:
        print("Not enough data after feature + sentiment merging.")
        continue
    features = [
        'Close','Volume','Return','LogReturn','Lag1',
        'Volume_Change','BB_upper','BB_lower','MACD','Signal_Line','Sentiment'
    ]
    scaler = MinMaxScaler()
    close_scaler = MinMaxScaler()
    df_feat_clean = df_feat.dropna()
    scaler.fit(df_feat_clean[features])
    close_scaler.fit(df_feat_clean[['Close']])
    scaled_all = scaler.transform(df_feat_clean[features])
    n_steps = 30
    X, y = [], []
    for i in range(n_steps, len(scaled_all) - 1):
        X.append(scaled_all[i - n_steps:i])
        y.append(scaled_all[i, features.index('Close')])
    X = np.array(X)
    y = np.array(y)
    if len(X) < 1:
        print("Not enough data to train.")
        continue
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    model = Sequential([
        LSTM(32, input_shape=(n_steps, len(features))),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    print("\nThe model is being trained with technicals + sentiment. Please wait...\n")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        shuffle=False,
        verbose=0
    )
    y_val_pred = model.predict(X_val, verbose=0)
    mae_scaled = mean_absolute_error(y_val, y_val_pred)
    close_min, close_max = df_feat_clean['Close'].min(), df_feat_clean['Close'].max()
    mae_unscaled = mae_scaled * (close_max - close_min)
    last_hist_date = df_feat_clean.index[-1].date()
    all_future_bdays = pd.bdate_range(last_hist_date + timedelta(days=1),
                                      predict_date).date
    if not len(all_future_bdays):
        print(f"Requested date {predict_date} is not after last history date {last_hist_date}.")
        continue
    if len(all_future_bdays) >= 2:
        to_forecast = [all_future_bdays[-2], all_future_bdays[-1]]
    else:
        to_forecast = [all_future_bdays[-1]]
    predictions = []
    hist = history.copy()
    for target_date in to_forecast:
        sentiment = fetch_sentiment_vader(symbol)
        dff = create_features(hist, sentiment)
        if len(dff) < n_steps:
            break
        scaled = scaler.transform(dff[features])
        seq = scaled[-n_steps:].reshape(1, n_steps, len(features))
        pred_scaled = model.predict(seq, verbose=0)[0, 0]
        pred_price = close_scaler.inverse_transform([[pred_scaled]])[0, 0]
        lo = pred_price - mae_unscaled
        hi = pred_price + mae_unscaled
        hist.loc[pd.to_datetime(target_date)] = [pred_price, hist['Volume'].iloc[-1]]
        predictions.append((target_date, pred_price, lo, hi))
    print("\nForecast:")
    for d, p, lo, hi in predictions:
        print(f"{d}: ${p:.2f}  Price range: ${lo:.2f} – ${hi:.2f}")
    plt.figure(figsize=(10, 5))
    plt.plot(history.index[-100:], history['Close'].values[-100:], label='History')
    if predictions:
        fd, fp, lo_vals, hi_vals = zip(*predictions)
        plt.plot(pd.to_datetime(fd), fp, '--', color='red', label='Forecast')
        plt.fill_between(
            pd.to_datetime(fd),
            lo_vals,
            hi_vals,
            color='red',
            alpha=0.2,
            label='Price Range'
        )
    plt.title(f"{symbol} Forecast for {', '.join(str(x) for x in to_forecast)}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    save_choice = input("\nSave forecast to CSV? (yes/no): ").strip().lower()
    if save_choice == 'yes':
        out = pd.DataFrame(predictions, columns=['Date', 'Forecast', 'LowerBound', 'UpperBound'])
        filename = f"{symbol}_forecast_to_{predict_date}.csv"
        out.to_csv(filename, index=False)
        print(f"Saved to `{filename}`.")
    again = input("\nAnother symbol? (yes/no): ").strip().lower()
    if again != 'yes':
        print("Goodbye!")
        break
