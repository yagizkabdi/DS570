import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["lag_1", "lag_7", "roll7", "dow", "is_weekend"]


def add_features(data):
    df = data.copy()
    df["log_views"] = np.log1p(df["views"])
    g = df.groupby("title")["log_views"]
    df["lag_1"] = g.shift(1)
    df["lag_7"] = g.shift(7)
    df["roll7"] = g.shift(1).rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def naive(df):
    return df.groupby("title")["log_views"].shift(1)


def seasonal_naive(df):
    return df.groupby("title")["log_views"].shift(7)


def train_models(data, test_days=30):
    df = add_features(data)
    df["pred_naive"] = naive(df)
    df["pred_seasonal"] = seasonal_naive(df)
    df = df.dropna(subset=FEATURES + ["log_views"]).copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff]
    X_train, y_train = train[FEATURES], train["log_views"]
    linreg = LinearRegression().fit(X_train, y_train)
    df["pred_linreg"] = linreg.predict(df[FEATURES])
    df["is_test"] = df["date"] > cutoff
    return {"data": df, "cutoff": cutoff, "linreg": linreg}
