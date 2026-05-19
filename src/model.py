# we work with log(views) since the counts are very skewed

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

FEATURES = ["lag_1", "lag_7", "roll7", "dow", "is_weekend"]


def add_features(data):
    df = data.copy()
    df["log_views"] = np.log1p(df["views"])
    g = df.groupby("title")["log_views"]
    df["lag_1"] = g.shift(1)   # views yesterday
    df["lag_7"] = g.shift(7)   # views same day last week
    # shift before rolling so today is not included
    df["roll7"] = g.shift(1).rolling(7, min_periods=1).mean().reset_index(0, drop=True)
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def naive(df):
    return df.groupby("title")["log_views"].shift(1)


def seasonal_naive(df):
    return df.groupby("title")["log_views"].shift(7)


def train_models(data, test_days=30):
    # train on all days except the last test_days
    df = add_features(data)
    df["pred_naive"] = naive(df)
    df["pred_seasonal"] = seasonal_naive(df)
    df = df.dropna(subset=FEATURES + ["log_views"]).copy()
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff]
    X_train, y_train = train[FEATURES], train["log_views"]
    X_all = df[FEATURES]
    linreg = LinearRegression().fit(X_train, y_train)
    df["pred_linreg"] = linreg.predict(X_all)
    rf = RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1)
    rf.fit(X_train, y_train)
    df["pred_rf"] = rf.predict(X_all)
    df["is_test"] = df["date"] > cutoff
    return {"data": df, "cutoff": cutoff, "linreg": linreg, "rf": rf}


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    keep = y_true > 0
    if keep.sum() == 0:
        return 0.0
    return float(100 * np.mean(np.abs(y_true[keep] - y_pred[keep]) / y_true[keep]))


def evaluate(test_df):
    models = {
        "Naive (yesterday)": "pred_naive",
        "Seasonal naive (last week)": "pred_seasonal",
        "Linear Regression": "pred_linreg",
        "Random Forest": "pred_rf",
    }
    true_log = test_df["log_views"].to_numpy()
    true_views = np.expm1(true_log)
    rows = []
    for label, col in models.items():
        pred_log = test_df[col].to_numpy()
        pred_views = np.expm1(pred_log)
        rows.append({
            "Model": label,
            "MAE (log)": round(mae(true_log, pred_log), 3),
            "MAPE (%)": round(mape(true_views, pred_views), 1),
        })
    return pd.DataFrame(rows)
