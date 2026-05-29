import numpy as np
import pandas as pd
from src.model import add_features, train_models, find_spikes, mape


def fake_data(n_days=120):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for i, name in enumerate(["A", "B", "C"]):
        views = 100 + 20 * i + np.random.randint(0, 30, n_days)
        if name == "A" and n_days > 80:
            views[80] = 5000
        for d, v in zip(dates, views):
            rows.append({"title": name, "name": name, "category": "Physics",
                         "date": d, "views": int(v)})
    return pd.DataFrame(rows)


def test_mape_zero_when_perfect():
    assert mape([10, 20, 30], [10, 20, 30]) == 0.0


def test_features_use_only_past():
    df = fake_data()
    f1 = add_features(df)
    df2 = df.copy()
    df2.loc[df2.index[-10:], "views"] = 0
    f2 = add_features(df2)
    early = f1["date"] < df["date"].iloc[-10]
    cols = ["lag_1", "lag_7", "roll7"]
    pd.testing.assert_frame_equal(
        f1[early][cols].reset_index(drop=True),
        f2[early][cols].reset_index(drop=True),
    )


def test_train_test_split_by_date():
    result = train_models(fake_data(), test_days=20)
    df = result["data"]
    assert (df[df["is_test"]]["date"] > result["cutoff"]).all()


def test_spike_is_found():
    spikes = find_spikes(fake_data())
    a = spikes[spikes["title"] == "A"]
    assert a["is_spike"].sum() >= 1
