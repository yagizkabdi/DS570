import json
import time
from datetime import date, timedelta
from urllib.parse import quote
import pandas as pd
import requests

DATA_DIR = "data"
API = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
       "en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}")
HEADERS = {"User-Agent": "DS570-project (wiki pageview forecasting)"}


def load_scientists():
    with open(f"{DATA_DIR}/scientists.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["scientists"])


def _date_range(days):
    # API is ~1 day behind, so end 2 days ago to be safe
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _fetch_one(title, start, end, session):
    url = API.format(title=quote(title, safe=""), start=start, end=end)
    # try a couple of times in case of a network hiccup
    for attempt in range(3):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code == 404:
            return pd.DataFrame(columns=["date", "views"])
        if r.status_code == 200:
            items = r.json().get("items", [])
            df = pd.DataFrame(items)
            if df.empty:
                return pd.DataFrame(columns=["date", "views"])
            df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
            return df[["date", "views"]]
        time.sleep(2)
    return pd.DataFrame(columns=["date", "views"])


def get_pageviews(scientists=None, days=365, progress=None):
    if scientists is None:
        scientists = load_scientists()
    start, end = _date_range(days)
    session = requests.Session()
    frames = []
    total = len(scientists)
    for i, row in enumerate(scientists.itertuples(index=False), start=1):
        if progress:
            progress(i, total, row.title)
        df = _fetch_one(row.title, start, end, session)
        if df.empty:
            continue
        df["title"] = row.title
        df["name"] = row.display_name
        df["category"] = row.category
        frames.append(df)
    if not frames:
        raise RuntimeError("Could not download any data from Wikipedia.")
    data = pd.concat(frames, ignore_index=True)
    data = data[["title", "name", "category", "date", "views"]]
    return data.sort_values(["title", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = get_pageviews(days=120)
    print(df.head())
    print(f"{len(df)} rows, {df['title'].nunique()} scientists")
