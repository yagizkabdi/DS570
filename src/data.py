import json
from datetime import date, timedelta
from urllib.parse import quote
import pandas as pd
import requests

DATA_DIR = "data"
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"
HEADERS = {"User-Agent": "DS570-project"}


def load_scientists():
    with open(f"{DATA_DIR}/scientists.json", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data["scientists"])


def _fetch_one(title, start, end, session):
    url = API.format(title=quote(title, safe=""), start=start, end=end)
    r = session.get(url, headers=HEADERS, timeout=20)
    if r.status_code == 404:
        return pd.DataFrame(columns=["date", "views"])
    if r.status_code == 200:
        items = r.json().get("items", [])
        df = pd.DataFrame(items)
        if df.empty:
            return pd.DataFrame(columns=["date", "views"])
        df["date"] = pd.to_datetime(df["timestamp"].str[:8], format="%Y%m%d")
        return df[["date", "views"]]
    return pd.DataFrame(columns=["date", "views"])


def get_pageviews(scientists=None, days=365, progress=None):
    if scientists is None:
        scientists = load_scientists()
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=days - 1)
    start_s, end_s = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    session = requests.Session()
    frames = []
    for i, row in enumerate(scientists.itertuples(index=False), start=1):
        if progress:
            progress(i, len(scientists), row.title)
        df = _fetch_one(row.title, start_s, end_s, session)
        if df.empty:
            continue
        df["title"] = row.title
        df["name"] = row.display_name
        df["category"] = row.category
        frames.append(df)
    if not frames:
        raise RuntimeError("no data")
    data = pd.concat(frames, ignore_index=True)
    return data[["title", "name", "category", "date", "views"]].sort_values(["title", "date"]).reset_index(drop=True)
