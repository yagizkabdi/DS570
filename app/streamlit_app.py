import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data import get_pageviews
from src.model import train_models, evaluate, find_spikes, FEATURES

st.set_page_config(page_title="Wikipedia Fame Predictor", layout="wide")


@st.cache_data(show_spinner=False)
def load_data():
    bar = st.progress(0.0, text="Downloading Wikipedia pageviews...")
    def update(i, total, title):
        bar.progress(i / total, text=f"Downloading {title} ({i}/{total})")
    df = get_pageviews(progress=update)
    bar.empty()
    return df


@st.cache_data(show_spinner="training models...")
def run_models(df):
    result = train_models(df)
    import pandas as pd
    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance": result["rf"].feature_importances_,
    }).sort_values("importance", ascending=False)
    return result["data"], result["cutoff"], importance


@st.cache_data(show_spinner="looking for spikes...")
def run_spikes(df):
    return find_spikes(df)


st.title("Wikipedia Fame Predictor")
st.write("Daily Wikipedia pageviews for famous scientists. We try to predict "
         "tomorrows views and find the days a scientist suddenly trended.")

raw = load_data()
modeled, cutoff, importance = run_models(raw)
spikes = run_spikes(raw)

names = raw[["title", "name", "category"]].drop_duplicates().sort_values("name")
name_of = dict(zip(names["title"], names["name"]))

c1, c2, c3 = st.columns(3)
c1.metric("Scientists", names.shape[0])
c2.metric("Days of data", (raw["date"].max() - raw["date"].min()).days + 1)
c3.metric("Latest day", raw["date"].max().strftime("%Y-%m-%d"))

tab1, tab2, tab3 = st.tabs(["Explore", "Spikes", "Prediction"])

with tab1:
    st.subheader("Pageviews over time")
    title = st.selectbox("Pick a scientist", names["title"],
                         format_func=lambda t: name_of[t])
    log_scale = st.checkbox("Log scale", value=True)
    one = raw[raw["title"] == title]
    fig = px.line(one, x="date", y="views", title=name_of[title],
                  labels={"date": "Date", "views": "Daily pageviews"},
                  log_y=log_scale)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average daily views per category")
    by_cat = raw.groupby("category")["views"].mean().sort_values(ascending=False)
    fig2 = px.bar(by_cat, labels={"value": "Mean daily views", "category": "Category"})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
