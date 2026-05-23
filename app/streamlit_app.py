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

with tab2:
    st.subheader("Who trended recently?")
    st.write("A spike is a day with far more views than the previous weeks (z-score >= 3).")
    days = st.slider("Look back this many days", 30, 365, 120, step=30)
    recent = spikes[spikes["date"] >= spikes["date"].max() - pd.Timedelta(days=days)]
    top = recent[recent["is_spike"]].sort_values("zscore", ascending=False).head(20)
    if top.empty:
        st.info("No spikes found in this period.")
    else:
        table = top[["date", "name", "category", "views", "zscore"]].copy()
        table["views"] = table["views"].astype(int)
        table["zscore"] = table["zscore"].round(1)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("See spikes for one scientist")
    title2 = st.selectbox("Scientist", names["title"],
                          format_func=lambda t: name_of[t], key="spike_pick")
    one = spikes[spikes["title"] == title2]
    fig = px.line(one, x="date", y="views", log_y=True, title=name_of[title2],
                  labels={"date": "Date", "views": "Daily pageviews"})
    flagged = one[one["is_spike"]]
    fig.add_scatter(x=flagged["date"], y=flagged["views"], mode="markers",
                    name="Spike", marker=dict(color="red", size=8))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("How good are the predictions?")
    test = modeled[modeled["is_test"]]
    scores = evaluate(test)
    st.write(f"Tested on the last 30 days ({(cutoff + pd.Timedelta(days=1)).date()} "
             f"to {modeled['date'].max().date()}). Lower is better.")
    st.dataframe(scores, use_container_width=True, hide_index=True)
    best = scores.sort_values("MAPE (%)").iloc[0]
    naive_mape = scores[scores["Model"] == "Naive (yesterday)"]["MAPE (%)"].iloc[0]
    st.write(f"Best: **{best['Model']}** with {best['MAPE (%)']}% error (naive was {naive_mape}%).")

    st.subheader("Prediction vs reality")
    title3 = st.selectbox("Scientist", names["title"],
                          format_func=lambda t: name_of[t], key="pred_pick")
    one = modeled[modeled["title"] == title3].copy()
    one["Actual"] = np.expm1(one["log_views"])
    one["Random Forest"] = np.expm1(one["pred_rf"])
    plot_df = one[["date", "Actual", "Random Forest"]].melt(
        id_vars="date", var_name="series", value_name="pageviews")
    fig = px.line(plot_df, x="date", y="pageviews", color="series", log_y=True,
                  title=name_of[title3],
                  labels={"date": "Date", "pageviews": "Daily pageviews", "series": ""})
    fig.add_vline(x=cutoff.timestamp() * 1000, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The dashed line is where the 30-day test period starts.")

    st.subheader("What does the Random Forest use most?")
    fig = px.bar(importance, x="importance", y="feature", orientation="h",
                 labels={"importance": "Importance", "feature": "Feature"})
    st.plotly_chart(fig, use_container_width=True)

    st.write("Note: this model only looks at past views, so it cannot predict "
             "a sudden spike caused by a news event. Those show up in the Spikes tab.")
