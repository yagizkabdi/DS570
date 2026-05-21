import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.data import get_pageviews
from src.model import train_models, evaluate, find_spikes, FEATURES

st.set_page_config(page_title="Wikipedia Fame Predictor", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    return get_pageviews()

@st.cache_data(show_spinner="training models...")
def run_models(df):
    result = train_models(df)
    return result["data"], result["cutoff"]

st.title("Wikipedia Fame Predictor")
st.write("trying to predict wikipedia pageviews for scientists")

raw = load_data()
modeled, cutoff = run_models(raw)
