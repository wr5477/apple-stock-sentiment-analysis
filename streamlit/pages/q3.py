import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q3. Do sentiment spikes precede larger price volatility?")

df = build_quarterly_merged().copy()
df = df.sort_values("quarter_end_date")

st.subheader("Filters")
quarters = df["quarter_label"].unique().tolist()
start_q, end_q = st.select_slider(
    "Quarter range (for the *volatility* quarter)",
    options=quarters,
    value=(quarters[0], quarters[-1]),
)

lag = st.slider("Lag (quarters): use sentiment from how many quarters earlier?", 1, 4, 1, 1)

# Create lagged sentiment
df[f"sentiment_lag_{lag}"] = df["sentiment_score"].shift(lag)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()
xcol = f"sentiment_lag_{lag}"
f = f.dropna(subset=[xcol, "return_volatility"])

st.subheader("Visualization — Lagged sentiment vs volatility")
base = alt.Chart(f).encode(
    x=alt.X(f"{xcol}:Q", title=f"Sentiment (lag {lag} quarter(s))"),
    y=alt.Y("return_volatility:Q", title="Quarterly Return Volatility"),
    tooltip=["quarter_label", "sentiment_score", xcol, "return_volatility"],
)

use_reg = st.checkbox("Show regression line", value=True)
pts = base.mark_circle()

if use_reg and len(f) >= 2:
    reg = base.transform_regression(xcol, "return_volatility").mark_line()
    chart = (pts + reg).properties(height=420)
else:
    chart = pts.properties(height=420)

st.altair_chart(chart, use_container_width=True)

st.subheader("Summary")
corr = f[xcol].corr(f["return_volatility"]) if len(f) >= 2 else np.nan
st.write(f"- Rows: **{len(f)}**")
st.write(f"- Correlation (lagged sentiment vs volatility): **{corr:.3f}**" if np.isfinite(corr) else "- Correlation: N/A")

st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)