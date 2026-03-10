import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q2. Do extreme sentiment quarters correspond to higher trading volume?")

df = build_quarterly_merged()

st.subheader("Filters")
quarters = df["quarter_label"].unique().tolist()
start_q, end_q = st.select_slider(
    "Quarter range",
    options=quarters,
    value=(quarters[0], quarters[-1]),
)

thr = st.slider("Extreme sentiment threshold (abs value)", 0.0, 1.0, 0.4, 0.05)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()
f["is_extreme"] = (f["sentiment_score"].abs() >= thr)

st.subheader("Visualization A — Volume by Extreme vs Non-Extreme")
grp = (
    f.groupby("is_extreme", as_index=False)["trading_volume"]
    .mean()
    .rename(columns={"is_extreme": "Extreme sentiment (abs>=thr)", "trading_volume": "Avg trading volume"})
)
grp["Extreme sentiment (abs>=thr)"] = grp["Extreme sentiment (abs>=thr)"].map({True: "Extreme", False: "Non-extreme"})

bar = (
    alt.Chart(grp)
    .mark_bar()
    .encode(
        x=alt.X("Extreme sentiment (abs>=thr):N", title="Group"),
        y=alt.Y("Avg trading volume:Q", title="Average Trading Volume"),
        tooltip=["Extreme sentiment (abs>=thr)", "Avg trading volume"],
    )
    .properties(height=320)
)
st.altair_chart(bar, use_container_width=True)

st.subheader("Visualization B — Sentiment vs Volume scatter")
scatter = (
    alt.Chart(f)
    .mark_circle()
    .encode(
        x=alt.X("sentiment_score:Q", title="Quarterly Average News Sentiment"),
        y=alt.Y("trading_volume:Q", title="Trading Volume"),
        color=alt.Color("is_extreme:N", title="Extreme?"),
        tooltip=["quarter_label", "sentiment_score", "trading_volume", "news_count"],
    )
    .properties(height=420)
)
st.altair_chart(scatter, use_container_width=True)

st.subheader("Summary")
ext_mean = f.loc[f["is_extreme"], "trading_volume"].mean()
non_mean = f.loc[~f["is_extreme"], "trading_volume"].mean()
st.write(f"- Extreme mean volume: **{ext_mean:,.0f}**" if np.isfinite(ext_mean) else "- Extreme mean volume: N/A")
st.write(f"- Non-extreme mean volume: **{non_mean:,.0f}**" if np.isfinite(non_mean) else "- Non-extreme mean volume: N/A")

st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)