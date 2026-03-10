import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q1. Is quarterly news sentiment associated with Apple’s quarterly stock return?")

df = build_quarterly_merged()

# ---- Filters
st.subheader("Filters")
quarters = df["quarter_label"].unique().tolist()
start_q, end_q = st.select_slider(
    "Quarter range",
    options=quarters,
    value=(quarters[0], quarters[-1]),
)

use_reg = st.checkbox("Show regression line", value=True)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()

# ---- Visualization
st.subheader("Visualization")

# Base encoding with cleaner axes (reduced gridlines)
base = alt.Chart(f).encode(
    x=alt.X(
        "sentiment_score:Q",
        title="Quarterly average news sentiment",
        axis=alt.Axis(grid=False)  # remove x-grid to reduce clutter
    ),
    y=alt.Y(
        "q_return:Q",
        title="Quarterly stock return",
        axis=alt.Axis(grid=True)   # keep light y-grid for reading values
    ),
    tooltip=["quarter_label", "sentiment_score", "q_return", "news_count"],
)

# Simple, consistent point styling -> higher data-ink ratio
pts = base.mark_circle(
    size=70,
    opacity=0.75,
)

if use_reg and len(f) >= 2:
    # Strategic color highlighting: trend line in a contrasting color
    reg = base.transform_regression(
        "sentiment_score",
        "q_return"
    ).mark_line(
        strokeWidth=3,
    )
    chart = (pts + reg).properties(height=420)
else:
    chart = pts.properties(height=420)

st.altair_chart(chart, use_container_width=True)

# ---- Design improvements (Part 2)
st.markdown("""
**Design Improvements:**

- Simplified the scatter plot to a single point style with fixed size and opacity, removing legends and unnecessary encodings to increase the data–ink ratio.  
- Turned off x-axis gridlines and kept only a light y-grid so values are easy to read without overloading the background.  
- Used a single neutral color for data points and a contrasting, thicker regression line to strategically highlight the overall relationship between sentiment and return.  
- Kept detailed information in hover tooltips (quarter, sentiment, return, news count) instead of on-chart labels, reducing clutter while preserving access to precise values.  
""")

# ---- Summary
st.subheader("Summary")
corr = f["sentiment_score"].corr(f["q_return"]) if len(f) >= 2 else np.nan
st.write(f"- Rows: **{len(f)}**")
st.write(
    f"- Correlation (sentiment vs return): **{corr:.3f}**"
    if np.isfinite(corr)
    else "- Correlation: N/A"
)

# ---- Underlying data
st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)