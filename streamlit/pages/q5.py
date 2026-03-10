import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q5. Is news sentiment associated with stock performance?")

df = build_quarterly_merged().copy()

st.subheader("Filters")
quarters = df["quarter_label"].unique().tolist()
start_q, end_q = st.select_slider(
    "Quarter range",
    options=quarters,
    value=(quarters[0], quarters[-1]),
)

use_reg = st.checkbox("Show regression line", value=True)

metric = st.selectbox(
    "Stock metric",
    options=["q_return", "return_volatility"],
    format_func=lambda x: "Quarterly stock return" if x == "q_return" else "Return volatility",
)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()

# ---- Visualization
st.subheader("Visualization")
y_title = "Quarterly Stock Return" if metric == "q_return" else "Return Volatility"

base = alt.Chart(f).encode(
    x=alt.X("sentiment_score:Q", title="Sentiment score"),
    y=alt.Y(f"{metric}:Q", title=y_title),
    tooltip=["quarter_label", "sentiment_score", "q_return", "return_volatility", "news_count"],
)

pts = base.mark_circle(size=80)

if use_reg and len(f) >= 2:
    reg = base.transform_regression("sentiment_score", metric).mark_line()
    chart = (pts + reg).properties(height=420)
else:
    chart = pts.properties(height=420)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
**Design Improvements:**
- Clean scatterplot to directly show association  
- Optional regression line to reveal trend without hiding the raw data  
- Tooltips replace on-chart labels to reduce clutter  
- Clear axis titles to support quick interpretation  
""")

# ---- Summary
st.subheader("Summary")
rows = len(f)
corr = f["sentiment_score"].corr(f[metric]) if rows >= 2 else np.nan

st.write(f"- Rows: **{rows}**")
if np.isfinite(corr):
    st.write(f"- Correlation (sentiment vs {y_title.lower()}): **{corr:.3f}**")
else:
    st.write("- Correlation: N/A")

st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)