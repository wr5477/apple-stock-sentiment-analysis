import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q6. Does news volume relate to stock performance?")

df = build_quarterly_merged().copy()

# ---- Filters
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
    options=["return_volatility", "q_return"],
    format_func=lambda x: "Return volatility" if x == "return_volatility" else "Quarterly stock return",
)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()

# ---- Visualization
st.subheader("Visualization")
y_title = "Return Volatility" if metric == "return_volatility" else "Quarterly Stock Return"

base = alt.Chart(f).encode(
    x=alt.X("news_count:Q", title="News count (per quarter)"),
    y=alt.Y(f"{metric}:Q", title=y_title),
    tooltip=["quarter_label", "news_count", "q_return", "return_volatility", "sentiment_score"],
)

pts = base.mark_circle(size=80)

if use_reg and len(f) >= 2:
    reg = base.transform_regression("news_count", metric).mark_line()
    chart = (pts + reg).properties(height=420)
else:
    chart = pts.properties(height=420)

st.altair_chart(chart, use_container_width=True)

st.markdown("""
**Design Improvements:**
- Minimal encodings keep focus on the relationship  
- Regression toggle adds structure without hiding raw points  
- Tooltips provide detail-on-demand, keeping the plot clean  
""")

# ---- Summary
st.subheader("Summary")
rows = len(f)
corr = f["news_count"].corr(f[metric]) if rows >= 2 else np.nan

st.write(f"- Rows: **{rows}**")
if np.isfinite(corr):
    st.write(f"- Correlation (news count vs {y_title.lower()}): **{corr:.3f}**")
else:
    st.write("- Correlation: N/A")

st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)