import streamlit as st
import altair as alt
import numpy as np

from utils.data import build_quarterly_merged

st.title("Q4. How has Apple’s stock return evolved over time?")

df = build_quarterly_merged().copy()
df = df.sort_values("quarter_end_date")

# ---- Filters
st.subheader("Filters")
quarters = df["quarter_label"].unique().tolist()
start_q, end_q = st.select_slider(
    "Quarter range",
    options=quarters,
    value=(quarters[0], quarters[-1]),
)

# optional smoothing
use_roll = st.checkbox("Show rolling mean (4 quarters)", value=False)

f = df[(df["quarter_label"] >= start_q) & (df["quarter_label"] <= end_q)].copy()

if use_roll and len(f) > 1:
    f["q_return_roll4"] = f["q_return"].rolling(4, min_periods=1).mean()
    y_col = "q_return_roll4"
    y_title = "Quarterly Return (Rolling Mean, 4Q)"
else:
    y_col = "q_return"
    y_title = "Quarterly Return"

# ---- Chart
st.subheader("Visualization")
base = alt.Chart(f).encode(
    x=alt.X("quarter_end_date:T", title="Quarter end date"),
    y=alt.Y(f"{y_col}:Q", title=y_title),
    tooltip=["quarter_label", "quarter_end_date", "q_return"],
)

chart = base.mark_line().properties(height=420)
st.altair_chart(chart, use_container_width=True)

# ---- Quick stats
st.subheader("Summary")
rows = len(f)
min_r = f["q_return"].min() if rows > 0 else np.nan
max_r = f["q_return"].max() if rows > 0 else np.nan

st.write(f"- Rows: **{rows}**")
if np.isfinite(min_r) and np.isfinite(max_r):
    st.write(f"- Min quarterly return: **{min_r:.3f}**")
    st.write(f"- Max quarterly return: **{max_r:.3f}**")
else:
    st.write("- Min/Max quarterly return: N/A")

# ---- Underlying data
st.subheader("Underlying data")
st.dataframe(f, use_container_width=True)