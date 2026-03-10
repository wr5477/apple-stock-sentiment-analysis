import streamlit as st

q1 = st.Page("pages/q1.py", title="Is quarterly news sentiment associated with Apple's quarterly stock return?", icon="📈")
q2 = st.Page("pages/q2.py", title="Do extreme sentiment quarters correspond to higher trading volume?", icon="📊")
q3 = st.Page("pages/q3.py", title="Do sentiment spikes precede larger price volatility?", icon="⚡")
q4 = st.Page("pages/q4.py", title="How has Apple's stock return evolved over time?", icon="📉")
q5 = st.Page("pages/q5.py", title="Is news sentiment associated with stock performance?", icon="📰")
q6 = st.Page("pages/q6.py", title="Does news volume relate to stock performance?", icon="📦")

v1 = st.Page("pages/my_vis_agent.py", title="Visual AI Agent", icon="🎯")

pg = st.navigation(
    {
        "Quarterly Analysis": [q1, q2, q3, q4, q5, q6],
        "AI Tools": [v1],
    }
)

pg.run()

# import streamlit as st

# st.set_page_config(
#     page_title="Project 1 — Milestone 5 Dashboard",
#     page_icon="📊",
#     layout="wide",
# )

# # ---- Header
# st.title("📊 Project 1 — Milestone 5 Dashboard")
# st.caption("Interactive dashboard organized by research question (Q1–Q6).")

# # ---- Intro
# st.markdown(
#     """
# This Streamlit app organizes our analysis by **research question**.
# Use the **sidebar** to navigate across pages.

# Each page includes:
# - **Interactive controls** (sliders / dropdowns / toggles)
# - At least **one visualization**
# - The **underlying data table**
# """
# )

# # ---- Quick start
# st.success("✅ Quick start: Begin with **Q1–Q3** (Sentiment ↔ Return / Volume / Volatility).")

# # ---- What’s inside
# with st.expander("What’s on each page?", expanded=False):
#     st.markdown(
#         """
# **Q1–Q3 (Core sentiment analysis)**
# - Sentiment vs Return  
# - Extreme Sentiment vs Volume  
# - Sentiment vs Volatility  

# **Q4–Q6 (Additional relationships / supporting analysis)**
# - Time trend view + summary stats  
# - Relationship checks with optional regression  
# - News volume / volatility / return comparisons  
# """
#     )

# # ---- Notes
# st.info(
#     "Tip: If anything looks slow, narrow the **Quarter range** filters on each page.\n\n"
#     "All charts are interactive via tooltips; use hover for exact values."
# )