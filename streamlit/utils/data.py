# streamlit/utils/data.py
import streamlit as st
import pandas as pd

# ---- GCS paths (너 버킷 기준) ----
PRIM_FIN = "gs://wr5477_utds/prim/apple_financial_dataset.csv"
SUPPL_NEWS = "gs://wr5477_utds/suppl-1/1_apple_news_data.csv"


@st.cache_data(show_spinner=False)
def load_base_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw financial + news from GCS and apply the same core cleaning as M3/M4."""
    fin = pd.read_csv(PRIM_FIN)
    news = pd.read_csv(SUPPL_NEWS)

    # --- normalize column names ---
    news.columns = news.columns.str.strip()

    # --- make sentiment column consistent ---
    if "sentiment_score" not in news.columns:
        candidates = ["sentiment_polarity", "sentiment", "polarity", "compound", "score"]
        found = next((c for c in candidates if c in news.columns), None)
        if found is None:
            raise KeyError(
                f"Sentiment column not found. Available columns: {list(news.columns)}"
            )
        news = news.rename(columns={found: "sentiment_score"})

    # Rename to match your pipeline conventions
    fin = fin.rename(
        columns={
            "total_volume": "trading_volume",
            "return volatility": "return_volatility",
        }
    )
    news = news.rename(columns={"title": "headline", "link": "url"})

    # Datetime
    fin["quarter_end"] = pd.to_datetime(fin["quarter_end"], errors="coerce")
    fin["quarter_start"] = pd.to_datetime(fin["quarter_start"], errors="coerce")
    news["date"] = pd.to_datetime(news["date"], errors="coerce")

    # Feature selection (minimum needed for Q1~Q3)
    fin = fin[
        [
            "quarter_start",
            "quarter_end",
            "close_price",
            "trading_volume",
            "quarter_return",
            "return_volatility",
        ]
    ].copy()

    news = news[
        [
            "date",
            "sentiment_score",
            "sentiment_neg",
            "sentiment_neu",
            "sentiment_pos",
        ]
    ].copy()

    # Indexing
    fin = fin.set_index("quarter_end").sort_index()
    news = news.set_index("date").sort_index()

    return fin, news


@st.cache_data(show_spinner=False)
def build_quarterly_merged() -> pd.DataFrame:
    """
    Build the quarterly merged dataframe used for Q1~Q3:
    - fin: already quarterly (quarter_end)
    - news: daily -> quarterly aggregation
    """
    fin, news = load_base_data()

    fin_reset = fin.reset_index().copy()
    fin_reset["quarter_end"] = pd.to_datetime(fin_reset["quarter_end"], errors="coerce")
    fin_reset["q_period"] = fin_reset["quarter_end"].dt.to_period("Q")

    fin_q = fin_reset[
        ["q_period", "quarter_return", "trading_volume", "return_volatility"]
    ].copy()
    fin_q = fin_q.rename(columns={"quarter_return": "q_return"}).dropna(
        subset=["q_period", "q_return"]
    )

    # News daily -> quarterly
    news_idx = news.copy()
    news_idx.index = pd.to_datetime(news_idx.index, errors="coerce")
    if getattr(news_idx.index, "tz", None) is not None:
        news_idx.index = news_idx.index.tz_convert(None)

    news_idx["q_period"] = news_idx.index.to_period("Q")

    news_q = (
        news_idx.groupby("q_period")
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_neg=("sentiment_neg", "mean"),
            sentiment_neu=("sentiment_neu", "mean"),
            sentiment_pos=("sentiment_pos", "mean"),
            news_count=("sentiment_score", "size"),
        )
        .reset_index()
    )

    q = fin_q.merge(news_q, on="q_period", how="inner")

    # For Streamlit/Altair (Period -> datetime/string)
    q["quarter_end_date"] = q["q_period"].dt.end_time.dt.normalize()
    q["quarter_label"] = q["q_period"].astype(str)

    q = q.drop(columns=["q_period"], errors="ignore").copy()

    # Ensure numeric
    for col in ["sentiment_score", "q_return", "trading_volume", "return_volatility"]:
        if col in q.columns:
            q[col] = pd.to_numeric(q[col], errors="coerce")

    q = q.dropna(subset=["sentiment_score", "q_return"]).sort_values("quarter_end_date")
    return q