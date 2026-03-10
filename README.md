# Apple Stock Performance & News Sentiment Analysis 📈📰

An end-to-end data analytics and time-series project investigating the relationship between Apple's (AAPL) stock performance, financial news sentiment, and core business metrics (iPhone revenue and iOS market share). 

This project features a robust data pipeline, statistical time-series evaluation, and an interactive Streamlit dashboard equipped with a **Generative AI Visual Agent** for automated exploratory data analysis.

## 📌 Project Overview
The primary objective of this project is to determine whether public sentiment, measured through financial news, acts as a leading indicator or an attention mechanism for Apple's stock volatility, trading volume, and overall returns. The analysis integrates over 40 years of financial data with natural language sentiment scores and global smartphone market statistics.

## 🏗 Architecture & Data Pipeline
The data pipeline handles data ingestion, cleaning, transformation, and aggregation from multiple sources:
* **Cloud Storage Integration:** Raw datasets (primary stock data, news sentiment, iPhone revenue, and market share) are ingested directly from **Google Cloud Storage (GCS)**.
* **Data Transformation:** * Handled missing values using targeted imputation and forward-filling for time-series continuity.
  * Applied **7-day rolling means (smoothing)** to daily financial prices and sentiment scores to filter out short-term noise and highlight structural trends.
* **Temporal Aggregation:** Standardized disparate granularities by aggregating daily financial and sentiment data to quarterly and yearly levels, merging them into a single analytical parquet file.

## 🔬 Statistical & Time-Series Analysis
To ensure analytical rigor, domain-specific time-series methods from the `statsmodels` library were applied:
* **ACF & PACF (Autocorrelation & Partial Autocorrelation):** Evaluated Apple's yearly stock returns. Results showed minimal autocorrelation, indicating that yearly returns behave similarly to a random walk and are driven by new market information rather than serial dependence.
* **Augmented Dickey-Fuller (ADF) Unit Root Test:** Applied to quarterly average news sentiment to test for stationarity. The non-stationary result (p-value > 0.05) quantitatively confirmed that Apple's media sentiment exhibits a persistent, long-term upward drift.

## 📊 Interactive Dashboard & AI Integration
The findings are presented in a multipage **Streamlit** application containing:
* **Targeted Research Modules (Q1-Q6):** Interactive visualizations (built with Altair and Matplotlib) exploring specific hypotheses, such as the correlation between extreme sentiment and trading volume.
* **Visual AI Agent:** Integrated with **Google GenAI**, this module acts as an autonomous data visualization assistant, allowing users to dynamically generate charts and uncover hidden patterns through natural language prompts.

## 🛠 Tech Stack
* **Language:** Python
* **Data Engineering:** Pandas, Google Cloud Storage (`gcsfs`, `pyarrow`)
* **Machine Learning & Stats:** Scikit-Learn, Statsmodels
* **Visualization:** Altair, Matplotlib, Seaborn
* **Application Framework:** Streamlit
* **AI / LLM Integration:** Google GenAI SDK

## 💡 Key Findings
1. **Sentiment as an Attention Mechanism:** While quarterly news sentiment showed little predictive power for the direction of stock returns, extreme sentiment (both positive and negative) was moderately correlated (r = 0.34) with heightened trading volume.
2. **Volatility Lead-Lag:** Higher news sentiment does not cause immediate market instability; rather, it moderately correlates (r = -0.48) with *lower* return volatility in the subsequent quarter.
3. **Business Fundamentals:** Structural stock growth aligned heavily with Apple's transition to an iPhone-dominated revenue model and sustained iOS market share, confirming that core product dominance outweighs short-term media sentiment in long-term valuation.
