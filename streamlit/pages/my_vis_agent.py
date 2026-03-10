import os
import json
import time
import traceback

import pandas as pd
import altair as alt
import streamlit as st

from google import genai
from pydantic import BaseModel, Field
from utils.data import build_quarterly_merged


USE_VERTEXAI = os.environ.get("USE_VERTEXAI")

if USE_VERTEXAI in ("true", "True", "TRUE"):
    USE_VERTEXAI = True
    PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
else:
    USE_VERTEXAI = False
    API_KEY = os.environ.get("API_KEY")

GEMINI_FLASH_LITE = "gemini-2.5-flash-lite"
GEMINI_FLASH = "gemini-2.5-flash"
USE_MODEL = GEMINI_FLASH_LITE


class Feasibility(BaseModel):
    feasible: bool = Field(description="Whether the request can be answered with the available data.")
    reason: str = Field(description="One short explanation of the decision.")
    selected_dataset: str | None = Field(description="The best dataset key to use, or None if infeasible.")


class Chart(BaseModel):
    code: str | None = Field(description="Executable Python code that creates the final Altair chart.")
    desc: str = Field(description="A short explanation of the chart and what it shows.")


@st.cache_resource(show_spinner=False)
def get_genai_client():
    if USE_VERTEXAI:
        return genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
    return genai.Client(api_key=API_KEY)


class VisAgent:
    def __init__(self):
        self.client = get_genai_client()
        self.datasets = self._init_datasets()
        self.schemas = self._extract_schemas()
        self.dataset_guidance = self._build_dataset_guidance()

    def _init_datasets(self):
        base_df = build_quarterly_merged().copy()
        base_df = base_df.sort_values("quarter_end_date").reset_index(drop=True)

        datasets = {
            "q1_sentiment_vs_return": base_df.copy(),
            "q2_extreme_sentiment_vs_volume": base_df.copy(),
            "q3_lagged_sentiment_vs_volatility": base_df.copy(),
            "q4_return_over_time": base_df.copy(),
            "q5_sentiment_vs_performance": base_df.copy(),
            "q6_news_volume_vs_performance": base_df.copy(),
        }
        return datasets

    def _extract_schemas(self):
        schemas = {}
        for name, df in self.datasets.items():
            schemas[name] = df.dtypes.astype(str).to_dict()
        return schemas

    def _build_dataset_guidance(self):
        return {
            "q1_sentiment_vs_return": {
                "best_for": "sentiment vs quarterly stock return",
                "preferred_columns": ["sentiment_score", "q_return", "quarter_label", "quarter_end_date"],
                "example_requests": [
                    "scatter plot of sentiment score vs quarterly return",
                    "does sentiment relate to stock return",
                ],
            },
            "q2_extreme_sentiment_vs_volume": {
                "best_for": "extreme sentiment and trading volume",
                "preferred_columns": ["sentiment_score", "trading_volume", "quarter_label", "quarter_end_date"],
                "example_requests": [
                    "do extreme sentiment quarters have higher trading volume",
                    "compare sentiment and trading volume",
                ],
            },
            "q3_lagged_sentiment_vs_volatility": {
                "best_for": "sentiment and return volatility",
                "preferred_columns": ["sentiment_score", "return_volatility", "quarter_label", "quarter_end_date"],
                "example_requests": [
                    "does sentiment relate to volatility",
                    "plot sentiment against return volatility",
                ],
            },
            "q4_return_over_time": {
                "best_for": "return trends over time",
                "preferred_columns": ["quarter_end_date", "quarter_label", "q_return"],
                "example_requests": [
                    "show returns over time",
                    "line chart of quarterly return",
                ],
            },
            "q5_sentiment_vs_performance": {
                "best_for": "sentiment against broader performance metrics",
                "preferred_columns": ["sentiment_score", "q_return", "return_volatility", "quarter_label"],
                "example_requests": [
                    "compare sentiment and performance",
                    "relationship between sentiment and performance",
                ],
            },
            "q6_news_volume_vs_performance": {
                "best_for": "news volume and stock performance",
                "preferred_columns": ["news_count", "q_return", "return_volatility", "quarter_label"],
                "example_requests": [
                    "does article count relate to returns",
                    "news volume vs stock performance",
                ],
            },
        }

    def _evaluate_and_select(self, user_prompt: str):
        sys_prompt = f"""
You are a data visualization planning assistant.

Your tasks:
1. Decide whether the user's request is feasible with the available data.
2. Choose the single best dataset key.

Available dataset guidance:
{json.dumps(self.dataset_guidance, indent=2)}

Available schemas:
{json.dumps(self.schemas, indent=2)}

Available columns across the data include:
- quarter_end_date
- quarter_label
- sentiment_score
- sentiment_neg
- sentiment_neu
- sentiment_pos
- news_count
- q_return
- trading_volume
- return_volatility

Selection rules:
- sentiment + return -> q1_sentiment_vs_return
- extreme sentiment + trading volume -> q2_extreme_sentiment_vs_volume
- sentiment + volatility -> q3_lagged_sentiment_vs_volatility
- return trend over time -> q4_return_over_time
- sentiment + performance -> q5_sentiment_vs_performance
- news count/news volume + performance -> q6_news_volume_vs_performance

Feasibility rules:
- If the user asks for variables not in the schema, mark infeasible.
- If the user asks for unsupported entities like Tesla, inflation, GDP, geography, or causation proof, mark infeasible.
- If the user asks for a general chart that can be answered with the available columns, mark feasible.
"""

        response = self.client.models.generate_content(
            model=USE_MODEL,
            contents=[sys_prompt, user_prompt],
            config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
                "response_schema": Feasibility,
            },
        )
        return response.parsed

    def _generate_code(self, user_prompt: str, dataset_name: str, error_context: str = ""):
        sys_prompt = f"""
Write Python code using Altair to create a visualization for the user's request.

The data is already loaded in a pandas DataFrame named df.
Selected dataset key: {dataset_name}
Schema: {json.dumps(self.schemas[dataset_name], indent=2)}

Column meanings:
- quarter_end_date: datetime end-of-quarter date
- quarter_label: quarter label string
- sentiment_score: average sentiment score
- sentiment_neg: average negative sentiment
- sentiment_neu: average neutral sentiment
- sentiment_pos: average positive sentiment
- news_count: number of news articles in the quarter
- q_return: quarterly stock return
- trading_volume: quarterly trading volume
- return_volatility: quarterly return volatility

Requirements:
1. Return executable Python code only.
2. Use Altair only.
3. Do not import additional libraries.
4. Do not save files.
5. Do not call plt.show().
6. Assign the final chart to a variable named final_chart.
7. Add informative axis titles.
8. Include tooltips.
9. Make the chart clean and Streamlit-friendly.
10. If the request is a time trend, use quarter_end_date on the x-axis.
11. If suitable, prefer simple charts like line, bar, or scatter.
12. Keep the code concise and robust.

If the previous attempt failed, fix it using this error:
{error_context}
"""

        response = self.client.models.generate_content(
            model=USE_MODEL,
            contents=[sys_prompt, user_prompt],
            config={
                "temperature": 0.1,
                "response_mime_type": "application/json",
                "response_schema": Chart,
            },
        )
        return response.parsed

    def run(self, user_prompt: str, max_retries: int = 2):
        start_time = time.time()

        eval_result = self._evaluate_and_select(user_prompt)

        if not eval_result.feasible:
            elapsed = time.time() - start_time
            return {
                "status": "error",
                "message": f"Request not feasible: {eval_result.reason}",
                "dataset": None,
                "elapsed_seconds": round(elapsed, 2),
            }

        selected_dataset = eval_result.selected_dataset
        if selected_dataset not in self.datasets:
            elapsed = time.time() - start_time
            return {
                "status": "error",
                "message": "The agent could not match your request to a valid dataset.",
                "dataset": None,
                "elapsed_seconds": round(elapsed, 2),
            }

        df = self.datasets[selected_dataset].copy()
        error_msg = ""

        for _ in range(max_retries):
            code_result = self._generate_code(user_prompt, selected_dataset, error_context=error_msg)

            raw_code = code_result.code or ""
            code = raw_code.replace("```python", "").replace("```", "").strip()
            desc = code_result.desc

            local_vars = {"df": df, "pd": pd, "alt": alt}

            try:
                exec(code, {}, local_vars)
                chart_obj = local_vars.get("final_chart")

                if chart_obj is None:
                    raise ValueError("Generated code did not assign a final_chart variable.")

                elapsed = time.time() - start_time
                return {
                    "status": "success",
                    "dataset": selected_dataset,
                    "chart_obj": chart_obj,
                    "code": code,
                    "desc": desc,
                    "elapsed_seconds": round(elapsed, 2),
                }

            except Exception:
                error_msg = traceback.format_exc()

        elapsed = time.time() - start_time
        return {
            "status": "error",
            "message": "Failed to generate valid code within the retry limit.",
            "last_error": error_msg,
            "dataset": selected_dataset,
            "elapsed_seconds": round(elapsed, 2),
        }


st.title("🎯 Visual AI Agent")
st.write("Use natural language to generate a chart from the Apple quarterly news and stock dataset.")

st.markdown("""
### How to use this page
Type a visualization request in the chat box below.

### Sample prompts
- Show Apple quarterly returns over time
- Create a scatter plot of sentiment score vs quarterly return
- Does extreme sentiment correspond to higher trading volume?
- Plot news_count against q_return
- Show the relationship between sentiment score and return volatility

### Tips for best results
- Be specific about which variables you want to compare.
- Best supported topics are sentiment, returns, volatility, trading volume, article count, and time trends.
- Avoid requests for variables that are not in the dataset.
- Avoid very vague prompts like "show something interesting."
- Avoid asking the model to prove causation.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Ask for a visualization...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating visualization..."):
            va = VisAgent()
            result = va.run(user_prompt)

            if result.get("status") != "success":
                st.error(result.get("message", "Error"))
                if result.get("dataset"):
                    st.subheader("Dataset Considered")
                    st.code(result["dataset"])
                if result.get("last_error"):
                    st.subheader("Last Error")
                    st.code(result["last_error"], language="text")
                st.caption(f"Response time: {result.get('elapsed_seconds', 'N/A')} seconds")

                assistant_text = result.get("message", "Error generating visualization.")
            else:
                st.subheader("Description")
                st.write(result["desc"])

                st.subheader("Dataset Used")
                st.code(result["dataset"])

                st.subheader("Generated Chart")
                st.altair_chart(result["chart_obj"], use_container_width=True)

                st.subheader("Generated Code")
                st.code(result["code"], language="python")

                st.caption(f"Response time: {result['elapsed_seconds']} seconds")
                st.caption(
                    "This visualization was generated by an LLM-based agent that first checked whether the request "
                    "was feasible with the available data, selected the most appropriate dataset, generated Altair code, "
                    "verified that the code executed successfully, and then rendered the chart in Streamlit."
                )

                assistant_text = (
                    f"Generated a visualization using dataset `{result['dataset']}`.\n\n"
                    f"{result['desc']}"
                )

    st.session_state.messages.append({"role": "assistant", "content": assistant_text})
