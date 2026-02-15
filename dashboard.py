import os
import tempfile
import streamlit as st
import pandas as pd
import pm4py
from discovery_eval import discover_three_models, save_models, evaluate_all
from config import OUT_DIR
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ---- Groq helpers ----
@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


def ask_groq(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    client = get_groq_client()
    if client is None:
        return "GROQ_API_KEY missing. Please set it in .env file (GROQ_API_KEY=...) or environment variables."
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a Process Mining expert. Be concise and data-driven."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=700,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq error: {e}"


# ---- simple log analytics helpers ----
def compute_log_summary(df: pd.DataFrame) -> dict:
    case_col = "case:concept:name"
    act_col = "concept:name"
    ts_col = "time:timestamp"

    out = {
        "cases": int(df[case_col].nunique()) if case_col in df.columns else None,
        "events": int(len(df)),
        "activities": int(df[act_col].nunique()) if act_col in df.columns else None,
    }

    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], errors="coerce")
        out["time_start"] = str(ts.min())
        out["time_end"] = str(ts.max())
    else:
        out["time_start"] = None
        out["time_end"] = None

    if act_col in df.columns:
        out["top_activities"] = df[act_col].value_counts().head(10).to_dict()
    else:
        out["top_activities"] = {}

    return out


@st.cache_data(show_spinner=False)
def compute_case_durations(df: pd.DataFrame) -> pd.DataFrame:
    case_col = "case:concept:name"
    ts_col = "time:timestamp"

    d = df[[case_col, ts_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce")
    d = d.dropna(subset=[ts_col])

    g = d.groupby(case_col)[ts_col].agg(["min", "max"])
    g["duration_hours"] = (g["max"] - g["min"]).dt.total_seconds() / 3600.0
    g = g.sort_values("duration_hours", ascending=False)
    return g


# ---- Streamlit config ----
st.set_page_config(page_title="Process Mining Dashboard", layout="wide")
st.title("Process Mining Dashboard")

out_dir = os.path.join(OUT_DIR, "dashboard")
os.makedirs(out_dir, exist_ok=True)

# ---- sidebar ----
st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload XES log", type=["xes", "xes.gz"])
run = st.sidebar.button("Run Process Mining",
                        type="primary", disabled=(uploaded is None))

st.sidebar.markdown("---")
st.sidebar.subheader("LLM (Groq)")
if GROQ_API_KEY:
    st.sidebar.success("GROQ_API_KEY loaded.")
    st.sidebar.caption("Model: llama-3.1-8b-instant")
else:
    st.sidebar.warning("GROQ_API_KEY missing. Please set it in .env.")

# ---- state ----
for k, default in [
    ("log", None),
    ("df", None),
    ("results", None),
    ("ai_report", None),
    ("messages", []),
        ("summary", None),
    ("case_durations", None),
    ("anomalies", None),
    ("anomaly_threshold_hours", None),
]:
    if k not in st.session_state:
        st.session_state[k] = default

# ---- load log ----
if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    log = pm4py.read_xes(tmp_path)
    df = pm4py.convert_to_dataframe(log)

    st.session_state.log = log
    st.session_state.df = df
    st.session_state.summary = compute_log_summary(df)

# ---- empty state ----
if st.session_state.log is None:
    st.info("Upload an XES log file from the sidebar to get started.")
    st.stop()

# ---- tabs ----
tab_overview, tab_metrics, tab_models, tab_ai, tab_insights = st.tabs(
    ["Overview", "Metrics", "Models", "AI Assistant", "Insights"]
)

with tab_overview:
    df = st.session_state.df
    c1, c2, c3 = st.columns(3)
    c1.metric("Cases", int(df["case:concept:name"].nunique()))
    c2.metric("Events", int(len(df)))
    c3.metric("Activities", int(df["concept:name"].nunique()))
    st.caption(
        # ---- run discovery/eval ----
        "Run Process Mining to discover models and compute metrics. Then explore the tabs for details and AI insights.")
if run:
    with st.spinner("Running discovery + evaluation..."):
        log = st.session_state.log

        models = discover_three_models(log)
        save_models(models, out_dir)
        results = evaluate_all(log, models, out_dir)

        st.session_state.results = results
        st.session_state.messages = []
        st.session_state.ai_report = None
# ---- metrics tab ----
with tab_metrics:
    if st.session_state.results is None:
        st.info("No results yet. Click 'Run Process Mining' in the sidebar.")
    else:
        results = st.session_state.results
        table = [{"algorithm": alg, **m} for alg, m in results.items()]
        metrics_df = pd.DataFrame(table).sort_values("algorithm")
        st.dataframe(metrics_df, use_container_width=True)

# ---- models tab ----
with tab_models:
    if st.session_state.results is None:
        st.info("No results yet. Click 'Run Process Mining' in the sidebar.")
    else:
        t_alpha, t_heu, t_ind = st.tabs(["Alpha", "Heuristics", "Inductive"])

        with t_alpha:
            st.image(os.path.join(out_dir, "alpha.png"),
                     use_container_width=True)
            st.image(os.path.join(out_dir, "alpha_metrics.png"),
                     use_container_width=True)

        with t_heu:
            st.image(os.path.join(out_dir, "heuristics.png"),
                     use_container_width=True)
            st.image(os.path.join(out_dir, "heuristics_metrics.png"),
                     use_container_width=True)

        with t_ind:
            st.image(os.path.join(out_dir, "inductive.png"),
                     use_container_width=True)
            st.image(os.path.join(out_dir, "inductive_metrics.png"),
                     use_container_width=True)

# ---- AI Assistant tab ----
with tab_ai:
    st.subheader("AI Assistant")

    df = st.session_state.df
    summary = st.session_state.summary or compute_log_summary(df)

    metrics_csv = "Metrics not computed yet. Run Process Mining first."
    if st.session_state.results is not None:
        metrics_df = pd.DataFrame(
            [{"algorithm": alg, **m}
                for alg, m in st.session_state.results.items()]
        ).sort_values("algorithm")
        metrics_csv = metrics_df.to_csv(index=False)

    # --- Report section (top) ---
    st.markdown("### AI Process Report")
    if st.session_state.results is None:
        st.info("Run Process Mining first to compute metrics.")
    else:
        if st.button("Generate AI Process Report"):
            prompt = f"""
You are a Process Mining expert.
Event log summary:
{summary}
Model metrics (CSV, 0-1 scale):
{metrics_csv}
Write a short report:
- Describe the process at high level
- Recommend the best model and justify (fitness vs precision vs generalization vs simplicity)
- List 3 issues to investigate
- List 3 improvement actions
Be concise and do not invent numbers.
"""
            with st.spinner("Generating report..."):
                st.session_state.ai_report = ask_groq(
                    prompt, model="llama-3.1-8b-instant")

        if st.session_state.ai_report:
            st.text_area("AI Report", st.session_state.ai_report,
                         height=400, disabled=True)

    st.markdown("---")

    # --- Chat section (bottom) ---
    st.markdown("### Ask questions")
    st.caption(
        "Example: 'Which model has the best fitness and why?' or 'What are the trade-offs between the models?'")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Write a question about the process/models...")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        prompt = f"""
You are a Process Mining expert.
Event log summary:
{summary}
Model metrics (CSV, 0-1 scale):
{metrics_csv}
User question:
{user_msg}
Answer rules:
- Be concise and concrete.
- If asked 'best model', justify using trade-offs (fitness vs precision vs generalization vs simplicity).
- Do not invent numbers not present in the context.
"""

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_groq(prompt, model="llama-3.1-8b-instant")
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer})


# ---- Insights tab ----
with tab_insights:
    st.subheader("Insights")

    df = st.session_state.df
    summary = st.session_state.summary or compute_log_summary(df)

    st.caption(
        f"Time range: {summary.get('time_start')} → {summary.get('time_end')}")
    st.write("Top activities:", summary.get("top_activities", {}))

    st.markdown("---")
    st.markdown("### Anomaly detection (case duration)")

    if st.button("Compute / Refresh anomalies"):
        durations = compute_case_durations(df)
        st.session_state.case_durations = durations
        thr = float(durations["duration_hours"].quantile(0.95))
        st.session_state.anomaly_threshold_hours = thr
        st.session_state.anomalies = durations[durations["duration_hours"] > thr]

    anomalies = st.session_state.anomalies
    thr = st.session_state.anomaly_threshold_hours

    if anomalies is None or thr is None:
        st.info(
            "Click 'Compute / Refresh anomalies' to compute anomalous cases (very long durations).")
    else:
        st.write(f"Threshold (95th percentile): **{thr:.2f} hours**")
        st.write(f"Anomalous cases: **{len(anomalies)}**")
        st.dataframe(anomalies.head(30), use_container_width=True)

    st.markdown("---")
    st.markdown("### Root Cause Analysis on anomalies")

    if anomalies is None or len(anomalies) == 0:
        st.info("Compute anomalies first (case durations).")
    else:
        k = min(10, len(anomalies))
        sample_cases = anomalies.head(
            k).reset_index().to_dict(orient="records")

        if st.button("Generate RCA for anomalies"):
            prompt = f"""
You are a Process Mining expert.
We detected anomalously long cases (by duration_hours). Here is a sample:
{sample_cases}
Explain likely causes and suggest checks.
"""
            with st.spinner("Generating RCA..."):
                rca = ask_groq(prompt)
            st.text_area("RCA Output", rca, height=400, disabled=True)
