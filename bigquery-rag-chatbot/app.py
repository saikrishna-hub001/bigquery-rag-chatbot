"""
BigQuery AI Data Catalog Assistant
- Default mode: thelook_ecommerce BigQuery public dataset
- Custom mode: upload your own CSV and ask questions about it
Powered by Groq (Llama 3.3 70B) + TF-IDF RAG + BigQuery / pandas execution
"""

import json
import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BigQuery AI Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    color: white; box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem; color: white; }
.hero p  { font-size: 1rem; margin: 0; opacity: 0.9; color: white; }

.mode-badge {
    display: inline-block; border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600; margin-bottom: 1rem;
}
.mode-bq     { background: linear-gradient(135deg,#667eea,#764ba2); color: white; }
.mode-custom { background: linear-gradient(135deg,#11998e,#38ef7d); color: white; }

.stat-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 90px; border-radius: 12px; padding: 14px 16px;
    color: white; font-weight: 600; box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stat-card .val { font-size: 1.6rem; line-height: 1; }
.stat-card .lbl { font-size: 0.7rem; opacity: 0.85; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.card-purple { background: linear-gradient(135deg,#667eea,#764ba2); }
.card-pink   { background: linear-gradient(135deg,#f093fb,#f5576c); }
.card-teal   { background: linear-gradient(135deg,#4facfe,#00f2fe); }

.chip {
    display: inline-block; background: #a78bfa11; color: #a78bfa;
    border: 1px solid #a78bfa55; border-radius: 20px; padding: 3px 10px;
    font-size: 0.75rem; font-weight: 500; margin: 3px 2px;
}

.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important; border: none !important; border-radius: 20px !important;
    font-size: 0.8rem !important; font-weight: 500 !important; padding: 8px 16px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 3px 10px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(102,126,234,0.45) !important;
}
.run-btn > button {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    color: white !important; border-radius: 20px !important;
    font-weight: 600 !important; border: none !important;
    box-shadow: 0 3px 10px rgba(17,153,142,0.3) !important;
}
.result-banner {
    background: linear-gradient(135deg,#11998e22,#38ef7d22);
    border: 1px solid #11998e55; border-radius: 10px; padding: 8px 14px;
    color: #11998e; font-weight: 600; font-size: 0.85rem; margin-bottom: 8px;
}
.upload-box {
    background: linear-gradient(135deg,#667eea11,#764ba211);
    border: 2px dashed #667eea55; border-radius: 12px; padding: 1rem;
    text-align: center; margin-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#1a1a2e,#16213e,#0f3460) !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .chip { color:#a78bfa !important; border-color:#a78bfa55 !important; background:#a78bfa11 !important; }
.section-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.08em; color: #888; margin: 1rem 0 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
BQ_DATASET = "bigquery-public-data.thelook_ecommerce"
GEN_MODEL  = "llama-3.3-70b-versatile"
MAX_ROWS   = 100
MAX_BYTES  = 200 * 1024 * 1024


# ── Groq ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

groq_client = get_groq_client()

def generate(prompt: str) -> str:
    resp = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ── BigQuery ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_bq_client():
    creds_dict  = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])


# ── Default BQ index ──────────────────────────────────────────────────────────
@st.cache_resource
def load_bq_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "data", "metadata.json")) as f:
        metadata = json.load(f)
    docs = []
    for table in metadata["tables"]:
        col_text = ", ".join(
            f"{c['name']} ({c['type']}): {c['description']}" for c in table["columns"]
        )
        docs.append({
            "table_name": table["table_name"],
            "text": f"Table: {table['table_name']}\nDescription: {table['description']}\nColumns: {col_text}",
            "raw": table,
        })
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([d["text"] for d in docs])
    return docs, vec, mat, metadata


# ── CSV index builder ─────────────────────────────────────────────────────────
def build_csv_index(df: pd.DataFrame, filename: str):
    """Build a RAG index from an uploaded CSV dataframe."""
    col_descriptions = []
    for col in df.columns:
        dtype    = str(df[col].dtype)
        n_unique = df[col].nunique()
        sample   = df[col].dropna().head(3).tolist()
        col_descriptions.append(
            f"{col} ({dtype}, {n_unique} unique values, e.g. {sample})"
        )

    doc_text = (
        f"Table: {filename}\n"
        f"Rows: {len(df)}\n"
        f"Columns: {', '.join(col_descriptions)}"
    )
    docs = [{"table_name": filename, "text": doc_text, "raw": {"columns": [{"name": c} for c in df.columns]}}]
    vec  = TfidfVectorizer(stop_words="english")
    mat  = vec.fit_transform([doc_text])
    return docs, vec, mat


# ── Shared helpers ────────────────────────────────────────────────────────────
def retrieve(query, docs, vec, mat, top_k=3):
    sims    = cosine_similarity(vec.transform([query]), mat).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [docs[i] for i in top_idx]


def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_python(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def strip_code_blocks(text):
    text = re.sub(r"```sql\s*.*?```",    "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```python\s*.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def enforce_limit(sql):
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip(";") + f"\nLIMIT {MAX_ROWS}"


def run_bq_query(sql):
    safe_sql   = enforce_limit(sql)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES, use_query_cache=True)
    job        = get_bq_client().query(safe_sql, job_config=job_config)
    return job.result().to_dataframe(), safe_sql


def run_pandas_query(code: str, df: pd.DataFrame):
    """Execute LLM-generated pandas code safely. df is available as 'df'."""
    local_vars = {"df": df.copy(), "pd": pd}
    exec(compile(code, "<string>", "exec"), {"__builtins__": {}}, local_vars)
    result = local_vars.get("result", None)
    if result is None:
        for k, v in local_vars.items():
            if isinstance(v, pd.DataFrame) and k != "df":
                result = v
                break
    return result


def build_bq_prompt(query, context_docs):
    context = "\n\n".join(d["text"] for d in context_docs)
    return f"""You are a BigQuery data catalog assistant for dataset `{BQ_DATASET}`.
Use ONLY the schema below. For SQL, use fully-qualified table names and wrap in ```sql blocks. Be concise.

SCHEMA:
{context}

QUESTION: {query}"""


def build_csv_prompt(query, context_docs, df: pd.DataFrame):
    context  = "\n\n".join(d["text"] for d in context_docs)
    col_info = ", ".join(df.columns.tolist())
    return f"""You are a data analyst assistant. The user has uploaded a CSV file.
The dataframe is available as `df` in pandas. Columns: {col_info}

Schema details:
{context}

If the user wants to query or analyze the data, write a pandas code block (```python) that:
1. Performs the analysis on `df`
2. Stores the final result in a variable called `result` (as a DataFrame or value)
3. Never import anything or use file I/O

For schema/column questions, just answer in plain text.
Be concise.

QUESTION: {query}"""


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi! I'm your AI data assistant. Ask me about the default BigQuery dataset, or upload your own CSV in the sidebar to explore your own data!",
        "sql": None, "python": None, "df": None,
    }]
if "csv_df"   not in st.session_state: st.session_state.csv_df   = None
if "csv_name" not in st.session_state: st.session_state.csv_name = None
if "mode"     not in st.session_state: st.session_state.mode     = "bigquery"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>BigQuery AI</div>
        <div style='font-size:0.75rem; opacity:0.6; margin-top:2px;'>Data Catalog Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Mode toggle ──
    st.markdown('<div class="section-label">📂 Data source</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Choose mode",
        ["🔵 BigQuery Public Dataset", "🟢 Upload my own CSV"],
        index=0 if st.session_state.mode == "bigquery" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = "bigquery" if "BigQuery" in mode else "csv"

    # ── CSV upload ──
    if st.session_state.mode == "csv":
        st.markdown('<div class="section-label">📤 Upload CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8", errors="replace")))
                st.session_state.csv_df   = df
                st.session_state.csv_name = uploaded.name
                st.session_state.messages = [{
                    "role": "assistant",
                    "content": f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns. Ask me anything about this data!",
                    "sql": None, "python": None, "df": None,
                }]
                st.success(f"✅ {uploaded.name} loaded!")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        if st.session_state.csv_df is not None:
            df = st.session_state.csv_df
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card card-teal">
                    <div class="val">{len(df):,}</div><div class="lbl">Rows</div>
                </div>
                <div class="stat-card card-pink">
                    <div class="val">{len(df.columns)}</div><div class="lbl">Cols</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="section-label">📋 Columns</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="chip">⬡ {c}</span>' for c in df.columns)
            st.markdown(chips, unsafe_allow_html=True)
            with st.expander("Preview data"):
                st.dataframe(df.head(5), use_container_width=True)

    # ── BQ stats ──
    else:
        bq_docs, bq_vec, bq_mat, bq_meta = load_bq_index()
        total_cols = sum(len(t["columns"]) for t in bq_meta["tables"])
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card card-purple">
                <div class="val">{len(bq_docs)}</div><div class="lbl">Tables</div>
            </div>
            <div class="stat-card card-pink">
                <div class="val">{total_cols}</div><div class="lbl">Columns</div>
            </div>
            <div class="stat-card card-teal">
                <div class="val">RAG</div><div class="lbl">Method</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-label">📦 Dataset</div>', unsafe_allow_html=True)
        st.code(BQ_DATASET, language=None)
        st.markdown('<div class="section-label">🗄️ Tables</div>', unsafe_allow_html=True)
        chips = "".join(f'<span class="chip">📋 {d["table_name"]}</span>' for d in bq_docs)
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.8rem; color:#a78bfa; font-weight:600;'>Llama 3.3 70B · Groq</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "👋 Chat cleared! Ask me anything.",
            "sql": None, "python": None, "df": None,
        }]
        st.rerun()


# ── Hero ──────────────────────────────────────────────────────────────────────
mode_badge = (
    '<span class="mode-badge mode-bq">🔵 BigQuery Mode</span>'
    if st.session_state.mode == "bigquery"
    else '<span class="mode-badge mode-custom">🟢 Custom CSV Mode</span>'
)
st.markdown(f"""
<div class="hero">
    <h1>🔍 BigQuery AI Data Catalog</h1>
    <p>Ask anything about your data in plain English — explain tables, columns, and generate + run queries instantly.</p>
    {mode_badge}
</div>
""", unsafe_allow_html=True)


# ── Render messages ───────────────────────────────────────────────────────────
def render_message(idx, msg):
    with st.chat_message(msg["role"]):
        code     = msg.get("sql") or msg.get("python")
        lang     = "sql" if msg.get("sql") else "python"
        disp_txt = strip_code_blocks(msg["content"]) if code else msg["content"]

        if disp_txt:
            st.markdown(disp_txt)

        if code:
            st.code(code, language=lang)
            if msg.get("df") is not None:
                result = msg["df"]
                if isinstance(result, pd.DataFrame):
                    st.markdown(
                        f'<div class="result-banner">✅ Returned {len(result)} rows</div>',
                        unsafe_allow_html=True
                    )
                    st.dataframe(result, use_container_width=True)
                else:
                    st.markdown(f'<div class="result-banner">✅ Result: {result}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="run-btn">', unsafe_allow_html=True)
                btn_label = "▶ Run in BigQuery" if msg.get("sql") else "▶ Run on my data"
                if st.button(btn_label, key=f"run_{idx}"):
                    with st.spinner("⚡ Running..."):
                        try:
                            if msg.get("sql"):
                                df_result, _ = run_bq_query(msg["sql"])
                                msg["df"] = df_result
                            else:
                                result = run_pandas_query(msg["python"], st.session_state.csv_df)
                                msg["df"] = result
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)


for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg)


# ── Suggestions ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">✨ Try these</div>', unsafe_allow_html=True)

if st.session_state.mode == "bigquery":
    suggestions = [
        ("📋 List all tables",            "List all tables"),
        ("🔎 Orders table columns",        "What columns are in the orders table?"),
        ("📊 Revenue by month SQL",        "Generate SQL for total revenue by month"),
        ("👥 Top customers by spend",      "Top 10 customers by total spend SQL"),
        ("🛍️ Best selling products",       "Best selling products by quantity SQL"),
    ]
elif st.session_state.csv_df is not None:
    suggestions = [
        ("📋 Describe the dataset",        "Describe this dataset and its columns"),
        ("📊 Show summary statistics",     "Show summary statistics for all columns"),
        ("🔎 Find missing values",         "Which columns have missing values and how many?"),
        ("📈 Top 10 rows by first column", f"Show top 10 rows sorted by {st.session_state.csv_df.columns[0]}"),
        ("💡 Insights",                    "What interesting insights can you find in this data?"),
    ]
else:
    suggestions = [("📤 Upload a CSV", "Upload a CSV file to get started")]

clicked = None
cols    = st.columns(len(suggestions))
for col, (label, prompt_text) in zip(cols, suggestions):
    if col.button(label, use_container_width=True):
        clicked = prompt_text


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("💬 Ask about your data...") or clicked

if prompt:
    if st.session_state.mode == "csv" and st.session_state.csv_df is None:
        st.warning("⚠️ Please upload a CSV file first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "sql": None, "python": None, "df": None})

        with st.spinner("🧠 Thinking..."):
            if st.session_state.mode == "bigquery":
                bq_docs, bq_vec, bq_mat, _ = load_bq_index()
                context_docs = retrieve(prompt, bq_docs, bq_vec, bq_mat)
                answer       = generate(build_bq_prompt(prompt, context_docs))
                sql          = extract_sql(answer)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sql": sql, "python": None, "df": None,
                })
            else:
                df           = st.session_state.csv_df
                csv_docs, csv_vec, csv_mat = build_csv_index(df, st.session_state.csv_name)
                context_docs = retrieve(prompt, csv_docs, csv_vec, csv_mat)
                answer       = generate(build_csv_prompt(prompt, context_docs, df))
                python_code  = extract_python(answer)
                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                    "sql": None, "python": python_code, "df": None,
                })

        st.rerun()
