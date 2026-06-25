"""
BigQuery AI Data Catalog Assistant
- BigQuery mode: thelook_ecommerce public dataset
- CSV mode: upload your own CSV
- Dashboard: auto charts for any dataset
Powered by Groq (Llama 3.3 70B) + TF-IDF RAG
"""

import json, os, re, io
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="BigQuery AI Assistant", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.hero{background:linear-gradient(135deg,#667eea 0%,#764ba2 40%,#f093fb 100%);border-radius:16px;padding:1.5rem 2rem;margin-bottom:1rem;color:white;box-shadow:0 8px 32px rgba(102,126,234,0.35);}
.hero h1{font-size:1.8rem;font-weight:700;margin:0 0 0.2rem;color:white;}
.hero p{font-size:0.95rem;margin:0;opacity:0.9;color:white;}
.mode-badge{display:inline-block;border-radius:20px;padding:4px 14px;font-size:0.78rem;font-weight:600;margin-top:0.5rem;}
.mode-bq{background:rgba(255,255,255,0.25);color:white;}
.mode-custom{background:rgba(56,239,125,0.35);color:white;}
.stat-row{display:flex;gap:10px;margin-bottom:1rem;flex-wrap:wrap;}
.stat-card{flex:1;min-width:80px;border-radius:12px;padding:12px 14px;color:white;font-weight:600;box-shadow:0 4px 12px rgba(0,0,0,0.15);}
.stat-card .val{font-size:1.5rem;line-height:1;}
.stat-card .lbl{font-size:0.65rem;opacity:0.85;margin-top:3px;text-transform:uppercase;letter-spacing:0.05em;}
.card-purple{background:linear-gradient(135deg,#667eea,#764ba2);}
.card-pink{background:linear-gradient(135deg,#f093fb,#f5576c);}
.card-teal{background:linear-gradient(135deg,#4facfe,#00f2fe);}
.card-orange{background:linear-gradient(135deg,#f7971e,#ffd200);}
.chip{display:inline-block;background:#a78bfa11;color:#a78bfa;border:1px solid #a78bfa55;border-radius:20px;padding:3px 10px;font-size:0.72rem;font-weight:500;margin:2px;}
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:white!important;border:none!important;border-radius:20px!important;font-size:0.78rem!important;font-weight:500!important;padding:7px 14px!important;transition:transform 0.15s,box-shadow 0.15s!important;box-shadow:0 3px 10px rgba(102,126,234,0.3)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 16px rgba(102,126,234,0.45)!important;}
.run-btn button{background:linear-gradient(135deg,#11998e,#38ef7d)!important;color:white!important;border-radius:20px!important;font-weight:600!important;border:none!important;}
.result-banner{background:linear-gradient(135deg,#11998e22,#38ef7d22);border:1px solid #11998e55;border-radius:10px;padding:8px 14px;color:#11998e;font-weight:600;font-size:0.85rem;margin-bottom:8px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e,#0f3460)!important;}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
[data-testid="stSidebar"] .chip{color:#a78bfa!important;border-color:#a78bfa55!important;background:#a78bfa11!important;}
.section-label{font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#888;margin:0.8rem 0 0.3rem;}
.tab-header{font-size:1.1rem;font-weight:700;color:#667eea;margin-bottom:1rem;}
.chart-card{background:#f8f9ff;border:1px solid #e0e0ff;border-radius:12px;padding:1rem;margin-bottom:1rem;}
/* ── Big bold tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 2px solid #e0e0ff;
    padding-bottom: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: linear-gradient(135deg,#667eea22,#764ba222) !important;
    border-radius: 12px 12px 0 0 !important;
    border: 1px solid #667eea44 !important;
    border-bottom: none !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #667eea !important;
    transition: all 0.2s !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    color: white !important;
    border-color: #667eea !important;
    box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    background: linear-gradient(135deg,#667eea44,#764ba244) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
BQ_DATASET = "bigquery-public-data.thelook_ecommerce"
GEN_MODEL  = "llama-3.3-70b-versatile"
MAX_ROWS   = 100
MAX_BYTES  = 1024 * 1024 * 1024   # 1GB — stays within BigQuery free tier

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
        max_tokens=1024,
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

# ── BQ index ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_bq_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "data", "metadata.json")) as f:
        metadata = json.load(f)
    docs = []
    for table in metadata["tables"]:
        col_text = ", ".join(f"{c['name']} ({c['type']}): {c['description']}" for c in table["columns"])
        docs.append({"table_name": table["table_name"],
                     "text": f"Table: {table['table_name']}\nDescription: {table['description']}\nColumns: {col_text}",
                     "raw": table})
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([d["text"] for d in docs])
    return docs, vec, mat, metadata

# ── CSV index ─────────────────────────────────────────────────────────────────
def build_csv_index(df: pd.DataFrame, filename: str):
    parts = [f"File: {filename}", f"Rows: {len(df)}", f"Columns ({len(df.columns)}):"]
    for col in df.columns:
        dtype   = str(df[col].dtype)
        n_uniq  = df[col].nunique()
        samples = df[col].dropna().astype(str).head(3).tolist()
        parts.append(f"  - {col} | type:{dtype} | unique:{n_uniq} | examples:{samples}")
    doc_text = "\n".join(parts)
    docs = [{"table_name": filename, "text": doc_text}]
    vec  = TfidfVectorizer()
    mat  = vec.fit_transform([doc_text])
    return docs, vec, mat, doc_text

# ── Helpers ───────────────────────────────────────────────────────────────────
def retrieve(query, docs, vec, mat, top_k=3):
    sims = cosine_similarity(vec.transform([query]), mat).flatten()
    return [docs[i] for i in np.argsort(sims)[::-1][:top_k]]

def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_python(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else None

def strip_code_blocks(text):
    text = re.sub(r"```sql\s*.*?```",    "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"```python\s*.*?```", "", text, flags=re.DOTALL|re.IGNORECASE)
    return text.strip()

def enforce_limit(sql):
    return sql if re.search(r"\blimit\b", sql, re.IGNORECASE) else sql.rstrip(";")+f"\nLIMIT {MAX_ROWS}"

def run_bq_query(sql):
    safe = enforce_limit(sql)
    cfg  = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES, use_query_cache=True)
    return get_bq_client().query(safe, job_config=cfg).result().to_dataframe(), safe

def run_pandas_code(code: str, df: pd.DataFrame):
    """Execute LLM-generated pandas code with full builtins available."""
    import builtins
    local_ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<llm_code>", "exec"), {"__builtins__": builtins}, local_ns)
    # Look for 'result' variable first, then any new DataFrame
    if "result" in local_ns:
        return local_ns["result"]
    for k, v in local_ns.items():
        if k != "df" and isinstance(v, (pd.DataFrame, pd.Series)):
            return v
    return None

def build_bq_prompt(query, context_docs):
    context = "\n\n".join(d["text"] for d in context_docs)
    return f"""You are a BigQuery data catalog assistant for `{BQ_DATASET}`.
Use ONLY the schema below. Wrap SQL in ```sql blocks. Be concise.

SCHEMA:
{context}

QUESTION: {query}"""

def build_csv_prompt(query, schema_text, df: pd.DataFrame):
    cols  = list(df.columns)
    dtype = {col: str(df[col].dtype) for col in cols}
    return f"""You are a helpful data analyst assistant. The user uploaded a CSV as a pandas DataFrame called `df`.

DataFrame info:
- Shape: {df.shape[0]} rows x {df.shape[1]} columns
- Columns and types: {dtype}
- Schema: {schema_text}

Rules:
1. For data analysis, statistics, filtering, aggregation questions → write a ```python code block. Store the final answer in a variable called `result` (DataFrame or scalar). Only use `pd` and `np` (pre-imported). Do NOT import anything.
2. For questions about column meanings or descriptions → answer in plain text ONLY, no code block.
3. Always write a brief explanation before or after any code block.
4. Be concise.

QUESTION: {query}"""

# ── Dashboard helpers ─────────────────────────────────────────────────────────
def render_dashboard(df: pd.DataFrame, title: str = "Dataset Dashboard"):
    st.markdown(f'<div class="tab-header">📊 {title}</div>', unsafe_allow_html=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower() or "month" in c.lower() or "year" in c.lower()]

    # ── Stat cards ──
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card card-purple"><div class="val">{len(df):,}</div><div class="lbl">Rows</div></div>
        <div class="stat-card card-pink"><div class="val">{len(df.columns)}</div><div class="lbl">Columns</div></div>
        <div class="stat-card card-teal"><div class="val">{len(num_cols)}</div><div class="lbl">Numeric</div></div>
        <div class="stat-card card-orange"><div class="val">{df.isnull().sum().sum()}</div><div class="lbl">Nulls</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Missing values ──
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown("**🔴 Missing values by column**")
        st.bar_chart(missing)

    # ── Numeric distributions ──
    if num_cols:
        st.markdown("**📈 Numeric column distributions**")
        cols_per_row = 3
        for i in range(0, min(len(num_cols), 6), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(num_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    st.markdown(f"**{col}**")
                    st.bar_chart(df[col].dropna().value_counts().head(20) if df[col].nunique() < 20 else df[col].dropna().describe().to_frame())

    # ── Categorical top values ──
    if cat_cols:
        st.markdown("**🏷️ Top values in categorical columns**")
        cols_per_row = 2
        for i in range(0, min(len(cat_cols), 4), cols_per_row):
            row_cols = st.columns(cols_per_row)
            for j, col in enumerate(cat_cols[i:i+cols_per_row]):
                with row_cols[j]:
                    st.markdown(f"**{col}** — top values")
                    top = df[col].value_counts().head(8)
                    st.bar_chart(top)

    # ── Correlation heatmap ──
    if len(num_cols) >= 2:
        st.markdown("**🔗 Numeric correlation**")
        corr = df[num_cols].corr().round(2)
        st.dataframe(corr.style.background_gradient(cmap="RdYlGn", axis=None), use_container_width=True)

    # ── Data preview ──
    st.markdown("**🔍 Data preview**")
    st.dataframe(df.head(20), use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"  not in st.session_state: st.session_state.messages  = [{"role":"assistant","content":"👋 Hi! Ask me about the BigQuery dataset, or upload your own CSV in the sidebar!","sql":None,"python":None,"df":None}]
if "csv_df"    not in st.session_state: st.session_state.csv_df    = None
if "csv_name"  not in st.session_state: st.session_state.csv_name  = None
if "mode"      not in st.session_state: st.session_state.mode      = "bigquery"
if "active_tab" not in st.session_state: st.session_state.active_tab = "chat"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:2.2rem;'>🔍</div>
        <div style='font-size:1rem;font-weight:700;color:#a78bfa;'>BigQuery AI</div>
        <div style='font-size:0.7rem;opacity:0.6;'>Data Catalog Assistant</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">📂 Data source</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["🔵 BigQuery Public Dataset", "🟢 Upload my own CSV"],
                    index=0 if st.session_state.mode=="bigquery" else 1,
                    label_visibility="collapsed")
    st.session_state.mode = "bigquery" if "BigQuery" in mode else "csv"

    if st.session_state.mode == "csv":
        st.markdown('<div class="section-label">📤 Upload CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(io.StringIO(uploaded.read().decode("utf-8", errors="replace")))
                st.session_state.csv_df   = df
                st.session_state.csv_name = uploaded.name
                st.session_state.messages = [{"role":"assistant",
                    "content": f"✅ Loaded **{uploaded.name}** — {len(df):,} rows × {len(df.columns)} columns. Ask me anything about this data!",
                    "sql":None,"python":None,"df":None}]
                st.success(f"✅ {uploaded.name} loaded!")
            except Exception as e:
                st.error(f"Read failed: {e}")

        if st.session_state.csv_df is not None:
            df = st.session_state.csv_df
            st.markdown(f"""<div class="stat-row">
                <div class="stat-card card-teal"><div class="val">{len(df):,}</div><div class="lbl">Rows</div></div>
                <div class="stat-card card-pink"><div class="val">{len(df.columns)}</div><div class="lbl">Cols</div></div>
            </div>""", unsafe_allow_html=True)
            chips = "".join(f'<span class="chip">⬡ {c}</span>' for c in df.columns)
            st.markdown(chips, unsafe_allow_html=True)
    else:
        bq_docs, bq_vec, bq_mat, bq_meta = load_bq_index()
        total_cols = sum(len(t["columns"]) for t in bq_meta["tables"])
        st.markdown(f"""<div class="stat-row">
            <div class="stat-card card-purple"><div class="val">{len(bq_docs)}</div><div class="lbl">Tables</div></div>
            <div class="stat-card card-pink"><div class="val">{total_cols}</div><div class="lbl">Cols</div></div>
        </div>""", unsafe_allow_html=True)
        chips = "".join(f'<span class="chip">📋 {d["table_name"]}</span>' for d in bq_docs)
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.78rem;color:#a78bfa;font-weight:600;'>Llama 3.3 70B · Groq</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role":"assistant","content":"👋 Chat cleared!","sql":None,"python":None,"df":None}]
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
badge = '<span class="mode-badge mode-bq">🔵 BigQuery Mode</span>' if st.session_state.mode=="bigquery" else '<span class="mode-badge mode-custom">🟢 CSV Mode</span>'
st.markdown(f"""<div class="hero">
    <h1>🔍 BigQuery AI Data Catalog</h1>
    <p>Ask anything about your data — explain tables, generate SQL, run queries, and explore dashboards instantly.</p>
    {badge}
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_dashboard = st.tabs(["💬 Chat", "📊 Dashboard"])

# ══════════════════════════════════════════════════════════════════════════════
# CHAT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    def render_message(idx, msg):
        with st.chat_message(msg["role"]):
            code = msg.get("sql") or msg.get("python")
            lang = "sql" if msg.get("sql") else "python"

            # Always show text — strip code block for display but fall back to full content
            if code:
                stripped = strip_code_blocks(msg["content"]).strip()
                st.markdown(stripped if stripped else "Here's the code:")
                st.code(code, language=lang)
                if msg.get("df") is not None:
                    result = msg["df"]
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.markdown(f'<div class="result-banner">✅ {len(result)} rows returned</div>', unsafe_allow_html=True)
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.markdown(f'<div class="result-banner">✅ Result: **{result}**</div>', unsafe_allow_html=True)
                else:
                    btn_label = "▶ Run in BigQuery" if msg.get("sql") else "▶ Run on my data"
                    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
                    if st.button(btn_label, key=f"run_{idx}"):
                        with st.spinner("⚡ Running..."):
                            try:
                                if msg.get("sql"):
                                    df_res, _ = run_bq_query(msg["sql"])
                                    msg["df"] = df_res
                                else:
                                    msg["df"] = run_pandas_code(msg["python"], st.session_state.csv_df)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Pure text answer (schema questions, descriptions, etc.)
                st.markdown(msg["content"])

    for i, msg in enumerate(st.session_state.messages):
        render_message(i, msg)

    # Suggestions — dynamic based on mode and uploaded columns
    st.markdown('<div class="section-label" style="margin-top:1rem;">✨ Try these</div>', unsafe_allow_html=True)
    if st.session_state.mode == "bigquery":
        suggestions = [
            ("📋 List tables",      "List all tables and what they contain"),
            ("📊 Revenue by month", "Generate SQL for total revenue by month"),
            ("👥 Top customers",    "Top 10 customers by total spend SQL"),
            ("🛍️ Best products",    "Best selling products by revenue SQL"),
            ("🔎 Orders columns",   "What columns are in the orders table?"),
        ]
    elif st.session_state.csv_df is not None:
        df_cols   = st.session_state.csv_df.columns.tolist()
        num_cols  = st.session_state.csv_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols  = st.session_state.csv_df.select_dtypes(include=["object","category"]).columns.tolist()
        first_col = df_cols[0]
        # Build dynamic suggestions based on actual columns
        suggestions = [("📋 Describe data", "Describe this dataset and explain each column")]
        if num_cols:
            suggestions.append(("📊 Summary stats", f"Show summary statistics for {', '.join(num_cols[:3])}"))
            suggestions.append(("📈 Distribution",  f"Show the distribution of {num_cols[0]}"))
        if cat_cols:
            suggestions.append(("🏷️ Top categories", f"What are the top 10 values in the {cat_cols[0]} column?"))
        suggestions.append(("💡 Key insights", "What are the top 3 interesting insights from this data?"))
        suggestions = suggestions[:5]   # max 5
    else:
        suggestions = [("📤 Upload a CSV", "Please upload a CSV file in the sidebar to get started")]

    clicked = None
    cols    = st.columns(len(suggestions))
    for col, (label, prompt_text) in zip(cols, suggestions):
        if col.button(label, use_container_width=True):
            clicked = prompt_text

    prompt = st.chat_input("💬 Ask about your data...") or clicked

    if prompt:
        if st.session_state.mode == "csv" and st.session_state.csv_df is None:
            st.warning("⚠️ Please upload a CSV file first using the sidebar.")
        else:
            st.session_state.messages.append({"role":"user","content":prompt,"sql":None,"python":None,"df":None})
            with st.spinner("🧠 Thinking..."):
                if st.session_state.mode == "bigquery":
                    bq_docs, bq_vec, bq_mat, _ = load_bq_index()
                    ctx    = retrieve(prompt, bq_docs, bq_vec, bq_mat)
                    answer = generate(build_bq_prompt(prompt, ctx))
                    sql    = extract_sql(answer)
                    st.session_state.messages.append({"role":"assistant","content":answer,"sql":sql,"python":None,"df":None})
                else:
                    df = st.session_state.csv_df
                    _, csv_vec, csv_mat, schema_text = build_csv_index(df, st.session_state.csv_name)
                    answer = generate(build_csv_prompt(prompt, schema_text, df))
                    py     = extract_python(answer)
                    st.session_state.messages.append({"role":"assistant","content":answer,"sql":None,"python":py,"df":None})
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    if st.session_state.mode == "csv":
        if st.session_state.csv_df is None:
            st.info("📤 Upload a CSV file in the sidebar to see your dashboard.")
        else:
            render_dashboard(st.session_state.csv_df, st.session_state.csv_name)

    else:
        st.markdown('<div class="tab-header">📊 thelook_ecommerce Live Dashboard</div>', unsafe_allow_html=True)
        st.markdown("Click any card to load live data directly from BigQuery.")

        dash_col1, dash_col2 = st.columns(2)

        with dash_col1:
            if st.button("📦 Order Status Breakdown", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT status, COUNT(*) as total_orders
                            FROM `{BQ_DATASET}.orders`
                            GROUP BY status
                            ORDER BY total_orders DESC
                        """)
                        st.markdown("**Order Status Breakdown**")
                        st.bar_chart(df.set_index("status")["total_orders"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

            if st.button("🛍️ Top 10 Products by Revenue", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT p.name, ROUND(SUM(oi.sale_price),2) as revenue
                            FROM `{BQ_DATASET}.order_items` oi
                            JOIN `{BQ_DATASET}.products` p ON oi.product_id = p.id
                            WHERE oi.status NOT IN ('Cancelled','Returned')
                            GROUP BY p.name
                            ORDER BY revenue DESC
                            LIMIT 10
                        """)
                        st.markdown("**Top 10 Products by Revenue**")
                        st.bar_chart(df.set_index("name")["revenue"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

            if st.button("🏷️ Revenue by Product Category", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT p.category, ROUND(SUM(oi.sale_price),2) as revenue
                            FROM `{BQ_DATASET}.order_items` oi
                            JOIN `{BQ_DATASET}.products` p ON oi.product_id = p.id
                            WHERE oi.status NOT IN ('Cancelled','Returned')
                            GROUP BY p.category
                            ORDER BY revenue DESC
                        """)
                        st.markdown("**Revenue by Product Category**")
                        st.bar_chart(df.set_index("category")["revenue"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

        with dash_col2:
            if st.button("📈 Monthly Revenue Trend", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT
                                FORMAT_TIMESTAMP('%Y-%m', created_at) as month,
                                ROUND(SUM(sale_price),2) as revenue
                            FROM `{BQ_DATASET}.order_items`
                            WHERE status NOT IN ('Cancelled','Returned')
                            GROUP BY month
                            ORDER BY month
                        """)
                        st.markdown("**Monthly Revenue Trend**")
                        st.line_chart(df.set_index("month")["revenue"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

            if st.button("👥 Customers by Country", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT country, COUNT(*) as customers
                            FROM `{BQ_DATASET}.users`
                            GROUP BY country
                            ORDER BY customers DESC
                            LIMIT 15
                        """)
                        st.markdown("**Top 15 Countries by Customers**")
                        st.bar_chart(df.set_index("country")["customers"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

            if st.button("👫 Revenue by Gender", use_container_width=True):
                with st.spinner("Querying BigQuery..."):
                    try:
                        df, _ = run_bq_query(f"""
                            SELECT o.gender, ROUND(SUM(oi.sale_price),2) as revenue
                            FROM `{BQ_DATASET}.order_items` oi
                            JOIN `{BQ_DATASET}.orders` o ON oi.order_id = o.order_id
                            WHERE oi.status NOT IN ('Cancelled','Returned')
                            GROUP BY o.gender
                            ORDER BY revenue DESC
                        """)
                        st.markdown("**Revenue by Gender**")
                        st.bar_chart(df.set_index("gender")["revenue"])
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

        st.divider()
        if st.button("🚀 Load Full Overview — all 6 charts", use_container_width=True):
            queries = [
                ("📦 Order Status", f"SELECT status, COUNT(*) as count FROM `{BQ_DATASET}.orders` GROUP BY status ORDER BY count DESC", "status", "count", "bar"),
                ("📈 Monthly Revenue", f"SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) as month, ROUND(SUM(sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` WHERE status NOT IN ('Cancelled','Returned') GROUP BY month ORDER BY month", "month", "revenue", "line"),
                ("🛍️ Top Products", f"SELECT p.name, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id = p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.name ORDER BY revenue DESC LIMIT 10", "name", "revenue", "bar"),
                ("👥 By Country", f"SELECT country, COUNT(*) as customers FROM `{BQ_DATASET}.users` GROUP BY country ORDER BY customers DESC LIMIT 15", "country", "customers", "bar"),
                ("🏷️ By Category", f"SELECT p.category, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id = p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.category ORDER BY revenue DESC", "category", "revenue", "bar"),
                ("👫 By Gender", f"SELECT o.gender, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.orders` o ON oi.order_id = o.order_id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY o.gender ORDER BY revenue DESC", "gender", "revenue", "bar"),
            ]
            c1, c2 = st.columns(2)
            containers = [c1, c2] * 3
            for (title, sql, idx_col, val_col, chart_type), container in zip(queries, containers):
                with container:
                    with st.spinner(f"Loading {title}..."):
                        try:
                            df, _ = run_bq_query(sql)
                            st.markdown(f"**{title}**")
                            if chart_type == "bar":
                                st.bar_chart(df.set_index(idx_col)[val_col])
                            else:
                                st.line_chart(df.set_index(idx_col)[val_col])
                        except Exception as e:
                            st.error(f"{title} failed: {e}")
