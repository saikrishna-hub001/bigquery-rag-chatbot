"""
BigQuery AI Data Catalog — Groq + TF-IDF RAG
Supports BigQuery public dataset + custom CSV upload
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
.hero p{font-size:0.9rem;margin:0;opacity:0.9;color:white;}
.mode-badge{display:inline-block;border-radius:20px;padding:4px 14px;font-size:0.78rem;font-weight:600;margin-top:0.5rem;}
.mode-bq{background:rgba(255,255,255,0.25);color:white;}
.mode-csv{background:rgba(56,239,125,0.35);color:white;}
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
.run-btn>button{background:linear-gradient(135deg,#11998e,#38ef7d)!important;color:white!important;border-radius:20px!important;font-weight:600!important;border:none!important;box-shadow:0 3px 10px rgba(17,153,142,0.3)!important;}
.result-banner{background:linear-gradient(135deg,#11998e22,#38ef7d22);border:1px solid #11998e55;border-radius:10px;padding:8px 14px;color:#11998e;font-weight:600;font-size:0.85rem;margin-bottom:8px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e,#16213e,#0f3460)!important;}
[data-testid="stSidebar"] *{color:#e0e0e0!important;}
[data-testid="stSidebar"] .chip{color:#a78bfa!important;border-color:#a78bfa55!important;background:#a78bfa11!important;}
.section-label{font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#888;margin:0.8rem 0 0.3rem;}
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:8px;background:transparent;border-bottom:2px solid #e0e0ff;padding-bottom:4px;}
[data-testid="stTabs"] [data-baseweb="tab"]{background:linear-gradient(135deg,#667eea22,#764ba222)!important;border-radius:12px 12px 0 0!important;border:1px solid #667eea44!important;border-bottom:none!important;padding:12px 28px!important;font-size:1rem!important;font-weight:600!important;color:#667eea!important;transition:all 0.2s!important;}
[data-testid="stTabs"] [aria-selected="true"]{background:linear-gradient(135deg,#667eea,#764ba2)!important;color:white!important;box-shadow:0 4px 12px rgba(102,126,234,0.3)!important;}
[data-testid="stTabs"] [data-baseweb="tab"]:hover{background:linear-gradient(135deg,#667eea44,#764ba244)!important;transform:translateY(-2px)!important;}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
BQ_DATASET = "bigquery-public-data.thelook_ecommerce"
GEN_MODEL  = "llama-3.3-70b-versatile"
MAX_ROWS   = 100
MAX_BYTES  = 1024 * 1024 * 1024

# ── Groq ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def generate(prompt: str) -> str:
    # Create fresh client every time — cached client can go stale
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # smaller/faster model, less likely to timeout
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                return f"⚠️ Groq API error after 3 attempts: {str(e)}"
            import time
            time.sleep(3)

# ── BigQuery ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_bq_client():
    creds = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return bigquery.Client(credentials=credentials, project=creds["project_id"])

@st.cache_resource
def load_bq_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, "data", "metadata.json")) as f:
        meta = json.load(f)
    docs = []
    for t in meta["tables"]:
        col_text = ", ".join(f"{c['name']} ({c['type']})" for c in t["columns"])
        docs.append({"table_name": t["table_name"],
                     "text": f"Table: {t['table_name']}\n{t['description']}\nColumns: {col_text}"})
    vec = TfidfVectorizer(stop_words="english")
    mat = vec.fit_transform([d["text"] for d in docs])
    return docs, vec, mat, meta

def retrieve(query, docs, vec, mat, top_k=3):
    sims = cosine_similarity(vec.transform([query]), mat).flatten()
    return [docs[i] for i in np.argsort(sims)[::-1][:top_k]]

def enforce_limit(sql):
    return sql if re.search(r"\blimit\b", sql, re.IGNORECASE) else sql.rstrip(";") + f"\nLIMIT {MAX_ROWS}"

def run_bq_query(sql):
    safe = enforce_limit(sql)
    cfg  = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES, use_query_cache=True)
    return get_bq_client().query(safe, job_config=cfg).result().to_dataframe(), safe

def run_pandas_code(code, df):
    import builtins
    ns = {"df": df.copy(), "pd": pd, "np": np}
    exec(compile(code, "<llm>", "exec"), {"__builtins__": builtins}, ns)
    if "result" in ns:
        return ns["result"]
    for k, v in ns.items():
        if k != "df" and isinstance(v, (pd.DataFrame, pd.Series)):
            return v
    return None

def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else None

def extract_python(text):
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL|re.IGNORECASE)
    return m.group(1).strip() if m else None

def strip_code_blocks(text):
    text = re.sub(r"```sql\s*.*?```", "", text, flags=re.DOTALL|re.IGNORECASE)
    text = re.sub(r"```python\s*.*?```", "", text, flags=re.DOTALL|re.IGNORECASE)
    return text.strip()

def build_bq_prompt(query, ctx_docs):
    context = "\n\n".join(d["text"] for d in ctx_docs)
    return f"""You are a BigQuery assistant for `{BQ_DATASET}`. Use the schema below. Wrap SQL in ```sql blocks. Be concise.

SCHEMA:
{context}

QUESTION: {query}"""

def build_csv_prompt(query, df, filename):
    # Keep schema very short to avoid Groq timeouts
    cols = []
    for col in list(df.columns)[:15]:
        try:
            sample = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else "N/A"
            cols.append(f"{col} ({df[col].dtype}): e.g. {sample[:30]}")
        except Exception:
            cols.append(col)
    schema = "\n".join(cols)
    return f"""Dataset: "{filename}", {len(df)} rows, {len(df.columns)} columns.
Columns: {schema}
Question: {query}
Answer briefly and helpfully in plain English."""

# ── Smart dashboard charts ────────────────────────────────────────────────────
def smart_chart(df_chart: pd.DataFrame, title: str, chart_hint: str = "auto"):
    """Render the best chart type based on data shape."""
    st.markdown(f"**{title}**")
    if df_chart.empty:
        st.info("No data returned.")
        return
    num_rows = len(df_chart)
    val_col  = df_chart.columns[-1]
    idx_col  = df_chart.columns[0]
    # Pie: small number of categories (≤8), use st.plotly_chart via plotly express if available
    # fallback to bar since we don't have plotly. Use metric cards for 1-2 rows.
    if num_rows == 1:
        st.metric(label=str(df_chart[idx_col].iloc[0]), value=f"{df_chart[val_col].iloc[0]:,.2f}" if isinstance(df_chart[val_col].iloc[0], float) else df_chart[val_col].iloc[0])
    elif num_rows <= 2:
        c1, c2 = st.columns(num_rows)
        for i, (_, row) in enumerate(df_chart.iterrows()):
            [c1, c2][i].metric(label=str(row[idx_col]), value=f"{row[val_col]:,.0f}")
    elif chart_hint == "line" or "month" in idx_col.lower() or "year" in idx_col.lower() or "date" in idx_col.lower():
        st.line_chart(df_chart.set_index(idx_col)[val_col])
    elif num_rows <= 8:
        # Simulate pie with bar + percentage annotation
        total = df_chart[val_col].sum()
        df_chart = df_chart.copy()
        df_chart["pct"] = (df_chart[val_col] / total * 100).round(1)
        st.bar_chart(df_chart.set_index(idx_col)[val_col])
        # Show percentage table alongside
        st.dataframe(df_chart[[idx_col, val_col, "pct"]].rename(columns={"pct": "% share"}), use_container_width=True, hide_index=True)
        return   # already showed table, skip below
    else:
        st.bar_chart(df_chart.set_index(idx_col)[val_col])
    st.dataframe(df_chart, use_container_width=True, hide_index=True)

def render_csv_dashboard(df, title):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    st.markdown(f"### 📊 {title}")
    st.markdown(f"""<div class="stat-row">
        <div class="stat-card card-purple"><div class="val">{len(df):,}</div><div class="lbl">Rows</div></div>
        <div class="stat-card card-pink"><div class="val">{len(df.columns)}</div><div class="lbl">Columns</div></div>
        <div class="stat-card card-teal"><div class="val">{len(num_cols)}</div><div class="lbl">Numeric</div></div>
        <div class="stat-card card-orange"><div class="val">{int(df.isnull().sum().sum())}</div><div class="lbl">Nulls</div></div>
    </div>""", unsafe_allow_html=True)

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.markdown("**🔴 Missing values**")
        smart_chart(missing.reset_index().rename(columns={"index":"column", 0:"nulls"}), "Missing values per column")

    # Numeric distributions
    if num_cols:
        st.markdown("**📈 Numeric distributions**")
        cols3 = st.columns(min(3, len(num_cols)))
        for i, col in enumerate(num_cols[:6]):
            with cols3[i % 3]:
                if df[col].nunique() <= 15:
                    smart_chart(df[col].value_counts().reset_index().rename(columns={"index": col, col: "count"}), col)
                else:
                    st.markdown(f"**{col}**")
                    st.bar_chart(df[col].dropna().describe().to_frame())

    # Categorical top values — use pie-style for small cardinality
    if cat_cols:
        st.markdown("**🏷️ Categorical columns**")
        cols2 = st.columns(min(2, len(cat_cols)))
        for i, col in enumerate(cat_cols[:4]):
            with cols2[i % 2]:
                top = df[col].value_counts().head(8).reset_index()
                top.columns = [col, "count"]
                smart_chart(top, f"{col} — top values")

    # Correlation
    if len(num_cols) >= 2:
        st.markdown("**🔗 Numeric correlation**")
        corr = df[num_cols].corr().round(2)
        st.dataframe(corr, use_container_width=True)

    st.markdown("**🔍 Data preview**")
    st.dataframe(df.head(20), use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"👋 Hi! Ask me about the BigQuery dataset, or upload your own CSV in the sidebar!","sql":None,"python":None,"df":None}]
if "csv_df"   not in st.session_state: st.session_state.csv_df   = None
if "csv_name" not in st.session_state: st.session_state.csv_name = None
if "mode"     not in st.session_state: st.session_state.mode     = "bigquery"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0 0.5rem;'>
        <div style='font-size:2.2rem;'>🔍</div>
        <div style='font-size:1rem;font-weight:700;color:#a78bfa;'>BigQuery AI</div>
        <div style='font-size:0.7rem;opacity:0.6;'>Data Catalog Assistant</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">📂 Data source</div>', unsafe_allow_html=True)
    mode_choice = st.radio("Mode", ["🔵 BigQuery Public Dataset", "🟢 Upload my own CSV"],
                           index=0 if st.session_state.mode == "bigquery" else 1,
                           label_visibility="collapsed")
    new_mode = "bigquery" if "BigQuery" in mode_choice else "csv"
    if new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.messages = [{"role":"assistant","content":"👋 Mode switched! Ask me anything.","sql":None,"python":None,"df":None}]

    if st.session_state.mode == "csv":
        st.markdown('<div class="section-label">📤 Upload CSV</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                raw = uploaded.read()
                df_up = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")))
                st.session_state.csv_df   = df_up
                st.session_state.csv_name = uploaded.name
                st.session_state.messages = [{"role":"assistant",
                    "content": f"✅ Loaded **{uploaded.name}** — {len(df_up):,} rows × {len(df_up.columns)} columns. Ask me anything about this data!",
                    "sql":None,"python":None,"df":None}]
                st.success(f"✅ {uploaded.name} loaded!")
            except Exception as e:
                st.error(f"Upload failed: {e}")

        if st.session_state.csv_df is not None:
            df_s = st.session_state.csv_df
            st.markdown(f"""<div class="stat-row">
                <div class="stat-card card-teal"><div class="val">{len(df_s):,}</div><div class="lbl">Rows</div></div>
                <div class="stat-card card-pink"><div class="val">{len(df_s.columns)}</div><div class="lbl">Cols</div></div>
            </div>""", unsafe_allow_html=True)
            chips = "".join(f'<span class="chip">⬡ {c}</span>' for c in df_s.columns[:15])
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
badge_cls  = "mode-bq" if st.session_state.mode == "bigquery" else "mode-csv"
badge_text = "🔵 BigQuery Mode" if st.session_state.mode == "bigquery" else "🟢 CSV Mode"
st.markdown(f"""<div class="hero">
    <h1>🔍 BigQuery AI Data Catalog</h1>
    <p>Ask anything about your data — tables, columns, SQL queries, and live results.</p>
    <span class="mode-badge {badge_cls}">{badge_text}</span>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_dash = st.tabs(["💬  Chat", "📊  Dashboard"])

# ══════════════ CHAT TAB ══════════════════════════════════════════════════════
with tab_chat:

    # Suggestion chips
    if st.session_state.mode == "bigquery":
        suggs = [
            ("📋 List tables",      "List all tables and what they contain"),
            ("📊 Revenue by month", "Generate SQL for total revenue by month"),
            ("👥 Top customers",    "Top 10 customers by total spend SQL"),
            ("🛍️ Best products",    "Best selling products by revenue SQL"),
            ("🔎 Orders columns",   "What columns are in the orders table?"),
        ]
    elif st.session_state.csv_df is not None:
        _nc = st.session_state.csv_df.select_dtypes(include=np.number).columns.tolist()
        _cc = st.session_state.csv_df.select_dtypes(include=["object","category"]).columns.tolist()
        suggs = [("📋 Describe data", "Describe this dataset and explain each column")]
        if _nc:
            suggs.append(("📊 Summary stats", "Give me summary statistics for the numeric columns"))
        if _cc:
            suggs.append(("🏷️ Top categories", f"What are the most common values in the {_cc[0]} column?"))
        suggs.append(("💡 Key insights", "What are the top 3 interesting insights from this data?"))
        suggs.append(("🔎 Missing values", "Which columns have missing values and how many?"))
        suggs = suggs[:5]
    else:
        suggs = [("📤 Upload a CSV", "Please upload a CSV file in the sidebar")]

    st.markdown('<div class="section-label">✨ Try these</div>', unsafe_allow_html=True)
    clicked = None
    scols = st.columns(len(suggs))
    for sc, (label, ptxt) in zip(scols, suggs):
        if sc.button(label, use_container_width=True):
            clicked = ptxt

    # Scrollable message container
    chat_container = st.container(height=480)
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                has_sql = bool(msg.get("sql"))
                code    = msg.get("sql") or msg.get("python")
                lang    = "sql" if has_sql else "python"
                if code:
                    stripped = strip_code_blocks(msg["content"]).strip()
                    if stripped:
                        st.markdown(stripped)
                    st.code(code, language=lang)
                    if msg.get("df") is not None:
                        result = msg["df"]
                        if isinstance(result, (pd.DataFrame, pd.Series)):
                            st.success(f"✅ {len(result)} rows returned")
                            st.dataframe(result, use_container_width=True)
                        else:
                            st.success(f"✅ Result: {result}")
                    elif has_sql:
                        def make_runner(m):
                            def run_it():
                                try:
                                    res, _ = run_bq_query(m["sql"])
                                    m["df"] = res
                                except Exception as e:
                                    st.session_state["run_error"] = str(e)
                            return run_it
                        st.button("▶ Run in BigQuery", key=f"run_{idx}", on_click=make_runner(msg))
                else:
                    st.markdown(msg["content"])

    # Chat input — always at bottom below the container
    prompt = st.chat_input("💬 Ask about your data...") or clicked

    if prompt:
        if st.session_state.mode == "csv" and st.session_state.csv_df is None:
            st.warning("⚠️ Upload a CSV file first.")
        else:
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    placeholder.markdown("🧠 Thinking...")
                    try:
                        if st.session_state.mode == "bigquery":
                            docs, vec, mat, _ = load_bq_index()
                            ctx    = retrieve(prompt, docs, vec, mat)
                            answer = generate(build_bq_prompt(prompt, ctx))
                            sql    = extract_sql(answer)
                            py     = None
                        else:
                            answer = generate(build_csv_prompt(prompt, st.session_state.csv_df, st.session_state.csv_name))
                            sql    = None
                            py     = None
                    except Exception as e:
                        answer = f"⚠️ Error: {str(e)}"
                        sql    = None
                        py     = None

                    placeholder.empty()
                    if sql:
                        stripped = strip_code_blocks(answer).strip()
                        if stripped:
                            st.markdown(stripped)
                        st.code(sql, language="sql")
                        def run_new_query():
                            try:
                                res, _ = run_bq_query(sql)
                                st.session_state.messages[-1]["df"] = res
                            except Exception as e:
                                st.session_state["run_error"] = str(e)
                        st.button("▶ Run in BigQuery", key="run_new", on_click=run_new_query)
                    else:
                        st.markdown(answer)

            st.session_state.messages.append({"role":"user",      "content":prompt, "sql":None, "python":None, "df":None})
            st.session_state.messages.append({"role":"assistant", "content":answer, "sql":sql,  "python":py,   "df":None})

    # Show any run errors
    if st.session_state.get("run_error"):
        st.error(f"Query failed: {st.session_state.pop('run_error')}")

# ══════════════ DASHBOARD TAB ═════════════════════════════════════════════════
with tab_dash:
    if st.session_state.mode == "csv":
        if st.session_state.csv_df is None:
            st.info("📤 Upload a CSV file in the sidebar to see your automatic dashboard.")
        else:
            render_csv_dashboard(st.session_state.csv_df, st.session_state.csv_name)
    else:
        st.markdown("### 📊 thelook_ecommerce Live Dashboard")
        st.markdown("Click any button to load live data from BigQuery.")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("📦 Order Status", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT status, COUNT(*) as orders FROM `{BQ_DATASET}.orders` GROUP BY status ORDER BY orders DESC")
                        smart_chart(df, "Order Status Breakdown")
                    except Exception as e: st.error(str(e))

            if st.button("🛍️ Top Products", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT p.name, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id=p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.name ORDER BY revenue DESC LIMIT 10")
                        smart_chart(df, "Top 10 Products by Revenue")
                    except Exception as e: st.error(str(e))

            if st.button("🏷️ Revenue by Category", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT p.category, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id=p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.category ORDER BY revenue DESC")
                        smart_chart(df, "Revenue by Category")
                    except Exception as e: st.error(str(e))

        with c2:
            if st.button("📈 Monthly Revenue", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) as month, ROUND(SUM(sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` WHERE status NOT IN ('Cancelled','Returned') GROUP BY month ORDER BY month")
                        smart_chart(df, "Monthly Revenue Trend", chart_hint="line")
                    except Exception as e: st.error(str(e))

            if st.button("👥 Customers by Country", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT country, COUNT(*) as customers FROM `{BQ_DATASET}.users` GROUP BY country ORDER BY customers DESC LIMIT 12")
                        smart_chart(df, "Customers by Country")
                    except Exception as e: st.error(str(e))

            if st.button("👫 Revenue by Gender", use_container_width=True):
                with st.spinner("Loading..."):
                    try:
                        df, _ = run_bq_query(f"SELECT o.gender, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.orders` o ON oi.order_id=o.order_id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY o.gender ORDER BY revenue DESC")
                        smart_chart(df, "Revenue by Gender")
                    except Exception as e: st.error(str(e))

        st.divider()
        if st.button("🚀 Load All 6 Charts", use_container_width=True):
            charts = [
                ("📦 Order Status",       f"SELECT status, COUNT(*) as orders FROM `{BQ_DATASET}.orders` GROUP BY status ORDER BY orders DESC", "auto"),
                ("📈 Monthly Revenue",    f"SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) as month, ROUND(SUM(sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` WHERE status NOT IN ('Cancelled','Returned') GROUP BY month ORDER BY month", "line"),
                ("🛍️ Top Products",       f"SELECT p.name, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id=p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.name ORDER BY revenue DESC LIMIT 10", "auto"),
                ("👥 By Country",         f"SELECT country, COUNT(*) as customers FROM `{BQ_DATASET}.users` GROUP BY country ORDER BY customers DESC LIMIT 12", "auto"),
                ("🏷️ By Category",        f"SELECT p.category, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.products` p ON oi.product_id=p.id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY p.category ORDER BY revenue DESC", "auto"),
                ("👫 By Gender",          f"SELECT o.gender, ROUND(SUM(oi.sale_price),2) as revenue FROM `{BQ_DATASET}.order_items` oi JOIN `{BQ_DATASET}.orders` o ON oi.order_id=o.order_id WHERE oi.status NOT IN ('Cancelled','Returned') GROUP BY o.gender ORDER BY revenue DESC", "auto"),
            ]
            col_a, col_b = st.columns(2)
            conts = [col_a, col_b] * 3
            for (title, sql, hint), cont in zip(charts, conts):
                with cont:
                    with st.spinner(f"Loading {title}..."):
                        try:
                            df, _ = run_bq_query(sql)
                            smart_chart(df, title, chart_hint=hint)
                        except Exception as e:
                            st.error(f"{title}: {e}")
