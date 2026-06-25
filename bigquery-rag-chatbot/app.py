"""
BigQuery RAG Data Catalog Assistant
Colorful, LinkedIn-ready UI with Groq (Llama 3.3) + TF-IDF RAG + BigQuery live execution.
"""

import json
import os
import re
import numpy as np
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
/* ── Import font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Gradient hero header ── */
.hero {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 40%, #f093fb 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(102,126,234,0.35);
}
.hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem; color: white; }
.hero p  { font-size: 1rem; margin: 0; opacity: 0.9; color: white; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 1.2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 90px;
    border-radius: 12px;
    padding: 14px 16px;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.stat-card .val { font-size: 1.6rem; line-height: 1; }
.stat-card .lbl { font-size: 0.7rem; opacity: 0.85; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.card-purple { background: linear-gradient(135deg,#667eea,#764ba2); }
.card-pink   { background: linear-gradient(135deg,#f093fb,#f5576c); }
.card-teal   { background: linear-gradient(135deg,#4facfe,#00f2fe); }

/* ── Table chips in sidebar ── */
.chip {
    display: inline-block;
    background: linear-gradient(135deg,#667eea22,#764ba222);
    color: #667eea;
    border: 1px solid #667eea55;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 3px 2px;
}

/* ── Suggestion buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 3px 10px rgba(102,126,234,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 16px rgba(102,126,234,0.45) !important;
}

/* ── Run button ── */
.run-btn > button {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    color: white !important;
    border-radius: 20px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 3px 10px rgba(17,153,142,0.3) !important;
}

/* ── Chat bubbles ── */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    margin-bottom: 8px !important;
}

/* ── Success banner ── */
.result-banner {
    background: linear-gradient(135deg,#11998e22,#38ef7d22);
    border: 1px solid #11998e55;
    border-radius: 10px;
    padding: 8px 14px;
    color: #11998e;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 8px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%) !important;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .chip { color: #a78bfa !important; border-color: #a78bfa55 !important; background: #a78bfa11 !important; }

/* ── Section labels ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin: 1rem 0 0.4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DATASET = "bigquery-public-data.thelook_ecommerce"
GEN_MODEL      = "llama-3.3-70b-versatile"
MAX_ROWS       = 100
MAX_BYTES      = 200 * 1024 * 1024


# ── Groq ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

groq_client = get_groq_client()

def generate(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ── BigQuery ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])


# ── RAG index ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_index():
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

    vectorizer  = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform([d["text"] for d in docs])
    return docs, vectorizer, doc_vectors, metadata


def retrieve(query, docs, vectorizer, doc_vectors, top_k=3):
    sims    = cosine_similarity(vectorizer.transform([query]), doc_vectors).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [docs[i] for i in top_idx]


def build_prompt(query, context_docs):
    context = "\n\n".join(d["text"] for d in context_docs)
    return f"""You are a data catalog assistant for BigQuery dataset `{SOURCE_DATASET}`.
Use ONLY the schema below. For SQL, use fully-qualified table names and wrap in ```sql blocks.
Be concise.

SCHEMA:
{context}

QUESTION: {query}"""


def extract_sql(text):
    m = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None


def strip_sql_block(text):
    return re.sub(r"```sql\s*.*?```", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def enforce_limit(sql):
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip(";") + f"\nLIMIT {MAX_ROWS}"


def run_query_safely(sql):
    safe_sql   = enforce_limit(sql)
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=MAX_BYTES, use_query_cache=True)
    job        = get_bq_client().query(safe_sql, job_config=job_config)
    return job.result().to_dataframe(), safe_sql


# ── Load index ────────────────────────────────────────────────────────────────
docs, vectorizer, doc_vectors, metadata = load_index()
total_cols = sum(len(t["columns"]) for t in metadata["tables"])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-size:2.5rem;'>🔍</div>
        <div style='font-size:1.1rem; font-weight:700; color:#a78bfa;'>BigQuery AI</div>
        <div style='font-size:0.75rem; opacity:0.6; margin-top:2px;'>Data Catalog Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card card-purple">
            <div class="val">{len(docs)}</div>
            <div class="lbl">Tables</div>
        </div>
        <div class="stat-card card-pink">
            <div class="val">{total_cols}</div>
            <div class="lbl">Columns</div>
        </div>
        <div class="stat-card card-teal">
            <div class="val">RAG</div>
            <div class="lbl">Method</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📦 Dataset</div>', unsafe_allow_html=True)
    st.code(SOURCE_DATASET, language=None)

    st.markdown('<div class="section-label">🗄️ Tables</div>', unsafe_allow_html=True)
    chips = "".join(f'<span class="chip">📋 {d["table_name"]}</span>' for d in docs)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="section-label">🤖 Model</div>', unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.8rem; color:#a78bfa; font-weight:600;'>Llama 3.3 70B via Groq</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:0.7rem; opacity:0.5; line-height:1.5;'>🔒 Read-only · {MAX_ROWS} row cap · {MAX_BYTES//(1024*1024)}MB limit per query</div>",
        unsafe_allow_html=True
    )


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 BigQuery AI Data Catalog</h1>
    <p>Ask anything about your data in plain English — I'll explain tables, columns, and generate + run SQL instantly.</p>
</div>
""", unsafe_allow_html=True)


# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi! Ask me anything about this BigQuery dataset — I can explain tables, describe columns, or generate and run SQL queries for you!",
        "sql": None,
        "df": None,
    }]


def render_message(idx, msg):
    with st.chat_message(msg["role"]):
        display_text = strip_sql_block(msg["content"]) if msg.get("sql") else msg["content"]
        if display_text:
            st.markdown(display_text)

        if msg.get("sql"):
            st.code(msg["sql"], language="sql")

            if msg.get("df") is not None:
                st.markdown(
                    f'<div class="result-banner">✅ Returned {len(msg["df"])} rows (limited to {MAX_ROWS})</div>',
                    unsafe_allow_html=True
                )
                st.dataframe(msg["df"], use_container_width=True)
            else:
                with st.container():
                    st.markdown('<div class="run-btn">', unsafe_allow_html=True)
                    if st.button("▶ Run in BigQuery", key=f"run_{idx}"):
                        with st.spinner("⚡ Querying BigQuery..."):
                            try:
                                df, _ = run_query_safely(msg["sql"])
                                msg["df"] = df
                                st.rerun()
                            except Exception as e:
                                st.error(f"Query failed: {e}")
                    st.markdown('</div>', unsafe_allow_html=True)


for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg)


# ── Suggested prompts ─────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.5rem;">✨ Try these</div>', unsafe_allow_html=True)
suggestions = [
    "📋 List all tables",
    "🔎 What columns are in orders?",
    "📊 Revenue by month SQL",
    "👥 Top customers by spend",
    "🛍️ Best selling products",
]
clicked = None
cols = st.columns(len(suggestions))
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked = s.split(" ", 1)[1]   # strip the emoji prefix before sending as prompt


# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("💬 Ask about your BigQuery tables...") or clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sql": None, "df": None})

    with st.spinner("🧠 Thinking..."):
        context_docs = retrieve(prompt, docs, vectorizer, doc_vectors)
        answer       = generate(build_prompt(prompt, context_docs))

    sql = extract_sql(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sql": sql,
        "df": None,
    })
    st.rerun()
