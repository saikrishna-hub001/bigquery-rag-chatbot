"""
BigQuery RAG Data Catalog Assistant
A chatbot that answers natural language questions about a BigQuery dataset
using RAG (TF-IDF retrieval over table/column metadata) and Groq (Llama 3.3)
for generation, including SQL generation with optional safe execution.
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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Catalog Assistant", page_icon="🗂️", layout="centered")

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DATASET = "bigquery-public-data.thelook_ecommerce"
GEN_MODEL      = "llama-3.3-70b-versatile"
MAX_ROWS       = 100   # row limit enforced on any executed query
MAX_BYTES      = 200 * 1024 * 1024   # 200 MB cap per query - keeps cost at $0


# ── Setup: Groq ──────────────────────────────────────────────────────────────
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


# ── Setup: BigQuery (read-only service account) ─────────────────────────────
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])


# ── Load metadata + build TF-IDF index (cached once per session) ────────────
@st.cache_resource
def load_index():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(base_dir, "data", "metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    docs = []
    for table in metadata["tables"]:
        col_text = ", ".join(
            f"{c['name']} ({c['type']}): {c['description']}" for c in table["columns"]
        )
        doc = (
            f"Table: {table['table_name']}\n"
            f"Description: {table['description']}\n"
            f"Columns: {col_text}"
        )
        docs.append({"table_name": table["table_name"], "text": doc, "raw": table})

    # Build TF-IDF vectors over table/column metadata - no external API calls needed
    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform([d["text"] for d in docs])

    return docs, vectorizer, doc_vectors, metadata


def retrieve(query: str, docs, vectorizer, doc_vectors, top_k: int = 3):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, doc_vectors).flatten()
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [docs[i] for i in top_idx]


def build_prompt(query: str, context_docs) -> str:
    context = "\n\n".join(d["text"] for d in context_docs)
    return f"""You are a data catalog assistant for a BigQuery dataset called `{SOURCE_DATASET}`.

Use ONLY the table/column information below to answer. If the user asks for SQL,
write a valid BigQuery Standard SQL query using fully-qualified table names
(e.g. `{SOURCE_DATASET}.orders`). Wrap any SQL in a ```sql code block.
Keep explanations concise.

RELEVANT SCHEMA:
{context}

USER QUESTION:
{query}
"""


def extract_sql(text: str) -> str | None:
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def strip_sql_block(text: str) -> str:
    """Remove the ```sql ... ``` block from the answer text, since it's rendered separately."""
    return re.sub(r"```sql\s*.*?```", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def enforce_limit(sql: str) -> str:
    """Add a LIMIT clause if the query doesn't already have one."""
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip(";") + f"\nLIMIT {MAX_ROWS}"


def run_query_safely(sql: str):
    client = get_bq_client()
    safe_sql = enforce_limit(sql)
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=MAX_BYTES,
        use_query_cache=True,
    )
    job = client.query(safe_sql, job_config=job_config)
    return job.result().to_dataframe(), safe_sql


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🗂️ Data Catalog Assistant")
st.caption(f"Ask natural language questions about `{SOURCE_DATASET}` — powered by Groq (Llama 3.3) + RAG")

with st.sidebar:
    st.subheader("Connected dataset")
    st.code(SOURCE_DATASET, language=None)

    docs, vectorizer, doc_vectors, metadata = load_index()
    st.metric("Tables indexed", len(docs))
    total_cols = sum(len(t["columns"]) for t in metadata["tables"])
    st.metric("Columns embedded", total_cols)
    st.metric("Model", GEN_MODEL)

    st.divider()
    st.caption("Tables")
    for d in docs:
        st.markdown(f"`{d['table_name']}`")

    st.divider()
    st.caption(
        "Queries run with a read-only service account, capped at "
        f"{MAX_ROWS} rows and {MAX_BYTES // (1024*1024)} MB per query."
    )


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about tables, columns, or ask me to generate SQL for this dataset."}
    ]

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about tables, columns, or ask me to generate SQL for this dataset.", "sql": None, "df": None}
    ]


def render_message(idx: int, msg: dict):
    with st.chat_message(msg["role"]):
        display_text = strip_sql_block(msg["content"]) if msg.get("sql") else msg["content"]
        if display_text:
            st.markdown(display_text)

        if msg.get("sql"):
            st.code(msg["sql"], language="sql")

            if msg.get("df") is not None:
                st.success(f"Returned {len(msg['df'])} rows (limited to {MAX_ROWS})")
                st.dataframe(msg["df"], use_container_width=True)
            else:
                if st.button("▶ Run in BigQuery", key=f"run_{idx}"):
                    with st.spinner("Running query..."):
                        try:
                            df, _ = run_query_safely(msg["sql"])
                            msg["df"] = df
                            st.rerun()
                        except Exception as e:
                            st.error(f"Query failed: {e}")


for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg)


# Suggested prompts
cols = st.columns(3)
suggestions = [
    "List all tables",
    "What columns are in the orders table?",
    "Generate SQL for total revenue by month",
]
clicked = None
for col, s in zip(cols, suggestions):
    if col.button(s, use_container_width=True):
        clicked = s


prompt = st.chat_input("Ask about your BigQuery tables...") or clicked

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sql": None, "df": None})

    with st.spinner("Thinking..."):
        context_docs = retrieve(prompt, docs, vectorizer, doc_vectors)
        full_prompt = build_prompt(prompt, context_docs)
        answer = generate(full_prompt)

    sql = extract_sql(answer)
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sql": sql,
        "df": None,
    })
    st.rerun()
