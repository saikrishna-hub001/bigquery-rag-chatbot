"""
BigQuery RAG Data Catalog Assistant
A chatbot that answers natural language questions about a BigQuery dataset
using RAG (Gemini embeddings + cosine similarity retrieval) and Gemini for
generation, including SQL generation with optional safe execution.
"""

import json
import os
import re
import numpy as np
import streamlit as st
import google.generativeai as genai
from google.cloud import bigquery
from google.oauth2 import service_account

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Catalog Assistant", page_icon="🗂️", layout="centered")

# ── Config ────────────────────────────────────────────────────────────────────
SOURCE_DATASET = "bigquery-public-data.thelook_ecommerce"
EMBED_MODEL    = "models/embedding-001"
GEN_MODEL      = "gemini-2.0-flash"
MAX_ROWS       = 100   # row limit enforced on any executed query
MAX_BYTES      = 200 * 1024 * 1024   # 200 MB cap per query - keeps cost at $0


# ── Setup: Gemini ────────────────────────────────────────────────────────────
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gen_model = genai.GenerativeModel(GEN_MODEL)


# ── Setup: BigQuery (read-only service account) ─────────────────────────────
@st.cache_resource
def get_bq_client():
    creds_dict = dict(st.secrets["gcp_service_account"])
    credentials = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    return bigquery.Client(credentials=credentials, project=creds_dict["project_id"])


# ── Load metadata + build embeddings (cached once per session) ──────────────
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

    # Embed all table docs in one batch call
    embeddings = []
    for d in docs:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=d["text"],
            task_type="retrieval_document",
        )
        embeddings.append(result["embedding"])

    return docs, np.array(embeddings), metadata


def retrieve(query: str, docs, embeddings, top_k: int = 3):
    q_emb = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_query",
    )["embedding"]
    q_emb = np.array(q_emb)

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb)
    sims = embeddings @ q_emb / norms

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
st.caption(f"Ask natural language questions about `{SOURCE_DATASET}` — powered by Gemini + RAG")

with st.sidebar:
    st.subheader("Connected dataset")
    st.code(SOURCE_DATASET, language=None)

    docs, embeddings, metadata = load_index()
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            st.code(msg["sql"], language="sql")
            if msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)


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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_docs = retrieve(prompt, docs, embeddings)
            full_prompt = build_prompt(prompt, context_docs)
            response = gen_model.generate_content(full_prompt)
            answer = response.text

        sql = extract_sql(answer)
        st.markdown(answer)

        df = None
        if sql:
            st.code(sql, language="sql")
            if st.button("▶ Run in BigQuery", key=f"run_{len(st.session_state.messages)}"):
                with st.spinner("Running query..."):
                    try:
                        df, executed_sql = run_query_safely(sql)
                        st.success(f"Returned {len(df)} rows (limited to {MAX_ROWS})")
                        st.dataframe(df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Query failed: {e}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "df": df,
        })

