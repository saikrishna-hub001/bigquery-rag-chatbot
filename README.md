# 🔍 BigQuery AI Data Catalog Assistant

> An AI-powered chatbot that answers natural language questions about BigQuery datasets — explains tables, generates SQL, executes live queries, and auto-generates dashboards. Upload your own CSV to analyze any dataset instantly.

**🔗 Live Demo:** https://bigquery-rag.streamlit.app/

**Built by:** [Sai Krishna Pothana](https://github.com/saikrishna-hub001) · [Sandeep Reddy Gongati](https://github.com/sandeepgongati)

---

## 💡 The Problem

Data engineers and analysts waste hours answering the same questions:
- *"Which table has customer orders?"*
- *"What does this column mean?"*
- *"Can you write me a SQL query for monthly revenue?"*

This project solves that with a live AI assistant — ask in plain English, get an answer, generate SQL, and run it against real BigQuery data in one click.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 Natural language chat | Ask anything about tables, columns, or data relationships |
| 🔍 RAG retrieval | TF-IDF + cosine similarity over BigQuery metadata |
| ⚡ SQL generation | Llama 3.3 70B generates valid BigQuery SQL |
| ▶ Live execution | Run generated SQL against BigQuery with one click |
| 📊 Auto dashboard | Charts auto-generated based on data shape and column types |
| 📂 CSV upload | Upload your own CSV and ask questions about your data |
| 🔒 Safety guardrails | Read-only service account, auto LIMIT, 1GB byte cap |

---

## 🏗️ Architecture

```
User Question
      ↓
TF-IDF Retrieval (scikit-learn)
      ↓
Top-3 matching tables/columns → context
      ↓
Llama 3.3 70B via Groq API → answer + SQL
      ↓
[optional] BigQuery execution → live results
      ↓
Streamlit UI
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Retrieval (RAG) | TF-IDF + cosine similarity (scikit-learn) |
| Data warehouse | Google BigQuery (public dataset) |
| Query execution | GCP service account (read-only) |
| Frontend | Streamlit |
| Hosting | Streamlit Community Cloud |
| Language | Python 3.11 |

**Cost: $0** — entirely on free tiers (Groq free tier, BigQuery 1TB/month free, Streamlit Cloud free)

---

## 📁 Project Structure

```
bigquery-rag-chatbot/
├── app.py                        # Main Streamlit app
├── requirements.txt              # Python dependencies
├── data/
│   └── metadata.json            # Pre-built BigQuery schema metadata
└── scripts/
    └── extract_metadata.py      # Optional: regenerate metadata from live BigQuery
```

---

## Setup & Deployment

### Prerequisites
- GCP account (free tier)
- Groq API key (free): https://console.groq.com
- Streamlit Cloud account (free): https://share.streamlit.io

### Step 1 — Get a Groq API key
1. Go to https://console.groq.com
2. Sign in → API Keys → Create API Key
3. Copy the key (starts with `gsk_...`)

### Step 2 — Create a GCP Service Account
1. Go to https://console.cloud.google.com
2. IAM & Admin → Service Accounts → Create Service Account
3. Name: `bq-rag-readonly`
4. Grant roles: `BigQuery Data Viewer` + `BigQuery Job User`
5. Keys tab → Add Key → Create new key → JSON
6. Download the JSON file

### Step 3 — Deploy on Streamlit Cloud
1. Fork this repo to your GitHub
2. Go to https://share.streamlit.io → New app
3. Select your forked repo, branch `main`, file `bigquery-rag-chatbot/app.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
GROQ_API_KEY = "gsk_your_key_here"

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "bq-rag-readonly@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/bq-rag-readonly%40your-project.iam.gserviceaccount.com"
```

5. Click **Deploy** — live in ~2 minutes

---

## 🔒 Safety Guardrails

- Service account has **read-only** roles — cannot write, delete, or modify data
- Every query gets `LIMIT 100` automatically if missing
- `maximum_bytes_billed` capped at 1GB per query (well within free tier)
- RAG context restricted to indexed schema only

---

## 🤝 Contributors

| Name | GitHub | LinkedIn |
|---|---|---|
| Sai Krishna Pothana | [@saikrishna-hub001](https://github.com/saikrishna-hub001) | [linkedin.com/in/sai-krishna001](https://linkedin.com/in/sai-krishna001) |
| Sandeep Reddy Gongati | [@sandeepgongati](https://github.com/sandeepgongati) | [linkedin.com/in/sandeepgongati](https://www.linkedin.com/in/sandeepgongati/) |

---

## 📄 License

MIT License — free to use, modify, and distribute.
