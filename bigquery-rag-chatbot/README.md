# BigQuery RAG Data Catalog Assistant 🗂️

A chatbot that answers natural language questions about a BigQuery dataset —
"what table has customer orders?", "explain the products table", "generate
SQL for monthly revenue" — and can **safely execute the generated SQL** with
real results shown in the chat.

Built with **RAG** (Gemini embeddings + cosine similarity retrieval over
table/column metadata) and **Gemini 2.0 Flash** for generation. Deployed for
free on **Streamlit Community Cloud**.

**Cost: $0.** Dataset is `bigquery-public-data.thelook_ecommerce` (free public
data). Gemini API free tier covers embeddings + generation. Streamlit Cloud
hosting is free. Query execution is capped at 100 rows / 200MB so it stays
within BigQuery's free tier (1TB/month).

---

## Architecture

```
User question
     ↓
Gemini embeddings (retrieval_query)
     ↓
Cosine similarity vs. pre-embedded table/column metadata
     ↓
Top-3 relevant tables → context
     ↓
Gemini 2.0 Flash (generation) → answer + optional SQL
     ↓
[optional] "Run in BigQuery" → read-only service account
     ↓                          → LIMIT 100, max 200MB
Results table shown in chat
```

---

## Project structure

```
bigquery-rag-chatbot/
├── app.py                          # Streamlit app (main entry point)
├── requirements.txt
├── data/
│   └── metadata.json               # Pre-built table/column metadata
├── scripts/
│   └── extract_metadata.py         # Optional: regenerate metadata.json from live BigQuery
└── .streamlit/
    └── secrets.toml.example        # Template for required secrets
```

---

## Setup — Part 1: Get a Gemini API key 

1. Go to https://aistudio.google.com/app/apikey
2. Click **Create API Key**
3. Copy it — you'll need it in Part 3

---

## Setup — Part 2: Create a read-only GCP service account 

This lets the app run BigQuery queries on your behalf, safely.

1. Go to https://console.cloud.google.com
2. Select (or create) a project — e.g. `dq-monitor-499122` from before, or a new one
3. Enable the BigQuery API: search "BigQuery API" in the top search bar → **Enable**
4. Go to **IAM & Admin → Service Accounts → Create Service Account**
   - Name: `bq-rag-readonly`
   - Click **Create and Continue**
5. Grant these two roles (search and add each):
   - `BigQuery Data Viewer`
   - `BigQuery Job User`
   - Click **Done**
6. Click on the new service account → **Keys** tab → **Add Key → Create new key → JSON**
7. A `.json` file downloads — keep it safe, you'll paste its contents into Streamlit secrets next

> This service account can only **read** data and run jobs under your project's
> quota. It cannot modify, delete, or write anything — even if someone tries
> unusual prompts through the chatbot.

---

## Setup — Part 3: Push to GitHub

1. Create a new repo on GitHub, e.g. `bigquery-rag-chatbot`
2. Push this project folder:
```bash
cd bigquery-rag-chatbot
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/bigquery-rag-chatbot.git
git push -u origin main
```

> `.streamlit/secrets.toml` is git-ignored — never commit real secrets.

---

## Setup — Part 4: Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select your `bigquery-rag-chatbot` repo, branch `main`, file `app.py`
5. Click **Advanced settings → Secrets** and paste:

```toml
GEMINI_API_KEY = "paste-your-gemini-key-here"

[gcp_service_account]
type = "service_account"
project_id = "paste-from-json-file"
private_key_id = "paste-from-json-file"
private_key = "paste-from-json-file-including-BEGIN-END-lines"
client_email = "paste-from-json-file"
client_id = "paste-from-json-file"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "paste-from-json-file"
```

   Open the downloaded service account `.json` file and copy each field's
   value into the matching line above. For `private_key`, keep the `\n`
   characters as-is — paste it exactly as it appears in the JSON file
   (including quotes).

6. Click **Deploy**

In about 1-2 minutes you'll get a live URL like:
```
https://your-app-name.streamlit.app
```


## Tech stack

`Python` `Streamlit` `BigQuery` `Gemini API` `RAG` `Google Cloud`
