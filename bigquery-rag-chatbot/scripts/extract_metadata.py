"""
Extract table and column metadata from a BigQuery dataset's INFORMATION_SCHEMA.

Usage:
    python scripts/extract_metadata.py YOUR_GCP_PROJECT_ID

This queries the public `thelook_ecommerce` dataset's INFORMATION_SCHEMA.COLUMNS
view and writes the result to data/metadata.json in the same format the app expects.

Requires:
    - gcloud auth application-default login
    - YOUR_GCP_PROJECT_ID with BigQuery API enabled (used only for billing the
      metadata query, which is free/negligible)
"""

import sys
import json
from collections import defaultdict
from google.cloud import bigquery

SOURCE_DATASET = "bigquery-public-data.thelook_ecommerce"


def main(project_id: str):
    client = bigquery.Client(project=project_id)

    query = f"""
        SELECT table_name, column_name, data_type, ordinal_position
        FROM `{SOURCE_DATASET}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name, ordinal_position
    """

    rows = client.query(query).result()

    tables = defaultdict(list)
    for row in rows:
        tables[row.table_name].append({
            "name": row.column_name,
            "type": row.data_type,
            "description": ""   # fill in manually or via a separate LLM enrichment pass
        })

    output = {
        "dataset": SOURCE_DATASET,
        "tables": [
            {"table_name": t, "description": "", "columns": cols}
            for t, cols in tables.items()
        ]
    }

    with open("data/metadata.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote metadata for {len(output['tables'])} tables to data/metadata.json")
    print("Tip: add table/column descriptions manually for better RAG retrieval quality.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/extract_metadata.py YOUR_GCP_PROJECT_ID")
        sys.exit(1)
    main(sys.argv[1])
