"""
Fetch companyName, website, websiteStatus.tag from Elasticsearch (sellers_0726)
for a list of dwhIds, and write the results to an Excel file.

Requires a .env file (in the same directory you run this from, or set the path
in load_dotenv below) containing:
    ES_HOST=https://prodelastic.growthpal.com:443
    ES_API_TOKEN=<your token>

Usage:
    python fetch_es_data.py input.xlsx output.xlsx
"""

import sys
import os
import json
import requests
import urllib3
import pandas as pd
from dotenv import load_dotenv

urllib3.disable_warnings()  # cluster sits behind a proxy with a self-signed/mismatched cert

load_dotenv()  # loads ES_HOST and ES_API_TOKEN from .env in cwd

ES_HOST = os.environ["ES_HOST"].rstrip("/")
ES_TOKEN = os.environ["ES_API_TOKEN"]
INDEX = "sellers_0726"

BATCH_SIZE = 500  # dwhIds per request, tune down if you hit payload/timeout limits

HEADERS = {
    "Token": ES_TOKEN,
    "Content-Type": "application/json",
}


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fetch_batch(dwh_ids):
    """
    Query ES for a batch of dwhIds using a terms query.
    Adjust the field name below if dwhId is mapped differently
    (e.g. 'dwhId.keyword' vs 'dwhId').
    """
    query = {
        "size": len(dwh_ids),
        "_source": ["dwhId", "companyName", "website", "websiteStatus.tag"],
        "query": {"terms": {"dwhId.keyword": dwh_ids}},
    }

    resp = requests.post(
        f"{ES_HOST}/{INDEX}/_search",
        headers=HEADERS,
        data=json.dumps(query),
        verify=False,
        timeout=60,
    )
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]

    results = []
    for h in hits:
        src = h.get("_source", {})
        website_status = src.get("websiteStatus", {})
        tag = website_status.get("tag") if isinstance(website_status, dict) else None
        results.append(
            {
                "dwhId": src.get("dwhId"),
                "companyName": src.get("companyName"),
                "website": src.get("website"),
                "websiteStatus.tag": tag,
            }
        )
    return results


def main(input_path, output_path):
    df_in = pd.read_excel(input_path)
    dwh_ids = df_in["dwhId"].dropna().astype(str).tolist()
    print(f"Loaded {len(dwh_ids)} dwhIds from {input_path}")

    all_results = []
    found_ids = set()

    for i, batch in enumerate(chunked(dwh_ids, BATCH_SIZE), start=1):
        print(f"Fetching batch {i} ({len(batch)} ids)...")
        try:
            batch_results = fetch_batch(batch)
        except requests.HTTPError as e:
            print(f"  Batch {i} failed: {e} - {e.response.text[:500]}")
            continue
        all_results.extend(batch_results)
        found_ids.update(r["dwhId"] for r in batch_results if r["dwhId"])

    # Track dwhIds that returned no ES document, so nothing silently disappears
    missing_ids = [d for d in dwh_ids if d not in found_ids]
    for d in missing_ids:
        all_results.append(
            {
                "dwhId": d,
                "companyName": None,
                "website": None,
                "websiteStatus.tag": None,
            }
        )

    df_out = pd.DataFrame(
        all_results, columns=["dwhId", "companyName", "website", "websiteStatus.tag"]
    )
    df_out.to_excel(output_path, index=False)
    print(f"Wrote {len(df_out)} rows to {output_path}")
    print(f"  Matched: {len(found_ids)} | Missing from ES: {len(missing_ids)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fetch_es_data.py <input.xlsx> <output.xlsx>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
