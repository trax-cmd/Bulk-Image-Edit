#!/usr/bin/env python3
"""Replace a product's images on Shopify with the cleaned 1200x1200 files.

DRY RUN BY DEFAULT. Nothing is sent to Shopify unless --apply is given.

Usage:
  python3 scripts/shopify_upload.py <output_root> <image_manifest.csv> [--items 10570,10579] [--apply]

Needs: SHOPIFY_SHOP (e.g. traxnyc.myshopify.com) and SHOPIFY_ADMIN_TOKEN (Admin API
access token of a custom app with write_products + write_files) in the environment.

Per product:
  1. stagedUploadsCreate for every new image, PUT/POST the bytes to the staged URL
  2. productCreateMedia with the staged resource URLs, in manifest position order
  3. wait until the new media are READY
  4. productDeleteMedia for the media that existed before step 2
  5. productReorderMedia so the new images sit in position order
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

API_VERSION = "2025-07"


def gql(shop, token, query, variables=None):
    r = requests.post(f"https://{shop}/admin/api/{API_VERSION}/graphql.json",
                      headers={"X-Shopify-Access-Token": token, "Content-Type": "application/json"},
                      json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"]))
    return data["data"]


Q_MEDIA = """query($id: ID!) { product(id: $id) { title media(first: 50) { nodes { id alt mediaContentType status
  ... on MediaImage { image { url } } } } } }"""
M_STAGED = """mutation($input: [StagedUploadInput!]!) { stagedUploadsCreate(input: $input) {
  stagedTargets { url resourceUrl parameters { name value } } userErrors { field message } } }"""
M_CREATE = """mutation($id: ID!, $media: [CreateMediaInput!]!) { productCreateMedia(productId: $id, media: $media) {
  media { id status } mediaUserErrors { field message } } }"""
M_DELETE = """mutation($id: ID!, $ids: [ID!]!) { productDeleteMedia(productId: $id, mediaIds: $ids) {
  deletedMediaIds mediaUserErrors { field message } } }"""
M_REORDER = """mutation($id: ID!, $moves: [MoveInput!]!) { productReorderMedia(id: $id, moves: $moves) {
  job { id } mediaUserErrors { field message } } }"""


def upload_one(shop, token, path: Path):
    data = path.read_bytes()
    res = gql(shop, token, M_STAGED, {"input": [{"filename": path.name, "mimeType": "image/jpeg",
                                                   "resource": "IMAGE", "httpMethod": "POST",
                                                   "fileSize": str(len(data))}]})["stagedUploadsCreate"]
    if res["userErrors"]:
        raise RuntimeError(res["userErrors"])
    t = res["stagedTargets"][0]
    form = {p["name"]: p["value"] for p in t["parameters"]}
    r = requests.post(t["url"], data=form, files={"file": (path.name, data, "image/jpeg")}, timeout=120)
    r.raise_for_status()
    return t["resourceUrl"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_root")
    ap.add_argument("manifest")
    ap.add_argument("--items", default=None, help="comma-separated item numbers; default all")
    ap.add_argument("--apply", action="store_true", help="actually change the store")
    a = ap.parse_args()
    shop, token = os.environ.get("SHOPIFY_SHOP"), os.environ.get("SHOPIFY_ADMIN_TOKEN")
    if a.apply and not (shop and token):
        print("SHOPIFY_SHOP and SHOPIFY_ADMIN_TOKEN must be set for --apply", file=sys.stderr)
        return 2
    want = set(a.items.split(",")) if a.items else None
    by_product = defaultdict(list)
    for r in csv.DictReader(open(a.manifest)):
        if want and r["item_number"] not in want:
            continue
        by_product[(r["shopify_product_id"], r["item_number"])].append(r)

    for (pid, item), rows in sorted(by_product.items(), key=lambda kv: int(kv[0][1])):
        rows.sort(key=lambda r: int(r["position"]))
        files = [Path(a.output_root) / r["folder"] / r["output_file"] for r in rows]
        missing = [f for f in files if not f.exists()]
        if missing:
            print(f"[{item}] SKIP, missing {len(missing)} files")
            continue
        gid = f"gid://shopify/Product/{pid}"
        print(f"[{item}] product {pid}: {len(files)} images" + ("" if a.apply else "  (dry run)"))
        if not a.apply:
            continue
        before = gql(shop, token, Q_MEDIA, {"id": gid})["product"]
        old_ids = [m["id"] for m in before["media"]["nodes"]]
        media = [{"originalSource": upload_one(shop, token, f), "mediaContentType": "IMAGE",
                  "alt": f"{before['title']} - image {i + 1}"} for i, f in enumerate(files)]
        res = gql(shop, token, M_CREATE, {"id": gid, "media": media})["productCreateMedia"]
        if res["mediaUserErrors"]:
            raise RuntimeError(res["mediaUserErrors"])
        new_ids = [m["id"] for m in res["media"]]
        for _ in range(30):
            nodes = gql(shop, token, Q_MEDIA, {"id": gid})["product"]["media"]["nodes"]
            st = {m["id"]: m["status"] for m in nodes}
            if all(st.get(i) == "READY" for i in new_ids):
                break
            if any(st.get(i) == "FAILED" for i in new_ids):
                raise RuntimeError(f"[{item}] media failed to process: {st}")
            time.sleep(2)
        if old_ids:
            res = gql(shop, token, M_DELETE, {"id": gid, "ids": old_ids})["productDeleteMedia"]
            if res["mediaUserErrors"]:
                raise RuntimeError(res["mediaUserErrors"])
        moves = [{"id": mid, "newPosition": str(i)} for i, mid in enumerate(new_ids)]
        gql(shop, token, M_REORDER, {"id": gid, "moves": moves})
        print(f"[{item}] replaced {len(old_ids)} old images with {len(new_ids)} new")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
