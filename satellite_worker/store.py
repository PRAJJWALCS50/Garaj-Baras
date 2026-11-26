"""
Where fog products are kept.

SupabaseStore : Supabase Storage bucket (public read, secret-key write) - production
                accepts the new secret key (sb_secret_...) or a legacy service_role JWT
LocalStore    : a local folder - for test runs on your own PC (--local DIR)

Layout (same for both):
    latest.json                 manifest: newest scene + the 2 before it
    scenes/YYYYMMDD_HHMM.npz    fog map + distance + speed for one scene
"""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

import config as C


def pack(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def unpack(data: bytes) -> dict:
    with np.load(io.BytesIO(data)) as z:
        return {k: z[k] for k in z.files}


class SupabaseStore:
    def __init__(self, url=None, service_key=None, bucket=C.BUCKET):
        self.url = (url or os.environ.get("SUPABASE_URL", "")).rstrip("/")
        self.key = service_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        self.bucket = bucket
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    def _headers(self, content_type, cache_s):
        h = {"apikey": self.key, "Content-Type": content_type, "x-upsert": "true",
             "cache-control": f"max-age={cache_s}"}
        # Legacy service_role keys are JWTs and also go in Authorization.
        # New secret keys (sb_secret_...) are not JWTs: apikey header only.
        if not self.key.startswith("sb_"):
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def put(self, path, data: bytes, content_type="application/octet-stream", cache_s=31536000):
        r = requests.post(f"{self.url}/storage/v1/object/{self.bucket}/{path}",
                          data=data, headers=self._headers(content_type, cache_s), timeout=60)
        r.raise_for_status()

    def get(self, path):
        # public URL + cache-buster so a CDN never serves an old manifest
        r = requests.get(f"{self.url}/storage/v1/object/public/{self.bucket}/{path}",
                         params={"v": int(time.time())}, timeout=60)
        if r.status_code in (400, 404):
            return None
        r.raise_for_status()
        return r.content


class LocalStore:
    def __init__(self, folder):
        self.root = Path(folder)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, path, data: bytes, content_type=None, cache_s=None):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, path):
        p = self.root / path
        return p.read_bytes() if p.exists() else None


def read_manifest(store):
    raw = store.get(C.MANIFEST)
    return json.loads(raw) if raw else None


def write_manifest(store, manifest):
    store.put(C.MANIFEST, json.dumps(manifest, indent=1).encode(), "application/json", cache_s=30)
