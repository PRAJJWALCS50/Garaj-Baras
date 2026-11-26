"""Supabase Storage (production) or a local folder (--local DIR, PC tests)."""
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


class SupabaseStore:
    def __init__(self):
        self.url = os.environ.get("SUPABASE_URL", "").rstrip("/")
        self.key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not self.url or not self.key:
            raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set")

    def _headers(self, content_type, cache_s):
        h = {"apikey": self.key, "Content-Type": content_type, "x-upsert": "true",
             "cache-control": f"max-age={cache_s}"}
        if not self.key.startswith("sb_"):      # legacy JWT keys also go in Authorization
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def put(self, path, data: bytes, content_type="application/octet-stream", cache_s=31536000):
        r = requests.post(f"{self.url}/storage/v1/object/{C.BUCKET}/{path}", data=data,
                          headers=self._headers(content_type, cache_s), timeout=60)
        r.raise_for_status()

    def get(self, path):
        r = requests.get(f"{self.url}/storage/v1/object/public/{C.BUCKET}/{path}",
                         params={"v": int(time.time())}, timeout=60)
        if r.status_code in (400, 404):
            return None
        r.raise_for_status()
        return r.content


class LocalStore:
    def __init__(self, folder):
        self.root = Path(folder)

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
