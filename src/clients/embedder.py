"""BGE-M3 dense embedder — local model or HTTP API client."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


class BGEM3Embedder:
    def __init__(
        self,
        model_path: str = "",
        use_fp16: bool = True,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 4096,
        api_url: str | None = None,
    ):
        self.batch_size = batch_size
        self.max_length = max_length
        self.api_url = (api_url or "").rstrip("/") if api_url else None
        if self.api_url:
            import requests
            self._session = requests.Session()
            self._session.trust_env = False
            return
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            devices=[device],
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalized dense vectors. Shape (n, dim)."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        if self.api_url:
            if self.api_url.endswith("/v1/embeddings"):
                resp = self._session.post(
                    self.api_url,
                    json={"input": list(texts), "model": "bge-m3"},
                    timeout=300,
                )
                resp.raise_for_status()
                data = resp.json().get("data") or []
                vecs = [row.get("embedding") for row in sorted(data, key=lambda r: r.get("index", 0))]
            else:
                resp = self._session.post(
                    f"{self.api_url}/bge/encode",
                    json={"texts": list(texts)},
                    timeout=300,
                )
                resp.raise_for_status()
                vecs = resp.json()["vectors"]
            out = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            return out / norms
        out = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        out = np.asarray(out, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out / norms
