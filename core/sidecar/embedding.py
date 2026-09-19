"""Embedding 抽象层：本地模型优先 + 零依赖 mock 降级。

支持三种 provider（按优先级自动尝试）：
  1. local       —— gte-small-zh 本地推理（首选，离线可用）
  2. mock        —— numpy 伪 embedding（fallback，零 torch 依赖）

向量库统一使用 L2 归一化 + cosine 相似度，Faiss IP 索引直接吃。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


# 默认 embedding 模型路径（gte-small-zh，从 ModelScope 下载）
_DEFAULT_MODEL = r"C:\Users\wang8\.cache\modelscope\models\AI-ModelScope--gte-small-zh\snapshots\master"
_DIM = 512  # gte-small-zh 输出维度


class _LocalEmbedder:
    """gte-small-zh 本地推理，懒加载 + eval + no_grad。"""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path or _DEFAULT_MODEL
        self._tok = None
        self._model = None
        self._dim = _DIM

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        import torch.nn.functional as F
        from transformers import AutoTokenizer, AutoModel
        self._F = F
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(self._model_path)
        self._model.eval()
        # 半精度省显存
        if torch.cuda.is_available():
            self._model = self._model.to("cuda").half()

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> "np.ndarray":
        self._ensure()
        F = self._F
        torch = self._torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.inference_mode():
            batch = self._tok(
                texts, max_length=512, padding=True, truncation=True,
                return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = self._model(**batch)
            emb = F.normalize(
                out.last_hidden_state[:, 0].float(), p=2, dim=1)
        return emb.cpu().numpy()


class _MockEmbedder:
    """零依赖伪 embedding：hash → numpy 单位向量。
    仅当本地 torch/transformers 不可用时降级使用。"""

    def __init__(self, dim: int = _DIM):
        import hashlib
        import math
        import numpy as np
        self._hashlib = hashlib
        self._math = math
        self._np = np
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _one(self, text: str):
        np = self._np
        h = self._hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        norm = np.linalg.norm(v) or 1.0
        return v / norm

    def encode(self, texts: list[str]) -> "np.ndarray":
        import numpy as np
        return np.stack([self._one(t) for t in texts]).astype(np.float32)


_EMBEDDER = None


def get_embedder(model_path: str | None = None,
                 force: str | None = None):
    """全局 embedder 单例。force='local'/'mock' 可强制指定。"""
    global _EMBEDDER
    if _EMBEDDER is not None and (force is None or force == "local"):
        return _EMBEDDER

    if force == "mock":
        _EMBEDDER = _MockEmbedder()
        return _EMBEDDER

    # 优先 local，失败降级 mock
    try:
        e = _LocalEmbedder(model_path)
        e._ensure()  # 触发加载，失败会抛
        _EMBEDDER = e
        return _EMBEDDER
    except Exception as ex:
        import sys
        print(f"[embedding] 本地模型加载失败（{ex}），降级到 mock",
              file=sys.stderr)
        _EMBEDDER = _MockEmbedder()
        return _EMBEDDER


def reset_embedder():
    """重建 embedder（测试用）。"""
    global _EMBEDDER
    _EMBEDDER = None
