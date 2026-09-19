"""Sidecar 向量库：**内存自适应分片** + Faiss 8bit 量化 + LRU 缓存。

为 drift/shell/cross_file 三知识库提供真正的语义检索能力，并且：
- 自动探测可用 RAM/VRAM，按比例决定分片大小（每 KB 独立分片）；
- 冷分片按需加载/自动卸载，常驻内存不超过阈值；
- 跨分片查询自动 top-k merge；
- 几百条冷启动 <1s，十万条也不会 OOM。

和现有 knowledge.py 的三 query 函数无缝对接——API 签名完全不变。

使用：
    from core.sidecar.vector_store import SidecarVectorStore
    vs = SidecarVectorStore()          # 自动探测内存、加载/构建索引
    results = vs.search("drift", "git push --force 跳过 staging", top_k=3)
"""
from __future__ import annotations

import gc
import json
import pickle
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .embedding import get_embedder, _DEFAULT_MODEL, _DIM


_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
_INDEX_DIR = _KNOWLEDGE_DIR / "vector_index"

# —— 内存自适应分片常量 ——
# 单条量化向量的近似字节数（ScalarQuant 8bit: dim × 1 byte + overhead）
_BYTES_PER_ENTRY = _DIM  # 512 B（8bit 量化）
# 向量库最多占可用内存的比例（安全阈值）
_MEMORY_FRACTION = 0.20
# 常驻内存的最大分片数（超出后 LRU 淘汰冷分片）
_MAX_RESIDENT_SHARDS = 8
# 最小分片条目数（少于这个不分片）
_MIN_SHARD_SIZE = 100

# 知识库 → 源 JSON 文件名 + 条目中"要做 embedding 的文本字段"
_KB_CONFIG = {
    "drift": {
        "file": "drift_knowledge.json",
        "text_fields": ["pattern", "description"],
    },
    "shell": {
        "file": "shell_intent_knowledge.json",
        "text_fields": ["label", "description", "keywords"],
    },
    "cross": {
        "file": "cross_file_patterns.json",
        "text_fields": ["pattern", "description", "trigger_files"],
    },
}


def _entry_text(entry: dict, fields: list) -> str:
    """把一条知识条目拼成可做 embedding 的文本。"""
    parts = []
    for f in fields:
        v = entry.get(f, "")
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        elif not isinstance(v, str):
            v = str(v)
        if v:
            parts.append(v)
    return " | ".join(parts)


class _OneKB:
    """单个知识库的 Faiss 索引 + 元数据。"""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim
        self.entries: list[dict] = []       # 原始条目
        self.texts: list[str] = []          # 每条的 embedding 文本
        self.index = None                    # faiss.IndexFlatIP
        self._loaded = False

    @property
    def size(self) -> int:
        return len(self.entries)

    def build(self, entries: list[dict], embedder, quantize: bool = True,
              text_fields: list[str] | None = None):
        """首次构建或重建索引。quantize=True 用 8bit ScalarQuantizer 省显存。"""
        import faiss
        self.entries = entries
        # 支持显式传 text_fields（ShardedKB 分片调用时）；否则从 _KB_CONFIG 查
        fields = text_fields
        if fields is None:
            # 去掉分片后缀查原 KB 名
            base_name = re.sub(r"_s\d+$", "", self.name)
            if base_name in _KB_CONFIG:
                fields = _KB_CONFIG[base_name]["text_fields"]
            else:
                fields = ["pattern", "description"]
        self.texts = [_entry_text(e, fields) for e in entries]
        if not self.texts:
            self.index = None
            return
        emb = embedder.encode(self.texts).astype(np.float32)
        if quantize and emb.shape[0] >= 4:
            self.index = faiss.IndexScalarQuantizer(
                self.dim, faiss.ScalarQuantizer.QT_8bit,
                faiss.METRIC_INNER_PRODUCT)
            self.index.train(emb)   # ScalarQuantizer 必须先 train
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)
        self._loaded = True

    def save(self, path: Path):
        """持久化索引 + 元数据（用 buffer 绕开 Windows 中文路径 fopen 问题）。"""
        import faiss
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            buf = faiss.serialize_index(self.index)
            with open(path / f"{self.name}.bin", "wb") as f:
                f.write(buf)
        meta = {"entries": self.entries, "texts": self.texts}
        with open(path / f"{self.name}.pkl", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, name: str, path: Path, dim: int) -> "_OneKB | None":
        import faiss
        pkl_path = path / f"{name}.pkl"
        if not pkl_path.exists():
            return None
        obj = cls(name, dim)
        meta = pickle.load(open(pkl_path, "rb"))
        obj.entries = meta["entries"]
        obj.texts = meta["texts"]
        bin_path = path / f"{name}.bin"
        if bin_path.exists():
            buf = open(bin_path, "rb").read()
            obj.index = faiss.deserialize_index(
                np.frombuffer(buf, dtype=np.uint8))
        obj._loaded = True
        return obj

    def search(self, query_vec: np.ndarray, top_k: int,
               score_threshold: float = 0.0) -> list[tuple[float, dict]]:
        """余弦相似度检索。返回 [(score, entry), ...]。"""
        if self.index is None or self.size == 0:
            return []
        k = min(top_k, self.size)
        scores, ids = self.index.search(
            query_vec.reshape(1, -1).astype(np.float32), k)
        results = []
        for s, i in zip(scores[0], ids[0]):
            if i < 0 or i >= self.size:
                continue
            if score_threshold and s < score_threshold:
                continue
            results.append((float(s), self.entries[i]))
        return results

    def unload(self):
        """释放 Faiss 索引内存（保留 entries/texts 元数据）。"""
        self.index = None
        self._loaded = False
        gc.collect()


# —— 内存自适应分片 ——

def _detect_available_memory() -> int:
    """探测当前环境可用于向量库的内存（bytes）。取 RAM 和 VRAM 之和的
    _MEMORY_FRACTION。两者都探测不到时 fallback 到 512MB。"""
    import os
    avail_ram = 0
    avail_vram = 0
    # RAM
    try:
        import psutil
        avail_ram = psutil.virtual_memory().available
    except Exception:
        # Windows fallback
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong)]
            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            avail_ram = int(m.ullAvailPhys * 0.6)  # 保守只取 60%
        except Exception:
            avail_ram = 512 * 1024 * 1024
    # VRAM（如有）
    try:
        import torch
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            avail_vram = int(free)
    except Exception:
        pass
    return int((avail_ram + avail_vram) * _MEMORY_FRACTION)


def _calc_shard_size(total_entries: int, bytes_per_entry: int = _BYTES_PER_ENTRY,
                      safe_mem: int | None = None) -> int:
    """根据可用内存和条目总数计算每个分片的最大条目数。

    返回: 每个分片最多容纳的条目数（至少 _MIN_SHARD_SIZE）。
    """
    if safe_mem is None:
        safe_mem = _detect_available_memory()
    max_bytes_per_kb = safe_mem // 3  # drift/shell/cross 三分
    shard_capacity = max(max_bytes_per_kb // bytes_per_entry, _MIN_SHARD_SIZE)
    # 如果总条目比分片容量小，就不分片
    if total_entries <= shard_capacity:
        return max(total_entries, _MIN_SHARD_SIZE)
    return shard_capacity


class ShardedKB:
    """内存自适应分片向量库。

    - 构建时自动按内存分片，每个分片独立保存；
    - 查询时 LRU 加载分片、跨分片 top-k merge；
    - 常驻分片数超过 _MAX_RESIDENT_SHARDS 时自动淘汰最久未用的。
    """

    def __init__(self, name: str, dim: int, index_dir: Path,
                 max_resident: int = _MAX_RESIDENT_SHARDS):
        self.name = name
        self.dim = dim
        self.index_dir = index_dir
        self._lru: OrderedDict[int, _OneKB] = OrderedDict()  # shard_idx -> _OneKB
        self._shard_count = 0
        self._total_entries = 0
        self._lock = threading.Lock()
        self._max_resident = max_resident
        # 不立即加载——按需

    # -------- 构建 --------

    def build(self, entries: list[dict], embedder, quantize: bool = True,
              text_fields: list[str] | None = None):
        """按内存自适应分片后构建每个分片的索引。"""
        total = len(entries)
        self._total_entries = total
        shard_size = _calc_shard_size(total)
        if total <= shard_size:
            shard_size = total

        # 推断 text_fields（如果没传）
        if text_fields is None:
            base = re.sub(r"_s\d+$", "", self.name)
            text_fields = _KB_CONFIG.get(base, {}).get(
                "text_fields", ["pattern", "description"])

        shards = [entries[i:i + shard_size]
                  for i in range(0, total, shard_size)]
        self._shard_count = len(shards)

        for idx, shard_entries in enumerate(shards):
            kb = _OneKB(f"{self.name}_s{idx}", self.dim)
            kb.build(shard_entries, embedder, quantize=quantize,
                     text_fields=text_fields)
            shard_dir = self.index_dir / "shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            kb.save(shard_dir)
            if idx == 0:
                self._lru[idx] = kb
        meta = {"shard_count": self._shard_count,
                "total_entries": self._total_entries,
                "shard_size": shard_size}
        with open(self.index_dir / f"{self.name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    # -------- 按需加载 --------

    def _load_shard(self, idx: int) -> _OneKB | None:
        """按需加载一个分片；超出 resident 上限时 LRU 淘汰。"""
        with self._lock:
            # 已加载 → 移到末尾（最近使用）
            if idx in self._lru:
                self._lru.move_to_end(idx)
                return self._lru[idx]
            # 未加载 → 从磁盘读
            shard_dir = self.index_dir / "shards"
            kb = _OneKB.load(f"{self.name}_s{idx}", shard_dir, self.dim)
            if kb is None:
                return None
            # LRU 淘汰
            while len(self._lru) >= self._max_resident:
                evicted_idx, evicted_kb = self._lru.popitem(last=False)
                evicted_kb.unload()
            self._lru[idx] = kb
            return kb

    def unload_all(self):
        """释放所有常驻分片。"""
        with self._lock:
            for kb in self._lru.values():
                kb.unload()
            self._lru.clear()

    @property
    def size(self) -> int:
        return self._total_entries

    # -------- 查询（跨分片 top-k merge） --------

    def search(self, query_vec: np.ndarray, top_k: int,
               score_threshold: float = 0.0) -> list[tuple[float, dict]]:
        if self._total_entries == 0:
            return []
        # 每个分片查 top_k，再全局 merge
        all_hits: list[tuple[float, dict]] = []
        for idx in range(self._shard_count):
            kb = self._load_shard(idx)
            if kb is None:
                continue
            hits = kb.search(query_vec, top_k, score_threshold=score_threshold)
            all_hits.extend(hits)
        # 全局 top-k
        all_hits.sort(key=lambda x: x[0], reverse=True)
        return all_hits[:top_k]


class SidecarVectorStore:
    """Sidecar 向量库门面。**内存自适应分片**管理 drift/shell/cross 三个子库。

    内部用 ShardedKB（自动分片 + LRU 缓存 + 跨分片 top-k merge），API 签名
    完全不变，knowledge.py 的三 query 函数零改动。
    """

    def __init__(self, index_dir: Path | str | None = None):
        self.index_dir = Path(index_dir) if index_dir else _INDEX_DIR
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = get_embedder()
        self.dim = self.embedder.dim
        self._kbs: dict[str, ShardedKB] = {}
        self._loaded = False
        # 内存探测（一次计算，后续复用）
        self._safe_mem = _detect_available_memory()
        self._memory_info = {
            "safe_mem_mb": self._safe_mem / 1e6,
            "shard_size_per_kb": _calc_shard_size(1 << 30,  # 足够大 → 算出 shard 容量
                                                   safe_mem=self._safe_mem),
            "fraction": _MEMORY_FRACTION,
            "max_resident": _MAX_RESIDENT_SHARDS,
        }

    # -------- 加载 / 构建 --------

    def load_or_build(self):
        """从分片持久化文件加载；首次运行则从 knowledge/*.json 构建分片。"""
        if self._loaded:
            return
        for name in _KB_CONFIG:
            meta_path = self.index_dir / f"{name}_meta.json"
            shard_dir = self.index_dir / "shards"
            kb = ShardedKB(name, self.dim, self.index_dir)
            # 有 meta + 分片 bin → 直接加载（只加载 shard 0）
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    kb._shard_count = meta.get("shard_count", 1)
                    kb._total_entries = meta.get("total_entries", 0)
                    # 预加载 shard 0（冷启动加速）
                    first = _OneKB.load(f"{name}_s0", shard_dir, self.dim)
                    if first is not None:
                        kb._lru[0] = first
                except Exception:
                    kb = None
            else:
                kb = None
            if kb is None or kb._total_entries == 0:
                # 从 knowledge 源文件读取 + 分片构建
                src = _KNOWLEDGE_DIR / _KB_CONFIG[name]["file"]
                if not src.exists():
                    continue
                entries = json.loads(src.read_text(encoding="utf-8"))
                kb = ShardedKB(name, self.dim, self.index_dir)
                kb.build(entries, self.embedder)
            self._kbs[name] = kb
        self._loaded = True

    def rebuild_all(self):
        """强制重建所有分片（知识文件更新后调用）。"""
        self._kbs.clear()
        # 清除旧分片
        shard_dir = self.index_dir / "shards"
        if shard_dir.exists():
            import shutil
            shutil.rmtree(shard_dir)
        for name in _KB_CONFIG:
            src = _KNOWLEDGE_DIR / _KB_CONFIG[name]["file"]
            if not src.exists():
                continue
            entries = json.loads(src.read_text(encoding="utf-8"))
            kb = ShardedKB(name, self.dim, self.index_dir)
            kb.build(entries, self.embedder)
            self._kbs[name] = kb
        self._loaded = True

    def unload_all(self):
        """主动释放所有分片内存（给 LoRA 推理让道）。"""
        for kb in self._kbs.values():
            kb.unload_all()
        gc.collect()

    def status(self) -> dict:
        """返回各子库统计 + 内存信息。"""
        self.load_or_build()
        out = {
            "memory": self._memory_info,
            "kbs": {},
        }
        for name, kb in self._kbs.items():
            out["kbs"][name] = {
                "entries": kb.size,
                "shards": kb._shard_count,
                "resident": len(kb._lru),
            }
        return out

    # -------- 检索 --------

    def search(self, kb_name: str, query: str,
               top_k: int = 3, score_threshold: float = 0.0) -> list[dict]:
        """向量检索（自动跨分片 top-k merge）。"""
        self.load_or_build()
        kb = self._kbs.get(kb_name)
        if kb is None or kb.size == 0:
            return []
        q_vec = self.embedder.encode([query])[0]
        hits = kb.search(q_vec, top_k, score_threshold)
        return [{"score": s, "entry": e} for s, e in hits]

    # -------- 兼容 knowledge.py 原有 API --------

    def search_drift(self, task_desc: str, anchor: dict,
                     tool_name: str, args: str,
                     top_k: int = 3) -> list[dict]:
        query = f"{task_desc} {json.dumps(anchor, ensure_ascii=False)} {tool_name} {args}"
        return [r["entry"] for r in self.search("drift", query, top_k)]

    def search_shell(self, cmd: str, top_k: int = 3) -> list[dict]:
        return [r["entry"] for r in self.search("shell", cmd, top_k)]

    def search_cross(self, changed_files: list[str], top_k: int = 3) -> list[dict]:
        query = " ".join(changed_files)
        return [r["entry"] for r in self.search("cross", query, top_k)]
