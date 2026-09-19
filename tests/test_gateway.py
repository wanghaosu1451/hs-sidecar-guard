"""支柱A 网关与供应商目录冒烟测试（不联网）。"""
from __future__ import annotations

import pytest

from core.llm.provider_catalog import PROVIDERS, GROUPS, count


def test_providers_count_at_least_80():
    assert count() >= 80, f"供应商数量不足: {count()}"


def test_local_backends_present():
    assert any(k.startswith("ollama/") for k in PROVIDERS)
    assert any(k.startswith("vllm/") for k in PROVIDERS)
    assert any(k.startswith("custom/") for k in PROVIDERS)


def test_groups_cover_all():
    covered = set()
    for items in GROUPS.values():
        covered.update(items)
    assert set(PROVIDERS) == covered


def test_groups_nonempty():
    for name, items in GROUPS.items():
        assert items, f"分组 {name} 为空"


def test_keys_defaults_load():
    from core.llm import keys
    data = keys.load()
    assert "llm" in data and "provider" in data["llm"]