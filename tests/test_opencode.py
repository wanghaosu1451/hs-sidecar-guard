"""OpenCode(opencode.json) 配置兼容解析测试。"""
import json

from core.llm import opencode as oc


def _write(tmp, data: dict, name: str = "opencode.json"):
    p = tmp / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def test_parse_default_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MYKEY", "sk-abc")
    p = _write(tmp_path, {
        "provider": {
            "q": {"baseURL": "http://x/v1", "apiKey": "{env:MYKEY}",
                  "models": {"qwen3:8b": {"options": {"thinking": False}}}},
            "other": {"baseURL": "http://y"},
        },
        "model": "q/qwen3:8b",
    })
    r = oc.parse(p)
    assert r["provider"] == "q/qwen3:8b"
    assert r["base_url"] == "http://x/v1"
    assert r["api_key"] == "sk-abc"
    assert r["model_kwargs"] == {"thinking": False}
    assert r["meta"]["pid"] == "q"
    assert r["meta"]["mid"] == "qwen3:8b"


def test_parse_no_model_returns_none(tmp_path):
    p = _write(tmp_path, {"provider": {"q": {"models": {}}}})
    assert oc.parse(p) is None


def test_parse_missing_provider_returns_none(tmp_path):
    p = _write(tmp_path, {"provider": {"q": {"models": {}}}, "model": "zz/m"})
    assert oc.parse(p) is None


def test_resolve_key_literal_and_env(monkeypatch):
    assert oc._resolve_key("plain-key") == "plain-key"
    monkeypatch.setenv("TOK", "v")
    assert oc._resolve_key("{env:TOK}") == "v"
    assert oc._resolve_key("{env:MISSING_VAR_XYZ}") == ""


def test_import_to_settings_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("HSKEY", "sk-hs")
    _write(tmp_path, {
        "provider": {"openai": {"baseURL": "https://api.test/v1",
                                "apiKey": "{env:HSKEY}",
                                "models": {"gpt-4o": {}}}},
        "model": "openai/gpt-4o",
    })
    from core.llm import keys as k
    r = oc.import_to_settings(tmp_path / "opencode.json")
    assert r["imported"] is True
    updated = k.load()
    assert updated["llm"]["provider"] == "openai/gpt-4o"
    assert updated["llm"]["base_url"] == "https://api.test/v1"
    assert updated["llm"]["api_key"] == "sk-hs"


def test_import_missing_returns_not_imported(tmp_path):
    from core.llm import keys as k
    r = oc.import_to_settings(tmp_path / "nope.json")
    assert r["imported"] is False