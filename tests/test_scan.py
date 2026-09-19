"""定时安全巡检 + 审计可视化测试：scan 检测/告警落盘、audit 统计聚合、Web 端点、CLI。"""
from __future__ import annotations

import json
from pathlib import Path

from core.project import scan
from core.project import audit


def _bad_project(tmp_path) -> Path:
    root = tmp_path / "proj"
    root.mkdir(parents=True, exist_ok=True)
    (root / "app.py").write_text(
        "import os\n"
        "api_key = 'sk-abcdefgh1234567890abcdef123456'\n"      # 密钥硬编码
        "os.system('rm -rf ~')\n"                               # 高危 shell
        "import shutil; shutil.rmtree('/data')\n", encoding="utf-8")
    (root / "sub").mkdir(exist_ok=True)
    (root / "sub" / "clean.py").write_text("x = 1\n", encoding="utf-8")
    (root / "requirements.txt").write_text("requests==0.0.0\nnumpy\n", encoding="utf-8")
    return root


# ---------------- 核心扫描 ----------------
def test_scan_detects_all_four_categories(tmp_path):
    root = _bad_project(tmp_path)
    result = scan.scan_project(root)
    cats = {f["category"] for f in result["findings"]}
    assert "secrets" in cats   # api_key 硬编码
    assert "shell" in cats     # rm -rf ~
    assert "filesystem" in cats  # rmtree
    assert "deps" in cats      # ==0.0.0
    # 干净子目录不产生误报
    assert result["files"] > 1


def test_scan_clean_project_no_findings(tmp_path):
    root = tmp_path / "ok"
    root.mkdir()
    (root / "main.py").write_text("print(1)\n", encoding="utf-8")
    result = scan.scan_project(root)
    assert result["total"] == 0
    assert result["counts"] == {"secrets": 0, "shell": 0, "filesystem": 0, "deps": 0}


def test_scan_single_file(tmp_path):
    root = _bad_project(tmp_path)
    result = scan.scan_project(tmp_path / "proj", path="app.py")
    assert result["files"] == 1
    assert any("app.py" == f["file"] for f in result["findings"])


def test_scan_persist_writes_alerts(tmp_path):
    root = _bad_project(tmp_path)
    result = scan.scan_and_persist(root)
    last = root / ".hs" / "scan_last.json"
    assert last.is_file()
    alerts = root / ".hs" / "scan_alerts.jsonl"
    assert alerts.is_file()
    # 只落高危（secrets/filesystem）告警；shell 中危不进告警日志
    n_high = sum(1 for f in result["findings"] if f["severity"] == "high")
    lines = [json.loads(x) for x in alerts.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(lines) == n_high
    assert all(l["severity"] == "high" for l in lines)


def test_scan_skips_binary_and_hs_dir(tmp_path):
    root = tmp_path / "p"
    root.mkdir()
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    hidden = root / ".hs"
    hidden.mkdir()
    (hidden / "secret.txt").write_text("password='x'\n", encoding="utf-8")
    result = scan.scan_project(root)
    assert result["total"] == 0  # .bin 二进制跳过、.hs 目录排除


# ---------------- 审计统计聚合 ----------------
def test_audit_stats_aggregation(tmp_path):
    from core.project.audit import record
    record(str(tmp_path), "tool_write_file", "x.py")
    record(str(tmp_path), "tool_delete_file", "y.py")
    record(str(tmp_path), "tool_run_shell", "ls")
    record(str(tmp_path), "read", "file")
    d = audit.stats(tmp_path)
    assert d["total"] == 4
    assert d["by_action"].get("write_file") == 1
    assert d["by_action"].get("delete_file") == 1
    assert "write_file" in d["high_risk"] and "delete_file" in d["high_risk"]
    assert "read" not in d["high_risk"]
    assert sum(d["by_day"].values()) == 4


def test_audit_stats_empty(tmp_path):
    d = audit.stats(tmp_path)
    assert d["total"] == 0
    assert d["by_action"] == {} and d["by_day"] == {}


# ------------- Web 端点测试已随 Web(IDE) 版移除 -------------