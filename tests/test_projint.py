"""项目智能测试：语义图谱、持久记忆、Code Review/漏洞扫描、审计。"""
from core.project import graph, repo_memory, audit, review
from core.agent import tools as tool_registry


def _make_project(tmp_path):
    (tmp_path / "a.py").write_text(
        "import b\n\ndef parse(x):\n    return b.norm(x)\n"
        "\ndef helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "import a\n\ndef norm(y):\n    v = a.helper()\n    return v\n", encoding="utf-8")
    (tmp_path / "c.py").write_text(
        "def guarded():\n    password = 's3cret'\n    return password\n", encoding="utf-8")
    return tmp_path


# ---------------- 语义图谱 ----------------
def test_build_graph_finds_files_and_edges(tmp_path):
    _make_project(tmp_path)
    g = graph.build_graph(tmp_path, force=True)
    assert len(g["files"]) == 3
    # a.py 定义 parse，b.py 定义 norm，跨文件调用应成边
    syms = {s for e in g["edges"] for s in [e["symbol"]]}
    assert {"parse", "norm", "helper"}.intersection(syms)
    assert g["def_index"].get("parse")
    assert g["def_index"].get("norm")


def test_query_impact(tmp_path):
    _make_project(tmp_path)
    g = graph.build_graph(tmp_path, force=True)
    out = graph.query_impact(g, "norm")
    assert "norm" in out and "调用它的文件" in out


def test_graph_summary_nonempty(tmp_path):
    _make_project(tmp_path)
    s = graph.graph_summary(graph.build_graph(tmp_path, force=True))
    assert "语义图谱" in s and "3 文件" in s


def test_tool_graph_query_registered():
    names = {t["function"]["name"] for t in tool_registry.tools_schema()}
    for n in ("graph_query", "graph_query_file", "repo_remember", "repo_read",
              "code_review", "vuln_scan", "audit_read"):
        assert n in names


# ---------------- 持久记忆 ----------------
def test_repo_memory_roundtrip(tmp_path):
    assert repo_memory.is_empty(tmp_path)
    repo_memory.remember(tmp_path, "编码规范", "禁止 magic number", kind="convention")
    assert not repo_memory.is_empty(tmp_path)
    out = repo_memory.read(tmp_path, topic="编码规范")
    assert "禁止 magic number" in out
    out2 = repo_memory.read(tmp_path, "不存在主题")
    assert "项目暂无" not in out2  # 过滤后仍有记录（按最近返回）


def test_repo_memory_persists_across_reload(tmp_path):
    repo_memory.remember(tmp_path, "决策", "用 FastAPI 替代 Flask")
    assert "FastAPI" in repo_memory.read(tmp_path)


# ---------------- Code Review / 漏洞扫描 ----------------
def test_code_review_detects(tmp_path):
    (tmp_path / "dirty.py").write_text(
        "def f():\n    try:\n        pass\n    except:\n        pass\n", encoding="utf-8")
    out = review.code_review(tmp_path, "dirty.py")
    assert "裸 except" in out


def test_vuln_scan_detects_hardcoded_secret(tmp_path):
    _make_project(tmp_path)
    out = review.vuln_scan(tmp_path, "c.py")
    assert "密钥/口令硬编码" in out


def test_vuln_scan_clean(tmp_path):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    out = review.vuln_scan(tmp_path, "ok.py")
    assert "未发现明显安全问题" in out


# ---------------- 审计 ----------------
def test_audit_record_and_read(tmp_path):
    audit.record(tmp_path, "project_write", "wrote app.py", "app.py")
    out = audit.read(tmp_path)
    assert "project_write" in out and "app.py" in out


def test_invoke_records_audit_when_root_set(tmp_path, monkeypatch):
    # 设置项目根后，写类工具调用会进入审计
    from core.agent import tools as T
    old = T._PROJECT_ROOT
    T.set_project_root(str(tmp_path))
    try:
        T.invoke("run_shell", {"command": "echo hi", "timeout": 5})
        out = audit.read(tmp_path)
        assert "tool_run_shell" in out
    finally:
        monkeypatch.setattr(T, "_PROJECT_ROOT", old)