"""交付闭环测试：快照/回滚/diff、测试/文档/部署生成。"""
from core.project import snapshot, generators
from core.agent import tools as tool_registry


def _mk(root):
    f = root / "app.py"
    f.write_text("def parse(x):\n    return int(x)\n", encoding="utf-8")
    return f


# ---------------- 快照 / 回滚 / diff ----------------
def test_snapshot_created_on_overwrite(tmp_path):
    _mk(tmp_path)
    # 首次写不建快照（无旧版）
    assert snapshot.backup(tmp_path, "app.py", "NEW") is not None
    ls = snapshot.list_snapshots(tmp_path)
    assert "app.py" in ls


def test_rollback_restores_previous(tmp_path):
    _mk(tmp_path)
    name = snapshot.backup(tmp_path, "app.py", "NEW CONTENT")
    # 写新内容
    (tmp_path / "app.py").write_text("NEW CONTENT", encoding="utf-8")
    out = snapshot.rollback(tmp_path, name)
    assert "已回滚" in out
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "def parse(x):\n    return int(x)\n"


def test_rollback_missing_snapshot(tmp_path):
    out = snapshot.rollback(tmp_path, "nope.json")
    assert "快照不存在" in out


def test_diff_shows_changes(tmp_path):
    _mk(tmp_path)
    snapshot.backup(tmp_path, "app.py", "def parse(x):\n    return int(x)\n  + x")
    (tmp_path / "app.py").write_text("def parse(x):\n    return int(x)  # changed\n",
                                     encoding="utf-8")
    out = snapshot.diff(tmp_path, "app.py")
    assert "+++" in out and ("+" in out or "-" in out)


def test_diff_no_history(tmp_path):
    _mk(tmp_path)
    out = snapshot.diff(tmp_path, "app.py")
    assert "历史快照" in out


def test_tools_registered():
    names = {t["function"]["name"] for t in tool_registry.tools_schema()}
    for n in ("snapshot_list", "snapshot_rollback", "diff_file",
              "gen_tests", "gen_docker", "gen_doc"):
        assert n in names


# ---------------- 生成器 ----------------
def test_gen_tests(tmp_path):
    _mk(tmp_path)
    out = generators.gen_tests(tmp_path, "app.py", "test_app.py")
    assert "已生成" in out
    body = (tmp_path / "test_app.py").read_text(encoding="utf-8")
    assert "import app" in body and "def test_app_parse" in body


def test_gen_tests_missing_module(tmp_path):
    out = generators.gen_tests(tmp_path, "nope.py")
    assert "错误" in out


def test_gen_docker(tmp_path):
    _mk(tmp_path)
    out = generators.gen_docker(tmp_path)
    assert "Dockerfile" in out
    assert (tmp_path / "Dockerfile").is_file()
    assert (tmp_path / ".dockerignore").is_file()


def test_gen_doc(tmp_path):
    _mk(tmp_path)
    out = generators.gen_doc(tmp_path)
    assert "README.auto.md" in out
    body = (tmp_path / "README.auto.md").read_text(encoding="utf-8")
    assert "parse" in body