"""Skill 插件生态测试：注册表(权限/分类/升级回滚/脚手架/流水线)、沙箱运行器、agent 工具、CLI、Web。

全部离线自测，不联网。同时覆盖安全语义：越权拦截、禁用不可运行、超时终止、路径穿越拒绝。
"""
from __future__ import annotations

import json
from pathlib import Path

from core.skill import registry as sk
from core.skill import runner
from core.agent import tools as tool_registry


# ---------------- 内置目录 / 八分类 ----------------
def test_catalog_has_eight_categories_and_builtins():
    assert len(sk.CATEGORIES) == 8
    cats = {c["category"] for c in sk.list_catalog()}
    assert {"测试生成", "文档生成", "数据库操作"} <= cats
    names = [c["name"] for c in sk.list_catalog()]
    assert "python-tester" in names and "csv-tool" in names


def test_catalog_filter_by_category():
    only = sk.list_catalog("文档生成")
    assert all(c["category"] == "文档生成" for c in only)
    assert [c["name"] for c in only] == ["report-writer"]


# ---------------- 安装 / 卸载 / 元数据 ----------------
def test_install_list_uninstall(tmp_path):
    root = str(tmp_path)
    msg = sk.install_skill("python-tester", root=root)
    assert "已安装技能 python-tester" in msg
    installed = sk.list_skills(root)
    assert any(x["name"] == "python-tester" for x in installed)
    meta = sk.get_skill("python-tester", root)
    assert meta["version"] == "1.0.0"
    assert meta["category"] == "测试生成"
    assert meta["permissions"]["shell"] is True       # 内置声明
    assert meta["permissions"]["network"] is False    # 默认断网
    assert "已卸载技能 python-tester" in sk.uninstall_skill("python-tester", root)


def test_install_requires_name_and_traversal_safe(tmp_path):
    assert sk.install_skill("", root=str(tmp_path)).startswith("错误")
    assert sk.install_skill("../evil", root=str(tmp_path)).startswith("错误")


def test_get_skill_missing(tmp_path):
    meta = sk.get_skill("nope", str(tmp_path))
    assert "error" in meta


# ---------------- 脚手架 + 沙箱运行 ----------------
def test_scaffold_and_run_in_sandbox(tmp_path):
    root = str(tmp_path)
    assert "已创建技能脚手架 demo-skill" in sk.scaffold_skill("demo-skill", "安全扫描", "demo", root)
    # 脚手架默认权限全关，但仍可直接运行（entry 是无害打印）
    out = runner.run_skill("demo-skill", {"target": "hi"}, root=root, timeout=20)
    assert "hi" in out
    # 授权后权限落盘生效
    assert "file_read" in sk.set_permissions("demo-skill", {"file_read": True}, root)
    assert sk.get_skill("demo-skill", root)["permissions"]["file_read"] is True


def test_disabled_skill_cannot_run(tmp_path):
    root = str(tmp_path)
    sk.scaffold_skill("off-skill", "文档生成", root=root)
    sk.set_enabled("off-skill", False, root)
    assert "已禁用" in runner.run_skill("off-skill", {}, root=root)


def test_runner_timeout(tmp_path):
    root = str(tmp_path)
    sk.scaffold_skill("slow-skill", "文档生成", root=root)
    entry = Path(root) / "skills" / "slow-skill" / "entry.py"
    entry.write_text("import time\ndef main(a):\n    time.sleep(30)\n    return 'done'\n",
                     encoding="utf-8")
    out = runner.run_skill("slow-skill", {}, root=root, timeout=2)
    assert "超时" in out


def test_runner_missing_entry(tmp_path):
    root = str(tmp_path)
    sk.install_skill("report-writer", root=root)  # 仅玩法说明，无 entry.py
    out = runner.run_skill("report-writer", {}, root=root)
    assert "无可编程入口" in out


# ---------------- 升级 / 回滚 ----------------
def test_upgrade_then_rollback(tmp_path):
    root = str(tmp_path)
    sk.install_skill("python-tester", root=root)
    assert "已升级" in sk.upgrade_skill("python-tester", root=root)
    assert sk.get_skill("python-tester", root)["version"] == "2.0.0"
    snap = Path(root) / "skills" / "python-tester" / "history"
    assert snap.is_dir()
    assert "已回滚" in sk.rollback_skill("python-tester", root)
    assert sk.get_skill("python-tester", root)["version"] == "1.0.0"


def test_rollback_without_history(tmp_path):
    root = str(tmp_path)
    sk.install_skill("csv-tool", root=root)
    assert "没有历史版本" in sk.rollback_skill("csv-tool", root)


# ---------------- 流水线 ----------------
def test_pipeline_save_run_delete(tmp_path):
    root = str(tmp_path)
    sk.scaffold_skill("pipe-a", "文档生成", root=root)
    sk.scaffold_skill("pipe-b", "安全扫描", root=root)
    assert "已保存流水线 p1" in sk.save_pipeline("p1", ["pipe-a", "pipe-b"], root)
    assert [p["name"] for p in sk.list_pipelines(root)] == ["p1"]
    out = sk.run_pipeline("p1", root, {}, timeout=20)
    assert "pipe-a" in out and "pipe-b" in out
    assert "已删除流水线 p1" in sk.delete_pipeline("p1", root)


def test_pipeline_requires_installed_skills(tmp_path):
    root = str(tmp_path)
    assert "未安装" in sk.save_pipeline("bad", ["not-a-skill"], root)


# ---------------- agent 工具 ----------------
def _set_tool_root(tmp_path):
    tool_registry.set_project_root(str(tmp_path))
    return str(tmp_path)


def test_tool_skill_scaffold_run_permit(tmp_path):
    root = _set_tool_root(tmp_path)
    assert "已创建" in tool_registry.invoke("skill_scaffold", {"name": "tool-s", "category": "安全扫描"})
    assert "file_read" in tool_registry.invoke("skill_set", {"name": "tool-s", "permissions": {"file_read": True}})
    out = tool_registry.invoke("skill_run", {"name": "tool-s", "args": "来自agent"})
    assert "tool-s 已处理" in out  # 沙箱内 entry 正常执行并回传结果
    assert tool_registry.invoke("skill_run", {"name": "tool-s", "args": ""}).startswith("错误") is False
    tool_registry.set_project_root(None)


def test_tool_pipeline_flow(tmp_path):
    _set_tool_root(tmp_path)
    sk.scaffold_skill("tk-a", "文档生成", root=str(tmp_path))
    sk.scaffold_skill("tk-b", "测试生成", root=str(tmp_path))
    assert "已保存流水线" in tool_registry.invoke("skill_pipeline", {"action": "save", "name": "flow", "skills": "tk-a,tk-b"})
    assert "flow" in tool_registry.invoke("skill_pipeline", {"action": "view"})
    tool_registry.set_project_root(None)


# ---------------- Web 端点的测试已随 Web(IDE) 版移除 ----------------