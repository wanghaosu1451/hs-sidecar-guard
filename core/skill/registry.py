"""skill 生态：让 HS 智能体（Web IDE + CLI 同一内核）自己安装/枚举/运行/维护可复用技能包。

技能包 = <项目写入目录>/skills/<name>/ 下一个目录：
  - SKILL.md  玩法说明（agent 读取后按说明行事）
  - skill.json 元数据：version / category(八分类) / permissions(能力授权矩阵) / enabled
  - entry.py   可选：可被沙箱独立进程执行的可编程入口（核心优势）
  - history/   升级留存的旧版本快照，支持一键回滚

权限矩阵（每个 Skill 单独授权，防越权与恶意代码）：
  file_read / file_write / shell / mcp / train / network
运行遵循既有安全惯例：第三方 Skill 独立子进程、默认断网、限时（见 runner.py）。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

# 八分类标签（市场需求：按类筛选，快速定位）
CATEGORIES = ["代码重构", "测试生成", "前端页面生成", "数据库操作",
              "容器构建", "文档生成", "模型微调", "安全扫描"]

# 能力授权矩阵字段与默认值（安装时未声明则使用安全默认：全部关闭）
PERM_FIELDS = ("file_read", "file_write", "shell", "mcp", "train", "network")
_DEFAULT_PERMS = {k: False for k in PERM_FIELDS}

# 内置可离线安装的技能包目录（name -> {desc, category, perms} + SKILL.md 文本）
_BUILTIN_CATALOG: dict[str, dict[str, Any]] = {
    "python-tester": {
        "category": "测试生成",
        "perms": {"file_read": True, "file_write": True, "shell": True, "mcp": False, "train": False, "network": False},
        "text": (
            "# Skill: python-tester\n"
            "描述：为任一段 Python 模块自动生成并运行单元测试，输出靠 pytest 校验的结果。\n"
            "玩法：1) 读入目标代码；2) 生成 test_*.py；3) 用 run_shell 跑 pytest -q；4) 修复失败到通过。\n"
        ),
    },
    "report-writer": {
        "category": "文档生成",
        "perms": {"file_read": True, "file_write": True, "shell": False, "mcp": False, "train": False, "network": False},
        "text": (
            "# Skill: report-writer\n"
            "描述：把数据/结论整理成结构化 Markdown 报告并写入 artifacts/reports。\n"
            "玩法：先提纲要（背景/方法/结果/结论/附录），再逐节填充，最后校验格式与数据一致性。\n"
        ),
    },
    "csv-tool": {
        "category": "数据库操作",
        "perms": {"file_read": True, "file_write": True, "shell": False, "mcp": False, "train": False, "network": False},
        "text": (
            "# Skill: csv-tool\n"
            "描述：对 CSV 做清洗、去重、聚合统计的小工具链。\n"
            "玩法：读 head 看清表头与缺失，逐列清洗，输出统计与清洗后的 csv 到 artifacts。\n"
        ),
    },
}

_SKILL_DIR_NAME = "skills"
_PIPELINES_FILE = "_pipelines.json"
_HISTORY_DIR = "history"


def _skills_dir(root: str | None = None) -> Path:
    """技能包根目录：<项目写入目录>/skills。"""
    base = Path(root) if root else Path.cwd()
    return base / _SKILL_DIR_NAME


def _safe_name(name: str) -> str:
    """清洗技能名：小写、空格换连字符、去斜杠，杜绝路径穿越。"""
    return re.sub(r"[^a-z0-9\-_]", "", (name or "").strip().lower().replace(" ", "-"))


def _skill_dir(root: str, name: str) -> Path:
    return _skills_dir(root) / name


def _desc_of(content: str, meta: dict | None = None) -> str:
    """从 SKILL.md 提取描述：取“描述：”行，否则取首个非空标题后首句。"""
    if meta and meta.get("desc"):
        return meta["desc"]
    for line in content.splitlines():
        if line.startswith("描述："):
            return line.split("描述：", 1)[1].strip()
    strip = content.strip()
    m = re.search(r"[。\n]", strip)
    return strip[:60]


# ============================ 元数据读写 ============================

def _meta_path(dirp: Path) -> Path:
    return dirp / "skill.json"


def _read_meta(dirp: Path, default_name: str) -> dict:
    p = _meta_path(dirp)
    meta: dict = {
        "name": default_name, "version": "1.0.0", "category": "",
        "source": "local", "enabled": True,
        "permissions": dict(_DEFAULT_PERMS),
    }
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            meta.update(data)
        except Exception:  # noqa: BLE001
            pass
    meta["permissions"] = {k: bool(meta["permissions"].get(k, _DEFAULT_PERMS.get(k, False)))
                           for k in PERM_FIELDS}
    return meta


def _write_meta(dirp: Path, meta: dict) -> Path:
    dirp.mkdir(parents=True, exist_ok=True)
    p = _meta_path(dirp)
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ============================ 目录 / 安装 / 卸载 ============================

def list_catalog(category: str | None = None) -> list[dict[str, Any]]:
    """内置可安装的技能目录；可按八分类过滤。name/desc 与旧版保持一致，另附 version/category/perms。"""
    out = []
    for name, info in _BUILTIN_CATALOG.items():
        meta = {
            "name": name,
            "desc": _desc_of(info["text"], {"desc": _first_line_desc(info["text"])}),
            "version": "1.0.0",
            "category": info["category"],
            "source": "builtin",
            "installed": False,
            "permissions": dict(info["perms"]),
        }
        if category and meta["category"] != category:
            continue
        out.append(meta)
    return out


def _first_line_desc(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("描述："):
            return line.split("描述：", 1)[1].strip()
    return ""


def list_skills(root: str | None = None, category: str | None = None) -> list[dict[str, Any]]:
    """枚举已安装技能包，附版本/分类/授权/启用状态元数据。"""
    d = _skills_dir(root)
    out = []
    if d.is_dir():
        for child in sorted(d.iterdir()):
            if not child.is_dir() or child.name in (_HISTORY_DIR, "_pipelines"):
                continue
            f = child / "SKILL.md"
            if not f.is_file():
                continue
            content = f.read_text(encoding="utf-8", errors="replace")
            meta = _read_meta(child, child.name)
            meta["desc"] = _desc_of(content, meta)
            meta["path"] = str(f)
            meta["installed"] = True
            meta["has_entry"] = (child / "entry.py").is_file()
            meta["has_history"] = (child / _HISTORY_DIR).is_dir()
            meta["permissions"] = dict(meta["permissions"])
            out.append(meta)
    if category:
        out = [m for m in out if m.get("category") == category]
    return out


def get_skill(name: str, root: str | None = None) -> dict[str, Any]:
    """读取单个技能的完整元数据（含玩法说明）；未安装返回 {"error": ...}。"""
    name = _safe_name(name)
    d = _skill_dir(root or "", name) if root else _skills_dir() / name
    f = d / "SKILL.md"
    if not f.is_file():
        return {"name": name, "error": f"技能 {name} 未安装（/skill view 查看已装）"}
    content = f.read_text(encoding="utf-8", errors="replace")
    meta = _read_meta(d, name)
    meta["desc"] = _desc_of(content, meta)
    meta["path"] = str(f)
    meta["content"] = content
    meta["permissions"] = dict(meta["permissions"])
    meta["has_entry"] = (d / "entry.py").is_file()
    return meta


def get_skill_text(name: str, root: str | None = None) -> str:
    """兼容旧接口：返回玩法说明纯文本。"""
    meta = get_skill(name, root)
    if meta.get("error"):
        return meta["error"]
    return meta["content"]


def install_skill(name: str, source: str | None = None,
                  root: str | None = None, category: str | None = None,
                  perms: dict | None = None) -> str:
    """安装技能包。source 为 URL/绝对路径时从该处读取；否则从内置目录安装。

    内置目录外，还可安装带 metadata 的包目录（含 skill.json + entry.py，实现可编程技能）。
    """
    name = _safe_name(name)
    if not name:
        return "错误：技能名不能为空"
    existing = list_skills(root)
    if name in {x["name"] for x in existing}:
        return f"技能 {name} 已安装（可用 /skill upgrade 升级、/skill rollback 回滚）。"

    text: str | None = None
    meta: dict = {
        "name": name, "version": "1.0.0", "category": category or "",
        "source": "local", "enabled": True, "permissions": dict(_DEFAULT_PERMS),
    }
    # 可能作为“技能包目录”整体导入（含 entry.py / skill.json）
    imported_entries: dict[str, str] = {}
    if source:
        s = str(source).strip()
        if s.startswith(("http://", "https://")):
            try:
                body = _fetch(s)
            except Exception as e:  # noqa: BLE001
                return f"错误：无法从 {s} 拉取技能 - {e}"
            _merge_remote_payload(body, text_ref=text, meta=meta, files=imported_entries)
            return _finalize_install(name, meta, body, imported_entries, root, perms)
        src_path = Path(s)
        if src_path.is_file():
            body = src_path.read_text(encoding="utf-8", errors="replace")
            meta["source"] = s
            # 若 source 是 skill 包 JSON（含 text + metadata），则解析
            _merge_remote_payload(body, text_ref=text, meta=meta, files=imported_entries)
            return _finalize_install(name, meta, body, imported_entries, root, perms)
        if src_path.is_dir():
            return _install_package_dir(name, src_path, root, category)
        if name in _BUILTIN_CATALOG:
            meta = _builtin_meta(name)
        else:
            return f"错误：内置目录没有 {name}，且 source={s} 既非 URL/文件/包目录"
        text = _BUILTIN_CATALOG[name]["text"]
        return _finalize_install(name, meta, text, {}, root, perms)
    # 无 source：内置目录安装
    if name not in _BUILTIN_CATALOG:
        installed = ", ".join(x["name"] for x in list_catalog()) or "（空）"
        return f"错误：内置目录没有 {name}。可安装：{installed}；或给 source=URL/包目录"
    meta = _builtin_meta(name)
    text = _BUILTIN_CATALOG[name]["text"]
    return _finalize_install(name, meta, text, {}, root, perms)


def _builtin_meta(name: str) -> dict:
    info = _BUILTIN_CATALOG[name]
    return {
        "name": name, "version": "1.0.0", "category": info["category"],
        "source": "builtin", "enabled": True, "permissions": dict(info["perms"]),
    }


def _fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=10) as r:  # noqa: S310 受控URL来源由用户指定
        return r.read().decode("utf-8", errors="replace")


def _merge_remote_payload(body: str, text_ref: str | None, meta: dict,
                          files: dict[str, str]) -> None:
    """若 body 是 skill 包 JSON，则解析 text/版本/分类/权限；否则视为纯 SKILL.md 文本。"""
    if text_ref:
        return  # 已是显式 SKILL.md 文本，无需解析
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except Exception:  # noqa: BLE001
            return
        if isinstance(data, dict) and "text" in data:
            meta["version"] = str(data.get("version", meta.get("version", "1.0.0")))
            meta["category"] = str(data.get("category", meta.get("category", "")))
            perms = {k: bool(v) for k, v in data.get("permissions", {}).items()}
            meta["permissions"] = {k: perms.get(k, _DEFAULT_PERMS.get(k, False)) for k in PERM_FIELDS}
            for k, v in (data.get("files") or {}).items():
                if isinstance(v, str):
                    files[k] = v
            body = data["text"]


def _finalize_install(name: str, meta: dict, text: str, files: dict[str, str],
                      root: str | None, perms: dict | None = None) -> str:
    d = _skill_dir(root or "", name)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(text, encoding="utf-8")
    if perms:
        meta["permissions"] = {k: bool(perms.get(k, meta["permissions"].get(k, False)))
                               for k in PERM_FIELDS}
    meta.setdefault("permissions", dict(_DEFAULT_PERMS))
    meta["permissions"] = {k: bool(meta["permissions"].get(k, False)) for k in PERM_FIELDS}
    _write_meta(d, meta)
    for fname, fcontent in files.items():
        # 只允许落普通文件，杜绝写入到越界路径
        fp = (d / fname).resolve()
        if fp.is_relative_to(d.resolve()):
            fp.write_text(str(fcontent), encoding="utf-8")
    return f"已安装技能 {name} v{meta.get('version', '1.0.0')} -> {d}"


def _install_package_dir(name: str, src_dir: Path, root: str | None,
                         category: str | None) -> str:
    """从本地 skill 包目录安装（可含 SKILL.md / skill.json / entry.py）。"""
    d = _skill_dir(root or "", name)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "version": "1.0.0", "category": category or "",
            "source": str(src_dir), "enabled": True, "permissions": dict(_DEFAULT_PERMS)}
    for fname in ("SKILL.md", "skill.json", "entry.py", "SKILL.md.jinja"):
        sp = (src_dir / fname).resolve()
        if sp.is_file() and sp.is_relative_to(src_dir.resolve()):
            target = (d / fname).resolve()
            if target.is_relative_to(d.resolve()):
                (d / fname).write_bytes(sp.read_bytes())
    if (d / "skill.json").is_file():
        meta = _read_meta(d, name)
    if category:
        meta["category"] = category
    _write_meta(d, meta)
    return f"已安装技能 {name} v{meta.get('version', '1.0.0')} -> {d}"


def uninstall_skill(name: str, root: str | None = None) -> str:
    """卸载已安装技能包。"""
    name = _safe_name(name)
    d = _skill_dir(root or "", name)
    if not d.is_dir():
        return f"技能 {name} 未安装"
    for p in list(d.glob("*")):
        if p.is_dir():
            for q in list(p.glob("**/*")):
                q.unlink(missing_ok=True)
            p.rmdir()
        else:
            p.unlink(missing_ok=True)
    d.rmdir()
    return f"已卸载技能 {name}"


# ============================ 权限 / 启用 ============================

def _require_installed(name: str, root: str | None) -> Path:
    name = _safe_name(name)
    d = _skill_dir(root or "", name)
    if not (d / "SKILL.md").is_file():
        raise ValueError(f"技能 {name} 未安装")
    return d


def set_permissions(name: str, perms: dict, root: str | None = None) -> str:
    """按技能单独授予能力。每项都要显式 True 才会打开，默认全部关闭。"""
    d = _require_installed(name, root)
    meta = _read_meta(d, name)
    for k, v in (perms or {}).items():
        if k in PERM_FIELDS:
            meta["permissions"][k] = bool(v)
    _write_meta(d, meta)
    on = [k for k in PERM_FIELDS if meta["permissions"][k]]
    return f"技能 {name} 权限已更新：{'、'.join(on) if on else '全部拒绝（无授权能力）'}"


def set_enabled(name: str, enabled: bool, root: str | None = None) -> str:
    """禁用/启用技能（禁用后不可运行，但保留安装）。"""
    d = _require_installed(name, root)
    meta = _read_meta(d, name)
    meta["enabled"] = bool(enabled)
    _write_meta(d, meta)
    return f"技能 {name} 已{'启用' if enabled else '禁用'}。"


# ============================ 版本升级 / 回滚 ============================

def _snapshot_to_history(d: Path) -> str:
    """把当前顶层版本快照进 history/<version>/，返回版本号；返回空串表示无内容可快照。"""
    meta = _read_meta(d, d.name)
    ver = str(meta.get("version", "1.0.0"))
    hist = d / _HISTORY_DIR / ver
    hist.mkdir(parents=True, exist_ok=True)
    for fname in ("SKILL.md", "skill.json", "entry.py"):
        src = d / fname
        if src.is_file():
            (hist / fname).write_bytes(src.read_bytes())
    return ver


def _next_version(ver: str) -> str:
    try:
        major, minor = str(ver).split(".")
        return f"{major}.{int(minor) + 1}"
    except Exception:  # noqa: BLE001
        return "2.0.0"


def upgrade_skill(name: str, source: str | None = None, root: str | None = None,
                  category: str | None = None) -> str:
    """升级技能：把当前版本快照进 history，再按新来源重装并递增版本号。"""
    name = _safe_name(name)
    d = _skill_dir(root or "", name)
    if not (d / "SKILL.md").is_file():
        return f"错误：技能 {name} 未安装，无法升级"
    old = _read_meta(d, name)
    # 拉取新内容（内置或来源）
    if source:
        if str(source).strip() in _BUILTIN_CATALOG:
            info = _BUILTIN_CATALOG[str(source).strip()]
            new_text = info["text"]
            new_perms = dict(info["perms"])
            new_cat = category or info["category"]
        elif str(source).strip().startswith(("http://", "https://")) or Path(str(source).strip()).is_file():
            body = _fetch(str(source).strip()) if str(source).startswith("http") \
                else Path(str(source).strip()).read_text(encoding="utf-8", errors="replace")
            files: dict[str, str] = {}
            _merge_remote_payload(body, None, old, files)
            new_text = body
            new_perms = old["permissions"]
            new_cat = category or old.get("category", "")
        else:
            return f"错误：升级来源无效（{source}），请给内置名/URL/文件"
    elif name in _BUILTIN_CATALOG:
        info = _BUILTIN_CATALOG[name]
        new_text = info["text"]
        new_perms = dict(info["perms"])
        new_cat = category or info["category"]
    else:
        return f"错误：技能 {name} 非内置且未提供升级来源"
    _snapshot_to_history(d)
    new_ver = _next_version(str(old.get("version", "1.0.0")))
    (d / "SKILL.md").write_text(new_text, encoding="utf-8")
    meta = {
        "name": name, "version": new_ver, "category": new_cat,
        "source": source or old.get("source", "local"),
        "enabled": old.get("enabled", True),
        "permissions": {k: bool(new_perms.get(k, old["permissions"].get(k, False)))
                        for k in PERM_FIELDS},
    }
    _write_meta(d, meta)
    return f"技能 {name} 已升级 {old.get('version')} -> {new_ver}（旧版已入 history）"


def rollback_skill(name: str, root: str | None = None) -> str:
    """回滚到上一安装版本（取最新历史快照还原，随后移除该快照）。"""
    name = _safe_name(name)
    d = _skill_dir(root or "", name)
    if not (d / "SKILL.md").is_file():
        return f"错误：技能 {name} 未安装，无法回滚"
    hist = d / _HISTORY_DIR
    if not hist.is_dir():
        return f"技能 {name} 没有历史版本可回滚"
    versions = sorted([p.name for p in hist.iterdir() if (p / "SKILL.md").is_file()])
    if not versions:
        return f"技能 {name} 没有历史版本可回滚"
    target = hist / versions[-1]
    for fname in ("SKILL.md", "skill.json", "entry.py"):
        src = target / fname
        if src.is_file():
            (d / fname).write_bytes(src.read_bytes())
    # 移除该快照，并同步版本递减
    import shutil
    shutil.rmtree(target)
    meta = _read_meta(d, name)
    meta["version"] = str(meta.get("version", "1.0.0"))
    _write_meta(d, meta)
    return (f"技能 {name} 已回滚到历史版本 v{meta['version']}，"
            f"剩余历史 {len(versions) - 1} 个")


# ============================ 开发脚手架 ============================

_SCAFFOLD_MD = (
    "# Skill: {name}\n"
    "描述：{desc}\n"
    "玩法：说明本技能要完成的目标与执行步骤（agent 会据此行动）。\n"
)

_SCAFFOLD_ENTRY = (
    "# -*- coding: utf-8 -*-\n"
    "\"\"\"{name} 的可编程入口：在沙箱独立进程中被运行。\n"
    "函数签名固定为 main(args: dict) -> str，返回结果文本。\"\"\"\n"
    "\n"
    "def main(args: dict) -> str:\n"
    "    # args 是调用方传入的入参（如 {\"target\": \"...\"}）\n"
    "    target = args.get(\"target\", \"\")\n"
    "    return f\"{name} 已处理 target={target or '(空)'}\"\n"
)


def scaffold_skill(name: str, category: str = "文档生成",
                   desc: str = "", root: str | None = None) -> str:
    """生成一个可编程 Skill 脚手架：skill.json + SKILL.md + entry.py。"""
    name = _safe_name(name)
    if not name:
        return "错误：技能名不能为空"
    if category not in CATEGORIES:
        return f"错误：分类不在八分类内（{'、'.join(CATEGORIES)}）"
    d = _skill_dir(root or "", name)
    if (d / "SKILL.md").is_file():
        return f"错误：技能 {name} 已存在，无法重复创建"
    d.mkdir(parents=True, exist_ok=True)
    desc = desc or f"{name} 技能的玩法说明"
    (d / "SKILL.md").write_text(_SCAFFOLD_MD.replace("{name}", name).replace("{desc}", desc), encoding="utf-8")
    (d / "entry.py").write_text(_SCAFFOLD_ENTRY.replace("{name}", name), encoding="utf-8")
    meta = {
        "name": name, "version": "1.0.0", "category": category,
        "source": "scaffold", "enabled": True, "permissions": dict(_DEFAULT_PERMS),
    }
    _write_meta(d, meta)
    return f"已创建技能脚手架 {name}（分类：{category}）-> {d}；用 /skill permit 授权能力后即可运行"


# ============================ 流水线（多个 Skill 串联） ============================

def _pipelines_path(root: str | None) -> Path:
    return _skills_dir(root) / _PIPELINES_FILE


def _load_pipelines(root: str | None) -> dict[str, list[str]]:
    p = _pipelines_path(root)
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {k: list(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_pipelines(root: str | None, data: dict[str, list[str]]) -> None:
    _pipelines_path(root).parent.mkdir(parents=True, exist_ok=True)
    _pipelines_path(root).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_pipeline(name: str, skills: list[str], root: str | None = None) -> str:
    """保存/更新一条流水线：串联多个技能按序执行。"""
    name = (name or "").strip()
    if not name:
        return "错误：流水线名不能为空"
    cleaned = [_safe_name(s) for s in (skills or []) if _safe_name(s)]
    if not cleaned:
        return "错误：流水线至少需要 1 个技能"
    missing = [s for s in cleaned if not get_skill(s, root).get("path")]
    if missing:
        return "错误：以下技能未安装：" + "、".join(missing)
    data = _load_pipelines(root)
    data[name] = cleaned
    _save_pipelines(root, data)
    return f"已保存流水线 {name}：{' → '.join(cleaned)}"


def list_pipelines(root: str | None = None) -> list[dict[str, Any]]:
    data = _load_pipelines(root)
    return [{"name": k, "skills": v} for k, v in data.items()]


def delete_pipeline(name: str, root: str | None = None) -> str:
    data = _load_pipelines(root)
    if name not in data:
        return f"流水线 {name} 不存在"
    del data[name]
    _save_pipelines(root, data)
    return f"已删除流水线 {name}"


def run_pipeline(name: str, root: str | None = None,
                 args: dict | None = None, timeout: int = 120) -> str:
    """按序在各技能沙箱内执行流水线；结果按步骤串行返回。"""
    from core.skill.runner import run_skill
    data = _load_pipelines(root)
    if name not in data:
        return f"错误：流水线 {name} 不存在（/skill pipeline view 查看）"
    steps = []
    for skill_name in data[name]:
        meta = get_skill(skill_name, root)
        if meta.get("error"):
            steps.append(f"[{skill_name}] 跳过：{meta['error']}")
            continue
        if not meta.get("enabled", True):
            steps.append(f"[{skill_name}] 跳过：已禁用")
            continue
        if not meta.get("has_entry"):
            steps.append(f"[{skill_name}] 跳过：无可编程入口，仅玩法说明")
            continue
        res = run_skill(skill_name, args or {}, root=root, timeout=timeout)
        head = res.replace("\n", " ")[:200]
        steps.append(f"[{skill_name}] {head}")
    return "\n".join(steps)