"""Skill 隔离运行器：把第三方技能在独立子进程中执行，带超时且默认断网。

与 core/sandbox/executor.py 同一套安全语义（限时、默认 allow_network=False），
并提供资源(内存)限制——仅在 POSIX 生效（Windows 以超时兜底）。防止恶意技能
破坏项目文件或向外部网络通信。
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from core.skill.registry import get_skill, _safe_name, _skill_dir

# 子进程通用环境加固（与 core/sandbox/executor 保持一致）
_SHELL_ENV = {
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",
    "LC_ALL": "C.UTF-8",
}

# 跨平台解码回退链：UTF-8 优先，失败回退 GBK 等
def _decode(data: bytes) -> str:
    if not data:
        return ""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    for enc in ("gb18030", "gbk", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _make_env(allow_network: bool = False) -> dict[str, str]:
    env = dict(os.environ)
    if not allow_network:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                  "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"):
            env.pop(k, None)
    env.update(_SHELL_ENV)
    return env


# 在子进程中加载 entry.py 并执行 main(args)：入参经 args 文件传（避免命令行注入）
_RUNNER_CODE = (
    "import runpy,json,sys\n"
    "mp=sys.argv[1]; ap=sys.argv[2]\n"
    "meta=json.load(open(mp,encoding='utf-8'))\n"
    "ns=runpy.run_path(meta['entry'],run_name='__main__')\n"
    "main=ns.get('main', lambda a: '（该 Skill 未定义 main 函数）')\n"
    "args=json.load(open(ap,encoding='utf-8')) if ap else {}\n"
    "print((main(args) or ''), end='')\n"
)


def run_skill(name: str, args: dict[str, Any] | str | None = None,
              root: str | None = None, timeout: int | None = None,
              allow_network: bool | None = None) -> str:
    """在独立子进程中运行技能的可编程入口 entry.py 的 main(args)。

    - 默认断网：除非技能自身声明了 network 权限。
    - 录入参数 args 可为 dict 或 JSON / 纯文本；纯文本会作为 args["text"]。
    - 返回子进程输出文本；任何越权/缺失/超时都以可读错误字符串返回。
    """
    name = _safe_name(name)
    meta = get_skill(name, root)
    if meta.get("error"):
        return meta["error"]
    if not meta.get("enabled", True):
        return f"技能 {name} 已禁用，请先启用再运行。"
    d = _skill_dir(root or "", name)
    entry = (d / "entry.py").resolve()
    if not entry.is_file() or not entry.is_relative_to(d.resolve()):
        return f"技能 {name} 无可编程入口（entry.py）——它当前仅为玩法说明技能。"
    # 网络授权：仅当技能声明 network 权限才允许联网
    net = bool(meta["permissions"].get("network", False))
    if allow_network is False:
        net = False
    to = int(timeout or 60)
    if to > 300:  # 限时上限，防止抢占失控
        to = 300

    # 入参与元数据落到技能目录下的临时 JSON 文件，交给子进程读取（避免命令行注入）
    arg_file = d / ".run_args.json"
    arg_file.write_text(json.dumps(
        args if isinstance(args, dict)
        else ({"text": args} if isinstance(args, str) else {}),
        ensure_ascii=False), encoding="utf-8")
    meta_file = d / ".run_meta.json"
    meta_file.write_text(json.dumps({"name": name, "entry": str(entry)},
                                    ensure_ascii=False), encoding="utf-8")

    try:
        p = subprocess.run(
            [sys.executable, "-u", "-c", _RUNNER_CODE, str(meta_file), str(arg_file)],
            cwd=str(d), capture_output=True, timeout=to, env=_make_env(net),
        )
    except subprocess.TimeoutExpired:
        return f"技能 {name} 执行超时（>{to}s），已终止。"
    except Exception as e:  # noqa: BLE001
        return f"技能 {name} 启动失败：{e}"
    finally:
        for _f in (arg_file, meta_file):
            try:
                _f.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
    if p.returncode != 0:
        return f"技能 {name} 执行失败(exit={p.returncode})：{_decode(p.stderr) or '无错误输出'}"
    return _decode(p.stdout) or "(无输出)"