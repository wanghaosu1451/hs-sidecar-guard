"""HS 终端版智能体。

用法示例：
  python cli.py                          # 交互式，每轮输入目标任务并自动执行
  python cli.py -d D:/my_proj "做一个计算器 web 项目"
  python cli.py -m deepseek/deepseek-chat -d ./ai_projects
  python cli.py --mode code -d ./ai_projects "写一个 CLI 工具"
  python cli.py --mode design "设计一个登录页"

前后端已统一为终端版（Web 已移除）：自主执行环 + 隔离沙箱跑码 + project_write 写到指定文件夹。
模式系统提示词单一来源：core.agent.agent.MODE_PROMPTS。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.markup import escape

from core.agent.agent import Agent, MODE_PROMPTS
from core.agent import tools as tool_registry
from core.llm.provider_catalog import PROVIDERS
from core.llm.keys import load as load_cfg

console = Console()

# 模式 -> 一句话角色说明，用于启动提示与 /mode 展示
_MODE_ROLES = {
    "work": "工作台模式：面向日常任务/报告/文档，先规划再执行、真实落盘。",
    "code": "代码模式：生产级全栈工程师，注重结构完整、沙箱验证跑通再交付。",
    "design": "设计模式：资深 UI/UX 与前端实现专家，产出高保真原型与页面文件。",
    "plan": "规划只读模式：只分析项目并输出开发方案，绝不修改任何本地文件。",
    "build": "构建执行模式：方案确认后全自动多文件开发、编译/测试、报错自动反思重试。",
}
_MODE_LABEL = {"work": "Work", "code": "Code", "design": "Design",
               "plan": "Plan", "build": "Build"}

# 写指令菜单（输入 / 即显示）
_SLASH_COMMANDS = [
    ("/mode work|code|design|plan|build", "切换内核角色模式"),
    ("/context", "查看当前注入模型的本地上下文（代码库地图/项目配置）"),
    ("/init", "首启自动配置：校验路径/模型/沙箱，缺则生成 AGENTS.md"),
    ("/model [status|probe|set]", "查看/探测/切换 LLM 模型配置"),
    ("/config set <sec> <key> <val>｜view [sec]", "通用 JSON 直配/查看（免手改 settings.json）"),
    ("/config opencode [path]｜/config（无参向导）", "导入 OpenCode 配置 或 终端交互式配模型"),
    ("/budget [retry on|off] [max_steps N] [max_tokens N]", "查看/设置 llm.retry 与预算上限"),
    ("/toollist", "列出智能体当前可调用的全部工具"),
    ("/pending", "查看待我确认的删除/不可逆操作清单"),
    ("/approve <序号>", "同意待确认操作（用户点击式确认删除等）"),
    ("/trust on|off", "完全托管模式开关：on 放行不可逆操作全自动执行"),
    ("/thinking on|off", "开关云端推理模型(如 DeepSeek-R1)思考过程的折叠展示"),
    ("/sub spawn <prompt>", "后台并行下发一个子任务（返回 task_id）"),
    ("/sub list", "列出所有子任务状态"),
    ("/sub show <id> [--keep]", "取回某子任务结果（默认取走即删，--keep 保留）"),
    ("/about", "关于 HS（版本 / 内核角色）"),
    ("/authors", "作者与致谢（AUTHORS.md）"),
    ("/mcp view", "查看 MCP 生态市场"),
    ("/mcp add <name> <cmd> [args]", "添加 MCP 服务器"),
    ("/mcp del <name>", "删除自定义 MCP 服务器"),
    ("/train status", "训练环境自检"),
    ("/train ced|fp4|opd", "研究创新：CED-MoE / FP4-QAT / 策略蒸馏"),
    ("/train lora <数据> [基座]", "LoRA 微调"),
    ("/train full <数据>", "完整训练"),
    ("/skill view [分类]", "查看已装/内置技能（可按八分类筛选）"),
    ("/skill install <name> [source]", "安装技能包（内置或 URL/文件/包目录）"),
    ("/skill uninstall <name>", "卸载技能"),
    ("/skill run <name> [args]｜/skill pipeline <op>", "运行技能/流水线（沙箱执行）"),
    ("/skill permit <name> k=v k=v｜/skill disable <name> [on|off]", "授权能力/启用禁用"),
    ("/skill new <name> [分类]", "创建自定义技能脚手架"),
    ("/skill upgrade <name> [source]｜/skill rollback <name>", "升级到新版本/回滚旧版"),
    ("/computer view|shot [x] [y]|type <文本>|key <快捷键>|open <应用>", "Computer Use 桌面测试"),
    ("/scan", "定时安全巡检：密钥/高危shell/危险读写/漏洞依赖"),
    ("/audit", "操作审计可视化报表（按日/周/动作/高危统计）"),
    ("/undo [序号]", "基于自动快照一键回滚写坏的文件（显示最近快照，默认回滚最新）"),
    ("/diff <文件>", "对比当前文件与最近快照的差异，回滚前先看清改动"),
    ("/health", "系统自检：Python/配置/网关/沙箱/MCP/出网 逐项✓✗+修复建议"),
    ("/help", "显示全部写指令"),
    ("q | exit", "退出"),
]


def _show_slash_menu(prefix: str | None = None) -> None:
    """打印匹配前缀的写指令；prefix 为空或未知时输出全部。"""
    p = (prefix or "").strip()
    items = _SLASH_COMMANDS
    if p and p != "/":
        q = p.lstrip("/")
        items = [(cmd, d) for cmd, d in items if q in cmd]
    if not items:
        console.print(f"[yellow]未找到匹配指令：{escape(p)}[/]")
        items = _SLASH_COMMANDS
    lines = [f"  [bold cyan]{escape(cmd)}[/]  [dim]— {escape(desc)}[/]" for cmd, desc in items]
    console.print(Panel("\n".join(lines), title="写指令（键入 / 前缀）", border_style="blue"))


def _authors_text() -> str:
    """读取仓库根 AUTHORS.md 内容；缺失时回退内置精简信息。"""
    from pathlib import Path
    p = Path(__file__).resolve().parent / "AUTHORS.md"
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    return ("# HS — 极简工程风白底 AI 智能体工具\n"
            "版本 0.1.0 · 内核角色 Work / Code / Design\n"
            "作者：HS")


def _mask_key(k: str) -> str:
    k = str(k or "")
    if not k:
        return "(未设置)"
    return k[:3] + "****" + k[-2:] if len(k) > 8 else "****"


def _parse_cfg_value(raw: str):
    """把命令行字符串参数转成合适的类型：bool/int/float/json，否则当字符串。"""
    s = (raw or "").strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low == "null":
        return None
    # 纯数字
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # JSON 结构（数组/对象）
    if s[:1] in ("[", "{"):
        import json as _json
        try:
            return _json.loads(s)
        except Exception:  # noqa: BLE001
            pass
    return s


def _cmd_model(parts: list[str]) -> None:
    """模型配置：/model [status] 查看；/model probe [base_url] 探测；/model set <provider> [base_url] 切换。"""
    import urllib.request
    from core.llm.keys import update as update_cfg
    act = parts[1].strip().lower() if len(parts) > 1 else "status"
    if act in ("set", "switch"):
        if len(parts) < 3:
            console.print("[yellow]用法：/model set <provider> [base_url][/]")
            return
        provider = parts[2]
        update_cfg("llm", "provider", provider)
        if len(parts) > 3:
            update_cfg("llm", "base_url", parts[3])
        console.print(f"[bold green]已切换模型 → {provider}[/]")
        return
    if act == "probe":
        base = parts[2] if len(parts) > 2 else ""
        if not base:
            base = load_cfg().get("llm", {}).get("base_url", "")
        base = (base or "").strip().rstrip("/")
        api_key = load_cfg().get("llm", {}).get("api_key", "")
        if not base:
            console.print("[yellow]未配置 base_url，请先 /model set <provider> <base_url>[/]")
            return
        # 带鉴权头探测：优先 OpenAI 兼容 /models，兼容 Ollama /api/tags
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload, kind = None, None
        for ep, ep_kind in ((base + "/models", "openai"), (base + "/api/tags", "ollama")):
            try:
                req = urllib.request.Request(ep, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310
                    payload = json.loads(r.read().decode("utf-8", errors="replace"))
                    kind = ep_kind
                    break
            except Exception:  # noqa: BLE001
                continue
        if payload is None:
            console.print("[red]探测失败：端点不可达或鉴权失败（请确认 llm.api_key 正确）[/]")
            return
        names = []
        if kind == "openai":
            for m in payload.get("data", []):
                if m.get("id"):
                    names.append("openai/" + str(m["id"]))
        else:
            for m in payload.get("models", []):
                if m.get("name"):
                    names.append("ollama/" + str(m["name"]))
        if not names:
            console.print("[yellow]端点可达，但未发现模型[/]")
            return
        lines = [f"  [bold cyan]{escape(n)}[/]  [dim]→  /model set {escape(n)}[/]" for n in names]
        console.print(Panel("\n".join(lines),
                            title=f"探测到 {len(names)} 个模型（{escape(base)}）",
                            border_style="green"))
        return
    # status（默认）
    llm = load_cfg().get("llm", {})
    kwargs = llm.get("model_kwargs")
    console.print(Panel(
        f"Provider：    [bold cyan]{escape(llm.get('provider', '(未设置)'))}[/]\n"
        f"Base URL：    [dim]{escape(llm.get('base_url') or '(未设置)')}[/]\n"
        f"API Key：     {_mask_key(llm.get('api_key'))}\n"
        f"温度 / max_tokens：{llm.get('temperature', 0.7)} / {llm.get('max_tokens', 2048)}\n"
        f"model_kwargs：{escape(json.dumps(kwargs, ensure_ascii=False)) if kwargs else '(无)'}\n"
        f"\n[dim]/model probe [base_url] 探测可用模型；/model set <provider> [base_url] 切换[/]",
        title="模型配置", border_style="blue"))


def _cmd_config(parts: list[str]) -> None:
    """OpenCode 配置兼容 + 终端直配：/config opencode [path] 或 /config（无参向导）。

    - /config opencode [path]：导入 opencode.json 的远端 provider 配置。
    - /config（无参/交互向导）：在终端一条条问 base_url / api_key / provider，
      边填边落盘，配完即生效，无需手改文件。
    """
    from core.llm import opencode as _oc
    from core.llm.keys import load as _cfg_load
    act = parts[1].strip().lower() if len(parts) > 1 else ""
    # ---------------- 通用 JSON 直配：/config set <section> <key> <value> ----------------
    if act in ("set", "=", "write"):
        if len(parts) < 5:
            console.print("[yellow]用法：/config set <section> <key> <value>[/]"
                          "\n        例：/config set sandbox timeout_seconds 120"
                          " ｜ /config set llm temperature 0.9"
                          " ｜ /config set sandbox allow_network true")
            return
        section, key = parts[2], parts[3]
        raw = " ".join(parts[4:])
        val = _parse_cfg_value(raw)
        from core.llm.keys import update as _upd
        # 敏感键做脱敏回显
        if key.lower() in ("api_key", "apikey", "token", "secret", "password"):
            _upd(section, key, str(val))
            console.print(f"[bold green]{section}.{key}[/] 已写入（脱敏）")
        else:
            try:
                val_show = json.dumps(val, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                val_show = str(val)
            _upd(section, key, val)
            console.print(f"[bold green]{section}.{key}[/] → [cyan]{escape(val_show)}[/]")
        return
    # ---------------- 通用 JSON 查看：/config view [section] ----------------
    if act in ("view", "get", "cat", "status"):
        section = parts[2].strip().lower() if len(parts) > 2 else ""
        data = _cfg_load()
        if section:
            seg = data.get(section, {})
            body = "\n".join(f"  [bold]{escape(str(k))}[/] = {escape(json.dumps(v, ensure_ascii=False))}"
                             for k, v in seg.items()) if isinstance(seg, dict) else str(seg)
            console.print(Panel(body or "（空）", title=f"配置 · {section}",
                                border_style="blue"))
        else:
            body = "\n".join(
                f"  [bold]{escape(str(k))}[/]: {escape(json.dumps(v, ensure_ascii=False))}"
                for k, v in data.items())
            console.print(Panel(body, title="全部配置（settings.json）", border_style="blue"))
        return
    if act in ("opencode", "import"):
        path = parts[2] if len(parts) > 2 else None
        if path:
            from pathlib import Path
            resolved = Path(path).expanduser()
            if not resolved.is_file():
                console.print(f"[red]找不到配置文件：{path}[/]")
                return
            path = resolved
        # 先解析，预览后落盘
        parsed = _oc.parse(path)
        if not parsed:
            console.print("[yellow]未找到可用的 opencode.json 或其默认模型（model=pid/mid）[/]")
            return
        console.print(Panel(
            f"来源：        [dim]{parsed['meta']['path']}[/]\n"
            f"Provider：    [bold cyan]{escape(parsed['provider'])}[/]\n"
            f"Base URL：    [dim]{escape(parsed['base_url'] or '(未设置)')}[/]\n"
            f"API Key：     {_mask_key(parsed['api_key'])}\n"
            f"model_kwargs：{escape(json.dumps(parsed['model_kwargs'], ensure_ascii=False)) or '(无)'}",
            title="OpenCode 配置预览（确认后落盘）", border_style="blue"))
        from core.llm.keys import update as _upd
        _upd("llm", "provider", parsed["provider"])
        _upd("llm", "base_url", parsed["base_url"])
        _upd("llm", "api_key", parsed["api_key"])
        if parsed["model_kwargs"]:
            _upd("llm", "model_kwargs", parsed["model_kwargs"])
        console.print(f"[bold green]已导入 OpenCode 模型 → {parsed['provider']}[/]"
                      f"（/model status 查看）")
        return
    # ---------------- 无参：终端交互式直配 ----------------
    from core.llm.keys import update as _upd
    llm = _cfg_load().get("llm", {})
    console.print(Panel("终端直配模型：留空则保留当前值，输入 q 取消",
                        title="/config 交互向导", border_style="blue"))
    base = console.input("Base URL（OpenAI 兼容端点，如 https://xxx/v1）> ").strip()
    if base.lower() == "q":
        console.print("[dim]已取消。[/]")
        return
    if base:
        _upd("llm", "base_url", base)
        console.print(f"[dim]  base_url → {escape(base)}[/]")
    api_key = console.input("API Key（取不到则留空）> ").strip()
    if api_key.lower() == "q":
        console.print("[dim]已取消。[/]")
        return
    if api_key:
        _upd("llm", "api_key", api_key)
        console.print("[dim]  api_key 已更新（已脱敏存储）[/]")
    prov = console.input("Provider（如 openai/gpt-4o，可先 /model probe 探测）> ").strip()
    if prov.lower() == "q":
        console.print("[dim]已取消。[/]")
        return
    if prov:
        _upd("llm", "provider", prov)
        console.print(f"[dim]  provider → {escape(prov)}[/]")
    console.print("[bold green]配置已生效[/]（/model status 查看 /model probe 探测）")


# ---------------- /init 首启自动配置 ----------------
# AGENTS.md 骨架文案：项目用途占位 + 项目约定（与本项目工具/能力绑定）。
_AGENTS_SKELETON = """# AGENTS.md —— 项目规则（本智能体每次会话自动注入，优先级最高）

## 项目用途（占位，请补充描述）
- 本项目要解决什么问题、面向谁、核心功能是什么。

## 项目约定（按需增删）
1. 写文件一律用 `project_write`（相对项目根路径）真实落盘。
2. 涉及代码先用 `verify_code` 在隔离沙箱跑通再采用。
3. 每次改动文件后会自动 git 提交（autocommit），确保每个版本可追溯。
4. 目标可拆分为相互独立的子片段时，用 `/sub spawn` 下发给后台并行子智能体。
5. 失败按真实报错反思修正；据实作答、不编造。
6. 回答用中文，简洁直白。
""".strip() + "\n"


def _first_run_wizard(force: bool = False) -> None:
    """首次启动配置向导：settings 里 base_url 未配置时自动弹出，三种来源可选，幂等。

    直接复用 core.llm.keys.update 落盘，配置过一次后不再打扰；可用 /model set 重配。
    """
    from core.llm.keys import load as _load, update as _update
    llm = _load().get("llm", {})
    base = (llm.get("base_url") or "").strip()
    needs = force or (not base)
    if not needs:
        return  # 已配置，静默跳过
    console.print(Panel(
        "[bold]欢迎使用 HS 智能体[/bold]\n\n"
        "检测到尚未配置大模型。首次使用请选择模型来源（以后可随时用 /model set 重配）：\n"
        "  [bold cyan]1[/] 远程 OpenAI 兼容（填 base_url + API Key，如你的模型网关）\n"
        "  [bold cyan]2[/] 本地 Ollama（127.0.0.1，无需密钥）\n"
        "  [bold cyan]3[/] 暂不配置，稍后用 /model set 或 /init",
        title="首次启动向导", border_style="green"))
    import builtins as _b
    try:
        choice = (_b.input("\n请选择 1/2/3 [默认 3] > ").strip() or "3")
    except (EOFError, KeyboardInterrupt):
        choice = "3"
    if choice == "1":
        try:
            b64 = _b.input("base_url（如 http://127.0.0.1:11434/v1）> ").strip()
            kw = _b.input("API Key（无则回车）> ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        _update("llm", "provider", "openai/gpt-4o-mini")
        if b64:
            _update("llm", "base_url", b64)
        if kw:
            _update("llm", "api_key", kw)
        console.print("[bold green]已保存。可运行 /model probe 校验连通，或直接开始任务。[/]")
    elif choice == "2":
        _update("llm", "provider", "ollama/qwen3")
        _update("llm", "base_url", "http://127.0.0.1:11434")
        console.print("[bold green]已设为本地 Ollama。请确认已在 127.0.0.1:11434 启动 Ollama，可 /model probe 校验。[/]")
    else:
        console.print("[dim]已跳过，可用 /model set 或 /init 随时配置。[/]")


def _cmd_init(root: str | None = None) -> None:
    """首启自动配置（幂等、可重复跑）：只读分析 + 写一个 AGENTS.md，不做破坏性操作。

    1) 确定项目根；2) 缺 AGENTS.md 则生成骨架；3) 校验展示配置并给补齐提示；
    4) 探测可用的沙箱与 hooks 规则文件。
    已有 AGENTS.md 时不覆盖（保留用户内容）并给出提示。
    """
    from core.llm.keys import load as ld_cfg
    _root = root or tool_registry.get_project_root() or "."
    rootp = Path(_root).expanduser().resolve()

    console.print(f"[dim]项目根：{rootp}[/]")

    # ---- 2) 生成 AGENTS.md（不覆盖已有） ----
    agents = rootp / "AGENTS.md"
    if agents.is_file():
        console.print("[yellow]已存在 AGENTS.md（未覆盖，保留你的约定）：[/]"
                      + f"[dim]{agents}[/]")
        console.print("  [dim]提示：如需重新生成，可手动删除后再次运行 /init[/]")
    else:
        try:
            rootp.mkdir(parents=True, exist_ok=True)
            agents.write_text(_AGENTS_SKELETON, encoding="utf-8")
            console.print(f"[green]✔ 已生成 AGENTS.md → {agents}[/]")
        except OSError as e:  # noqa: BLE001
            console.print(f"[red]写 AGENTS.md 失败：{e}[/]")

    # ---- 3) 校验并展示当前配置（友好清单，缺失项给补齐提示） ----
    cfg = ld_cfg()
    llm = cfg.get("llm", {}) or {}
    sandbox = cfg.get("sandbox", {}) or {}
    budget = cfg.get("budget", {}) or {}
    provider = llm.get("provider") or "(未设置)"
    base_url = llm.get("base_url") or "(未设置)"
    api_key = llm.get("api_key")
    key_txt = "✔ 已填" if api_key else "（未填）"
    if not api_key:
        key_txt += " 可 /model set <provider> 配置，或直接编辑 config/settings.json 的 llm.api_key"
    console.print(Panel(
        f"模型 provider：      [bold cyan]{escape(str(provider))}[/]\n"
        f"base_url：          [dim]{escape(str(base_url))}[/]"
        + ("" if base_url != "(未设置)" else "  /model set <provider> <base_url> 补齐\n"
           "   [dim]（本地 ollama 可用 /model set deepseek/deepseek-chat http://127.0.0.1:11434）[/]\n")
        + f"api_key：           {key_txt}\n"
        f"sandbox.allow_network：{'✔ 允许' if sandbox.get('allow_network') else '关闭（推荐，隔离更安全）'}\n"
        f"budget.max_steps：  [bold cyan]{budget.get('max_steps', 300)}[/]"
        f"   llm.retry：{llm.get('retry')}\n"
        f"\n[dim]查看/调整：/model status ｜ /budget ｜ 编辑 config/settings.json[/]",
        title="配置自检", border_style="blue"))

    # ---- 4) 探测可用的沙箱与 hooks 文件 ----
    try:
        from core.sandbox.sandbox import Sandbox, exec_cmd
        Sandbox, exec_cmd  # 仅验证可导入（依赖齐全），不真正执行
        sbox = {"available": True, "allow_network": sandbox.get("allow_network", False)}
    except Exception:  # noqa: BLE001
        sbox = None
    try:
        from core.agent import hooks as _hooks_mod
    except Exception:  # noqa: BLE001
        _hooks_mod = None
    lines = [f"隔离沙箱：{'✔ 可用（allow_network=' + str(sbox.get('allow_network')) + '）'
                    if isinstance(sbox, dict) and sbox.get('available')
                    else ('⚠ 未就绪（缺依赖或未配置）' if sbox is not None else '未知（跳过探测）')}"]
    if _hooks_mod is not None:
        hk_rules = _hooks_mod.rules_path(rootp)
        hk_script = _hooks_mod.script_path(rootp)
        if hk_rules.is_file() or hk_script.is_file():
            lines.append("hooks 钩子规则：✔ 已加载"
                          + (f"（{hk_rules.name}）" if hk_rules.is_file() else "")
                          + ("；脚本 已就绪" if hk_script.is_file() else ""))
            console.print(Panel("\n".join(lines), title="功能探测", border_style="green"))
        else:
            lines.append("hooks 钩子规则：未设置（可在项目根放 .hs_hooks.json 或 hooks.py）")
            console.print(Panel("\n".join(lines), title="功能探测", border_style="green"))
    else:
        console.print(Panel("\n".join(lines), title="功能探测", border_style="green"))
    console.print("[dim]提示：/init 仅做只读分析并生成 AGENTS.md，不做任何破坏性操作。[/]")


def _cmd_sub(parts: list[str]) -> None:
    """子任务命令：/sub spawn <prompt> ｜ /sub list ｜ /sub show <id> [--keep]。"""
    from core.agent import subagent
    act = parts[1].strip().lower() if len(parts) > 1 else ""
    if act == "spawn":
        prompt = " ".join(parts[2:]).strip()
        if not prompt:
            console.print("[yellow]用法：/sub spawn <prompt>（后台并行下发一个子任务）[/]")
            return
        console.print(subagent.spawn_task(prompt))
        return
    if act == "list":
        console.print(subagent.sub_list())
        return
    if act == "show":
        tid = parts[2].strip() if len(parts) > 2 else ""
        keep = "--keep" in parts[3:]
        if not tid:
            console.print("[yellow]用法：/sub show <id> [--keep]（可用 /sub list 查看可选 id）[/]")
            return
        console.print(subagent.sub_result(tid, keep=keep))
        return
    console.print("[yellow]用法：/sub spawn <prompt> ｜ /sub list ｜ /sub show <id> [--keep][/]")


def _cmd_thinking(parts: list[str]) -> None:
    """云端推理展示开关：/thinking on|off（折叠显示/隐藏 DeepSeek-R1 等的思考过程）。"""
    from core.llm.keys import update as _upd
    val = None
    if len(parts) > 1:
        s = parts[1].strip().lower()
        if s in ("on", "1", "true", "yes", "开", "显示"):
            val = True
        elif s in ("off", "0", "false", "no", "关", "隐藏"):
            val = False
    if val is None:
        console.print(f"[dim]当前：{'on（折叠展示思考）' if _thinking_enabled() else 'off（隐藏思考）'}[/]"
                      " ｜ 用法：/thinking on|off")
        return
    _upd("llm", "show_thinking", val)
    console.print(f"[bold green]思考过程折叠展示 → {'on' if val else 'off'}[/]"
                  "（下一次任务生效；本地/无推理模型时不受影响）")


def _cmd_budget(parts: list[str]) -> None:
    """预算/重试配置：/budget [retry on|off] [max_steps N] [max_tokens N]，缺省仅查看当前值。"""
    from core.llm.keys import update as update_cfg
    expr = " ".join(parts[1:])
    if expr:
        toks = expr.split()
        applied = []
        i = 0
        while i < len(toks):
            k = toks[i].lower()
            if k == "retry" and i + 1 < len(toks):
                val = toks[i + 1].lower() in ("1", "true", "on", "yes")
                update_cfg("llm", "retry", val)
                applied.append(f"llm.retry={val}")
                i += 2
            elif k in ("max_steps", "max_tokens") and i + 1 < len(toks):
                try:
                    val = int(toks[i + 1])
                except ValueError:
                    console.print(f"[red]{k} 需要整数[/]")
                    return
                update_cfg("budget", k, val)
                applied.append(f"budget.{k}={val}")
                i += 2
            else:
                console.print(f"[yellow]未知设置：{toks[i]}（可用 retry｜max_steps｜max_tokens）[/]")
                i += 1
        if applied:
            console.print("[bold green]已更新：[/]" + "，".join(applied))
    # 展示当前生效值
    cfg = load_cfg()
    llm = cfg.get("llm", {})
    budget = cfg.get("budget", {})
    console.print(Panel(
        f"llm.retry：        [bold cyan]{llm.get('retry', True)}[/]\n"
        f"budget.max_steps： [bold cyan]{budget.get('max_steps', 300)}[/]\n"
        f"budget.max_tokens：[bold cyan]{budget.get('max_tokens', 0)}[/]（0=不下发预算）\n"
        f"\n[dim]示例：/budget retry on max_steps 100 max_tokens 4000[/]",
        title="预算 / 重试配置", border_style="blue"))


def _cmd_toollist() -> None:
    """列出智能体当前可调用的全部工具（核心常用 + 按需扩展）。"""
    tools = tool_registry.list_all_tools()
    if not tools:
        console.print("[yellow]（暂无已注册工具）[/]")
        return
    core = [t for t in tools if t["core"]]
    ext = [t for t in tools if not t["core"]]
    lines = [f"  [bold cyan]{escape(t['name'])}[/]  [dim]— {escape(t['description'])}[/]"
             for t in core]
    console.print(Panel("\n".join(lines), title=f"常用工具（核心，共 {len(core)}）",
                        border_style="green"))
    if ext:
        el = [f"  [bold]{escape(t['name'])}[/]  [dim]— {escape(t['description'])}[/]"
              for t in ext]
        console.print(Panel("\n".join(el), title=f"扩展工具（按需调用，共 {len(ext)}）",
                            border_style="blue"))


def _cmd_about(which: str) -> None:
    """/about 或 /authors：输出版本信息与作者致谢。"""
    if which == "authors":
        console.print(Panel(_authors_text(), title="作者 / AUTHORS", border_style="blue"))
        return
    modes = " / ".join(_MODE_LABEL.get(m, m) for m in MODE_PROMPTS if m != "default")
    console.print(Panel(
        f"标识：HS [dim]v0.1.0[/]\n"
        f"定位：极简工程风白底 AI 智能体（主色 #4F46E5）\n"
        f"内核角色：{modes}\n"
        f"能力：对话 / 代码工程 / 语义图谱 / 模型训练 / MCP 生态 / 安全审计\n"
        f"运行/写目录：{tool_registry.get_project_root()}\n"
        f"\n[dim]输入 [/][bold cyan]/authors[/][dim] 查看作者与致谢。[/]",
        title="关于 HS", border_style="blue"))


def _read_task(prompt: str) -> str:
    """读取一行指令/任务。TTY 下实时识别 / 前缀并提示写指令；非 TTY（管道/测试）退回标准输入，保持可测。"""
    try:
        is_tty = sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        is_tty = False
    if not is_tty:
        # 非 TTY（管道/测试）：退回标准输入；菜单在 interactive 中按需弹出。
        return console.input(prompt).strip()
    # ---- 逐键读取，支持退格/Enter，键入 / 时实时弹出指令 ----
    _is_posix = sys.platform.startswith("win") is False
    _fd = None
    _old = None
    try:
        if _is_posix:
            import termios
            import tty as _tty
            _fd = sys.stdin.fileno()
            _old = termios.tcgetattr(_fd)
            try:
                _tty.setraw(_fd)
            except Exception:  # noqa: BLE001
                pass

            def getch() -> str:
                return sys.stdin.read(1)
        else:
            import msvcrt  # Windows

            def getch() -> str:
                return msvcrt.getwch()
    except Exception:  # noqa: BLE001
        getch = None

    def restore() -> None:
        if _is_posix and _fd is not None and _old is not None:
            try:
                import termios
                termios.tcsetattr(_fd, termios.TCSADRAIN, _old)
            except Exception:  # noqa: BLE001
                pass

    buf: list[str] = []
    try:
        while True:
            k = getch() if getch else input()
            if k in ("\r", "\n"):
                console.print()
                break
            if k in ("\x03", "\x04"):  # Ctrl+C / Ctrl+D
                raise KeyboardInterrupt
            if k in ("\x08", "\x7f"):  # 退格
                if buf:
                    buf.pop()
                    console.print("\b \b", end="")
                continue
            if k in ("\x1b", "\t"):  # ESC / Tab
                continue
            buf.append(k)
            console.print(k, end="")
            if "".join(buf) == "/":
                console.print()
                _show_slash_menu(None)
                console.print(prompt[:-1] + "/", end="")
    finally:
        restore()
    return "".join(buf).strip()


def _pick_provider(model: str | None) -> str:
    if model and model in PROVIDERS or (model and "/" in model):
        return model
    cfg = load_cfg()
    return model or cfg.get("llm", {}).get("provider", "openai/gpt-4o-mini")


# HS-Droid 吉祥物：紧凑版像素机器人（双天线 + 发光护目），主体用 █ 块构成。
_MASCOT = (
    "   ▀▄▀██▀▄▀   ",
    " ▄▀██████▀▄  ",
    "████████████ ",
    "██▀▀██▀▀██  ",
    "██▄▄████▄▄██ ",
    "████████████ ",
    " ▀▀▀▀▀▀▀▀▀▀ ",
)


def _banner(model: str = "", cwd: str | None = None) -> str:
    """HS 终端智能体 v1.0 入口横幅。

    布局：左侧保留 _MASCOT 吉祥物（用户指定不动），右侧新增信息面板：
        HS v1.0.0
        主模型  ·  使用方式
        基准agent  ·  版本
        向量库  ·  量化  ·  条目数
        工作目录...
    """
    import os
    from pathlib import Path

    provider = model or load_cfg().get("llm", {}).get("provider", "本地模型")
    cur = cwd or os.getcwd()

    # —— 基准agent 状态 ——
    # 优先读 hs_core_v4（刚重训的 LoRA），fallback 到 sidecar_v3，最后纯规则
    v4_path = Path("artifacts/hs_core_v4/adapter")
    v3_path = Path("artifacts/sidecar_v3/adapter")
    if v4_path.is_dir():
        sidecar_info = "hs_core_v4 · LoRA"
    elif v3_path.is_dir():
        sidecar_info = "sidecar_v3 · LoRA"
    else:
        sidecar_info = "规则级"

    # —— 向量库状态 ——
    try:
        from core.sidecar.vector_store import SidecarVectorStore
        vs = SidecarVectorStore()
        vs.load_or_build()
        st = vs.status()
        drift_n = st.get("drift", {}).get("entries", 0)
        shell_n = st.get("shell", {}).get("entries", 0)
        cross_n = st.get("cross", {}).get("entries", 0)
        vec_info = f"已就绪 · ScalarQuant 8bit · {drift_n+shell_n+cross_n} 条"
    except Exception:
        vec_info = "未加载"

    # —— 主模型标签 ——
    label = provider.split("/")[-1] if provider else "本地"
    usage = "终端 Sidecar"
    base_url = (load_cfg().get("llm", {}) or {}).get("base_url") or ""
    src_tag = ("API: " + base_url.replace("https://", "").split("/")[0]) if base_url else "本地推理"

    # —— right column ——
    right_lines = [
        f"  [bold]HS v1.0.0[/]",
        f"  [cyan]{label}[/] · {usage}",
        f"  [dim]{src_tag}[/]",
        f"  [yellow]基准agent[/] · {sidecar_info}",
        f"  [green]向量库[/] · {vec_info}",
        f"  [dim]{cur}[/]",
    ]
    right_text = "\n".join(right_lines)

    # —— 左右拼接 ——
    mascot_lines = list(_MASCOT)
    # 对齐到相同高度
    n = max(len(mascot_lines), len(right_lines))
    mascot_lines += [""] * (n - len(mascot_lines))

    # rich markup 的 Panel 直接用，不手动拼 width
    from rich.columns import Columns
    from rich.text import Text
    return Columns(
        [
            Text("\n".join(mascot_lines), style="cyan"),
            Panel(right_text, border_style="blue", expand=True),
        ],
        equal=False,
    )


# 实时状态：工具名 → 清晰的中文动作描述（供执行中进度面板实时展示"正在干什么"）
_TOOL_ACTIONS = {
    "project_write": "✍️ 正在写入/新建文件",
    "write_file": "✍️ 正在写入文件",
    "patch_file": "🩹 正在修改/打补丁到文件",
    "read_file": "📄 正在读取文件",
    "list_dir": "📂 正在浏览目录",
    "list_project_files": "📂 正在列出项目文件",
    "delete_file": "🗑️ 正在删除文件",
    "verify_code": "🧪 正在沙箱验证代码",
    "run_shell": "⚙️ 正在沙箱运行命令",
    "getenv": "🔧 正在读取环境变量",
    "file_upload": "📤 正在上传/登记文件",
    "graph_query": "🧭 正在查询项目语义图谱",
    "graph_query_file": "🧭 正在查询文件关联图谱",
    "repo_read": "📚 正在读取仓库上下文",
    "read_history": "📚 正在读取历史记录",
    "diff_file": "📑 正在查看文件差异",
    "repo_remember": "🧠 正在记录仓库记忆",
    "code_review": "🛡️ 正在代码审查",
    "vuln_scan": "🛡️ 正在漏洞/危险扫描",
    "audit_read": "🧾 正在读取操作审计",
    "gen_tests": "🧪 正在生成测试",
    "gen_docker": "🐳 正在生成 Dockerfile",
    "gen_doc": "📝 正在生成文档",
    "sub_spawn": "🧩 正在下发后台子任务",
    "sub_list": "📋 正在列出子任务状态",
    "sub_result": "📥 正在取回子任务结果",
    "mcp_call": "🔗 正在调用 MCP 工具",
    "approve_destructive": "🔐 正在确认删除/不可逆操作",
    "pending_destructive": "🔐 正在查看待确认清单",
    "set_destructive_policy": "🔐 正在调整删除托管策略",
    "snapshot_list": "📸 正在列出快照",
    "snapshot_rollback": "⏪ 正在回滚快照",
    "hello": "👋 正在执行 hello 自检",
}
# 路由子智能体：router:planner / router:coder / router:checker
_ROUTER_ACTIONS = {
    "planner": "🗂️ 规划中…（子智能体）",
    "coder": "🖥️ 编码中…（子智能体）",
    "checker": "🧪 校验中…（子智能体）",
}

# 从工具参数里抽取"目标文件/命令"，让"正在干什么"写得更清楚
_ACTION_ARG_KEYS = ("relative_path", "file_path", "path", "local_path", "target")


def _action_arg_hint(name: str, args) -> str:
    if not args:
        return ""
    try:
        # 部分参数已经是 dict，部分可能是 JSON 字符串
        obj = json.loads(args) if isinstance(args, str) else args
    except Exception:  # noqa: BLE001
        return ""
    if not isinstance(obj, dict):
        return ""

    def _pick(*keys):
        for k in keys:
            v = obj.get(k)
            if v:
                return str(v)
        return None

    if name in ("run_shell",):
        cmd = _pick("command", "cmd", "script")
        return f"：{cmd[:60]}" if cmd else ""
    if name in ("project_write", "write_file", "patch_file", "read_file",
                "delete_file", "file_upload"):
        p = _pick(*_ACTION_ARG_KEYS)
        return f"：{p[:60]}" if p else ""
    return ""


def _action_text(name: str, args=None) -> str:
    """返回当前动作的清晰中文描述（含目标文件，尽量"写清楚"）。"""
    if name.startswith("router:"):
        role = name.split(":", 1)[1]
        return _ROUTER_ACTIONS.get(role, f"🔀 子智能体协作中…（{role}）")
    base = _TOOL_ACTIONS.get(name, f"🛠️ 正在调用工具 {name}")
    if name in ("project_write", "write_file", "patch_file", "read_file",
                "delete_file", "file_upload", "run_shell"):
        return base + _action_arg_hint(name, args)
    return base


def _thinking_enabled() -> bool:
    from core.llm.keys import get as _cfg_get
    try:
        return bool(_cfg_get("llm", "show_thinking"))
    except Exception:  # noqa: BLE001
        return True


_RESUME: list[dict | None] = [None]  # 记录被 /run 中止的任务信息，供 /resume 续跑


def _run_task(agent, task: str) -> None:
    import time as _time
    import datetime as _dt
    t0 = _time.monotonic()
    console.print(f"[dim]⯈ {escape(task)}[/]")
    tool_names: list[str] = []
    reasoning_buf: list[str] = []
    show_thinking = _thinking_enabled()
    answer = {"answer": ""}

    def on_step(name: str, result: str, args=None) -> None:
        tool_names.append(name)
        # 进度面板实时更新为"正在干什么"
        cur_status[0] = f"[cyan]{_action_text(name, args)}[/]"
        _live_status()
        head = result.replace("\n", " ")[:1500]
        console.print(f"[cyan]  ↳ 调用[/] [bold]{name}[/] {_action_arg_hint(name, args)}")
        if head:
            console.print(f"    [dim]{escape(head)}[/]")
        # 子任务相关工具调用时给轻提示，便于用户跟上并行子任务进度
        if name == "sub_spawn":
            console.print("    [magenta]子任务已后台下发，可 /sub list 查看、/sub show <id> 取回结果[/]")

    cur_status = ["[green]⚡ 准备中…（正在初始化 / 分析任务）[/]"]
    last_live_t: list[float] = [0.0]

    def _live_status() -> None:
        now = _time.monotonic()
        if now - last_live_t[0] >= 0.12:  # 限频刷新，避免每 token 重绘
            last_live_t[0] = now
            st.update(cur_status[0])

    def on_stream(delta: str) -> None:
        # 收到模型生成增量：此刻在"推理/生成"
        cur_status[0] = "[yellow]🤔 思考中 / 生成中…（正在推理，请耐心等待）[/]"
        _live_status()

    def on_reasoning(chunk: str) -> None:
        # 云端推理模型（DeepSeek-R1 等）：思考内容实时流入进度面板
        reasoning_buf.append(chunk)
        cur_status[0] = ("[magenta]🧠 云端思考中…（推理流中，稍后可在底部折叠查看完整思考，"
                         "/thinking off 关闭）[/]")
        _live_status()

    with console.status(cur_status[0]) as st:
        try:
            answer = agent.run_task(task, on_step=on_step,
                                    on_stream=on_stream, on_reasoning=on_reasoning)
        except KeyboardInterrupt:
            # 优雅中止：给出“进行到哪一步”，并记录续跑信息供 /resume 用
            console.print("\n[red]⏹ 已中止当前任务。[/]")
            if tool_names:
                done = " → ".join(tool_names[:8]) + (" …" if len(tool_names) > 8 else "")
                console.print(f"[dim]已执行到第 {len(tool_names)} 步：{done}[/]")
            _RESUME[0] = {"task": task, "tools": list(tool_names), "n": len(tool_names)}
            console.print("[dim]想继续原任务，可输入 /resume[/]")
            return

    reasoning = "".join(reasoning_buf).strip()
    if not reasoning:
        _ra = getattr(agent, "_reasoning_buf", None) or []
        reasoning = "".join(_ra).strip()

    ans = answer.get("answer", "")
    el = _time.monotonic() - t0
    console.print()
    # 云端推理：以可折叠面板形式展示（/thinking off 隐藏收起）
    if reasoning and show_thinking:
        body = reasoning if len(reasoning) <= 2000 else reasoning[:2000] + "\n…（内容较长已截断）"
        console.print(Panel(escape(body),
                            title=f"🤔 思考过程（{len(reasoning)} 字 · /thinking off 折叠隐藏）",
                            border_style="magenta", subtitle="云端推理"))
    console.print(f"● {escape(ans)}")
    console.print(f"\n[dim]✻ Crunched for {el:.0f}s · done {_dt.datetime.now():%H:%M}[/]")
    console.print("\n" + "─" * 78, style="dim")
    _show_files()


def _show_files() -> None:
    try:
        files = tool_registry.invoke("list_project_files", {})
        console.print(Panel(escape(files), title=f"项目写入目录：{tool_registry.get_project_root()}",
                            border_style="blue"))
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]列出文件失败: {e}[/]")


# ---------------- 训练 / 微调（与 Web 版对齐） ----------------
def _train_out(name: str) -> Path:
    return Path(tool_registry.get_project_root()).resolve() / "artifacts" / name


def _cmd_train(action: str, dataset: str | None = None, base: str | None = None) -> None:
    if action == "status":
        deps = {m: False for m in ("torch", "transformers", "peft", "trl", "datasets")}
        for m in deps:
            try:
                __import__(m)
                deps[m] = True
            except Exception:  # noqa: BLE001
                deps[m] = False
        gpu = "CPU"
        try:
            from core.training.optimize import detect_gpu
            g = detect_gpu()
            if g.get("available"):
                gpu = g.get("name", "GPU")
        except Exception:  # noqa: BLE001
            pass
        ready = deps["torch"] and deps["transformers"]
        console.print(Panel(
            f"训练依赖: {deps}\n设备: {gpu}" +
            ("" if ready else "\n[red]缺少训练依赖，请 pip install torch transformers[red]"),
            title="训练环境", border_style="blue"))
        return

    logger = lambda s: console.print(f"  [cyan]{s}[/]")  # noqa: E731
    root = Path(tool_registry.get_project_root()).resolve()
    out: dict = {}
    try:
        if action == "ced":
            with console.status("运行 CED-MoE…"):
                from core.training.ced_moe import CedMoeSession
                out = CedMoeSession(log=logger).run(str(_train_out("research") / "ced"))
        elif action == "fp4":
            with console.status("运行 FP4 QAT…"):
                from core.training.fp4_qat import Fp4QatSession
                out = Fp4QatSession(log=logger).run(str(_train_out("research") / "fp4"))
        elif action == "opd":
            with console.status("教师蒸馏生成数据…"):
                from core.training.opd_distill import OpdDistillSession
                out = OpdDistillSession(tasks=[dataset] if dataset else None,
                                        output_dir=str(_train_out("research") / "opd"),
                                        log=logger).run()
        elif action in ("lora", "full"):
            if not dataset:
                console.print(f"[red]/train {action} 需要数据集路径（相对项目根或绝对）[/]")
                return
            ds = dataset if os.path.isabs(dataset) else str(root / dataset)
            if not Path(ds).is_file():
                console.print(f"[yellow]数据集不存在: {ds}[/]")
                return
            with console.status(("LoRA 微调" if action == "lora" else "完整训练") + "…"):
                if action == "lora":
                    from core.training.finetune import FinetuneSession
                    out = FinetuneSession(base_model=base,
                                          output_dir=str(_train_out("lora")),
                                          log=logger).tune(ds)
                else:
                    from core.training.trainer import TrainerSession
                    out = TrainerSession(output_dir=str(_train_out("full")),
                                         log=logger).train(ds)
        else:
            console.print("[yellow]未知操作。可用：status | ced | fp4 | opd | lora <数据> | full <数据>[/]")
            return
    except Exception as e:  # noqa: BLE001
        hint = ""
        if action == "opd" or "litellm" in str(e).lower() or "调用模型" in str(e):
            hint = "（/train opd 需要配置教师模型 API 密钥，请在对话/网关中配置后再试）"
        console.print(f"[red]训练异常: {e}{hint}[/]")
        return

    if out.get("ok"):
        console.print(f"\n[bold green]✔ 完成[/] 产物 -> [bold]{out.get('plan') or out.get('adapter')
                        or out.get('model') or out.get('dataset')}[/]"
                      + (f"（样本 {out.get('samples')}）" if out.get("samples") else ""))
    else:
        console.print(f"[red]✖ 失败:[/] {out.get('error', out)}".replace("\n", " ")[:400])


# ---------------- MCP 市场 / 生态（与 Web 版对齐） ----------------
def _cmd_mcp(action: str = "view", name: str = "", command: str = "", args: list[str] | None = None,
             desc: str = "") -> None:
    from core.mcp import registry as mcp_registry
    from core.mcp.client import list_tools
    if action == "add":
        if not name or not command:
            console.print("[yellow]用法：/mcp add <name> <command> [args...]（描述可选，用 | 分隔）[/]")
            return
        try:
            srv = mcp_registry.add_server(name, command, args, desc)
            console.print(f"[green]✔ 已添加 MCP 服务器:[/] {srv.get('name')} → {srv.get('command')}")
        except ValueError as e:
            console.print(f"[red]{e}[/]")
        return
    if action == "del":
        if not name:
            console.print("[yellow]用法：/mcp del <name>[/]")
            return
        ok = mcp_registry.remove_server(name)
        console.print(f"[green]✔ 已删除[/] {name}" if ok else f"[yellow]未找到 {name}（仅可删自定义）[/]")
        return
    if action == "approve":
        if not name:
            console.print("[yellow]用法：/mcp approve <name>[/]")
            return
        out = mcp_registry.approve(name)
        if out.get("ok"):
            console.print(f"[green]✔ 已启用[/] {name}：{out['message']}")
        else:
            console.print(f"[yellow]{out['message']}[/]")
        return
    if action == "revoke":
        if not name:
            console.print("[yellow]用法：/mcp revoke <name>[/]")
            return
        ok = mcp_registry.revoke(name)
        console.print(f"[green]✔ 已撤销启用[/] {name}" if ok else f"[yellow]未找到可撤销的自定义服务器 {name}[/]")
        return
    # view
    servers = mcp_registry.list_servers()
    if not servers:
        console.print("暂无已配置 MCP 服务器。内置市场目录：" +
                      ", ".join(s["name"] for s in mcp_registry.builtin_catalog()))
        return
    lines = []
    for s in servers:
        try:
            tools = list_tools(s)
            ready = bool(tools)
        except Exception:  # noqa: BLE001
            tools, ready = [], False
        tag = "内置" if s.get("builtin") else "自定义"
        apr_tag = "✔已启用" if s.get("approved") else "⚠待审批"
        lines.append(f"{s.get('name')} [{tag}|{apr_tag}] "
                     f"{'🟢 就绪' if ready else '🔴 未就绪'} "
                     f"命令={s.get('command')} {' '.join(s.get('args') or [])} "
                     f"工具={len(tools)}{('：' + ','.join(tools)) if tools else ''}"
                     + (f"  | {s.get('desc')}" if s.get("desc") else ""))
    console.print(Panel("\n".join(lines), title=f"MCP 市场（共 {len(servers)} 台）",
                        border_style="blue"))
    pending = mcp_registry.pending_activations()
    if pending:
        console.print("[yellow]⚠ 待审批启用: " +
                      ", ".join(f"{p['name']}（请求工具 {p['tool']}）" for p in pending) +
                      "[/] [dim]→ /mcp approve <name>[/]")
    catalog = mcp_registry.builtin_catalog()
    if catalog:
        console.print("[dim]市场可安装: " + ", ".join(f"{s['name']} ({s['command']})"
                      for s in catalog) + "[/]")


def _cmd_plugin_install(url: str) -> None:
    """从 GitHub 安装开源 MCP 插件并登记（deny-first，需再 /mcp approve 启用）。"""
    if not url:
        console.print("[yellow]用法：/plugin install <github-url-or-owner/repo>[/]")
        return
    from core.mcp import installer
    try:
        msg = installer.install_from_github(url)
        console.print(f"[green]{msg}[/]")
    except ValueError as e:
        console.print(f"[red]✖ {e}[/]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]✖ 安装失败：{e}[/]")


def _cmd_skill(parts: list[str]) -> None:
    """/skill 命令：查看/安装/卸载/运行/授权/脚手架/流水线/升级回滚。parts 已按空格拆分。"""
    from core.skill import registry as sk
    root = tool_registry.get_project_root()
    action = parts[1].strip().lower() if len(parts) > 1 else "view"

    if action in ("install", "ins"):
        name = parts[2] if len(parts) > 2 else ""
        source = parts[3] if len(parts) > 3 else None
        if not name:
            console.print("[yellow]用法：/skill install <name> [source]｜内置：python-tester / report-writer / csv-tool[/]")
            return
        console.print(sk.install_skill(name, source, root))
        return
    if action in ("uninstall", "rm"):
        name = parts[2] if len(parts) > 2 else ""
        if not name:
            console.print("[yellow]用法：/skill uninstall <name>[/]")
            return
        console.print(sk.uninstall_skill(name, root))
        return
    if action in ("run", "r"):
        name = parts[2] if len(parts) > 2 else ""
        args = " ".join(parts[3:]) if len(parts) > 3 else ""
        if not name:
            console.print("[yellow]用法：/skill run <name> [args][/]")
            return
        from core.skill.runner import run_skill
        console.print(run_skill(name, args or None, root))
        return
    if action == "permit":
        name = parts[2] if len(parts) > 2 else ""
        kv = {}
        for p in parts[3:]:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip().lower() in ("1", "true", "yes", "on")
        if not name or not kv:
            console.print("[yellow]用法：/skill permit <name> k=v k=v ｜如 /skill permit x file_read=true shell=true[/]")
            return
        console.print(sk.set_permissions(name, kv, root))
        return
    if action in ("disable", "enable"):
        name = parts[2] if len(parts) > 2 else ""
        on = action == "enable"
        if len(parts) > 3:
            on = parts[3].strip().lower() in ("1", "on", "true", "yes")
        if not name:
            console.print("[yellow]用法：/skill disable|enable <name> [on|off][/]")
            return
        console.print(sk.set_enabled(name, on, root))
        return
    if action in ("new", "create", "scaffold"):
        name = parts[2] if len(parts) > 2 else ""
        cat = parts[3] if len(parts) > 3 else "文档生成"
        if not name:
            console.print("[yellow]用法：/skill new <name> [八分类][/]")
            return
        console.print(sk.scaffold_skill(name, cat, root=root))
        console.print("[dim]然后 /skill permit <name> file_read=true file_write=true 授权能力，/skill install <绝对目录> 可安装[/]")
        return
    if action in ("upgrade", "up"):
        name = parts[2] if len(parts) > 2 else ""
        source = parts[3] if len(parts) > 3 else None
        if not name:
            console.print("[yellow]用法：/skill upgrade <name> [source][/]")
            return
        console.print(sk.upgrade_skill(name, source, root))
        return
    if action in ("rollback", "rb"):
        name = parts[2] if len(parts) > 2 else ""
        if not name:
            console.print("[yellow]用法：/skill rollback <name>[/]")
            return
        console.print(sk.rollback_skill(name, root))
        return
    if action == "pipeline":
        act = parts[2].strip().lower() if len(parts) > 2 else "view"
        _cmd_skill_pipeline(act, parts[3:], root)
        return

    # 默认 view：已装 + 内置目录，支持按分类过滤
    category = parts[2] if len(parts) > 2 else None
    installed = sk.list_skills(root, category)
    lines = [f"[green]●[/] [bold]{x['name']} v{x['version']}[/] "
             f"[dim][{x['category'] or '未分类'}]{'(禁用)' if not x['enabled'] else ''}[/] — {x['desc']}"
             for x in installed] or ["（尚未安装任何技能）"]
    console.print(Panel("\n".join(lines), title=f"已装技能（共 {len(installed)}）", border_style="#4F46E5"))
    if not category:
        catalog = sk.list_catalog()
        if catalog:
            console.print(Panel(" | ".join(sk.CATEGORIES), title="八分类", border_style="dim"))
            console.print("[dim]内置可安装: "
                          + ", ".join(f"{x['name']}（{x['category']}）" for x in catalog)
                          + " ｜ /skill install <name>[/]")


def _cmd_skill_pipeline(act: str, rest: list[str], root: str) -> None:
    from core.skill import registry as sk
    if act == "save":
        name = rest[0] if rest else ""
        seq = [s.strip() for s in " ".join(rest[1:]).split(",") if s.strip()]
        if not name or not seq:
            console.print("[yellow]用法：/skill pipeline save <名> <技能A>,<技能B>,…[/]")
            return
        console.print(sk.save_pipeline(name, seq, root))
        return
    if act == "del":
        name = rest[0] if rest else ""
        console.print(sk.delete_pipeline(name, root) if name else "[yellow]用法：/skill pipeline del <名>[/]")
        return
    if act == "run":
        name = rest[0] if rest else ""
        args = " ".join(rest[1:]) if len(rest) > 1 else ""
        console.print(sk.run_pipeline(name, root, {"text": args} if args else {}))
        return
    pipes = sk.list_pipelines(root)
    if not pipes:
        console.print("[dim]尚无流水线。/skill pipeline save <名> <技能A>,<技能B> 创建。[/]")
        return
    for p in pipes:
        console.print(f"[bold cyan]{p['name']}[/]  {' → '.join('[green]' + s + '[/]' for s in p['skills'])}")


def _cmd_scan() -> None:
    """/scan：执行一次安全巡检并打印报告。"""
    from core.project.scan import summary
    root = tool_registry.get_project_root() or "."
    console.print(f"[dim]对项目目录执行安全巡检：{root}[/]")
    text = summary(root)
    # 简单上色严重度
    lines = [("\n" if ln.startswith("发现") else "") + (ln.replace("[high]", "[bold red]高危[/]")
             .replace("[medium]", "[yellow]中危[/]").replace("[low]", "[dim]低危[/]"))
             for ln in text.split("\n")]
    console.print(Panel("\n".join(l.strip() for l in lines), title="安全巡检报告",
                        border_style="orange_red1"))


def _cmd_audit() -> None:
    """/audit：审计可视化报表（按日/周/动作/高危）。"""
    from core.project.audit import stats
    root = tool_registry.get_project_root() or "."
    d = stats(root)
    def _kv(label: str, obj: dict) -> str:
        return f"[bold {label}]{label}[/]: " + (", ".join(f"{k}={v}" for k, v in (obj or {}).items()) or "（无）")
    console.print(Panel(
        f"审计记录总数：[bold]{d['total']}[/]\n\n"
        + _kv("按日", d["by_day"]) + "\n" + _kv("按周", d["by_week"])
        + "\n" + _kv("按动作", d["by_action"]) + "\n"
        + "[bold red]高危动作[/]: " + (", ".join(f"{k}={v}" for k, v in (d["high_risk"] or {}).items()) or "（无）"),
        title="操作审计报表", border_style="blue"))


def _cmd_undo(parts: list[str]) -> None:
    """/undo [序号]：基于自动快照的可逆回滚。
    每次 project_write 覆盖写前都会自动备份上一版本到 .hs/snapshots/。
    /undo 列出最近快照，默认回滚最新一条；/undo <序号> 回滚指定倒数序号。
    用于“AI 把文件写坏了/写偏了 → 一键还原原版”。
    """
    from core.project import snapshot
    root = tool_registry.get_project_root() or "."
    d = (Path(root).resolve() / ".hs" / "snapshots")
    snaps = sorted(d.glob(snapshot.PREFIX + "*.json"),
                   key=lambda p: p.stat().st_mtime) if d.is_dir() else []
    if not snaps:
        console.print("[yellow]暂无可回滚的快照（需先有覆盖写文件的动作才会留痕）。[/]")
        return
    idx = parts[1].strip().lower() if len(parts) > 1 else ""
    if idx not in ("", "list", "ls"):
        if idx.isdigit():
            n = int(idx)
            if 0 <= n < len(snaps):
                console.print(snapshot.rollback(Path(root).resolve(), snaps[n].name))
                return
            console.print(f"[yellow]序号越界：共 {len(snaps)} 条快照（从 0 开始）。[/]")
            return
        console.print("[yellow]用法：/undo（回滚最新一条）｜ /undo <序号> ｜ /undo list[/]")
        return
    # 展示快照列表（新→旧），默认回滚最新
    lines = []
    for i, p in enumerate(reversed(snaps)):
        try:
            import json as _j
            e = _j.loads(p.read_text(encoding="utf-8"))
            mark = " ← 最新" if i == 0 else ""
            lines.append(f"[dim]{len(snaps)-1-i}[/]  [{e.get('ts')}] {e.get('path')}{mark}")
        except Exception:  # noqa: BLE001
            lines.append(f"[dim]{p.name}[/]（快照损坏，跳过）")
    if idx in ("list", "ls"):
        console.print(Panel("\n".join(lines) or "(无)", title="快照列表（序号 0 为最新）",
                            border_style="dim"))
        return
    # 默认回滚最新一条
    newest = snaps[-1]
    try:
        import json as _j
        e = _j.loads(newest.read_text(encoding="utf-8"))
        console.print(Panel(
            f"将还原文件：[bold cyan]{e.get('path')}[/]（快照于 {e.get('ts')}，序号 {len(snaps)-1}）\n"
            f"[dim]内容较长不直接打印；如需预览差异可用 /diff {e.get('path')}[/]\n"
            "输入 [bold]确认 y[/] 或任意键取消…", title="/undo 预览",
            border_style="orange_red1"))
        if input().strip().lower() not in ("y", "yes", "确认", "是"):
            console.print("[yellow]已取消。[/]")
            return
        console.print(snapshot.rollback(Path(root).resolve(), newest.name))
    except Exception as e2:  # noqa: BLE001
        console.print(f"[red]回滚失败：{e2}[/]")


def _cmd_diff(file: str) -> None:
    """/diff <文件>：对比当前文件与最近一次快照的差异，回滚前先看清改动。"""
    from core.project import snapshot
    root = tool_registry.get_project_root() or "."
    f = file.strip().lstrip("/\\")
    if not f:
        console.print("[yellow]用法：/diff <相对路径，如 backend/app.py>[/]")
        return
    console.print(snapshot.diff(Path(root).resolve(), f))


def _cmd_health() -> None:
    """/health：端点自检——逐项 ✓/✗ + 修复建议，全离线可跑，不做写操作。"""
    checks: list[tuple[str, bool, str]] = []

    # 1) Python 版本
    import sys as _sys
    ok = _sys.version_info >= (3, 8)
    checks.append(("Python 版本", ok,
                   f"{_sys.version.split()[0]}" + ("" if ok else " 建议 ≥ 3.8")))

    # 2) 配置文件可读
    from core.llm.keys import load as _load_cfg
    try:
        cfg = _load_cfg()
        checks.append(("配置加载 settings.json", True, "已读取且合法"))
    except Exception as e:  # noqa: BLE001
        checks.append(("配置加载 settings.json", False, f"读取失败：{e}"))

    # 3) LLM 网关可达（真实 ping，超时短）
    try:
        from core.llm import gateway
        base = cfg.get("llm", {}).get("base_url") or "(默认/未设)"
        console.print(f"[dim]  · 探测模型网关 {escape(str(base))} …[/]")
        resp = gateway.chat([{"role": "user", "content": "ping"}], max_tokens=1,
                            temperature=0)
        err = resp.get("error")
        ok3 = not err
        checks.append(("LLM 网关可达", ok3,
                       (f"base_url={base}  OK" if ok3 else f"调用失败：{str(err)[:120]}\n"
                        "  → 本机 Ollama 默认 http://127.0.0.1:11434，可 /model set 改")))
    except Exception as e3:  # noqa: BLE001
        checks.append(("LLM 网关可达", False, f"探测异常：{e3}"))

    # 4) 隔离沙箱可导入
    try:
        from core.sandbox.sandbox import Sandbox, exec_cmd  # noqa: F401
        network = (cfg.get("sandbox", {}) or {}).get("allow_network")
        checks.append(("隔离沙箱可用", True,
                       f"可用（allow_network={network}；推荐保持关闭以增强隔离）"))
    except Exception as e4:  # noqa: BLE001
        checks.append(("隔离沙箱可用", False, f"缺依赖/未配置：{e4}\n  → 可 pip install 相关依赖"))

    # 5) MCP 注册表加载 + 待审批项
    try:
        from core.mcp import registry as mcp_registry
        pending = mcp_registry.pending_activations()
        n_pending = len(pending) if isinstance(pending, list) else 0
        checks.append(("MCP 注册表可加载", True,
                       f"OK；待审批启用 {n_pending} 项" + ("（/mcp approve）" if n_pending else "")))
    except Exception as e5:  # noqa: BLE001
        checks.append(("MCP 注册表可加载", False, f"加载失败：{e5}"))

    # 6) webfetch 出网探测（超时快，断网提示加代理指引）
    try:
        console.print("[dim]  · 探测 webfetch 出网 …[/]")
        _probe = tool_registry.invoke("webfetch", {"url": "https://example.com"})
        if "connect" in str(_probe).lower() and "error" in str(_probe).lower() or \
           "timed out" in str(_probe).lower() or "超时" in str(_probe):
            checks.append(("webfetch 出网", False,
                           "出网失败/超时（当前机器若断网属预期）\n"
                           "  → 可设置 WEBFETCH_PROXY 或 /model set 带代理网关"))
        elif "error" in str(_probe).lower():
            checks.append(("webfetch 出网", False, f"出网异常：{str(_probe)[:120]}"))
        else:
            checks.append(("webfetch 出网", True, "可正常拉取网页（或已走代理）"))
    except Exception as e6:  # noqa: BLE001
        checks.append(("webfetch 出网", False, f"探测异常：{e6}"))

    # 汇总
    lines = [f"{'✔' if ok else '✗'}  {label}" + ("  " + ("" if ok else f"[red]{detail}[/]" if not ok else "")) 
             for label, ok, detail in checks]
    ok_count = sum(1 for _, ok, _ in checks if ok)
    console.print(Panel(
        "\n".join(lines),
        title=f"系统自检（通过 {ok_count}/{len(checks)}）",
        border_style=("green" if ok_count == len(checks) else "yellow")))


def _cmd_computer(parts: list[str]) -> None:
    """/computer：真机桌面自动化自测入口。
    用法：/computer view           查看前台窗口文本快照
          /computer windows        列出可见窗口
          /computer shot [路径]    保存全屏截图
          /computer click <x> <y>   点击坐标
          /computer type <文本>     键入文本
          /computer key <快捷键>    按键，如 ctrl+s、enter
          /computer open <应用/网址> 用默认程序打开
          /computer focus <标题>    切换窗口到前台
    """
    from core.computer import screen, input as ci, windows
    act = parts[1].lower() if len(parts) > 1 else "view"
    rest = parts[2:]

    def _show(text: str) -> None:
        console.print(Panel(str(text if text else "(空)"), title=f"/computer {act}",
                            border_style="blue"))

    try:
        if act == "view":
            _show(screen.screenshot_text())
        elif act == "windows":
            _show(windows.list_windows())
        elif act in ("shot", "screenshot"):
            _show(screen.save_screenshot(rest[0] if rest else None))
        elif act == "click":
            if len(rest) >= 2:
                _show(ci.click(int(rest[0]), int(rest[1])))
            else:
                _show("用法：/computer click <x> <y>")
        elif act == "type":
            _show(ci.type_text(" ".join(rest)))
        elif act in ("key", "keys"):
            _show(ci.key(rest[0] if rest else "enter"))
        elif act == "open":
            _show(windows.open_app(" ".join(rest)))
        elif act == "focus":
            _show(windows.focus_window(" ".join(rest)))
        else:
            console.print("[yellow]用法：/computer view|windows|shot|click|type|key|open|focus[/]")
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]执行失败：{e}[/]")


def interactive(agent: Agent) -> None:
    console.print(_banner())
    console.print("─" * 78, style="dim")
    # 首次启动自动弹配置向导（已配置则静默跳过；幂等）
    _first_run_wizard()
    # 真正在终端里运行时，先展示全部写指令，方便直接用
    _show_slash_menu(None)
    while True:
        try:
            task = _read_task("\n[bold] > [/]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n再见。")
            return
        if task.lower() in ("q", "exit", "quit"):
            console.print("再见。")
            return
        if task in ("/", "/help", "help"):
            _show_slash_menu(None)
            continue
        if task in ("/about", "about"):
            _cmd_about("about")
            continue
        if task in ("/authors", "authors"):
            _cmd_about("authors")
            continue
        if task.startswith("/model"):
            _cmd_model(task.split())
            continue
        if task.startswith("/config"):
            _cmd_config(task.split())
            continue
        if task.startswith("/sub"):
            _cmd_sub(task.split())
            continue
        if task.startswith("/budget"):
            _cmd_budget(task.split())
            continue
        if task in ("/toollist", "toollist", "/tools", "tools"):
            _cmd_toollist()
            continue
        if task.startswith("/train"):
            parts = task.split()
            act = parts[1].strip().lower() if len(parts) > 1 else "status"
            dataset = parts[2] if len(parts) > 2 else None
            base = parts[3] if len(parts) > 3 else None
            _cmd_train(act, dataset=dataset, base=base)
            continue
        if task.startswith("/mcp"):
            parts = task.split()
            act = parts[1].strip().lower() if len(parts) > 1 else "view"
            name = parts[2] if len(parts) > 2 else ""
            def _split_args(rest: list[str]) -> tuple[str, list[str]]:
                desc = ""
                a = []
                for r in rest:
                    if r == "|":
                        desc = "|".join(rest[rest.index(r) + 1:])
                        break
                    a.append(r)
                return desc, a
            cmd_bits = parts[3:] if len(parts) > 3 else []
            desc, mcp_args = _split_args(cmd_bits)
            command = parts[3] if len(parts) > 3 else ""
            _cmd_mcp(act, name=name, command=command,
                     args=(mcp_args if command else None), desc=desc)
            continue
        if task.startswith("/plugin"):
            parts = task.split()
            act = parts[1].strip().lower() if len(parts) > 1 else "help"
            url = parts[2] if len(parts) > 2 else ""
            if act in ("install", "i"):
                _cmd_plugin_install(url)
            else:
                console.print("[yellow]用法：/plugin install <github-url-or-owner/repo>[/]")
            continue
        if task.startswith("/skill"):
            _cmd_skill(task.split())
            continue
        if task.startswith("/computer"):
            _cmd_computer(task.split())
            continue
        if task in ("/scan", "scan"):
            _cmd_scan()
            continue
        if task in ("/audit", "audit"):
            _cmd_audit()
            continue
        if task in ("/undo", "undo") or task.startswith("/undo "):
            _cmd_undo(task.split())
            continue
        if task.startswith("/diff"):
            _cmd_diff(task[5:].strip())
            continue
        if task in ("/health", "health"):
            _cmd_health()
            continue
        if task in ("/context", "context"):
            try:
                console.print(Panel(agent._system(), title="注入模型的本地上下文",
                                    border_style="dim"))
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]构建上下文失败：{e}[/]")
            continue
        if task in ("/init", "init"):
            _cmd_init()
            continue
        if task.startswith("/pending"):
            # 查看待确认的不可逆/删除操作
            console.print(tool_registry.pending_destructive())
            continue
        if task.startswith("/approve"):
            parts = task.split()
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if idx is None:
                console.print("[yellow]用法：/approve <序号>（序号来自 /pending）[/]")
                continue
            console.print(tool_registry.invoke("approve_destructive", {"index": idx}))
            continue
        if task.startswith("/trust"):
            parts = task.split()
            on = (len(parts) > 1 and parts[1].strip().lower()
                  in ("1", "on", "true", "yes", "托管", "放行"))
            console.print(tool_registry.invoke("set_destructive_policy", {"enable": on}))
            continue
        if task.startswith("/thinking"):
            _cmd_thinking(task.split())
            continue
        if task.startswith("/mode"):
            parts = task.split()
            new_mode = parts[1].strip().lower() if len(parts) > 1 else ""
            if new_mode not in MODE_PROMPTS or new_mode == "default":
                console.print("[yellow]用法：/mode work|code|design|plan|build[/]")
                continue
            agent.system_prompt = MODE_PROMPTS.get(new_mode, MODE_PROMPTS["work"])
            agent.mode = new_mode  # 供只读(Plan)模式做工具门控
            agent.reset(keep_system=True)
            console.print(f"[bold]已切换模式：{_MODE_LABEL.get(new_mode, new_mode)}[/] — "
                          f"{_MODE_ROLES.get(new_mode, '')}")
            continue
        if task in ("/resume", "resume"):
            if not _RESUME[0]:
                console.print("[yellow]没有可续跑的任务（当前会话未中断过任务）。[/]")
                continue
            info = _RESUME[0]
            note = (f"\n\n[续跑注记] 上次任务『{info['task'][:80]}』在完成第 {info['n']} 步后被中断。"
                    f"已完成的工具序列：{', '.join(info['tools'][:12])}。"
                    f"请在此基础之上继续完成原任务并给出收敛总结，不要从头重复。")
            console.print("[cyan]↻ 续跑上次任务…[/]")
            try:
                _run_task(agent, info["task"] + note)
            except KeyboardInterrupt:
                console.print("\n[red]已停止续跑。[/]")
            continue
        if task.startswith("/"):
            # 未知写指令：弹出完整菜单
            _show_slash_menu(None)
            continue
        if not task:
            continue
        # 拖拽/路径上传：输入是现有本地文件(夹)则先拷入项目 uploads/ 并登记，让 agent 拿到
        if os.path.exists(task.strip().strip('"').strip("'")):
            _dump = tool_registry.invoke(
                "file_upload", {"local_path": task.strip().strip('"').strip("'")})
            console.print(f"[blue]{_dump}[/]")
            task = _dump + "\n\n请基于以上已上传的文件继续我的请求。"
        try:
            _run_task(agent, task)
        except KeyboardInterrupt:
            console.print("\n[red]已停止当前任务。[/]")


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="hs",
        description="HS 终端智能体（在真实终端里交互运行；输入 / 弹出写指令菜单）",
        epilog=(
            "示例：\n"
            "  python cli.py --mode work                  # 进入交互模式\n"
            "  python cli.py '帮我写一个计算器'            # 一次性任务\n"
            "  python cli.py --train lora sft-code.jsonl # LoRA 微调后退出\n"
            "交互内可用写指令：/about ｜ /authors ｜ /mode ｜ /mcp ｜ /train ｜ /help ｜ q\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("task", nargs="?", default=None, help="目标任务（省略则进入交互模式）")
    ap.add_argument("-m", "--model", default=None,
                    help='模型 provider，如 "deepseek/deepseek-chat"')
    ap.add_argument("-d", "--dir", default=None, help="AI 写项目的目标文件夹（默认：--cwd 或 ./ai_projects）")
    ap.add_argument("--cwd", default=None, help="工作区目录（上下文）")
    ap.add_argument("--mode", default="work",
                    choices=["work", "code", "design", "plan", "build"],
                    help='智能体模式（Work/Code/Design/Plan/Build），默认 work')
    ap.add_argument("--train", default=None,
                    choices=["status", "ced", "fp4", "opd", "lora", "full"],
                    help="运行训练/微调并退出：status/ced/fp4/opd/lora/full")
    ap.add_argument("--dataset", default=None, help="--train lora/full 的数据集路径")
    ap.add_argument("--base-model", default=None, help="--train lora 的基座模型")
    args = ap.parse_args()

    # 设置写入目录
    target = args.dir
    if not target:
        base = args.cwd or "."
        target = str((Path(base).resolve() / "ai_projects"))
    root_msg = tool_registry.set_project_root(target)

    if args.train:
        console.print(f"[dim]{root_msg}[/]")
        _cmd_train(args.train, dataset=args.dataset, base=args.base_model)
        return 0

    mode = (args.mode or "work").strip() or "work"
    if mode not in MODE_PROMPTS:
        mode = "work"
    system_prompt = MODE_PROMPTS.get(mode, MODE_PROMPTS["work"])

    # 设置写入目录
    target = args.dir
    if not target:
        base = args.cwd or "."
        target = str((Path(base).resolve() / "ai_projects"))
    root_msg = tool_registry.set_project_root(target)

    agent = Agent(system_prompt=system_prompt, provider_model=_pick_provider(args.model),
                  mode=mode)
    # 让模型可向操作者提问（人类决策权威）：收到 ask_user 时在终端展示问题并等待回车输入。
    def _ask_handler(q: str) -> str:
        console.print(f"\n[bold cyan]🤔 需要你决定：[/]{q}")
        try:
            return console.input("[bold yellow]你的回答 > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
    tool_registry.set_ask_handler(_ask_handler)
    if args.cwd:
        from core.project.workspace import Workspace
        try:
            agent.attach_workspace(Workspace(args.cwd))
            console.print(f"[dim]工作区上下文: {Path(args.cwd).resolve()}[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]工作区注入失败: {e}[/]")
    console.print(f"[dim]{root_msg}[/]")
    console.print(f"[bold]模式：{_MODE_LABEL.get(mode, mode)}[/] — {_MODE_ROLES.get(mode, '')}")

    if args.task:
        console.print(_banner(_pick_provider(args.model), args.cwd))
        console.print("─" * 78, style="dim")
        _run_task(agent, args.task)
    else:
        interactive(agent)
    return 0


if __name__ == "__main__":
    sys.exit(main())