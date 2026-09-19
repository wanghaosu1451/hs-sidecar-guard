"""智能体推理循环：多轮对话 + function calling 调度。

流程：把 memory 传给网关 → 模型返回文本或 tool_calls →
若是 tool_calls 就执行工具、回填结果、再次请求，直到模型不再请求工具。
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Callable, Iterator


def _args_normalized(args) -> str:
    """把工具参数规范化为按键排序的紧凑字符串，用于相似度比较。"""
    try:
        obj = json.loads(args) if isinstance(args, str) else args
    except Exception:  # noqa: BLE001
        return str(args or "")

    def _s(x):
        if isinstance(x, dict):
            return {str(k): _s(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_s(v) for v in x]
        return x

    return json.dumps(_s(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _args_similar(a: str, b: str) -> float:
    """两段参数规范化串的相似度 0~1（完全样则恒为 1）。"""
    if a == b:
        return 1.0
    try:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
    except Exception:  # noqa: BLE001
        return 0.0


# 守卫4 适用工具：写/不可逆类。read/list 等只读工具即使重复也无破坏性，不禁。
_SIM_TOOLS = ("project_write", "patch_file", "run_shell", "delete_file")
_SIM_SIMILARITY = 0.75   # 相邻参数相似度阈值
_SIM_RUN_LIMIT = 3       # 连续相似多少次判定为隐蔽循环

from ..llm import gateway
from ..llm.gateway import LLMError
from ..llm.keys import load as load_cfg
from .memory import Memory
from . import tools as tool_registry

SYSTEM_PROMPT = (
    "你是运行在本机的自主智能体。执行目标任务时：\n"
    "1. 先拆解步骤；写文件一律用 project_write 真实落盘到用户指定项目目录（relative_path 相对项目根）；\n"
    "2. 代码先用 verify_code 在隔离沙箱跑通再采用；工具结果已注入，据实作答不编造；\n"
    "3. 失败时按真实报错反思修正；全自动推进，任务达成给出简洁结论并列出生成文件；\n"
    "4. 约束保真：用户明确给过的约束必须逐字遵守，冲突时说明并提供替代方案。\n"
    "5. 高影响决策先问人：遇到删/改关键文件、产生大额网络或推理调用、两种方向分歧且后果难回退时，"
    "先用 ask_user 向操作者提问征询确认，得到回答后再继续；低影响、可用合理默认解决的小事不必问。\n"
    "回答用中文，简洁直白。"
)

# 各模式共用的“指令遵循 / 生态”收尾规则，保持约束精度
_CONSTRAINT_RULE = (
    "约束保真：只依据用户明确给出的约束执行，不臆测、不自作主张地扩展范围；"
    "遇到多重约束冲突时先说明冲突，再提供替代方案，而不要悄悄弱化任一约束。"
)

# ---------------- Work / Code / Design 三种模式的系统提示词 ----------------
# 单一来源：Web(GUI) 与 CLI 共用此词典，按 mode 选择不同的智能体角色。
# 每个模式角色、交付物、质量标准各不相同；二者共用「真实落盘 + 沙箱验证」底座。
# "default" 为默认别名，指向 work（工作台模式）。
MODE_PROMPTS = {
    "work": (
        "HS「工作台」智能体：先拆解步骤再执行；文件一律用 project_write 写入用户指定项目目录"
        "（相对项目根路径），可生成 README.md；涉及代码先 verify_code 沙箱验证跑通再采用；"
        "失败按真实报错反思重试，不重复同一错误；据实作答、不编造；全自动推进。中文简洁。" + _CONSTRAINT_RULE
    ),
    "code": (
        "HS「代码」智能体，生产级全栈工程师：先给可执行计划；工程结构完整"
        "（入口、requirements、测试、README），逐文件用 project_write 写入项目根；"
        "每个可测函数用 verify_code 隔离沙箱跑通再交付；处理空值/并发/超时/边界；"
        "不编造不存在的 API/函数/表；失败按真实报错反思修正。中文。" + _CONSTRAINT_RULE
    ),
    "design": (
        "HS「设计」智能体，资深 UI/UX 与前端实现专家：先给设计要点（配色/布局/交互）"
        "再产出高保真原型；交付可用 HTML/CSS/JS 用 project_write 写入项目根"
        "（如 index.html、style.css、app.js）；给出预览要点与改进项；JS 用 verify_code 校验语法，"
        "报错则修正重试。中文并说明生成的文件路径。" + _CONSTRAINT_RULE
    ),
    "plan": (
        "HS「规划」智能体（只读模式）：只分析项目、输出开发方案，绝不修改任何本地文件、"
        "不执行任何写/破坏性操作（read_file/list_project_files/graph_query 等只读工具除外）。"
        "输出结构化的开发方案：目标→现状与关键文件→实现步骤（精确到文件路径）→风险→验证方式；"
        "不编造不存在的文件/函数；看到有写操作冲动时改回只读分析。中文。" + _CONSTRAINT_RULE
    ),
    "build": (
        "HS「构建」智能体（执行模式）：方案已确认后进入全自动开发执行。编排多文件编辑，"
        "用 project_write 落盘、patch_file 迭代，verify_code/run_shell 做隔离沙箱编译与测试；"
        "报错按真实输出反思并重试，不重复同一错误，循环直到全部通过；"
        "自主完成从骨架到可运行的全流程开发，结束时说明改动的文件与验证结果。中文。" + _CONSTRAINT_RULE
    ),
}
MODE_PROMPTS.setdefault("default", MODE_PROMPTS["work"])


def _fire_stop_hook(fn):
    """装饰器：在会话/任务方法返回时触发 Stop 生命周期钩子（异常静默）。

    采用 try/finally 保证无论正常返回还是中途 return，Stop 都会在结束路径触发。
    """
    def _wrapped(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        finally:
            try:
                from . import hooks as _hooks
                _hooks.stop()
            except Exception:  # noqa: BLE001 - 钩子出错不影响主流程
                pass
    return _wrapped


def _is_failure(result: str) -> bool:
    """依据工具返回与内建报错约定，粗略判断该步是否执行失败（用于触发反思）。"""
    head = (result or "").strip()
    if not head:
        return False
    return head.startswith(("错误：", "❌", "拒绝", "命令执行失败")) or \
        ("(exit=" in head and "exit=0)" not in head)


def _tool_calls_from_text(content: str) -> list | None:
    """部分本地模型在流式下不走原生 tool_calls，而是把工具调用以 JSON 文本返回。

    尝试把该文本解析为一组工具调用；解析不出则返回 None（当作普通回答）。
    兼容：单个 {"name","arguments"}、数组、以及带 "tool_calls" 键的对象。
    """
    if not content:
        return None
    text = content.strip()
    # 去掉常见的 ```json 围栏
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return None

    def _make(name, arguments) -> object:
        return SimpleNamespace(
            type="function",
            id="text_call",
            function=SimpleNamespace(name=name,
                                     arguments=arguments if isinstance(arguments, str)
                                     else json.dumps(arguments, ensure_ascii=False)),
        )

    if isinstance(data, dict):
        if "tool_calls" in data and isinstance(data["tool_calls"], list):
            calls = []
            for tc in data["tool_calls"]:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                calls.append(_make(fn.get("name", ""), fn.get("arguments", "{}")))
            return calls or None
        name = data.get("name") or data.get("function", {}).get("name", "")
        args = data.get("arguments", data.get("function", {}).get("arguments", "{}"))
        if name:
            return [_make(name, args)]
        return None
    if isinstance(data, list):
        calls = []
        for tc in data:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            if name:
                calls.append(_make(name, fn.get("arguments", "{}")))
        return calls or None
    return None


class Agent:
    def __init__(self, system_prompt: str = SYSTEM_PROMPT,
                 provider_model: str | None = None, max_tool_rounds: int = 8,
                 workspace=None, mode: str = "work",
                 forbidden_tools: set | frozenset = frozenset()):
        self.memory = Memory()
        self.system_prompt = system_prompt
        self.provider_model = provider_model or load_cfg().get("llm", {}).get(
            "provider", "openai/gpt-4o-mini")
        self.max_tool_rounds = max_tool_rounds
        self.workspace = workspace
        self.mode = mode  # work/code/design/plan/build
        self.forbidden_tools = set(forbidden_tools)  # 子 agent 禁用的编排类工具
        self._in_compress = False      # 压缩重入守卫：防止 _summarize_with_model 内部递归
        self._compress_counter = 0     # 全局压缩限频计数
        self._reasoning_buf: list[str] = []  # 云端推理内容累计（供 UI 可折叠展示）

    def _auto_compress(self) -> None:
        """全局自动上下文压缩：任意会话入口（chat/chat_stream/run_task）
        在每次发送给网关前，若上下文超过阈值则自动触发一次压缩。

        - 用 _in_compress 守卫避免 _messages()→压缩→_messages() 无限递归；
        - 用计数限频，避免同一次会话内过度压缩造成抖动；
        - 压缩本身走 _maybe_summarize（含多模型回退与降级审计），绝不静默丢上下文。
        """
        if self._in_compress:
            return
        self._in_compress = True
        try:
            self._compress_counter += 1
            if self._compress_counter % 5 == 0 and self.memory.should_summarize():
                self._maybe_summarize()
        finally:
            self._in_compress = False

    def _invoke_tool(self, name: str, args) -> str:
        """统一工具执行入口：Plan(只读) 模式下对写/执行类工具做门控，其余直通 invoke。

        被拒时不执行任何写操作，返回一条可见的“只读已拒绝”提示，引导模型切回只读分析。
        """
        if self.mode == "plan" and not tool_registry.is_readonly_tool(name):
            return (f"❌ 只读(Plan)模式下已拒绝写/执行类工具 `{name}({args})`："
                    "本模式仅分析与规划，不修改文件、不执行危险操作。"
                    "请改用 read_file/list_project_files/graph_query/repo_read 等只读工具继续分析。")
        if name in self.forbidden_tools:
            return (f"❌ 子智能体不支持 `{name}`（编排类工具仅主控可用）。"
                    "请直接基于现场文件与上下文继续执行本环节任务，不要尝试再下发子任务。")
        return tool_registry.invoke(name, args)

    # ==================== 双信道压缩 · 信道A：长工具结果指纹化 ====================
    # 过长工具结果（如整文件 read_file 输出）会占满热上下文。超过阈值时把全文落盘到
    # .hs/tool_logs/，热上下文只留“1行指纹 + 摘要头 + 磁盘指针”，模型需要细节再 read_file——
    # 显著省 token、保持关键结论仍在眼前。阈值内结果原样保留，不影响失败分析/校验逻辑。
    _TOOL_LOG_CAP = 4000

    def _dump_tool_log(self, name: str, result: str) -> str | None:
        """把过长工具结果落盘，返回 `.hs/...` 相对路径以嵌入指纹。未设项目根时不落盘。"""
        try:
            from pathlib import Path
            from core.agent import tools as _tr
            root = _tr.get_project_root()
            if not root or root == ".":
                return None
            rootp = Path(root).resolve()
            d = rootp / ".hs" / "tool_logs"
            d.mkdir(parents=True, exist_ok=True)
            import time as _t
            rel = f".hs/tool_logs/{name}_{int(_t.time() * 1000)}.txt"
            (rootp / rel).write_text(result, encoding="utf-8")
            return rel
        except Exception:  # noqa: BLE001
            return None

    def _add_tool(self, tool_call_id: str, name: str, result: str) -> None:
        """把工具结果写入 memory；过长结果做指纹化（存盘 + 摘要头），保持热上下文精简。"""
        result = result or ""
        if len(result) > self._TOOL_LOG_CAP:
            rel = self._dump_tool_log(name, result)
            head = result[:800].replace("\n", " ") + ("…" if len(result) > 800 else "")
            if rel:
                self.memory.add_tool(
                    tool_call_id, name,
                    f"[长结果 {len(result)} 字符已落盘 → {rel}；如需原始全文请 read_file "
                    f"{rel}。摘要头：{head}]")
                return
        self.memory.add_tool(tool_call_id, name, result)

    def _load_project_config(self, root: str) -> str:
        """读取项目专属配置文件（hside.json / .hside.toml / project.json），收敛为一段约定文本。

        与 AGENTS.md（侧重规则）互补，此处侧重项目结构/命令/技术栈等“代码库地图”级信息。
        """
        from pathlib import Path
        if not root or root == ".":
            root = tool_registry.get_project_root()
        rootp = Path(root)
        # 按优先级尝试：hside.json > .hside.toml > project.json
        for fname in ("hside.json", ".hside.toml", "project.json"):
            f = rootp / fname
            if not f.is_file():
                continue
            try:
                raw = (f.read_text(encoding="utf-8").strip() or "")
            except Exception:  # noqa: BLE001
                continue
            if not raw:
                continue
            if fname.endswith(".toml"):
                return f"[{fname}]\n" + raw
            # JSON：仅取基础字段，避免把大 JSON 灌入上下文
            import json
            try:
                data = json.loads(raw)
                keep = {k: v for k, v in data.items()
                        if isinstance(v, (str, int, float, bool)) or k in
                        ("commands", "scripts", "targets", "notes", "architecture")}
                return f"[{fname}]\n" + json.dumps(keep, ensure_ascii=False, indent=2)
            except Exception:  # noqa: BLE001
                return f"[{fname}]\n" + raw[:2000]
        return ""

    def attach_workspace(self, workspace) -> None:
        """注入工作区，使模型能看到全仓上下文而非只看片段。"""
        self.workspace = workspace

    def _load_agents_rules(self, root: str, max_files: int = 30) -> str:
        """递归收集项目规则文件（AGENTS.md · 子目录层级）。

        根部规则在最前，子目录规则按【目录相对路径】分组追加在后，
        子目录越深优先级越高（放得更靠后，压过更粗粒度规则）。
        为防上下文膨胀：最多收 max_files 个、单文件截断，跳过 .git/.venv 等。
        """
        from pathlib import Path
        if not root or root == ".":
            root = tool_registry.get_project_root()
        rootp = Path(root)
        _SKIP = {".git", ".venv", "venv", "__pycache__", "node_modules",
                 "ai_projects", ".hside", "dist", "build"}
        parts: list[str] = []
        root_agent = rootp / "AGENTS.md"
        if root_agent.is_file():
            try:
                c = root_agent.read_text(encoding="utf-8").strip()
                if c:
                    parts.append("# 根目录 AGENTS.md\n" + c)
            except Exception:  # noqa: BLE001
                pass
        # 仅向下两层，避免扫描极深目录树拖慢启动
        visited = 0
        for base in (rootp,):
            for p in sorted(rootp.rglob("AGENTS.md")):
                if p == root_agent:
                    continue
                if visited >= max_files - len(parts):
                    break
                if any(_s in p.parts for _s in _SKIP):
                    continue
                try:
                    rel = str(p.relative_to(rootp)).replace("\\", "/")
                    c = p.read_text(encoding="utf-8").strip()
                    if not c:
                        continue
                    visited += 1
                    parts.append(f"# 子目录规则：{rel}\n{c[:3000]}")
                except Exception:  # noqa: BLE001
                    continue
        return "\n\n".join(parts)

    def reset(self, keep_system: bool = False) -> None:
        self.memory.messages.clear()
        if keep_system and self.system_prompt:
            self.memory.messages.append(
                {"role": "system", "content": self.system_prompt})

    def set_provider(self, provider_model: str) -> None:
        self.provider_model = provider_model

    def _system(self) -> str:
        sys = self.system_prompt or ""
        # 注入完整工具清单（= /toollist）：核心 + 扩展各附一句话，
        # 让模型知道全部可用工具；核心工具的完整 schema 随每次 chat 的 tools 传入。
        try:
            sys += "\n\n" + tool_registry.tools_manifest()
        except Exception:  # noqa: BLE001
            pass
        if self.workspace is not None:
            try:
                from ..project.workspace import build_repo_context
                ctx = build_repo_context(self.workspace)
                if ctx:
                    sys = sys + "\n\n# 当前工作区上下文（只读索引，请据实作答，不要编造不存在的名称）\n" + ctx
            except Exception:  # noqa: BLE001
                pass
        if self.workspace is not None:
            try:
                from ..project.graph import build_graph, graph_summary
                from ..project.repo_memory import is_empty
                if not is_empty(self.workspace):
                    from ..project.repo_memory import read
                    m = read(self.workspace, limit=5)
                    if m and not m.startswith("(项目暂无"):
                        sys += "\n\n# 项目持久记忆（跨会话，优先遵守其约定）\n" + m
                try:
                    g = build_graph(self.workspace)
                    s = graph_summary(g)
                    if "为空" not in s:
                        sys += "\n\n# 项目语义图谱（离线索引）\n" + s
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        # 项目级规则文件 AGENTS.md：递归收集根与子目录层级规则，越深目录优先级越高
        try:
            root = self.workspace if self.workspace is not None else tool_registry.get_project_root()
            rules = self._load_agents_rules(str(root))
            if rules:
                sys += "\n\n# 项目规则(AGENTS.md · 含子目录层级)\n" + rules
        except Exception:  # noqa: BLE001
            pass
        # 项目本地配置注入：读取 hside.json / .hside.toml 等项目专属配置，合并为“代码库地图/约定”
        try:
            import os as _os2
            from pathlib import Path as _Path2
            root = self.workspace if self.workspace is not None else tool_registry.get_project_root()
            cfgtext = self._load_project_config(str(root))
            if cfgtext:
                sys += "\n\n# 项目配置(代码库地图补充)\n" + cfgtext
        except Exception:  # noqa: BLE001
            pass
        # 并行子任务引导：目标可拆成互不依赖片段时优先下发给后台子 agent，保持主上下文精简
        sys += ("\n\n提示：若当前目标可拆分为彼此独立、互不依赖的子片段，"
                "优先用 sub_spawn 下发给后台并行的子智能体执行，再用 sub_result "
                "收集各子任务结果后统一汇总收尾，避免主上下文无谓膨胀；"
                "相互依赖的步骤仍按顺序在主循环内完成。")
        return sys

    def _tokens(self, resp: dict) -> int:
        """从 gateway 归一化结果里提取本轮 token 消耗，取不到按 0 算。"""
        usage = resp.get("usage")
        if usage is None:
            return 0
        try:
            if hasattr(usage, "total_tokens"):
                return int(usage.total_tokens)
            if isinstance(usage, dict):
                return int(usage.get("total_tokens") or
                           (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)))
        except (TypeError, ValueError):
            return 0
        return 0

    def _budget(self) -> tuple[int, int]:
        """读配置里的预算：返回 (max_steps, max_tokens)。max_tokens<=0 表示不限额。"""
        try:
            b = load_cfg().get("budget", {})
            return int(b.get("max_steps", 300) or 300), int(b.get("max_tokens", 0) or 0)
        except Exception:  # noqa: BLE001
            return 300, 0

    def _audit_budget(self, reason: str, steps: int, tokens: int) -> None:
        try:
            from core.project import audit
            root = tool_registry.get_project_root()
            if root and root != ".":
                audit.record(root, "agent_budget_" + reason,
                             f"steps={steps} tokens={tokens}")
        except Exception:  # noqa: BLE001
            pass

    def _summarize_with_model(self, text: str, provider_model: str):
        """调用网关压缩一次；失败/cfg 报错则抛出让上层回退。"""
        resp = gateway.chat(
            self._messages() + [{"role": "user", "content": text}],
            provider_model=provider_model)
        if resp.get("error"):
            raise LLMError(resp["error"])
        return resp.get("content") or ""

    def _maybe_summarize(self) -> None:
        """上下文已达阈值时，调用网关把早期对话压缩成摘要，避免全量回放丢全局观。

        压缩失败的多模型回退：默认压缩模型失败时用 provider_model 重试一次；
        仍失败则显式降级为“不回写摘要、保留原文继续”，并把失败原因记入审计，
        绝不静默丢上下文。
        """
        if not self.memory.should_summarize():
            return
        compact_model = self.provider_model
        try:
            compact_model = (load_cfg().get("llm", {}).get("compact_model")
                             or self.provider_model)
        except Exception:  # noqa: BLE001
            compact_model = self.provider_model

        models = [compact_model, self.provider_model]

        def _best_effort(text: str) -> str:
            last_err = None
            for model in models:
                try:
                    out = self._summarize_with_model(text, model)
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    continue
                if out:
                    self._audit_summarize("ok", model)
                    return out
            # 显式降级：不写摘要、保留原文继续；说明失败原因，避免静默丢上下文
            self._audit_summarize("degraded_keep_original",
                                  f"压缩失败: {last_err}")
            return ""

        self.memory.summarize(_best_effort)

    def _audit_summarize(self, status: str, detail: str) -> None:
        try:
            from core.project import audit
            root = tool_registry.get_project_root()
            if root and root != ".":
                audit.record(root, "summarize_" + status, str(detail)[:200])
        except Exception:  # noqa: BLE001
            pass

    def _messages(self) -> list[dict]:
        self._auto_compress()  # 全局自动压缩：发送前超阈值即压，所有入口统一生效
        msgs = self.memory.get()
        sys = self._system()
        if sys:
            return [{"role": "system", "content": sys}] + msgs
        return msgs

    @_fire_stop_hook
    def chat(self, user_input: str) -> str:
        """同步对话，返回最终文本。若发生工具调用，自动多轮直到结束。"""
        self.memory.add_user(user_input)
        # 同类死循环守卫（与 run_task 一致）：连续同工具同签名重复 3 次则终止
        last_sig, repeat_cnt = None, 0
        REPEAT_LIMIT = 3
        step_cap, token_budget = self._budget()
        step_cap = min(step_cap, self.max_tool_rounds)
        total_tokens = 0
        for steps in range(step_cap):
            if token_budget > 0 and total_tokens >= token_budget:
                self._audit_budget("token_exhausted", steps, total_tokens)
                return ("预算耗尽已停止：本轮累计消耗 token 超过阈值 "
                        f"({total_tokens} >= {token_budget})。请精简任务或调大 budget.max_tokens。")
            resp = gateway.chat(self._messages(),
                                provider_model=self.provider_model,
                                tools=tool_registry.core_tools_schema())
            total_tokens += self._tokens(resp)
            content = resp.get("content")
            tool_calls = resp.get("tool_calls")
            self.memory.add_assistant(content, tool_calls)
            if not tool_calls:
                return content or ""
            for tc in tool_calls:
                fn = getattr(tc, "function", None) or {}
                name = getattr(fn, "name", None) or ""
                args = getattr(fn, "arguments", "{}") or "{}"
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                sig = f"{name}::{args}"
                repeat_cnt = repeat_cnt + 1 if sig == last_sig else 1
                last_sig = sig
                if repeat_cnt >= REPEAT_LIMIT:
                    return f"检测到工具 `{name}` 连续重复调用相同参数 {REPEAT_LIMIT} 次，判定死循环，已停止。"
            self._dispatch_tool_calls(tool_calls)
            self.memory.compact()
            self._maybe_summarize()
        return "(工具调用轮次过多，已中止)"

    def chat_stream(self, user_input: str,
                    on_text: Callable[[str], None] | None = None) -> str:
        """流式对话。on_text 每收到一段增量回调一次。返回完整最终文本。"""
        self.memory.add_user(user_input)
        full = []
        for _ in range(self.max_tool_rounds):
            collected = []
            for delta in gateway.chat_stream(
                    self._messages(), provider_model=self.provider_model,
                    tools=tool_registry.core_tools_schema()):
                collected.append(delta)
                if on_text:
                    on_text(delta)
            text = "".join(collected)
            full.append(text)
            self.memory.add_assistant(text, None)
            if text.strip():
                # 流式下简化处理：有非空文本即视为一轮作答，返回
                return "".join(full)
            self.memory.compact()
        return "".join(full)

    def _detect_cycle(self, tool_names: list[str]) -> str | None:
        """检测工具调用模式是否陷入循环：2-3 步短序列连续重复 2+ 次即触发。

        例: [write, verify, run, write, verify, run] → 检测到 "write-verify-run" 循环
        返回循环描述字符串，未检测到返回 None。
        """
        n = len(tool_names)
        # 至少需要 4 个工具名才能检测到 2+2 循环
        if n < 4:
            return None
        # 尝试 2~3 步的短序列是否连续出现 2+ 次
        for seq_len in (2, 3):
            # 从后往前找：最近的 seq_len*2 个中是否前半 == 后半
            tail = tool_names[-(seq_len * 2):]
            if len(tail) == seq_len * 2:
                first_half = tuple(tail[:seq_len])
                second_half = tuple(tail[seq_len:])
                if first_half == second_half:
                    desc = "-".join(first_half)
                    return f"检测到工具循环模式: {desc}"
        # 额外：同一单一工具连续出现 4+ 次
        if n >= 4 and len(set(tool_names[-4:])) == 1:
            return f"检测到单一工具连续调用: {tool_names[-1]} × 4"
        return None

    @_fire_stop_hook
    def run_task(self, task: str, max_steps: int = 30,
                 on_step=None, on_stream=None, on_reasoning=None,
                 should_stop=None) -> dict:
        """自主任务执行：规划 → 执行工具/沙箱 → 验证，全自动循环直到产出最终答案。

        - 默认在隔离沙箱中真实执行代码（run_shell/verify_code 走 Executor）。
        - 不向用户逐条确认（全自动）。
        - on_step(名称, 结果, 参数) 回传每一步进度；on_stream(文本) 逐字回传模型流式生成；
          on_reasoning(文本) 逐字回传云端推理模型(如 DeepSeek-R1)的思考内容；
          should_stop() 返回 True 则提前停止。
        - 三重死循环守卫：精确签名重复、模式循环、工具多样性耗尽。
        - 收尾决策：核心验证通过后注入收敛提示，让模型直接总结而非无休止调用工具。
        """
        self._reasoning_buf = []
        self.memory.add_user(task if task else "开始")
        # 双信道压缩 · 信道B启用：项目根已设时，压缩丢弃的早期消息落盘 .hs/checkpoints/
        try:
            from core.agent import tools as _tr
            _root = _tr.get_project_root()
            if _root and _root != ".":
                from pathlib import Path
                self.memory.set_checkpoint_dir(str(Path(_root).resolve() / ".hs" / "checkpoints"))
        except Exception:  # noqa: BLE001
            pass
        outcome = {"steps": 0, "tools": []}
        # 多 agent 决策已改为「模型主动」：不再在此按任务大小启发式预拆链。
        # 模型在执行循环中自行判断复杂度，需要时主动调用 route_chain 工具
        # 把任务拆成 planner→coder→checker 链协作；简单任务则由模型直接单 agent 完成。
        # 子 agent（_run_sub 线程）已禁用 route_chain/sub_spawn 等编排工具，防递归。
        # 守卫1：精确签名连续重复 3 次（原逻辑保留）
        last_sig, repeat_cnt = None, 0
        REPEAT_LIMIT = 3
        # 守卫2：模式循环 + 工具多样性追踪
        step_tool_names: list[str] = []
        # 守卫4 状态：写类工具「相似参数重复」检测（跨工具独立计数）
        _sim_prev: dict[str, str] = {}
        _sim_run: dict[str, int] = {}
        # 收尾决策：记录是否已成功完成核心验证，避免模型继续空转
        verified_ok, convergence_injected = False, False
        noassert_nudge = False  # "请补断言"提示仅注入一次，防重复轰炸
        # 预算守卫（仿 codex rollout_budget）：轮次 + token 双阈值
        step_cap, token_budget = self._budget()
        step_cap = min(step_cap, max_steps)
        total_tokens = 0
        for steps in range(step_cap):
            if token_budget > 0 and total_tokens >= token_budget:
                self._audit_budget("token_exhausted", steps, total_tokens)
                return {"answer": ("预算耗尽已停止：本轮累计 token "
                                   f"{total_tokens} 超过预算 {token_budget}。"
                                   "请精简任务或调大 budget.max_tokens。"), **outcome}
            if should_stop and should_stop():
                return {"answer": "已手动停止任务。", **outcome}
            resp = gateway.chat_structured_stream(
                self._messages(),
                provider_model=self.provider_model,
                tools=tool_registry.core_tools_schema(),
                on_delta=on_stream,
                on_reasoning=on_reasoning)
            total_tokens += self._tokens(resp)
            rz = resp.get("reasoning")
            if rz:
                self._reasoning_buf.append(rz)
            content = resp.get("content") or ""
            tool_calls = resp.get("tool_calls")
            # 流式回退：本地模型可能不走原生 tool_calls，而是把工具调用 JSON 当文本返回，
            # 此时把文本解析为工具调用继续执行，而不是误当最终答案结束。
            if not tool_calls:
                parsed = _tool_calls_from_text(content)
                if parsed:
                    tool_calls = parsed
                    content = ""
            self.memory.add_assistant(content, tool_calls)
            if tool_calls:
                for tc in tool_calls:
                    if should_stop and should_stop():
                        return {"answer": "已手动停止任务。", **outcome}
                    if tc.type != "function":
                        continue
                    fn = getattr(tc, "function", None) or {}
                    name = getattr(fn, "name", None) or ""
                    args = getattr(fn, "arguments", "{}") or "{}"
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    sig = f"{name}::{args}"
                    # 守卫1：精确签名连续重复
                    repeat_cnt = repeat_cnt + 1 if sig == last_sig else 1
                    last_sig = sig
                    if repeat_cnt >= REPEAT_LIMIT:
                        return {"answer": (f"检测到工具 `{name}` 连续重复调用相同参数 "
                                           f"{REPEAT_LIMIT} 次，判定陷入死循环，已强制终止任务。"),
                                **outcome}
                    # 守卫4：写类工具「相似参数重复」——抓“每次微调后重复”的隐蔽绕圈
                    if name in _SIM_TOOLS:
                        norm = _args_normalized(args)
                        prev = _sim_prev.get(name)
                        if prev is not None and _args_similar(prev, norm) >= _SIM_SIMILARITY:
                            _sim_run[name] = _sim_run.get(name, 0) + 1
                        else:
                            _sim_run[name] = 1
                        _sim_prev[name] = norm
                        if _sim_run[name] >= _SIM_RUN_LIMIT:
                            return {"answer": (f"检测到工具 `{name}` 连续 {_sim_run[name]} 次调用参数高度相似"
                                               f"（每次仅小幅改动后重复），判定陷入隐蔽循环，已强制终止任务。"),
                                    **outcome}
                    result = self._invoke_tool(name, args)
                    self._add_tool(tc.id, name, result)
                    outcome["tools"].append(name)
                    step_tool_names.append(name)
                    outcome["steps"] += 1
                    self._maybe_summarize()
                    # 收尾决策：verify_code/run_shell 成功通过即视为已产出并验证。
                    # 断言式验证：verify_code 只有"带断言且全过"才算强通过；
                    # 无断言的弱验证不视为强通过，仅注入一次"请补断言"的引导，避免把
                    # "能跑通"误当成"结果正确"而草草收敛（同时不重复轰炸模型）。
                    if name == "verify_code":
                        _noassert = "__NOASSERT__" in result
                        if not _is_failure(result):
                            if _noassert:
                                if not noassert_nudge:
                                    noassert_nudge = True
                                    self.memory.add_user(
                                        "你刚才的 verify_code 虽能跑通，但代码里没有任何断言"
                                        "（弱验证），只能证明'能运行'、不能证明'结果正确'。"
                                        "请给核心函数补充 assert 断言式自测，再次调用 verify_code 验证，"
                                        "断言全部通过后再交付。")
                            else:
                                verified_ok = True
                    elif name == "run_shell" and not _is_failure(result):
                        verified_ok = True
                    if _is_failure(result):
                        # 失败反思注入：把真实报错回填，要求模型定位原因并修正，避免重复踩坑
                        verified_ok = False
                        self.memory.add_user(
                            f"你刚调用工具 {name} 失败，真实返回：\n{result}\n"
                            "请先反思失败原因（只依据上面的真实报错，不要臆测），"
                            "给出修正方案，然后用修正后的做法重试；不要原样重复该失败调用。")
                    if on_step:
                        on_step(name, result, args)
                # 收尾决策：核心验证通过后收敛，避免模型无意义空转
                if verified_ok and outcome["steps"] >= 2:
                    if not convergence_injected:
                        convergence_injected = True
                        self.memory.add_user(
                            "核心产出已生成并通过验证。请不要再调用任何工具，"
                            "直接用简洁中文总结：完成了什么、生成了哪些文件、验证结果如何，然后结束。")
                    else:
                        # 已提示收敛却仍连续调用工具：视为空转，强制收尾
                        return {"answer": ("核心产出已生成并通过验证（见上方工具执行日志与生成文件）。"
                                           "已完成本轮任务。"), **outcome}
                # 守卫2：模式循环检测（每轮结束后检查）
                cycle = self._detect_cycle(step_tool_names)
                if cycle and outcome["steps"] >= 8:
                    return {"answer": (f"{cycle}。已强制终止，请检查是否缺少最终回答步骤。"
                                       "提示：写出代码并验证通过后，应直接给出最终答案并结束，"
                                       "而不是继续调用工具。"), **outcome}
                # 守卫3：工具多样性耗尽（15+ 步只用了 ≤2 种工具，强制收敛）
                unique_tools = set(step_tool_names)
                if outcome["steps"] >= 15 and len(unique_tools) <= 2:
                    self.memory.add_user(
                        "你已连续调用了 15 步以上但只用了极少数工具，任务很可能已经完成。"
                        "请不要再调用任何工具，直接用简洁中文总结你完成了什么、生成了哪些文件、"
                        "验证结果如何，然后结束。")
                continue
            if content.strip():
                return {"answer": content, **outcome}
        return {"answer": "已达到最大执行步数，任务未收敛，已自动停止。请查看上方进度日志。",
                **outcome}

    def _dispatch_tool_calls(self, tool_calls) -> None:
        for tc in tool_calls:
            if tc.type != "function":
                continue
            fn = getattr(tc, "function", None) or {}
            name = getattr(fn, "name", None) or ""
            args = getattr(fn, "arguments", "{}") or "{}"
            tid = tc.id
            # 注意：LiteLLM 里 arguments 可能是对象或字符串，统一处理
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            result = self._invoke_tool(name, args)
            self._add_tool(tid, name, result)

    def generate_code(self, task: str, max_attempts: int = 3) -> dict:
        """生成代码 + 自动验证循环（防幻觉关键）。

        1. 让模型产出代码；
        2. 抽取代码 → 隔离进程实际运行（verify_code）；
        3. 若失败，把真实报错回填，要求模型修正后重试，直到跑通或达到次数上限。
        """
        from .tools import _extract_python
        outcome = {"code": "", "verified": False, "attempts": 0, "report": ""}
        base_prompt = (task + "\n请只返回可直接运行的 Python 代码（放 python 代码块中，"
                               "不要附带解释）。")
        self.memory.add_user(base_prompt)
        for attempt in range(1, max_attempts + 1):
            outcome["attempts"] = attempt
            resp = gateway.chat(self._messages(),
                                provider_model=self.provider_model,
                                tools=tool_registry.core_tools_schema())
            content = resp.get("content") or ""
            tool_calls = resp.get("tool_calls")
            self.memory.add_assistant(content, tool_calls)
            if tool_calls:
                self._dispatch_tool_calls(tool_calls)
                continue
            if not content.strip():
                continue
            code = _extract_python(content)
            # 隔离执行验证
            report = tool_registry.invoke(
                "verify_code", {"code": code, "timeout": 30})
            self.memory.add_tool("verify-" + str(attempt), "verify_code", report)
            outcome["code"] = code
            outcome["report"] = report
            outcome["verified"] = "✅" in report and "未定义" not in report
            if outcome["verified"]:
                return outcome
            # 失败：把验证报错注入，要求修正
            self.memory.add_user(
                f"你上次生成的代码验证未通过，请按下面报错修正后再重新给出完整代码：\n{report}")
        return outcome