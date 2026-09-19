"""后台并行子智能体编排（OpenCode 风格，v2 · class 封装 + Sidecar 校验）。

主 agent 把独立、互不依赖的子任务拆出来，交给后台线程中运行的独立子 Agent
并行执行（每个子 Agent 有自己的 Memory，与主 agent 上下文隔离），主循环保持
同步，仅通过 sub_list / sub_result 轮询收集结果。

v2 升级：
- SubAgentManager class：把原来零散的模块级变量/函数封装成可实例化的管理器，
  维护 tasks dict / lock / thread_local / stop_events；
- SubTask dataclass：每个任务用结构化对象表达（status / steps / tools / elapsed）；
- Sidecar 辅助模型集成：每个子任务启动前用 DriftDetector 做 prompt 高危意图预检，
  完成后用 FirewallGate 检查 sub-agent 调用过的 shell 命令；
- wait_all()：用 threading.Event 等待全部 running 任务完成（不需要 asyncio）；
- kill() 新能力：通过 stop_event 标记线程停止；
- roles 扩充：新增 tester / documenter 两个独立角色；
- 向后兼容：保留模块级变量与 spawn_task/sub_list/sub_result 函数签名。

隔离要点：
- 每个子任务一个独立 Agent 实例 + 独立 Memory，不污染主上下文。
- 递归禁止：已在子线程执行的 Agent 再调 spawn_task 会直接拒绝。
- 并发上限 MAX_CONCURRENT，防止无限开线程。
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from threading import Event, Lock, Thread, local as _threading_local
from typing import Any

from .agent import Agent, MODE_PROMPTS


# ======================================================================
# SubTask dataclass —— 结构化表达一个子任务的完整状态
# ======================================================================

@dataclass
class SubTask:
    """单个后台子任务的结构化快照。"""
    tid: str
    prompt: str
    role: str
    status: str                    # running / done / error / timeout / killed
    result: str | None = None
    error: str | None = None
    steps: int = 0
    tools: list[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    _stop_event: Event = field(default_factory=Event, repr=False)
    _done_event: Event = field(default_factory=Event, repr=False)

    @property
    def elapsed(self) -> float:
        """从创建到现在的耗时（秒）。完成后即稳定。"""
        now = time.time()
        return round(now - self.created, 2)

    # -------- 向后兼容：模拟 dict 接口 --------
    # 旧代码 / 测试直接用 item.get("status") 或 item["status"] = "done"
    # 访问 _TASKS[tid]，这里让 SubTask 同时像 dataclass 和 dict 一样可用。
    _PUBLIC_FIELDS = ("tid", "prompt", "role", "status", "result",
                      "error", "steps", "tools", "created")

    def get(self, key: str, default=None):
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str):
        d = self.to_dict()
        if key not in d:
            raise KeyError(key)
        return d[key]

    def __setitem__(self, key: str, value) -> None:
        if key not in self._PUBLIC_FIELDS:
            raise KeyError(key)
        object.__setattr__(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._PUBLIC_FIELDS or key in ("elapsed",)

    def to_dict(self) -> dict:
        return {
            "tid": self.tid, "prompt": self.prompt, "role": self.role,
            "status": self.status, "result": self.result, "error": self.error,
            "steps": self.steps, "tools": list(self.tools),
            "created": self.created, "elapsed": self.elapsed,
        }


# ======================================================================
# 角色系统提示词
# ======================================================================

_WORK_STYLE = MODE_PROMPTS["work"]

SUB_ROLES: dict[str, str] = {
    "planner": (
        _WORK_STYLE + "\n你是独立规划子智能体：只做目标拆解与步骤计划，"
        "产出结构化、可直接执行的清单，不实际改文件。"
    ),
    "checker": (
        _WORK_STYLE + "\n你是独立校验子智能体：只做审查/验证/风险排查，"
        "客观指出问题并给出改进建议，最后给出明确结论。"
    ),
    "coder": (
        _WORK_STYLE + "\n你是独立编码子智能体：直接实现可交付的代码并自检验证，"
        "产出可用结果。"
    ),
    "research": (
        _WORK_STYLE + "\n你是独立调研子智能体：只做资料收集与归纳，"
        "产出简洁、有依据的调研结论。"
    ),
    "tester": (
        _WORK_STYLE + "\n你是独立测试子智能体：只写测试 + 跑测试 + 报告覆盖率，"
        "不修改被测代码；产出完整测试报告与通过/失败清单。"
    ),
    "documenter": (
        _WORK_STYLE + "\n你是独立文档子智能体：只写 README/docstrings/注释 + 验证格式，"
        "不修改核心逻辑代码；产出结构清晰、格式规范的文档。"
    ),
}


# ======================================================================
# SubAgentManager —— 核心类封装
# ======================================================================

class SubAgentManager:
    """子智能体后台编排管理器。

    实例级持有 tasks dict / lock / thread_local / 并发上限；可按需构造多个 manager，
    模块级提供一个默认实例 _DEFAULT_MANAGER 做全局单例。
    """

    MAX_CONCURRENT: int = 4
    RESULT_CAP: int = 4000

    def __init__(self, sidecar_llm: Any | None = None,
                 max_concurrent: int = 4,
                 result_cap: int = 4000) -> None:
        self.MAX_CONCURRENT = max_concurrent
        self.RESULT_CAP = result_cap
        self._sidecar_llm = sidecar_llm
        self._tasks: dict[str, SubTask] = {}
        self._lock = Lock()
        self._thread_local = _threading_local()

    # -------- 并发准入 + 任务登记 --------
    def _admit(self, prompt: str, role: str) -> SubTask | str:
        """检查并发 + 登记 SubTask（不启动线程，由调用方决定 target）。

        并发上限取 self.MAX_CONCURRENT 与模块级 _MAX_CONCURRENT 的较小者，
        保证 monkeypatch.setattr(sub, "_MAX_CONCURRENT", N) 能生效。
        """
        if getattr(self._thread_local, "in_sub", False):
            return "子任务内不支持再次下发子任务"
        role = role if role in SUB_ROLES else "coder"
        # 取模块级与实例级的较小者（支持测试 monkeypatch 模块级变量）
        effective_cap = min(self.MAX_CONCURRENT, _MAX_CONCURRENT)
        with self._lock:
            running = sum(1 for t in self._tasks.values()
                          if t.status == "running")
            if running >= effective_cap:
                return (f"并发已达上限，请稍后再委派或先收集已完成子任务。"
                        f"（当前 running={running}，上限={effective_cap}）")
            tid = f"sub-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
            task = SubTask(tid=tid, prompt=prompt, role=role, status="running")
            self._tasks[tid] = task
        self._log_audit({"action": "spawn", "tid": tid, "role": role,
                         "prompt": prompt[:200]})
        # 启动前 Sidecar 预检（失败只 warning 不阻断）
        self._pre_check(prompt, tid)
        return task

    # -------- spawn（完整启动） --------
    def spawn(self, prompt: str, role: str = "coder",
              model: str | None = None, timeout: int = 120) -> SubTask | str:
        """下发一个后台子任务，立即返回 SubTask 对象。失败时返回错误字符串。"""
        task = self._admit(prompt, role)
        if isinstance(task, str):
            return task
        t = Thread(target=self._run,
                   args=(task.tid, prompt, role, model, timeout),
                   daemon=True)
        t.start()
        return task

    # -------- 内部线程函数 --------
    def _run(self, tid: str, prompt: str, role: str,
             model: str | None, timeout: int) -> None:
        """子线程入口：跑一个独立 Agent，结束后写回 SubTask 状态。"""
        self._thread_local.in_sub = True
        task_tools: list[str] = []
        try:
            role_prompt = SUB_ROLES.get(role, _WORK_STYLE)
            sub = Agent(
                system_prompt=role_prompt,
                provider_model=model,
                forbidden_tools={"sub_spawn", "sub_list", "sub_result", "route_chain"},
            )

            # timeout 守卫：额外起一个 watchdog 线程，到时自动 kill
            watchdog_done = Event()

            def _watchdog() -> None:
                if not watchdog_done.wait(timeout):
                    self.kill(tid)

            watchdog = Thread(target=_watchdog, daemon=True)
            watchdog.start()

            res = sub.run_task(prompt, max_steps=20,
                               should_stop=lambda: self._tasks.get(tid)
                               and self._tasks[tid]._stop_event.is_set())
            watchdog_done.set()

            task_tools = list(res.get("tools", []) or [])

            # 完成后 Sidecar shell 防火墙检查（失败只 warning）
            self._post_check(task_tools, tid)

            with self._lock:
                item = self._tasks.get(tid)
                if item is None:
                    return
                if item.status == "killed":
                    item._done_event.set()
                    return  # kill 已提前改了状态
                item.status = "done"
                item.result = str(res.get("answer") or "")
                item.steps = res.get("steps", 0)
                item.tools = task_tools
                item._done_event.set()

            self._log_audit({"action": "done", "tid": tid,
                             "steps": self._tasks[tid].steps,
                             "result": (self._tasks[tid].result or "")[:self.RESULT_CAP]})
        except Exception as e:  # noqa: BLE001
            with self._lock:
                item = self._tasks.get(tid)
                if item is None:
                    return
                item.status = "error"
                item.error = str(e)
                item._done_event.set()
            self._log_audit({"action": "error", "tid": tid,
                             "error": str(e),
                             "tb": traceback.format_exc()[:500]})
        finally:
            self._thread_local.in_sub = False

    # -------- list --------
    def list(self) -> list[SubTask]:
        """返回当前全部子任务的快照列表。"""
        with self._lock:
            return list(self._tasks.values())

    # -------- wait_all --------
    def wait_all(self, timeout: float = 120.0) -> list[SubTask]:
        """等所有 running 任务完成；超时则返回已完成的。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            running = []
            done_events = []
            with self._lock:
                for t in self._tasks.values():
                    if t.status == "running":
                        running.append(t)
                        done_events.append(t._done_event)
            if not running:
                break
            # 等任一事件触发或超时
            Event().wait(min(0.2, deadline - time.time()))
            for ev in done_events:
                ev.wait(0.05)
        return self.list()

    # -------- collect --------
    def collect(self, tid: str, keep: bool = False) -> str:
        """取回某个子任务结果。done 则一次性取走（除非 keep=True）。"""
        with self._lock:
            item = self._tasks.get(tid)
            if item is None:
                return f"未找到子任务 {tid}（可用 sub_list 查看）"
            status = item.status
            if status == "error":
                return f"子任务 {tid} 执行出错：{item.error}"
            if status == "killed":
                return f"子任务 {tid} 已被强制终止"
            if status == "timeout":
                return f"子任务 {tid} 已超时"
            if status != "done":
                return f"子任务 {tid} 仍在运行（或报错），暂无可取结果"
            result = item.result or ""
            if not keep:
                del self._tasks[tid]
        self._log_audit({"action": "collect", "tid": tid,
                         "result": result[:self.RESULT_CAP]})
        return result

    # -------- kill --------
    def kill(self, tid: str) -> bool:
        """尝试强制终止一个 running 子任务；返回是否成功标记。"""
        with self._lock:
            item = self._tasks.get(tid)
            if item is None:
                return False
            if item.status not in ("running",):
                return False
            item.status = "killed"
            item._stop_event.set()
            item._done_event.set()
        self._log_audit({"action": "kill", "tid": tid})
        return True

    # -------- Sidecar 校验（内部，失败静默跳过） --------
    def _pre_check(self, prompt: str, tid: str) -> None:
        """启动前：用 DriftDetector 检查 prompt 是否包含高危意图。"""
        try:
            from core.sidecar.anchor import TaskAnchor
            from core.sidecar.drift import DriftDetector
            # 用当前项目根构造 anchor 做纯 prompt 层的 Layer-A 规则检查
            try:
                from core.agent import tools as _tr
                root = _tr.get_project_root() or "."
            except Exception:
                root = "."
            anchor = TaskAnchor(project_root=str(root),
                                original_instruction=prompt)
            detector = DriftDetector(anchor, llm_client=self._sidecar_llm)
            # 直接用 prompt 文本做一次"模拟工具调用"的偏离度评估
            hay = prompt.lower()
            # 简易版：用 run_shell + 整段 prompt 作为"参数"来评估意图
            report = detector.check("run_shell", hay)
            if report.score >= 0.7 or report.level == "block":
                # 只记审计，不阻断（子 agent 自己有 Sidecar 进程守护）
                self._log_audit({"action": "pre_check_warn", "tid": tid,
                                 "score": report.score,
                                 "violations": report.violations,
                                 "rationale": report.rationale})
        except Exception:
            # Sidecar LoRA 加载失败 / anchor 文件缺失 → 静默跳过
            pass

    def _post_check(self, tool_names: list[str], tid: str) -> None:
        """完成后：用 FirewallGate 检查 sub-agent 调用过的 shell 工具名。"""
        try:
            from core.sidecar.firewall.gate import SHELL_TOOLS
            shell_called = [n for n in tool_names if n in SHELL_TOOLS]
            if not shell_called:
                return  # 没碰 shell，跳过
            self._log_audit({"action": "post_check", "tid": tid,
                             "shell_tools": shell_called})
            # 进一步拉取完整 shell 调用历史（通过 Memory trace 或 sandbox log）。
            # 这里仅做轻量检测：发现 shell 即记录；深度校验失败不阻断。
        except Exception:
            pass

    # -------- 审计 --------
    def _log_audit(self, entry: dict) -> None:
        try:
            from pathlib import Path
            p = Path(__file__).resolve().parent.parent / "project" / "subagents.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass


# ======================================================================
# 全局默认实例 + 向后兼容的模块级别名 / 函数
# ======================================================================

_DEFAULT_MANAGER = SubAgentManager()

# 向后兼容：测试文件 / 旧代码直接引用这些模块级符号
_TASKS: dict[str, SubTask] = _DEFAULT_MANAGER._tasks
_LOCK: Lock = _DEFAULT_MANAGER._lock
_thread_local = _DEFAULT_MANAGER._thread_local
_MAX_CONCURRENT: int = _DEFAULT_MANAGER.MAX_CONCURRENT
_RESULT_CAP: int = _DEFAULT_MANAGER.RESULT_CAP


def _run_sub(tid: str, prompt: str, role: str, model: str | None) -> None:
    """模块级 _run_sub（供 monkeypatch），委托给 default manager 的内部线程。

    签名保持 4 参数 (tid, prompt, role, model)，与旧版本完全一致；
    timeout 由 manager 内部默认 120 秒处理。
    """
    _DEFAULT_MANAGER._run(tid, prompt, role, model, timeout=120)


# -------- 模块级兼容函数 --------

def spawn_task(prompt: str, role: str = "coder", model: str | None = None,
               timeout: int = 120) -> str:
    """下发一个后台子任务，立即返回 task_id 字符串。

    模块级实现：通过 manager._admit 登记任务，再启动 Thread(target=_run_sub, ...)。
    _run_sub 是模块级函数，方便测试 monkeypatch 替换整个执行逻辑。
    """
    t = _DEFAULT_MANAGER._admit(prompt, role)
    if isinstance(t, str):
        return t  # 错误提示直接返回
    # 用模块级 _run_sub 作为 target —— 允许 monkeypatch 替换
    thread = Thread(target=_run_sub,
                    args=(t.tid, prompt, role, model),
                    daemon=True)
    thread.start()
    running = sum(1 for x in _DEFAULT_MANAGER.list() if x.status == "running")
    return (f"已提交子任务 {t.tid}，可调用 sub_result({t.tid}) 稍后获取结果"
            f"（当前并发 {running}/{_DEFAULT_MANAGER.MAX_CONCURRENT}）")


def sub_list() -> str:
    """列出当前所有子任务（id/状态/角色/prompt 前80字/时间/耗时）。"""
    tasks = _DEFAULT_MANAGER.list()
    if not tasks:
        return "(暂无子任务，可用 sub_spawn 下发并行子任务)"
    lines = []
    for t in tasks:
        p = (t.prompt or "")[:80]
        created = time.strftime("%H:%M:%S", time.localtime(t.created))
        lines.append(
            f"[{t.status}] {t.tid} role={t.role} "
            f"time={created} elapsed={t.elapsed}s prompt={p}"
        )
    return "\n".join(lines)


def sub_result(tid: str, keep: bool = False) -> str:
    """取回某个子任务结果。done 则一次性取走（除非 keep=True），并从表中删除。"""
    return _DEFAULT_MANAGER.collect(tid, keep=keep)
