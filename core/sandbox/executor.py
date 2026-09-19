"""任务执行器：在隔离子进程中运行代码/命令，带超时与资源限制。"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field

from .security import sanitize_env

# 子进程单一输出的最大字符数，防止超长输出撑爆上层上下文
_MAX_OUTPUT = 20000


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    resource: str = ""  # 资源限制触发原因（内存/进程数超额等），非空表示被资源守卫终止

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.resource

    @property
    def text(self) -> str:
        body = self.stdout + (("\n" + self.stderr) if self.stderr else "")
        if len(body) > _MAX_OUTPUT:
            body = body[:_MAX_OUTPUT] + "\n...[输出过长已截断]"
        if self.timed_out:
            body = (body + "\n[超时]").strip()
        if self.resource:
            body = (body + f"\n[资源受限] {self.resource}").strip()
        return body


class Executor:
    """零配置本地沙箱：纯 subprocess 隔离子进程。

    - 工作目录锁定在 allowed_dirs（项目根），读写不出项目根；
    - 默认全断网：不注入任何代理/凭证，并打断网标记；
    - 剥离宿主密钥/凭据类环境变量；
    - timeout 到期强制杀掉整棵进程树（Windows taskkill /T /F），返回 timed_out；
    - 输出统一 utf-8→gb18030→gbk→cp1252→latin-1 回退解码，超长截断。
    """

    def __init__(self, policy):
        self.policy = policy

    def run_python(self, code: str, exec_path: str | None = None) -> ExecResult:
        """在隔离 python 子进程运行一段代码字符串。"""
        cmd = [sys.executable, "-c", code]
        return self._run(cmd, cwd=exec_path)

    def run_command(self, command: str, cwd: str | None = None) -> ExecResult:
        cmd = self.policy.clamp_command(command)
        return self._run(cmd, shell=True, cwd=cwd)

    def _run(self, cmd, shell: bool = False, cwd: str | None = None) -> ExecResult:
        # 守护仅针对“可读写的参数内容”：shell=True 看整串；非 shell 跳过解释器本体，只看参数
        cmd_text = cmd if shell else " ".join(str(x) for x in cmd[1:])
        blocked = self.policy.path_block_reason(cmd_text)
        if blocked:
            return ExecResult(-1, "",
                              f"⚠️ 已拦截：命令引用了项目目录外的路径 {blocked}（"
                              "沙箱读写锁定在项目根内）。")
        eff_cwd = self.policy.resolve_workdir(cwd)
        env = self._env()
        try:
            proc = subprocess.Popen(cmd, shell=shell, cwd=eff_cwd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, env=env,
                                    preexec_fn=_posix_rlimits(self.policy))
        except Exception as e:  # noqa: BLE001
            return ExecResult(-1, "", f"启动失败: {e}")
        # 资源守卫：后台线程按策略监控内存/进程数，超限即杀整棵进程树
        state: dict = {}
        stop = threading.Event()
        watch = threading.Thread(target=self._resource_watch,
                                 args=(proc, stop, state), daemon=True)
        watch.start()
        try:
            out_b, err_b = proc.communicate(timeout=self.policy.timeout_seconds)
            res = ExecResult(proc.returncode, _decode(out_b), _decode(err_b))
        except subprocess.TimeoutExpired:
            self._terminate_tree(proc)
            res = ExecResult(-1, "", "执行超时", timed_out=True)
        except Exception as e:  # noqa: BLE001
            self._terminate_tree(proc)
            res = ExecResult(-1, "", str(e))
        finally:
            stop.set()
            watch.join(timeout=2)
        if state.get("reason"):
            res.resource = state["reason"]
            res.exit_code = -1
        return res

    def _resource_watch(self, proc: subprocess.Popen, stop: threading.Event,
                        state: dict) -> None:
        """后台监控子进程资源占用：内存 RSS、派生进程数，超限强杀。"""
        try:
            import psutil
        except Exception:  # noqa: BLE001 - 无 psutil 时降级为仅超时守护
            stop.wait()
            return
        try:
            p = psutil.Process(proc.pid)
        except Exception:  # noqa: BLE001
            return
        while not stop.is_set():
            try:
                info = p.as_dict(attrs=["memory_info", "num_children"])
                mem = (info.get("memory_info") or _MemInfo()).rss / (1024 * 1024)
                children = info.get("num_children", 0)
            except Exception:  # noqa: BLE001 - 进程已退出即停止监控
                break
            if self.policy.max_memory_mb and mem > self.policy.max_memory_mb:
                state["reason"] = (f"内存 {mem:.0f}MB 超过上限 "
                                   f"{self.policy.max_memory_mb}MB")
                self._terminate_tree(proc)
                break
            if self.policy.max_processes and children > self.policy.max_processes:
                state["reason"] = (f"派生进程数 {children} 超过上限 "
                                   f"{self.policy.max_processes}")
                self._terminate_tree(proc)
                break
            stop.wait(0.3)

    @staticmethod
    def _terminate_tree(proc: subprocess.Popen) -> None:
        """超时/异常时强制杀掉整棵进程树，防止僵尸子进程残留。"""
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True, timeout=10)
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass

    def _env(self) -> dict:
        # 零配置本地化：先从宿主环境剥离密钥/凭据 + 默认剥离代理，再合入加固变量
        env = sanitize_env(os.environ, self.policy.allow_network)
        env.update(_SHELL_ENV)
        return env


def _decode(data: bytes) -> str:
    """跨平台解码子进程输出：优先 UTF-8，失败回退 GBK/gb18030，避免小众编码崩溃。"""
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


class _MemInfo:
    """psutil 缺 memory_info 时的兜底（看门人拿不到就按 0 计，避免误杀）。"""
    rss = 0


def _posix_rlimits(policy):
    """POSIX 下在 fork 出的子进程里设置硬性资源上限：CPU 秒 + 地址空间(MB)。

    Windows 无 resource 模块，依赖后台 _resource_watch + timeout 兜底。
    返回 preexec_fn 可调用对象；进程直接 exec 前生效，宿主进程不受影响。
    """
    try:
        import resource as _res
    except Exception:  # noqa: BLE001 - Windows/受限环境无 resource
        return None

    def _set():
        try:
            if policy.max_memory_mb:
                _res.setrlimit(_res.RLIMIT_AS,
                               (policy.max_memory_mb * 1024 * 1024,
                                policy.max_memory_mb * 1024 * 1024))
            if policy.max_cpu_percent:
                # RLIMIT_CPU 按进程中 CPU 运行秒数硬限制，超限触发 SIGXCPU/被杀
                _res.setrlimit(_res.RLIMIT_CPU,
                               (policy.max_cpu_percent,
                                policy.max_cpu_percent))
        except Exception:  # noqa: BLE001
            pass
    return _set


# 子进程通用环境加固：关 git pager（防 git log/diff 挂死）、禁止交互提示、强制 UTF-8 输出
_SHELL_ENV = {
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUNBUFFERED": "1",
    "NO_COLOR": "1",
    "LC_ALL": "C.UTF-8",
}