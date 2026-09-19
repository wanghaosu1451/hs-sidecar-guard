"""沙箱管理：模型/代码在隔离子进程运行。

提供两类沙箱执行：
- run_code  ：运行任意 Python 代码片段（隔离进程 + 超时）
- run_model ：把一次模型推理放在隔离子进程执行（走统一网关，落库归管理）
记录资源占用（psutil 可选）。
"""
from __future__ import annotations

import json

from ..llm import gateway
from .executor import Executor, ExecResult
from .security import SandboxPolicy


class Sandbox:
    def __init__(self, policy: SandboxPolicy | None = None):
        self.policy = policy or SandboxPolicy()
        self.executor = Executor(self.policy)
        self._history: list[dict] = []

    def run_code(self, code: str, cwd: str | None = None) -> ExecResult:
        r = self.executor.run_python(code, cwd)
        self._history.append({"kind": "code", "ok": r.ok, "out": r.text})
        return r

    def run_model(self, prompt: str, provider_model: str | None = None) -> ExecResult:
        """在子进程里执行一次模型推理，把结果返回。"""
        payload = json.dumps({
            "provider_model": provider_model or "openai/gpt-4o-mini",
            "prompt": prompt,
        })
        script = (
            "import json,sys\n"
            "p=json.loads(sys.argv[1])\n"
            "from core.llm import gateway\n"
            "r=gateway.chat([{'role':'user','content':p['prompt']}], "
            "provider_model=p['provider_model'])\n"
            "print(r.get('content') or '')\n"
        )
        r = self.executor._run([exec_cmd(), "-c", script, payload],
                               cwd=None)
        self._history.append({"kind": "model", "ok": r.ok, "out": r.text})
        return r

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def apply_policy(self, policy: SandboxPolicy) -> None:
        self.policy = policy
        self.executor.policy = policy

    def policy_dict(self) -> dict:
        return self.policy.to_dict()


def exec_cmd():
    import sys
    return sys.executable