"""Sidecar 独立进程：FastAPI server 暴露 HTTP 接口，主 Agent 通过本地端口调用。

主进程（主 Agent） ←--HTTP 127.0.0.1:PORT--→ 子进程（Sidecar server）

物理隔离：
  - 独立 OS 进程（subprocess.Popen 拉起）
  - 独立 Python 解释器
  - 独立 GPU 显存（各加载各的模型/适配器）
  - 独立上下文（主 Agent 对话历史不传给 Sidecar，Sidecar 只看锚点 + 当前工具调用）

启动：python -m core.sidecar.server --port 8765 --adapter artifacts/sidecar_lora/adapter --project-root .
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .anchor import TaskAnchor
from .drift import DriftDetector, DriftReport
from .firewall.gate import FirewallGate
from .progress import ProgressTracker, TraceLogger
from .ui import TerminalUI
from ..sandbox.branch import SandboxManager, SandboxBranch


# ---------------------------------------------------------
# 轻量本地 LLM client（加载 Sidecar 专用 LoRA 适配器）
# ---------------------------------------------------------

class _LocalLlm:
    """Sidecar 进程内的本地推理 client。加载 LoRA adapater 到 Qwen2.5-1.5B 基座。"""

    def __init__(self, model_path: str, adapter_path: str,
                 device: str = "cuda") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            raise RuntimeError(f"缺少依赖: {e}")

        self.torch = torch
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 4bit 加载（和 finetune.py 一致）
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=qcfg, device_map="auto")
        if adapter_path and os.path.isdir(adapter_path):
            # 姿势2（唯一稳定方案）：4bit 基座 + PeftModel + merge_and_unload。
            # 直接 PeftModel 不 merge 会输出空换行（已知问题）。
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.1) -> str:
        # Sidecar 推理 prompt 格式（ChatML，**和训练 format_alpaca 默认 system 完全对齐**）。
        # 两边 system prompt 必须一致，否则 LoRA 对推理 prompt 的生成方向零引导。
        chat = (
            "<|im_start|>system\n"
            "你是终端 Agent 的安全校验器。给定原始任务、锚点（需求/约束/禁止）"
            "和当前操作，判断该操作是否偏离原始任务意图。"
            "只输出 JSON 格式的 score（0-1 越高越偏离）、violations（违规列表）、rationale（简短理由）。"
            "不知道就输出 score=0.5 的保守估计。"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        with self.torch.inference_mode():
            inputs = self.tokenizer(chat, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0, temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            gen = out[0][inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(gen, skip_special_tokens=True)
        return text.strip()


# ---------------------------------------------------------
# Sidecar Server 核心
# ---------------------------------------------------------

class SidecarServer:
    """独立进程内的 Sidecar 单例。HTTP 层只做序列化/反序列化，
    真正的校验逻辑都在这里——和 in-process Sidecar 完全同一份代码。"""

    def __init__(self, project_root: str | Path,
                 model_path: str | None,
                 adapter_path: str | None) -> None:
        self.project_root = str(Path(project_root).resolve())
        self.ui = TerminalUI()

        # Layer 3: Trace（启动就建，独立于 anchor）
        self.trace = TraceLogger(self.project_root)

        # 本地 LLM（可选，加载 LoRA 适配器）
        self.llm = None
        if model_path:
            try:
                self.ui.print("加载本地 LoRA 适配器...", "info")
                self.llm = _LocalLlm(model_path, adapter_path or "")
                self.ui.print("LoRA 加载完成", "ok")
            except Exception as e:
                self.ui.print(f"LoRA 加载失败，降级到规则级: {e}", "warn")

        # 安全组件（全部无 LLM 依赖，毫秒级）
        self.firewall = FirewallGate(project_root=self.project_root,
                                     llm_client=self.llm)

        # 锚点 + 漂移检测 + 会话进展（按需激活）
        self.anchor: TaskAnchor | None = None
        self.detector: DriftDetector | None = None
        self.progress: ProgressTracker | None = None

        # 沙盒分支管理器
        self.sandbox = SandboxManager(self.project_root)

        self.ui.print(f"Sidecar 独立进程启动 | PID={os.getpid()} | project={self.project_root}",
                     "ok")

    # ---------------- HTTP handlers ----------------

    def bootstrap(self, payload: dict) -> dict:
        """首次启动：固化锚点 + 创建 Progress + 记录 Trace。"""
        self.anchor = TaskAnchor.freeze(
            self.project_root,
            payload.get("original_instruction", ""),
            payload.get("requirements"),
            payload.get("constraints"),
            payload.get("forbidden"),
        )
        self.detector = DriftDetector(self.anchor, llm_client=self.llm)
        self.progress = ProgressTracker.create(
            self.project_root, self.anchor.instruction_hash)
        self.ui.anchor_frozen(self.anchor.summary())
        self.trace.append("bootstrap", details={
            "anchor_hash": self.anchor.instruction_hash,
            "requirements": self.anchor.requirements,
            "constraints": self.anchor.constraints,
            "forbidden": self.anchor.forbidden,
        })
        return {"ok": True, "anchor": self.anchor.summary()}

    def pre_tool_use(self, payload: dict) -> dict:
        """拦截入口。payload = {tool_name, arguments}
        返回 {"block": bool, "msg": str, "pause": bool}
        """
        tool_name = payload.get("tool_name", "")
        arguments = payload.get("arguments", "")
        args_preview = arguments if isinstance(arguments, str) \
            else json.dumps(arguments, ensure_ascii=False)

        # 1) Shell 防火墙（最前置，毫秒级）
        fw = self.firewall.judge(tool_name, arguments)
        if fw.blocked:
            self.ui.fw_blocked(tool_name, fw.reason)
            self.trace.append("firewall_block", tool=tool_name,
                              args=args_preview, details={"reason": fw.reason})
            return {"block": True, "msg": fw.reason, "pause": True,
                    "source": "firewall"}
        if fw.confirm_needed:
            self.ui.fw_confirm(tool_name, fw.reason)

        # 2) Goal Drift（有锚点才跑）
        if self.detector is not None:
            report: DriftReport = self.detector.check(tool_name, arguments)
            if report.level == "block":
                self.ui.drift_alert(report.score, report.level,
                                    report.violations, report.rationale)
                if self.progress is not None:
                    self.progress.record_drift(
                        report.score, report.violations,
                        tool_name, args_preview)
                self.trace.append("drift_block", tool=tool_name,
                                  args=args_preview,
                                  details={"violations": report.violations},
                                  score=report.score)
                return {"block": True, "msg": "目标漂移严重: " +
                        "; ".join(report.violations) or report.rationale,
                        "pause": True, "source": "drift",
                        "score": report.score}
            if report.level == "warn":
                self.ui.drift_alert(report.score, report.level,
                                    report.violations, report.rationale)
                if self.progress is not None:
                    self.progress.record_drift(
                        report.score, report.violations,
                        tool_name, args_preview)
                self.trace.append("drift_warn", tool=tool_name,
                                  args=args_preview,
                                  details={"violations": report.violations},
                                  score=report.score)
                return {"block": False, "pause": True,
                        "source": "drift", "score": report.score}

        self.ui.fw_allowed(tool_name)
        self.trace.append("allow", tool=tool_name, args=args_preview)
        return {"block": False, "pause": False}

    def post_tool_use(self, payload: dict) -> dict:
        """工具执行后：只做日志打点。"""
        if self.detector is not None:
            self.detector.check(
                payload.get("tool_name", ""),
                payload.get("arguments", ""))
        self.trace.append("post_tool_use",
                          tool=payload.get("tool_name", ""),
                          args=str(payload.get("arguments", ""))[:200])
        return {"ok": True}

    # ---------------- Sandbox HTTP API ----------------

    def sandbox_create(self, payload: dict) -> dict:
        hint = payload.get("hint", "sandbox")
        branch = self.sandbox.create(hint=hint)
        self.trace.append("sandbox_create", details={"name": branch.name})
        return {"ok": True, "branch": branch.name, "root": branch.root}

    def sandbox_list(self) -> dict:
        return {"ok": True, "branches": self.sandbox.list_branches()}

    def sandbox_diff(self, payload: dict) -> dict:
        branch = self.sandbox.get(payload.get("name", ""))
        if branch is None:
            return {"ok": False, "error": "branch not found"}
        diff_text = branch.generate_diff_text()
        changed = branch.diff_files()
        return {"ok": True, "diff": diff_text,
                "changed": {k: v["state"] for k, v in changed.items()}}

    def sandbox_merge(self, payload: dict) -> dict:
        branch = self.sandbox.get(payload.get("name", ""))
        if branch is None:
            return {"ok": False, "error": "branch not found"}
        applied = branch.merge()
        self.trace.append("sandbox_merge", details={
            "name": branch.name, "applied": applied})
        return {"ok": True, "applied": applied}

    def sandbox_discard(self, payload: dict) -> dict:
        branch = self.sandbox.get(payload.get("name", ""))
        if branch is None:
            return {"ok": False, "error": "branch not found"}
        name = branch.name
        branch.discard()
        self.trace.append("sandbox_discard", details={"name": name})
        return {"ok": True}

    def health(self) -> dict:
        return {
            "ok": True,
            "pid": os.getpid(),
            "llm_loaded": self.llm is not None,
            "anchor_frozen": self.anchor is not None,
            "project_root": self.project_root,
        }


# ---------------------------------------------------------
# HTTP server（用标准库 http.server，零依赖）
# ---------------------------------------------------------

def serve(server: SidecarServer, host: str, port: int) -> None:
    import http.server
    import json as _json

    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 静默，由 TerminalUI 负责日志
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = _json.loads(raw.decode("utf-8") or "{}")
            except Exception:
                payload = {}
            path = self.path.rstrip("/")

            if path == "/bootstrap":
                resp = server.bootstrap(payload)
            elif path == "/pre_tool_use":
                resp = server.pre_tool_use(payload)
            elif path == "/post_tool_use":
                resp = server.post_tool_use(payload)
            elif path == "/sandbox_create":
                resp = server.sandbox_create(payload)
            elif path == "/sandbox_list":
                resp = server.sandbox_list()
            elif path == "/sandbox_diff":
                resp = server.sandbox_diff(payload)
            elif path == "/sandbox_merge":
                resp = server.sandbox_merge(payload)
            elif path == "/sandbox_discard":
                resp = server.sandbox_discard(payload)
            elif path == "/health":
                resp = server.health()
            else:
                self.send_error(404); return

            body = _json.dumps(resp, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                return self.do_POST()
            self.send_error(404)

    srv = http.server.HTTPServer((host, port), _Handler)
    srv.timeout = 3600  # 长连接保活
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


# ---------------------------------------------------------
# 进程入口
# ---------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Sidecar 独立校验进程")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--project-root", required=True,
                     help="主项目根目录（锚点文件和安全规则文件放这里）")
    ap.add_argument("--model-path", default=None,
                     help="基座模型路径（可选；不提供就降级为规则级）")
    ap.add_argument("--adapter-path", default=None,
                     help="Sidecar 专用 LoRA 适配器路径")
    args = ap.parse_args()

    server = SidecarServer(
        project_root=args.project_root,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
    )

    # SIGTERM 优雅退出
    def _term(signum, frame):
        print(f"[Sidecar] 收到信号 {signum}，退出")
        sys.exit(0)
    signal.signal(signal.SIGTERM, _term)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _term)

    serve(server, args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())

