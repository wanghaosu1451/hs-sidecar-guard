"""会话记忆管理：多轮历史记录、上下文裁剪、可选落盘。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_MAX_MESSAGES = 40
DEFAULT_MAX_TOKENS = 12000


class Memory:
    def __init__(self, max_messages: int = DEFAULT_MAX_MESSAGES,
                 max_chars: int = DEFAULT_MAX_TOKENS * 4):
        self.messages: list[dict[str, Any]] = []
        self.max_messages = max_messages
        self.max_chars = max_chars
        # 双信道压缩的信道B：早期消息外部化到磁盘的目录（None=不落盘）。由 Agent 按项目根设置。
        self.checkpoint_dir: str | None = None

    def set_checkpoint_dir(self, path: str | None) -> None:
        """开启“早期消息外部化”：把被压缩丢弃的早期对话完整备份为 JSONL checkpoint，
        即使摘要丢了细节，也可在磁盘上随时 read_file 取回（解决长任务上下文压缩后的细节丢失）。"""
        self.checkpoint_dir = path

    def externalize(self, dropped: list[dict]) -> str | None:
        """把 dropped 早期消息追加写成一个 JSONL checkpoint，返回文件路径（无目录则不写）。"""
        if not dropped or not self.checkpoint_dir:
            return None
        try:
            import time as _t
            d = Path(self.checkpoint_dir)
            d.mkdir(parents=True, exist_ok=True)
            name = f"checkpoint_{int(_t.time() * 1000)}.jsonl"
            p = (d / name)
            with p.open("a", encoding="utf-8") as f:
                for m in dropped:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
            return str(p)
        except Exception:  # noqa: BLE001
            return None

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str | None, tool_calls=None) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def add_tool(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append({
            "role": "tool", "tool_call_id": tool_call_id, "name": name,
            "content": content,
        })

    def compact(self) -> None:
        """裁剪超长历史，保留系统提示（role=system 不动）。"""
        # 先按消息数裁剪
        user_msgs = [m for m in self.messages if m.get("role") != "system"]
        sys_msgs = [m for m in self.messages if m.get("role") == "system"]
        keep = user_msgs[-self.max_messages:]
        # 再按字符数从前往后裁剪
        total = sum(len(m.get("content", "")) for m in keep)
        while keep and total > self.max_chars:
            dropped = keep.pop(0)
            total -= len(dropped.get("content", ""))
        self.messages = sys_msgs + keep

    def should_summarize(self, ratio: float = 1.5) -> bool:
        """早期对话是否已积累到需要压缩（避免粗暴截断丢失全局观）。"""
        n = sum(1 for m in self.messages if m.get("role") != "system")
        return n > self.max_messages * ratio

    def summarize(self, summarizer, keep_last: int | None = None) -> bool:
        """把超出的早期对话压缩成“早期摘要”注入系统消息，而非直接丢弃。

        summarizer(原始文本) -> 摘要字符串（由 Agent 提供网关调用）。
        保留最近 keep_last 条原始消息，保证一致性。
        """
        keep_last = keep_last or self.max_messages
        sys_msgs = [m for m in self.messages if m.get("role") == "system"]
        user_msgs = [m for m in self.messages if m.get("role") != "system"]
        if len(user_msgs) <= keep_last:
            return False
        dropped = user_msgs[:-keep_last]
        # 双信道压缩 · 信道B：被丢弃的早期消息先完整落盘为 checkpoint（细节不打折）。
        ckpt = self.externalize(dropped)
        # 抽取可理解的语义行（用户问话 + 助手文本），忽略 tool 噪音
        lines = []
        for m in dropped:
            role = m.get("role")
            c = (m.get("content") or "").strip()
            if role == "tool" or not c:
                continue
            who = "用户" if role == "user" else "助手"
            lines.append(f"{who}: {c[:300]}")
        _checkpoint_hint = (f"\n（完整早期对话已备份至 {ckpt}，"
                            "如需要任何被压缩前的原始细节，可用 read_file 前往该文件查阅）"
                            if ckpt else "")
        if lines:
            try:
                summary = (summarizer(
                    "\n".join(lines)
                    + "\n\n以上是一次任务对话的早期内容。请用 ≤120 字压缩成要点："
                      "保留目标、关键决策、已生成/修改的文件与未竟事项，不要复述细节。"
                ) or "").strip()
            except Exception:  # noqa: BLE001
                summary = ""
            if summary:
                sys_msgs.append({"role": "system",
                                 "content": "（早期对话摘要，仅作背景参考，勿当作当前事实）"
                                 + summary + _checkpoint_hint})
        self.messages = sys_msgs + user_msgs[-keep_last:]
        return True

    def get(self) -> list[dict[str, Any]]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = []

    def to_json(self) -> str:
        return json.dumps(self.messages, ensure_ascii=False, indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self.messages = json.loads(Path(path).read_text(encoding="utf-8"))