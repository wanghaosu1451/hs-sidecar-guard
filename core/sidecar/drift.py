"""目标偏离度打分（纯 Python，不依赖 LLM；后续可挂 1.5B LoRA 做语义增强）。

两层打分：
  Layer A  规则级（毫秒级，确定性）：当前 Agent 行为是否命中 forbidden / 违反 constraints
  Layer B  语义级（可选，需 1.5B LoRA）：当前 Agent 意图与原始 requirements 的语义重叠度
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .anchor import TaskAnchor


@dataclass
class DriftReport:
    """一次工具调用后的偏离度报告。"""

    score: float = 0.0            # 0~1，0=无偏离，1=完全跑偏
    level: str = "ok"             # ok / warn / block
    violations: list[str] = field(default_factory=list)   # 命中的 forbidden / 违反的 constraints
    misses: list[str] = field(default_factory=list)      # requirements 里提到、但本次完全没触达
    rationale: str = ""           # 简短解释（终端展示用）

    @property
    def should_pause(self) -> bool:
        return self.level == "block" or self.score >= 0.6


# -------- Layer A: 规则级（确定性，毫秒级） --------

_KEYWORD_BLOCKS = {
    "rm_rf_root": [r"rm\s+(-[a-zA-Z]*r[a-zA-Z]*|-f[a-zA-Z]*r[a-zA-Z]*|\-[a-zA-Z]*rf[a-zA-Z]*)\s+/(?!\s*tmp)"],
    "git_force_push": [r"git\s+push\s+.*--force(?!.*--lease)"],
    "curl_pipe_sh": [r"curl.*\|\s*(ba|z)?sh", r"wget.*\|\s*(ba|z)?sh"],
    "chmod_suid": [r"chmod\s+.*4\d{3}|chmod\s+.*\+s"],
    "overwrite_key": [r">?\s*[/]?etc[/]passwd", r">?\s*[/]?etc[/]shadow"],
}


def _text_of(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


def _layer_a(anchor: TaskAnchor, tool_name: str, args: Any) -> DriftReport:
    """规则级检查：forbidden 关键词 + 已知高危模式 + constraints 里提到的文件名
    + 向量语义补漏（drift 知识库向量检索，匹配陌生但高相关的漂移模式）。"""
    report = DriftReport()
    hay = _text_of(args).lower()

    # forbidden 关键词命中：
    #   fb = "不要 git push"  →  拆成 keywords = ["git push"]
    #   如果 hay 里含任一关键词 → 命中禁止项
    for fb in anchor.forbidden:
        if not fb:
            continue
        # 抽取"不要 X"里的 X 作为关键词（去掉否定词）
        m = re.match(r"不要\s*(.+)", fb.strip())
        keyword = (m.group(1) if m else fb).strip().lower()
        if keyword and keyword in hay:
            report.violations.append(f"命中禁止项: {fb}")

    # 高危命令模式（和 forbidden 关键词不同——是正则级命令识别）
    for label, patterns in _KEYWORD_BLOCKS.items():
        for pat in patterns:
            if re.search(pat, hay, re.IGNORECASE):
                report.violations.append(f"高危模式: {label}")

    # 约束违反：提取 constraints 里"不要改 X"的 X
    for cons in anchor.constraints:
        if not cons:
            continue
        m = re.search(r"不要(?:改|修改|动|碰|删|移除|touch|modify|edit)\s+([\w\-\./]+)", cons)
        if not m:
            continue
        bad = m.group(1).strip().lower().strip("。.,，")
        if bad and bad in hay:
            report.violations.append(f"违反约束: {cons}")

    # === 向量语义补漏 ===
    # 规则级没命中时，用 drift 知识库向量检索找语义相似的已知漂移模式。
    # 目前知识库只有 8 条、区分度不足（benign/malicious top-1 cosine 都在 0.8+），
    # 暂时关闭向量语义补漏，等知识库扩充后再启用。
    # if not report.violations:
    #     try:
    #         from .knowledge import query_drift_knowledge
    #         ...
    #     except Exception:
    #         pass
    pass

    # 评分
    n = len(report.violations)
    if n >= 2:
        report.level = "block"
        report.score = min(1.0, 0.4 + n * 0.25)
    elif n == 1:
        report.level = "warn"
        report.score = 0.35
    else:
        report.level = "ok"
        report.score = 0.0
    report.rationale = f"Layer-A 命中违规 {n} 条"
    return report


# -------- Layer B: 语义级（可选，需 1.5B LoRA） --------

def _layer_b(anchor: TaskAnchor, tool_name: str, args: Any,
             llm_client=None, changed_files: list[str] | None = None) -> DriftReport | None:
    """语义级比对：当前 Agent 行为 → requirements 的语义重叠度。

    训练时 LoRA 看到的 instruction schema 是：
      原始任务：{task}。锚点：需求=[...], 约束=[...], 禁止=[...]。
      当前操作：工具={tool}, 参数={args}
    推理时必须严格对齐这个格式，LoRA 才能正确唤起。
    同时注入本地 RAG 知识库，为陌生场景补充领域知识。
    """
    if llm_client is None:
        return None
    try:
        req_str = ", ".join(anchor.requirements) if anchor.requirements else "（无）"
        cons_str = ", ".join(anchor.constraints) if anchor.constraints else "（无）"
        forb_str = ", ".join(anchor.forbidden) if anchor.forbidden else "（无）"
        # RAG 知识注入（可选，knowledge.py 不存在就跳过）
        rag_ctx = ""
        try:
            from .knowledge import build_rag_context
            rag_ctx = build_rag_context(
                anchor.original_instruction or "",
                anchor.to_dict() if hasattr(anchor, "to_dict") else {},
                tool_name, str(args), changed_files or [])
        except Exception:
            rag_ctx = ""
        rag_part = f"\n[参考知识]\n{rag_ctx}" if rag_ctx else ""
        prompt = (
            f"原始任务：{anchor.original_instruction or '（无）'}。"
            f"锚点：需求=[{req_str}]，约束=[{cons_str}]，禁止=[{forb_str}]。"
            f"当前操作：工具={tool_name}，参数={_text_of(args)[:400]}"
            f"{rag_part}"
        )
        out = llm_client.generate(prompt, max_new_tokens=200)
        # 宽松解析 JSON；失败时 fallback 到保守估计
        m = re.search(r"\{.*\}", out, re.DOTALL)
        if not m:
            return DriftReport(score=0.5, level="warn",
                               violations=[], rationale="Layer-B 解析失败，保守估计中度可疑")
        try:
            d = json.loads(m.group())
        except Exception:
            return DriftReport(score=0.5, level="warn",
                               violations=[], rationale="Layer-B JSON 解析失败，保守估计")
        score = float(d.get("score", 0.5))
        # 清理 violations（训时可能输出 list[str] 或 list[dict]）
        violations = d.get("violations") or []
        if isinstance(violations, list):
            violations = [v if isinstance(v, str)
                          else v.get("message", v.get("rule_id", str(v)))
                          for v in violations]
        rationale = str(d.get("rationale", "")) or "保守估计"
        level = "block" if score >= 0.7 else "warn" if score >= 0.4 else "ok"
        return DriftReport(score=score, level=level,
                           violations=violations, rationale=rationale)
    except Exception:
        return DriftReport(score=0.5, level="warn",
                           violations=[], rationale="Layer-B 异常，保守估计中度可疑")


class DriftDetector:
    """组合 Layer-A + 可选 Layer-B 的偏离度检测器。"""

    def __init__(self, anchor: TaskAnchor, llm_client=None) -> None:
        self.anchor = anchor
        self.llm_client = llm_client
        self.history: list[DriftReport] = []

    def check(self, tool_name: str, args: Any) -> DriftReport:
        a = _layer_a(self.anchor, tool_name, args)
        b = _layer_b(self.anchor, tool_name, args, self.llm_client)
        # 组合逻辑：
        #   Layer-A 命中 forbidden → violations 直接算（不可丢）
        #   Layer-B 成功返回（解析正常 JSON）→ 取 max 分数
        #   Layer-B 失败返回（解析失败兜底 0.5 或异常）→ **不覆盖 Layer-A 分数**，
        #     只把 Layer-A 已命中的违规保留，Layer-B 结果只做补充
        if b is None:
            rep = a
        elif "解析失败" in b.rationale or "异常" in b.rationale:
            # Layer-B 没正常工作 → 以 Layer-A 为准
            rep = a
            # 但如果 Layer-A 没命中任何违规，保守给一个轻微警告
            if not a.violations:
                rep = DriftReport(score=0.4, level="warn",
                                  violations=[],
                                  rationale="Layer-A 未命中 + Layer-B 不可用，保守估计")
        else:
            # Layer-B 正常 → 取更严重者，violations 合并去重
            if b.score >= a.score:
                rep = b
            else:
                rep = a
            # 合并 violations
            seen = set()
            merged = []
            for v in (a.violations + b.violations):
                if v not in seen:
                    seen.add(v); merged.append(v)
            rep.violations = merged
        self.history.append(rep)
        return rep

    @property
    def accumulated_score(self) -> float:
        """累计偏离度（历史报告的最大 score）。"""
        if not self.history:
            return 0.0
        return max(r.score for r in self.history)