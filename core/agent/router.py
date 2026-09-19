"""AI 路由器：按任务类型自动编排 sub-agent 链（排在主 agent 之前）。

工作方式：
1. classify() 用轻量关键词把任务分为 code / research / review / direct 四类；
2. needs_chain() 仅在命中复杂任务（避免把“写个 hello.py”这类简单任务拆链变慢）时返回 True；
3. run_chain() 依类型串条 agent 链（如 code → planner→coder→checker），
   顺序 spawn 子任务并一次性取回结果，最后汇总统一返回给主控。

不额外调用云端大模型做分类：规则判断零 token、即时、对本地 9B 足够。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

# 注意：这里不在模块顶层 import subagent——agent 在加载 router 时会触发
# agent→router→subagent→agent 的循环导入。改为函数内延迟导入。

KINDS = ("code", "research", "review", "direct")

# 关键词 → 任务类型
_CODE_HINTS = ("写", "实现", "开发", "做一个", "创建", "搭建", "flask", "django",
               "爬虫", "脚本", "重构", "后端", "前端", "modules", "模块", "类", "函数")
_RESEARCH_HINTS = ("调研", "调查", "研究", "查资料", "对比", "分析", "学习",
                   "梳理", "归纳", "总结", "科普", "介绍")
_REVIEW_HINTS = ("审查", "审核", "检查", "找bug", "找 bug", "风险评估",
                 "code review", "review", "评估", "code review")

# 只有命中“复杂项目类”词的 code 任务才值得拆链，避免小任务被拖慢
_COMPLEX_HINTS = ("系统", "项目", "完整", "多文件", "多模块", "多个文件", "服务",
                  "应用", "平台", "批量", "网站", "工具链", "全套", "容器")


def in_sub() -> bool:
    """当前线程是否在子 agent 内（_run_sub）。子 agent 不拆链，单 agent 执行即可，
    避免 主控→子 agent→再拆链 的无限递归。"""
    from . import subagent
    return bool(getattr(subagent._thread_local, "in_sub", False))


def classify(task: str) -> str:
    """返回任务类型。命中类型词即归类；否则直接单 agent 作答。"""
    t = (task or "").lower()
    if any(k in t for k in _REVIEW_HINTS):
        return "review"
    if any(k in t for k in _RESEARCH_HINTS):
        return "research"
    if any(k in t for k in _CODE_HINTS):
        return "code"
    return "direct"


def needs_chain(kind: str, task: str) -> bool:
    """是否值得走 agent 链。research/review 默认走链；code 需命中复杂/多文件特征。"""
    if kind == "direct":
        return False
    if kind == "code":
        t = (task or "").lower()
        # 复杂特征词 或 任务偏长（多步骤） 或 明确多文件/模块/系统级 → 强制走链，
        # 以治“跨文件一致性”短板（planner 先定清单，coder/checker 据清单推进）。
        if any(h in t for h in _COMPLEX_HINTS):
            return True
        if len(t) >= 60:  # 长任务基本需要拆解
            return True
        if any(k in t for k in ("多文件", "模块", "系统", "完整", "多模块", "整个")):
            return True
        return False
    return True


def _chain_for(kind: str, task: str) -> list[tuple[str, str]]:
    """按类型返回 (role, prompt) 链定义。"""
    if kind == "code":
        return [
            ("planner", "对下面的开发任务拆解成可执行的步骤计划，"
                        "并明确列出【需新建/改动的文件清单】（含每个文件职责一句话）。只输出计划与清单，不写文件：\n" + task),
            ("coder", "严格按上面的计划与文件清单实现代码，逐文件用 project_write 落盘并自检验证，"
                      "不要偏离清单擅自新增无关文件：\n" + task),
            ("checker", "对照上面的计划与文件清单审查已完成的成果，核对每个清单文件的实现与一致性，"
                        "指出问题/风险并给出明确结论：\n" + task),
        ]
    if kind == "research":
        return [("research", task)]
    return [("checker", "对下面内容做审查/评估，指出问题并给出明确结论：\n" + task)]


def _poll(tid: str, timeout: int = 300) -> str:
    """阻塞轮询某个子任务直到 done/error/超时，返回其结果文本。"""
    from . import subagent  # 延迟导入，避免循环导入
    start = time.time()
    while time.time() - start < timeout:
        res = subagent.sub_result(tid, keep=False)
        if "仍在运行" not in res and "暂无可取" not in res:
            return res
        time.sleep(2)
    return f"[{tid}] 取回超时"


# ============================ 契约文件（治跨文件一致性） ============================
# planner 会输出自由文本的“文件清单”。我们把清单里像文件路径的行解析成机器可读的
# ［file, aim］结构，写入 <project_root>/.hs/plan.json，再让 coder/checker 先读契约
# 核对清单推进——比“纯文本传递”更不易被多轮上下文稀释，从而改善跨文件一致性。
_FILE_RE = re.compile(
    r"(?<![0-9A-Za-z_.])([/\\]?(?:\w[\w.\-]*[/\\])*\w[\w.\-]*\."
    r"(?:py|js|ts|jsx|tsx|html|css|scss|json|md|yml|yaml|sh|bat|toml|txt"
    r"|c|cpp|h|hpp|go|java|rs|sql|ini|cfg))(?![\w.])",
    re.IGNORECASE)


def _plan_files(out: str) -> list[dict]:
    """从 planner 的自由文本产出里抽取“文件路径 + 一句职责”清单。
    文件路径可出现在行内任意位置（前面可有编号/说明），以 .扩展名 token 识别。"""
    files: list[dict] = []
    for line in (out or "").splitlines():
        m = _FILE_RE.search(line)
        if not m:
            continue
        path = m.group(1).lstrip("/\\").rstrip(",;，；")
        if not path:
            continue
        # 职责 = 路径之后去掉分隔符的部分（截到首个中文/西文逗号分号）
        rest = line[m.end():].lstrip(" \t:：——→->>")
        aim = re.split(r"[；;，,]", rest, maxsplit=1)[0].strip()
        files.append({"file": path, "aim": aim})
    # 去重保序
    seen: set[str] = set()
    out_files = []
    for f in files:
        if f["file"] in seen:
            continue
        seen.add(f["file"])
        out_files.append(f)
    return out_files


def _contract_root() -> str | None:
    """取项目写入根目录；未设置则返回 None（此时不落契约，仅走自由文本）。"""
    try:
        from . import tools
        root = tools.get_project_root()
        if not root or root == ".":
            return None
        return str(Path(root).resolve())
    except Exception:  # noqa: BLE001
        return None


def _write_contract(planner_out: str) -> str | None:
    """把 planner 文件清单写成 .hs/plan.json，返回文件路径；解析失败/无根目录返回 None。"""
    files = _plan_files(planner_out)
    if not files:
        return None
    root = _contract_root()
    if not root:
        return None
    try:
        d = Path(root) / ".hs"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "plan.json"
        p.write_text(json.dumps({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": files,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)
    except Exception:  # noqa: BLE001
        return None


def _contract_note(contract_path: str, kind: str) -> str:
    """为 coder/checker 生成“先读契约”的注入提示。"""
    if kind == "checker":
        return (f"（已生成机器可读契约文件 {contract_path}：文件清单与各自职责。"
                "审查前请先 read_file 读取该契约核对每个清单文件的实现与一致性，"
                "若与实际清单冲突，以契约为准并在结论中指出差异。）")
    return (f"（已生成机器可读契约文件 {contract_path}：文件清单与各自职责。"
            "开始实现前请先 read_file 读取该契约核对清单，逐清单文件落盘，"
            "不要偏离契约擅自新增无关文件；若需调整先说明。）")


def run_chain(task: str, kind: str, on_step: Callable | None = None) -> str:
    """按类型顺序执行 agent 链，返回汇总文本。任一环节失败则带错误继续/中断。

    前序环节的产出（如 planner 的文件清单/计划）会作为上下文追加到后续环节 prompt，
    保证 planner → coder → checker 信息贯穿，改善跨文件一致性。
    """
    from . import subagent  # 延迟导入，避免循环导入
    chain = _chain_for(kind, task)
    parts = []
    carry = ""  # 累积前序产出，注入到下一个环节
    contract_note = ""  # planner 写出的机器可读契约指引，注入后续环节
    for idx, (role, prompt) in enumerate(chain):
        p = (carry + "\n\n" + prompt) if carry else prompt
        if contract_note and role in ("coder", "checker"):
            p = p + "\n\n" + contract_note
        if on_step:
            on_step("router:" + role, "下发子任务…")
        r = subagent.spawn_task(p, role=role)
        if isinstance(r, str) and not r.startswith("已提交"):
            return f"路由失败（{role}）：{r}"
        tid = r.split("sub-")[1].split("，")[0] if "sub-" in r else ""
        tid = "sub-" + tid
        out = _poll(tid)
        parts.append(f"【{role}】\n{out}")
        carry = f"【{role}产出】\n{out}"  # 供下一环节参考
        # planner 环节：把文件清单解析成契约文件，供 coder/checker 先读核对
        if role == "planner":
            cp = _write_contract(out)
            if cp:
                contract_note = _contract_note(cp, next(
                    (r2 for r2, _ in chain[idx + 1:]), "coder"))
                carry = carry + f"\n（契约已落盘：{cp}，后续环节实现前先 read_file 读取核对）"
        if on_step:
            on_step("router:" + role, out[:120])
    return "\n\n".join(parts)