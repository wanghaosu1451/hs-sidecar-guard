"""Sidecar RAG 知识库：向量检索（gte-small-zh + Faiss）为主，
原有关键词/正则规则做兜底，合成语义检索 + 确定性匹配的双管线。

三个知识库（全部已向量索引）：
1. drift_knowledge.json       —— 目标漂移锚点模式（8 条 → 向量扩展）
2. shell_intent_knowledge.json —— Shell 命令意图语义分类（8 条）
3. cross_file_patterns.json   —— 跨文件隐性依赖模式（7 条）

Faiss 索引存 knowledge/vector_index/*.bin + *.pkl（Windows 中文路径 buffer 方案）。
冷启动自动加载，首次运行自动构建。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _ensure_knowledge_dir() -> Path:
    """确保知识库目录存在，首次运行写入默认知识。"""
    d = _KNOWLEDGE_DIR
    d.mkdir(parents=True, exist_ok=True)
    defaults = {
        "drift_knowledge.json": _DRIFT_KNOWLEDGE,
        "shell_intent_knowledge.json": _SHELL_KNOWLEDGE,
        "cross_file_patterns.json": _CROSS_FILE_KNOWLEDGE,
    }
    for fname, content in defaults.items():
        p = d / fname
        if not p.exists():
            p.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    return d


def load_knowledge(name: str) -> dict | list:
    """加载一个知识库文件（不带 .json 后缀）。"""
    d = _ensure_knowledge_dir()
    p = d / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


# ============ 检索函数 ============

# 向量库懒加载（避免 import 时就拉起 torch/faiss）
_vs = None


def _get_vs():
    global _vs
    if _vs is None:
        from .vector_store import SidecarVectorStore
        _vs = SidecarVectorStore()
        _vs.load_or_build()
    return _vs


def _entry_key(e: dict) -> str:
    """知识条目去重 key（三种知识库 schema 不同，取共有字段）。"""
    return (e.get("pattern") or e.get("label") or e.get("description", ""))


def _vector_then_rule(vs_name: str, rule_fn, rule_args,
                      query: str, top_k: int) -> list[dict]:
    """向量检索为主 + 规则匹配兜底 + 合并去重。"""
    try:
        vs = _get_vs()
        vec_hits = vs.search(vs_name, query, top_k * 2,
                              score_threshold=0.55)
    except Exception:
        vec_hits = []
    vec_entries = [h["entry"] for h in vec_hits]
    vec_keys = {_entry_key(e) for e in vec_entries}

    rule_entries = rule_fn(*rule_args)

    merged = list(vec_entries)
    for e in rule_entries:
        if _entry_key(e) not in vec_keys:
            merged.append(e)
    return merged[:top_k]


# -------- drift --------

def query_drift_knowledge(task_description: str, anchor: dict,
                          tool_name: str, args: str,
                          top_k: int = 3) -> list[dict]:
    """向量检索 + 关键词兜底：语义漂移知识检索。"""
    query = (f"{task_description} "
             f"{json.dumps(anchor, ensure_ascii=False)} "
             f"{tool_name} {args}")
    return _vector_then_rule(
        "drift", _rule_drift,
        (task_description, anchor, tool_name, args), query, top_k)


def _rule_drift(task_description, anchor, tool_name, args):
    """原有关键词漂移检索（兜底）。"""
    kb = load_knowledge("drift_knowledge")
    if not kb:
        return []
    hay = (task_description + " "
           + json.dumps(anchor, ensure_ascii=False)
           + " " + tool_name + " " + args).lower()
    scored = []
    for entry in kb:
        keywords = [kw.lower() for kw in entry.get("match_keywords", [])]
        hit = sum(1 for kw in keywords if kw in hay)
        if hit >= 1:
            scored.append((hit, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:3]]


# -------- shell intent --------

def query_shell_intent(cmd: str, top_k: int = 3) -> list[dict]:
    """向量检索 + 正则兜底：Shell 命令意图语义分类。"""
    return _vector_then_rule(
        "shell", _rule_shell, (cmd,), cmd, top_k)


def _rule_shell(cmd):
    """原有 shell 正则匹配（兜底）。"""
    kb = load_knowledge("shell_intent_knowledge")
    if not kb:
        return []
    hay = cmd.lower()
    scored = []
    for entry in kb:
        pats = entry.get("patterns", [])
        hit = sum(1 for pat in pats
                  if re.search(pat, hay, re.IGNORECASE))
        keywords = [kw.lower() for kw in entry.get("keywords", [])]
        hit += sum(1 for kw in keywords if kw in hay)
        if hit >= 1:
            scored.append((hit, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:3]]


# -------- cross file --------

def query_cross_file_knowledge(changed_files: list[str],
                              top_k: int = 3) -> list[dict]:
    """向量检索 + 文件名触发兜底：跨文件依赖语义匹配。"""
    query = " ".join(changed_files)
    return _vector_then_rule(
        "cross", _rule_cross, (changed_files,), query, top_k)


def _rule_cross(changed_files):
    """原有文件名触发匹配（兜底）。"""
    kb = load_knowledge("cross_file_patterns")
    if not kb:
        return []
    changed = {Path(f).name.lower() for f in changed_files}
    scored = []
    for entry in kb:
        triggers = {t.lower() for t in entry.get("trigger_files", [])}
        hit = len(triggers & changed)
        if hit >= 1:
            scored.append((hit, entry))
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:3]]


# -------- RAG context 拼装 --------

def build_rag_context(task_description: str, anchor: dict,
                      tool_name: str, args: str,
                      changed_files: list[str] | None = None,
                      tool_intent: str | None = None) -> str:
    """拼装 RAG 知识上下文（向量检索 + 规则兜底），
    注入到 drift.py Layer-B / firewall LLM prompt 里。"""
    parts: list[str] = []

    dk = query_drift_knowledge(task_description, anchor, tool_name, args)
    if dk:
        parts.append("[知识:常见漂移模式]")
        for e in dk:
            parts.append(
                f"- {e.get('pattern', '')}: "
                f"{e.get('violation_score', '?')}分, "
                f"{e.get('description', '')}")

    sk = query_shell_intent(args)
    if sk:
        parts.append("[知识:Shell 意图分类]")
        for e in sk:
            parts.append(
                f"- 命令 {e.get('label', '')}: "
                f"{e.get('description', '')}")

    if changed_files:
        ck = query_cross_file_knowledge(changed_files)
        if ck:
            parts.append("[知识:跨文件依赖模式]")
            for e in ck:
                parts.append(
                    f"- {e.get('pattern', '')}: "
                    f"{e.get('description', '')}")

    return "\n".join(parts) if parts else ""


# ============ 默认知识库内容（首次写入） ============

_DRIFT_KNOWLEDGE = [
    {
        "pattern": "跳过 staging 直接 push 生产",
        "match_keywords": ["staging", "prod", "push main", "skip staging", "deploy"],
        "violation_score": 0.85,
        "description": "未走 staging 验证直接上线，违反发布流程，属于高风险漂移"
    },
    {
        "pattern": "debug 过程 restart 生产服务",
        "match_keywords": ["debug", "restart", "systemctl", "prod", "线上"],
        "violation_score": 0.85,
        "description": "debug 要求只读不改，重启生产破坏现场且未达 debug 目的"
    },
    {
        "pattern": "debug 过程改数据库 schema",
        "match_keywords": ["debug", "alter table", "mysql", "schema", "index"],
        "violation_score": 0.9,
        "description": "debug 不应修改数据库结构，可能影响生产数据"
    },
    {
        "pattern": "部署任务执行 git push --force",
        "match_keywords": ["deploy", "git push", "force", "push --force"],
        "violation_score": 0.8,
        "description": "部署不应 force push，会覆盖历史且未走审核"
    },
    {
        "pattern": "重构任务删除被引用的文件",
        "match_keywords": ["refactor", "重构", "rm -rf", "删除", "import"],
        "violation_score": 0.75,
        "description": "重构应先改调用处再删旧文件，直接 rm 会导致 ImportError"
    },
    {
        "pattern": "清理缓存任务删除业务数据目录",
        "match_keywords": ["清理缓存", "clean", "rm -rf data", "删除 data", "删业务"],
        "violation_score": 0.9,
        "description": "清理缓存不应涉及 data/ 业务数据目录，是高危误操作"
    },
    {
        "pattern": "只读任务执行写操作",
        "match_keywords": ["只读", "read only", "只看", "不要改", "不要写"],
        "violation_score": 0.8,
        "description": "只读任务（读日志/查文档/explain）不应有写文件或修改代码操作"
    },
    {
        "pattern": "安全审计任务执行破坏性命令",
        "match_keywords": ["安全审计", "audit", "pentest", "渗透"],
        "violation_score": 0.85,
        "description": "安全审计应在隔离环境，不应在生产项目里执行破坏性命令"
    },
]

_SHELL_KNOWLEDGE = [
    {
        "label": "curl_sh_pipe",
        "patterns": [r"curl\s+.*\|\s*(ba)?sh", r"wget\s+.*\|\s*(ba)?sh"],
        "keywords": ["curl", "wget", "| sh", "| bash", "pipe sh"],
        "violation_score": 0.95,
        "description": "远程脚本直接 pipe 进 shell，是最常见的恶意命令注入方式"
    },
    {
        "label": "rm_data_directory",
        "patterns": [r"rm\s+-rf\s+\.?/data", r"rm\s+-rf\s+data/", r"rm\s+-rf\s+/var/lib/mysql"],
        "keywords": ["rm -rf data", "rm -rf ./data", "delete data", "删数据"],
        "violation_score": 0.9,
        "description": "删除业务数据目录，高危破坏性操作"
    },
    {
        "label": "rm_project_root",
        "patterns": [r"rm\s+-rf\s+\.\s*--no-preserve-root", r"rm\s+-rf\s+--no-preserve-root"],
        "keywords": ["rm -rf .", "no-preserve-root", "删项目"],
        "violation_score": 0.95,
        "description": "删除整个项目根目录，不可恢复"
    },
    {
        "label": "rm_temp_cache_benign",
        "patterns": [r"rm\s+-rf\s+\.?/tmp/\*", r"rm\s+-rf\s+\.?/__pycache__", r"rm\s+-rf\s+\.?/\.pytest_cache"],
        "keywords": ["rm -rf ./tmp", "rm __pycache__", "rm .pytest_cache"],
        "violation_score": 0.05,
        "description": "清理缓存/临时文件，正常维护操作"
    },
    {
        "label": "pip_install_benign",
        "patterns": [r"pip\s+install\s+[a-zA-Z0-9_\-]+", r"npm\s+install"],
        "keywords": ["pip install", "npm install", "装依赖"],
        "violation_score": 0.05,
        "description": "官方渠道装依赖，合规"
    },
    {
        "label": "systemctl_restart_prod",
        "patterns": [r"systemctl\s+restart", r"service\s+\w+\s+restart"],
        "keywords": ["systemctl restart", "restart prod", "重启生产"],
        "violation_score": 0.55,
        "description": "restart 生产服务，需确认是否在发布流程内"
    },
    {
        "label": "git_force_push",
        "patterns": [r"git\s+push\s+.*--force", r"git\s+push\s+-f"],
        "keywords": ["git push --force", "force push", "强推"],
        "violation_score": 0.7,
        "description": "force push 会覆盖历史，需在特定场景下才允许"
    },
    {
        "label": "tail_logs_benign",
        "patterns": [r"tail\s+-f\s+/var/log", r"tail\s+.*\.log"],
        "keywords": ["tail log", "读日志"],
        "violation_score": 0.05,
        "description": "读日志，正常 debug 操作"
    },
]

_CROSS_FILE_KNOWLEDGE = [
    {
        "pattern": "config_port_hardcode",
        "trigger_files": ["config.py", "server.py", "main.py"],
        "description": "改 config.py PORT 常量时，需检查 server.py 有没有硬编码 socket.listen(PORT) 调用"
    },
    {
        "pattern": "util_helper_rename",
        "trigger_files": ["util.py", "helper.py"],
        "description": "改 util.py 里的 helper 函数名/签名时，main.py、cli.py、tests/ 下所有 import 要同步改"
    },
    {
        "pattern": "env_var_rename",
        "trigger_files": [".env", "config.py"],
        "description": "改 .env / config.py 的 ENV 变量名时，docker-compose.yml、systemd service、部署脚本要同步"
    },
    {
        "pattern": "database_schema",
        "trigger_files": ["models.py", "alembic", "migrations"],
        "description": "改 ORM model schema 必须配套生成 alembic migration，不能直接改 model.py"
    },
    {
        "pattern": "api_endpoint",
        "trigger_files": ["router.py", "api.py", "client.py"],
        "description": "改后端 API endpoint（路由）时，前端 SDK / 调用方要同步"
    },
    {
        "pattern": "logger_format",
        "trigger_files": ["logger.py", "logging.py"],
        "description": "改 logger 输出格式时，所有下游 grep/awk 日志解析脚本要同步"
    },
    {
        "pattern": "docker_base_image",
        "trigger_files": ["Dockerfile"],
        "description": "升级 Dockerfile base image 时，CI 缓存要清，所有子镜像要 rebase"
    },
]
