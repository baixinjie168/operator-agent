"""Chat route: natural language intent parsing with rule engine + LLM fallback."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["chat"])


# ── Request / Response schemas ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""
    intent: dict | None = None


class ParseIntentRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str | None = None
    current_operator: str | None = None
    server_id: int | None = None


class SuggestedAction(BaseModel):
    label: str
    action: str
    params: dict = {}


class ParseIntentResponse(BaseModel):
    intent: dict
    readiness: dict | None = None
    response_type: str  # "direct" | "confirm" | "guide" | "error" | "unknown"
    response_message: str
    suggested_actions: list[SuggestedAction] = []


# ── In-memory session store ────────────────────────────────────────────────

_sessions: dict[str, dict[str, Any]] = {}
_MAX_HISTORY = 10


def _get_session(session_id: str | None) -> dict[str, Any]:
    if not session_id:
        session_id = uuid.uuid4().hex[:12]
    if session_id not in _sessions:
        _sessions[session_id] = {
            "session_id": session_id,
            "messages": [],
            "current_operator": None,
            "last_intent": None,
            "pending_action": None,
            "created_at": time.time(),
        }
    return _sessions[session_id]


def _add_message(session: dict, role: str, content: str, intent: dict | None = None) -> None:
    session["messages"].append({"role": role, "content": content, "intent": intent})
    if len(session["messages"]) > _MAX_HISTORY * 2:
        session["messages"] = session["messages"][-(_MAX_HISTORY * 2):]


# ── Rule engine ─────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, str]] = [
    # Confirm / cancel (multi-turn)
    (r"^(?:好的|确认|继续|执行吧|没问题|可以|ok|yes|对|是|行)$", "__confirm__"),
    (r"^(?:算了|取消|不要了|不了|no|不)$", "__cancel__"),

    # Compound actions (must come before single actions to match first)
    # operator name + extract + generate: e.g. "aclnnAbs重新提取约束生成测试用例"
    (
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:重新)?(?:提取|跑)\s*(?:算子)?约束\s*(?:并|然后|再)?\s*(?:生成|创建)\s*(?:测试)?用例",
        "extract_then_generate",
    ),
    # extract + generate: e.g. "重新提取约束生成测试用例", "提取约束并生成用例"
    (
        r"(?:重新)?(?:提取|跑)\s*(?:算子)?约束\s*(?:并|然后|再)?\s*(?:生成|创建)\s*(?:测试)?用例",
        "extract_then_generate",
    ),
    # operator name + generate + execute
    (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:生成|创建)\s*(?:测试)?用例\s*(?:并|然后|再)\s*(?:执行|跑|测试)", "generate_and_execute"),
    # generate + execute
    (r"(?:生成|创建)\s*(?:测试)?用例\s*(?:并|然后|再)\s*(?:执行|跑|测试)", "generate_and_execute"),

    # Data queries (must come before list_operators to match first)
    (
        r"(?:有哪些|列出|所有|当前)\s*(?:算子|operator)\s*(?:有|生成|创建|存在)\s*(?:测试)?用例",
        "query_operators_with_cases",
    ),
    (r"(?:哪些|多少)\s*算子\s*(?:有|生成|创建)\s*(?:了)?(?:测试)?用例", "query_operators_with_cases"),
    (
        r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:算子)?(?:的)?(?:最近|最新)\s*(?:的)?(?:测试)?(?:用例)?(?:执行)?结果",
        "query_exec_results",
    ),
    (r"(?:查看|查询)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:算子)?(?:的)?(?:执行)?结果", "query_exec_results"),

    # List operators (must come before view_cases to match first)
    (r"(?:有哪些|列出|所有|当前)\s*(?:算子|operator)", "list_operators"),
    (r"(?:哪些|多少)\s*算子\s*(?:有|生成|创建)", "list_operators"),

    # Read-only
    (r"(?:查看|看看|查一下|打开|显示)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?约束", "view_constraints"),
    (r"(?:查看|看看|有哪些|显示)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?用例", "view_cases"),
    (r"(?:查看|看看|显示)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?(?:执行)?结果", "view_results"),
    (r"(?:查看|看看|显示)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?文档", "view_document"),
    (r"(?:历史|记录|任务列表|任务)", "view_task_history"),
    (r"(?:帮助|怎么用|能做什么|help)", "help"),

    # Write operations
    (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:重新)?(?:提取|跑一下?)\s*(?:算子)?约束", "extract_constraints"),
    (r"(?:提取|重新提取|跑一下)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?约束", "extract_constraints"),
    (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:生成|创建)\s*(?:测试)?用例", "generate_cases"),
    (r"(?:生成|创建)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?(?:测试)?用例", "generate_cases"),
    (r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:执行|运行|跑)\s*(?:测试)?(?:用例|测试)", "execute_tests"),
    (r"(?:执行|运行|跑)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:的)?(?:测试)?(?:用例|测试)", "execute_tests"),
    (r"(?:上传|导入)\s*(?:算子)?文档", "upload_document"),
]

_TASK_TYPE_LABELS = {
    "constraint_extract": "约束提取",
    "case_generate": "用例生成",
    "test_execute": "测试执行",
}

# Actions that don't need confirmation
_READ_ONLY_ACTIONS = {
    "view_constraints", "view_cases", "view_results", "view_document",
    "list_operators", "view_task_history", "help",
    "query_operators_with_cases", "query_exec_results",
}

# Common Chinese words that should NOT be treated as operator names
_NON_OPERATOR_WORDS = {
    "算子", "测试", "用例", "约束", "文档", "所有", "哪些", "什么", "怎么",
    "当前", "已经", "可以", "能够", "是否", "有没有", "多少", "几个",
}


def _is_valid_operator_name(name: str | None) -> bool:
    """Check if the extracted name looks like a real operator name.
    
    Operator names typically:
    - Start with "aclnn", "acl", or similar prefixes
    - Are English identifiers (alphanumeric + underscore)
    - Are NOT common Chinese words
    """
    if not name:
        return False
    # Reject common Chinese words
    if name in _NON_OPERATOR_WORDS:
        return False
    # Reject if contains Chinese characters (operator names are English)
    return not any('\u4e00' <= c <= '\u9fff' for c in name)


def _rule_based_parse(text: str, current_operator: str | None) -> dict | None:
    """Rule engine: fast regex matching. Returns None if no match."""
    text = text.strip()
    for pattern, action in _INTENT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            op_name = None
            if match.groups():
                op_name = match.group(1)
            
            # Validate extracted operator name
            if op_name and not _is_valid_operator_name(op_name):
                # Invalid operator name, fall back to current_operator or None
                op_name = current_operator
            
            if not op_name:
                op_name = current_operator
            
            return {
                "action": action,
                "operator_name": op_name,
                "confidence": 0.95,
                "parameters": {},
            }
    return None


# ── LLM intent parsing (fallback) ──────────────────────────────────────────

_INTENT_PARSE_PROMPT = """你是一个昇腾算子测试平台的意图解析器。
根据用户输入和当前上下文，输出结构化意图 JSON。

## 当前上下文
- 当前选中算子: {current_operator}（可能为空）
- 已有算子列表: {operator_list}
- 对话历史: {conversation_history}

## 支持的意图

### 只读操作（直接执行）
- view_constraints: 查看约束
- view_cases: 查看用例
- view_results: 查看执行结果
- view_document: 查看文档
- list_operators: 列出算子
- view_task_history: 任务历史
- help: 帮助

### 复合操作（多步骤串行执行）
- extract_then_generate: 提取约束并生成用例（如"重新提取约束生成测试用例"、"提取约束并生成用例"）
- generate_and_execute: 生成用例并执行（如"生成用例并执行"、"生成然后跑一下"）

### 写操作（需要确认）
- extract_constraints: 提取约束
- generate_cases: 生成用例
- execute_tests: 执行测试
- generate_and_execute: 生成用例并执行
- upload_document: 上传文档

### 数据查询（直接返回查询结果）
- query_operators_with_cases: 查询哪些算子有测试用例（如"哪些算子有用例"、"当前有哪些算子生成了用例"）
- query_exec_results: 查询某个算子的执行结果（如"aclnnAbs的执行结果"、"xxx算子最近的测试情况"）

## 规则
1. 如果用户提到了算子名称，提取出来（支持模糊匹配，如"Ada"匹配"aclnnAdaLayerNorm"）
2. 如果用户没有提到算子名称，使用当前选中算子
3. 如果都没有，operator_name 为 null
4. 多轮对话中，用户说"好的"、"继续"、"执行吧"等，结合上文确定意图
5. 复合指令如"生成并执行"识别为 generate_and_execute
6. confidence 低于 0.6 时标记为 unknown

## 输出格式（严格 JSON，不要包含其他内容）
{{"action": "...", "operator_name": "...", "confidence": 0.95, "parameters": {{}}}}
"""


async def _llm_parse_intent(
    text: str,
    current_operator: str | None,
    operator_list: list[str],
    conversation_history: list[dict],
) -> dict:
    """Use LLM to parse intent when rule engine doesn't match."""
    try:
        from langchain_openai import ChatOpenAI

        from agent.core.config import settings

        llm = ChatOpenAI(
            api_key=settings.active_api_key,
            base_url=settings.active_base_url,
            model=settings.active_model,
            temperature=0.1,
        )

        history_text = ""
        if conversation_history:
            for msg in conversation_history[-6:]:
                role = "用户" if msg.get("role") == "user" else "系统"
                history_text += f"{role}: {msg.get('content', '')}\n"

        prompt = _INTENT_PARSE_PROMPT.format(
            current_operator=current_operator or "无",
            operator_list=", ".join(operator_list[:20]) if operator_list else "无",
            conversation_history=history_text or "无",
        )

        logger.info("=" * 60)
        logger.info("[LLM Intent Parse] Request:")
        logger.info("  User input: %s", text)
        logger.info("  Current operator: %s", current_operator or "无")
        logger.info("  System prompt:\n%s", prompt)
        logger.info("=" * 60)

        response = await llm.ainvoke([
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ])

        raw = response.content.strip() if hasattr(response, "content") else str(response)
        
        logger.info("[LLM Intent Parse] Response:")
        logger.info("  Raw response: %s", raw)
        
        # Extract JSON from response
        import contextlib
        import json
        result = None
        # Try direct parse first
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            result = json.loads(raw)
        # Fallback: extract JSON object from response text
        if not result:
            # Match balanced braces (supports nested {})
            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw)
            if json_match:
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    result = json.loads(json_match.group())
        if result and isinstance(result, dict) and result.get("action"):
            if not result.get("operator_name"):
                result["operator_name"] = current_operator
            logger.info("  Parsed intent: %s", json.dumps(result, ensure_ascii=False))
            logger.info("=" * 60)
            return result
        else:
            logger.warning("  Failed to extract JSON from response")
            logger.info("=" * 60)

    except Exception as e:
        logger.warning("LLM intent parsing failed: %s", e)

    return {
        "action": "unknown",
        "operator_name": current_operator,
        "confidence": 0.0,
        "parameters": {},
    }


# ── Readiness check ─────────────────────────────────────────────────────────

def _check_readiness(operator_name: str | None) -> dict:
    """Check operator data readiness from DB."""
    if not operator_name:
        return {"exists": False}

    from agent.db import (
        find_parent_task,
        query_run,
        query_test_cases,
    )
    from mcp_server.db import get_db

    db = get_db()

    # Check operator exists
    op_row = db.conn.execute(
        "SELECT id FROM operators WHERE name = ?", (operator_name,)
    ).fetchone()
    if not op_row:
        return {"exists": False}

    # Check document
    doc_row = db.conn.execute(
        "SELECT id, version FROM document_versions WHERE operator_id = ? ORDER BY version DESC LIMIT 1",
        (op_row[0],),
    ).fetchone()
    has_document = doc_row is not None

    # Check constraints
    has_constraints = False
    constraint_doc_id = None
    constraint_version = None
    if doc_row:
        jc_row = db.conn.execute(
            "SELECT json_constraints FROM document_versions WHERE id = ?", (doc_row[0],)
        ).fetchone()
        if jc_row and jc_row[0] and jc_row[0] != "{}":
            has_constraints = True
            constraint_doc_id = doc_row[0]
            constraint_version = doc_row[1]

    # Check parameters count
    parameters_count = 0
    if doc_row:
        p_row = db.conn.execute(
            "SELECT COUNT(*) FROM parameters WHERE doc_id = ?", (doc_row[0],)
        ).fetchone()
        parameters_count = p_row[0] if p_row else 0

    # Check cases
    has_cases = False
    cases_count = 0
    latest_cases_task_id = None
    latest_case_task = find_parent_task(operator_name, "case_generate")
    if latest_case_task:
        cases = query_test_cases(task_id=latest_case_task)
        if cases:
            has_cases = True
            cases_count = len(cases)
            latest_cases_task_id = latest_case_task

    # Check exec results
    has_exec_results = False
    latest_exec_task_id = None
    latest_exec_passed = 0
    latest_exec_total = 0
    latest_exec_task = find_parent_task(operator_name, "test_execute")
    if latest_exec_task:
        exec_run = query_run(latest_exec_task)
        if exec_run and exec_run.get("status") == "completed":
            has_exec_results = True
            latest_exec_task_id = latest_exec_task
            result_json = exec_run.get("result_json")
            if result_json:
                import json
                try:
                    result = json.loads(result_json)
                    er = result.get("exec_result", {})
                    latest_exec_passed = er.get("passed", 0)
                    latest_exec_total = er.get("total", 0)
                except (json.JSONDecodeError, TypeError):
                    pass

    return {
        "exists": True,
        "has_document": has_document,
        "has_constraints": has_constraints,
        "constraint_doc_id": constraint_doc_id,
        "constraint_version": constraint_version,
        "parameters_count": parameters_count,
        "has_cases": has_cases,
        "cases_count": cases_count,
        "latest_cases_task_id": latest_cases_task_id,
        "has_exec_results": has_exec_results,
        "latest_exec_task_id": latest_exec_task_id,
        "latest_exec_passed": latest_exec_passed,
        "latest_exec_total": latest_exec_total,
    }


# ── Data query functions ────────────────────────────────────────────────────

def _query_operators_with_cases() -> list[dict]:
    """Query all operators that have test cases in the database."""
    from mcp_server.db import get_db
    db = get_db()
    
    rows = db.conn.execute(
        "SELECT DISTINCT tc.operator_name, COUNT(*) as case_count, "
        "MAX(tc.created_at) as last_created "
        "FROM test_cases tc "
        "GROUP BY tc.operator_name "
        "ORDER BY last_created DESC"
    ).fetchall()
    
    return [
        {
            "operator_name": row[0],
            "case_count": row[1],
            "last_created": row[2],
        }
        for row in rows
    ]


def _query_exec_results_for_operator(operator_name: str) -> list[dict]:
    """Query recent execution results for a specific operator."""
    from agent.db import query_exec_results
    
    # Find all test_execute tasks for this operator
    from mcp_server.db import get_db
    db = get_db()
    
    task_rows = db.conn.execute(
        "SELECT run_id, status, created_at, completed_at "
        "FROM pipeline_runs "
        "WHERE operator_name = ? AND task_type = 'test_execute' "
        "ORDER BY created_at DESC LIMIT 5",
        (operator_name,)
    ).fetchall()
    
    results = []
    for task_row in task_rows:
        run_id, status, created_at, completed_at = task_row
        
        # Get execution results for this task
        exec_results = query_exec_results(task_id=run_id)
        
        passed = sum(1 for r in exec_results if r.get("passed"))
        total = len(exec_results)
        
        results.append({
            "task_id": run_id,
            "status": status,
            "created_at": created_at,
            "completed_at": completed_at,
            "total_cases": total,
            "passed_cases": passed,
            "failed_cases": total - passed,
            "pass_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "N/A",
        })
    
    return results


# ── Response strategy builder ───────────────────────────────────────────────

def _build_response(
    intent: dict,
    readiness: dict,
    session: dict,
    server_id: int | None = None,
) -> dict:
    """Build response with response_type, message, and suggested actions."""
    action = intent.get("action", "unknown")
    op_name = intent.get("operator_name") or session.get("current_operator")

    # Update session
    if op_name:
        session["current_operator"] = op_name
    session["last_intent"] = intent

    # Unknown intent
    if action == "unknown":
        suggestions = []
        if op_name:
            suggestions.extend([
                SuggestedAction(
                    label=f"查看 {op_name} 约束", action="view_constraints",
                    params={"operator_name": op_name},
                ),
                SuggestedAction(
                    label=f"生成 {op_name} 用例", action="generate_cases",
                    params={"operator_name": op_name},
                ),
                SuggestedAction(
                    label=f"执行 {op_name} 测试", action="execute_tests",
                    params={"operator_name": op_name},
                ),
            ])
        suggestions.append(SuggestedAction(label="查看任务历史", action="view_task_history"))
        suggestions.append(SuggestedAction(label="帮助", action="help"))
        return {
            "response_type": "unknown",
            "response_message": "抱歉，我没理解您的意思。您可以尝试以下操作：",
            "suggested_actions": suggestions,
        }

    # Help
    if action == "help":
        return {
            "response_type": "direct",
            "response_message": (
                "我可以帮您完成以下操作：\n"
                "• 查看算子约束 / 用例 / 执行结果\n"
                "• 提取算子约束\n"
                "• 生成测试用例\n"
                "• 执行测试\n"
                "• 查看任务历史\n\n"
                '您可以直接输入自然语言指令，如"查看 Ada 的约束"，"生成 xxx 的用例"等。'
            ),
            "suggested_actions": [],
        }

    # List operators
    if action == "list_operators":
        return {
            "response_type": "direct",
            "response_message": "正在获取算子列表...",
            "suggested_actions": [],
        }

    # View task history
    if action == "view_task_history":
        return {
            "response_type": "direct",
            "response_message": "正在获取任务列表...",
            "suggested_actions": [],
        }

    # Query operators with test cases (doesn't need operator)
    if action == "query_operators_with_cases":
        operators_with_cases = _query_operators_with_cases()
        if not operators_with_cases:
            return {
                "response_type": "direct",
                "response_message": "当前没有任何算子生成了测试用例。",
                "suggested_actions": [
                    SuggestedAction(label="列出所有算子", action="list_operators"),
                ],
            }
        
        lines = [f"共有 {len(operators_with_cases)} 个算子生成了测试用例：\n"]
        for op in operators_with_cases:
            lines.append(
                f"• {op['operator_name']}: {op['case_count']} 条用例 "
                f"(最后生成: {op['last_created'][:16] if op['last_created'] else 'N/A'})"
            )
        
        return {
            "response_type": "direct",
            "response_message": "\n".join(lines),
            "suggested_actions": [],
        }

    # Query execution results for operator (bypasses operator existence check)
    if action == "query_exec_results":
        if not op_name:
            return {
                "response_type": "error",
                "response_message": "请指定要查询执行结果的算子名称。",
                "suggested_actions": [
                    SuggestedAction(label="列出所有算子", action="list_operators"),
                ],
            }
        
        exec_results = _query_exec_results_for_operator(op_name)
        if not exec_results:
            return {
                "response_type": "direct",
                "response_message": f"算子 {op_name} 暂无执行结果。",
                "suggested_actions": [
                    SuggestedAction(
                        label=f"执行 {op_name} 测试",
                        action="execute_tests",
                        params={"operator_name": op_name},
                    ),
                ],
            }
        
        lines = [f"算子 {op_name} 最近 {len(exec_results)} 次执行结果：\n"]
        for i, result in enumerate(exec_results, 1):
            status_icon = "✅" if result["status"] == "completed" else "❌"
            lines.append(
                f"{i}. {status_icon} {result['created_at'][:16] if result['created_at'] else 'N/A'}\n"
                f"   通过率: {result['pass_rate']} ({result['passed_cases']}/{result['total_cases']})"
            )
        
        return {
            "response_type": "direct",
            "response_message": "\n".join(lines),
            "suggested_actions": [
                SuggestedAction(
                    label=f"再次执行 {op_name} 测试",
                    action="execute_tests",
                    params={"operator_name": op_name},
                ),
            ],
        }

    # No operator specified for actions that need one
    if not op_name and action not in ("upload_document",):
        return {
            "response_type": "error",
            "response_message": "请先选择一个算子，或在指令中指定算子名称。",
            "suggested_actions": [
                SuggestedAction(label="列出所有算子", action="list_operators"),
            ],
        }

    # Operator doesn't exist
    if not readiness.get("exists"):
        return {
            "response_type": "error",
            "response_message": f"未找到算子 {op_name}，请确认名称或上传文档。",
            "suggested_actions": [
                SuggestedAction(label="上传算子文档", action="upload_document"),
                SuggestedAction(label="列出所有算子", action="list_operators"),
            ],
        }

    # Read-only actions: direct execution
    if action in _READ_ONLY_ACTIONS:
        if action == "view_constraints" and not readiness.get("has_constraints"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 暂无约束数据，是否提取？",
                "suggested_actions": [
                    SuggestedAction(label="提取约束", action="extract_constraints", params={"operator_name": op_name}),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        if action == "view_cases" and not readiness.get("has_cases"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 暂无测试用例，是否生成？",
                "suggested_actions": [
                    SuggestedAction(label="生成用例", action="generate_cases", params={"operator_name": op_name}),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        if action == "view_results" and not readiness.get("has_exec_results"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 暂无执行结果，是否执行测试？",
                "suggested_actions": [
                    SuggestedAction(label="执行测试", action="execute_tests", params={"operator_name": op_name}),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        return {
            "response_type": "direct",
            "response_message": f"正在查看 {op_name} 的数据...",
            "suggested_actions": [],
        }

    # Write actions: need confirmation
    if action == "extract_then_generate":
        return {
            "response_type": "confirm",
            "response_message": (
                f"将为 {op_name} 执行以下操作：\n"
                f"  1. 重新提取算子约束\n"
                f"  2. 生成测试用例\n"
                f"确认？"
            ),
            "suggested_actions": [
                SuggestedAction(
                    label="确认执行",
                    action="extract_then_generate",
                    params={"operator_name": op_name},
                ),
                SuggestedAction(label="取消", action="cancel"),
            ],
        }

    if action == "extract_constraints":
        return {
            "response_type": "confirm",
            "response_message": f"将为 {op_name} 提取算子约束，确认？",
            "suggested_actions": [
                SuggestedAction(label="确认提取", action="extract_constraints", params={"operator_name": op_name}),
                SuggestedAction(label="取消", action="cancel"),
            ],
        }

    if action == "generate_cases":
        if not readiness.get("has_constraints"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 尚未提取约束，需要先提取约束。是否自动执行？",
                "suggested_actions": [
                    SuggestedAction(
                    label="自动提取约束并继续",
                    action="extract_then_generate",
                    params={"operator_name": op_name},
                ),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        pc = readiness.get("parameters_count", 0)
        cv = readiness.get("constraint_version", "?")
        return {
            "response_type": "confirm",
            "response_message": f"将为 {op_name} 生成测试用例，基于约束 v{cv} ({pc}个参数)，确认？",
            "suggested_actions": [
                SuggestedAction(label="确认生成", action="generate_cases", params={"operator_name": op_name}),
                SuggestedAction(label="取消", action="cancel"),
            ],
        }

    if action == "execute_tests":
        if not readiness.get("has_cases"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 暂无测试用例，是否先生成？",
                "suggested_actions": [
                    SuggestedAction(label="生成用例", action="generate_cases", params={"operator_name": op_name}),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        total_cc = readiness.get("cases_count", 0)

        # Filter by server's supported_product if server is specified
        server_product = None
        filtered_cc = total_cc
        if server_id:
            try:
                from agent.db import get_server as db_get_server
                srv = db_get_server(server_id)
                if srv and srv.get("supported_product"):
                    server_product = srv["supported_product"]
                    # Count cases matching this product
                    from agent.db import query_test_cases as db_query_test_cases
                    from agent.db import find_parent_task
                    parent_id = find_parent_task(op_name, "case_generate")
                    if parent_id:
                        cases = db_query_test_cases(task_id=parent_id)
                        filtered_cc = sum(
                            1 for c in cases
                            if (c.get("supported_product", "")
                                or (isinstance(c.get("case_data"), dict) and c["case_data"].get("supported_product", "")))
                            == server_product
                        )
            except Exception as e:
                logger.warning("Failed to filter cases by server product: %s", e)

        if server_product:
            msg = f"当前服务器支持「{server_product}」，算子 {op_name} 共有 {filtered_cc} 条匹配用例（总计 {total_cc} 条），确认执行？"
        else:
            msg = f"算子 {op_name} 共有 {total_cc} 条测试用例，将根据当前执行服务器的支持产品筛选后执行，确认？"

        return {
            "response_type": "confirm",
            "response_message": msg,
            "suggested_actions": [
                SuggestedAction(label="确认执行", action="execute_tests", params={"operator_name": op_name}),
                SuggestedAction(label="取消", action="cancel"),
            ],
        }

    if action == "generate_and_execute":
        if not readiness.get("has_constraints"):
            return {
                "response_type": "guide",
                "response_message": f"算子 {op_name} 尚未提取约束，需要完整流程：提取约束→生成用例→执行，是否继续？",
                "suggested_actions": [
                    SuggestedAction(label="全流程执行", action="full_pipeline", params={"operator_name": op_name}),
                    SuggestedAction(label="取消", action="cancel"),
                ],
            }
        return {
            "response_type": "confirm",
            "response_message": f"将为 {op_name} 生成用例并执行测试，确认？",
            "suggested_actions": [
                SuggestedAction(label="确认", action="generate_and_execute", params={"operator_name": op_name}),
                SuggestedAction(label="取消", action="cancel"),
            ],
        }

    if action == "upload_document":
        return {
            "response_type": "direct",
            "response_message": "请在左侧面板点击上传按钮，选择算子文档文件。",
            "suggested_actions": [],
        }

    # Fallback
    return {
        "response_type": "unknown",
        "response_message": "抱歉，暂不支持该操作。",
        "suggested_actions": [
            SuggestedAction(label="帮助", action="help"),
        ],
    }


# ── Route handler ───────────────────────────────────────────────────────────

@router.post("/chat/parse-intent", response_model=ParseIntentResponse)
async def parse_intent(body: ParseIntentRequest) -> ParseIntentResponse:
    """Parse user intent from natural language input."""
    text = body.text.strip()
    session = _get_session(body.session_id)

    # Update current operator from request
    if body.current_operator:
        session["current_operator"] = body.current_operator

    current_op = session.get("current_operator") or body.current_operator

    # Add user message to session
    _add_message(session, "user", text)

    # Handle confirm/cancel from multi-turn
    confirm_match = re.match(r"^(?:好的|确认|继续|执行吧|没问题|可以|ok|yes|对|是|行)$", text, re.IGNORECASE)
    cancel_match = re.match(r"^(?:算了|取消|不要了|不了|no|不)$", text, re.IGNORECASE)

    if confirm_match and session.get("pending_action"):
        intent = session["pending_action"]
        session["pending_action"] = None
        op_name = intent.get("operator_name") or current_op
        readiness = _check_readiness(op_name) if op_name else {"exists": False}
        _add_message(session, "assistant", f"正在执行 {intent['action']}...", intent)
        return ParseIntentResponse(
            intent=intent,
            readiness=readiness,
            response_type="direct",
            response_message=f"正在执行 {intent['action']}...",
            suggested_actions=[],
        )
    elif cancel_match:
        session["pending_action"] = None
        intent = {"action": "cancel", "operator_name": current_op, "confidence": 1.0, "parameters": {}}
        return ParseIntentResponse(
            intent=intent,
            readiness=None,
            response_type="direct",
            response_message="已取消操作。",
            suggested_actions=[],
        )
    else:
        # Step 1: Rule engine
        intent = _rule_based_parse(text, current_op)

        # Step 2: LLM fallback
        if intent is None:
            # Fetch operator list for LLM context
            operator_list = []
            try:
                from mcp_server.db import get_db
                db = get_db()
                rows = db.conn.execute("SELECT name FROM operators ORDER BY name LIMIT 20").fetchall()
                operator_list = [r[0] for r in rows]
            except Exception:
                pass

            intent = await _llm_parse_intent(text, current_op, operator_list, session["messages"])

    # Check readiness
    op_name = intent.get("operator_name") or current_op
    readiness = _check_readiness(op_name) if op_name else {"exists": False}

    # Build response
    response = _build_response(intent, readiness, session, server_id=body.server_id)

    # Store pending action for multi-turn confirm
    if response["response_type"] == "confirm":
        session["pending_action"] = intent

    # Add bot message to session
    _add_message(session, "assistant", response["response_message"], intent)

    return ParseIntentResponse(
        intent=intent,
        readiness=readiness,
        response_type=response["response_type"],
        response_message=response["response_message"],
        suggested_actions=response["suggested_actions"],
    )


@router.get("/operators/{operator_name}/readiness")
async def get_operator_readiness(operator_name: str):
    """Check operator data readiness status."""
    readiness = _check_readiness(operator_name)
    return {"success": True, **readiness}
