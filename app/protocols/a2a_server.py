"""
A2A (Agent-to-Agent) Protocol Server Adapter

Implements the A2A protocol (Google, Linux Foundation) for inter-agent
communication and collaboration.

Key Components:
  - Agent Card: /.well-known/agent.json — describes capabilities, skills, endpoint
  - JSON-RPC 2.0: All requests/responses follow JSON-RPC 2.0 spec
  - Task Lifecycle: submitted → working → input-required → completed / failed / canceled
  - Messages: user/agent turns with Parts (TextPart, DataPart)
  - Artifacts: Agent outputs (documents, data, files)

Spec: https://a2a-protocol.org/v0.2.6/specification/
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any

from app.config import settings

logger = logging.getLogger("agent_system")


TASK_STATES = ["submitted", "working", "input-required", "completed", "failed", "canceled"]


_TASKS: Dict[str, Dict[str, Any]] = {}
_TASK_MESSAGES: Dict[str, List[Dict[str, Any]]] = {}
_TASK_ARTIFACTS: Dict[str, List[Dict[str, Any]]] = {}


def get_agent_card() -> Dict[str, Any]:
    base_url = f"http://127.0.0.1:{settings.PORT}"
    return {
        "schemaVersion": "0.2.6",
        "name": "Agent IAM Security Server",
        "description": "Zero-trust identity and access management for AI agents. Provides prompt injection defense, alignment checks, credential brokering, 4-level revocation, and governance audit.",
        "url": f"{base_url}/api/protocols/a2a",
        "preferredTransport": "jsonrpc",
        "provider": {
            "organization": "Agent IAM",
            "url": "https://github.com/agent-iam",
        },
        "version": "2.4.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "securitySchemes": {
            "bearer": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            },
            "api_key": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            },
        },
        "security": [{"bearer": []}, {"api_key": []}],
        "skills": [
            {
                "id": "prompt-defense",
                "name": "Prompt Injection Defense",
                "description": "3-layer fusion engine (rules + semantic + behavioral) for detecting prompt injection attacks",
                "tags": ["security", "prompt-injection", "defense"],
                "examples": [
                    "Check if this prompt contains an injection attack",
                    "Analyze this message for security threats",
                ],
            },
            {
                "id": "alignment-check",
                "name": "Output Alignment Check",
                "description": "Verify agent output aligns with original user intent, detecting goal hijacking, indirect injection, and DLP leaks",
                "tags": ["security", "alignment", "output-defense"],
                "examples": [
                    "Check if this agent output is aligned with the user's intent",
                    "Detect if the agent was hijacked",
                ],
            },
            {
                "id": "iam-check",
                "name": "Identity & Access Management",
                "description": "RBAC + ABAC + time-based + dynamic policy engine with trust scoring",
                "tags": ["iam", "authorization", "trust"],
                "examples": [
                    "Check if agent has permission for this action",
                    "Get the trust score for this agent",
                ],
            },
            {
                "id": "delegation",
                "name": "Secure Delegation",
                "description": "JWT-based delegation chain with capability scoping, depth limits, and one-time use tokens",
                "tags": ["delegation", "chain", "token"],
                "examples": [
                    "Create a delegation token for data_agent",
                    "Delegate read access to another agent",
                ],
            },
            {
                "id": "revocation",
                "name": "4-Level Revocation",
                "description": "Token-level, Agent-level, Task-level, and Chain-cascade revocation",
                "tags": ["revocation", "security", "cascade"],
                "examples": [
                    "Revoke all tokens for this agent",
                    "Cascade revoke this delegation chain",
                ],
            },
            {
                "id": "credential-broker",
                "name": "Credential Broker",
                "description": "Agents never touch real credentials. Broker injects credentials at call time, returns only results.",
                "tags": ["credentials", "broker", "zero-trust"],
                "examples": [
                    "Request access to feishu API",
                    "Execute a bitable query via broker",
                ],
            },
            {
                "id": "governance-audit",
                "name": "Governance & Audit",
                "description": "SHA-256 hash chain audit logs, governance events, and compliance reporting",
                "tags": ["audit", "governance", "compliance"],
                "examples": [
                    "Get recent governance events",
                    "Retrieve audit log entries",
                ],
            },
        ],
    }




def handle_a2a_request(request: Dict[str, Any]) -> Dict[str, Any]:
    jsonrpc = request.get("jsonrpc", "2.0")
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    if method == "message/send":
        return _handle_send_task(req_id, params)
    elif method == "tasks/get":
        return _handle_get_task(req_id, params)
    elif method == "tasks/cancel":
        return _handle_cancel_task(req_id, params)
    elif method == "tasks/list":
        return _handle_list_tasks(req_id, params)
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def _handle_send_task(req_id: Any, params: Dict) -> Dict[str, Any]:
    task_id = params.get("id", f"task_{uuid.uuid4().hex[:12]}")
    message = params.get("message", {})
    context_id = params.get("contextId")

    _TASKS[task_id] = {
        "id": task_id,
        "status": {"state": "working"},
        "contextId": context_id,
        "history": [message],
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    _TASK_MESSAGES[task_id] = [message]

    user_text = _extract_text(message)
    agent_result = _process_agent_message(user_text, task_id)

    agent_message = {
        "role": "agent",
        "parts": [{"type": "text", "text": agent_result}],
        "messageId": f"msg_{uuid.uuid4().hex[:8]}",
    }
    _TASK_MESSAGES[task_id].append(agent_message)

    artifact = None
    if len(agent_result) > 100:
        artifact = {
            "name": "security-analysis",
            "parts": [{"type": "text", "text": agent_result}],
            "artifactId": f"art_{uuid.uuid4().hex[:8]}",
        }
        _TASK_ARTIFACTS.setdefault(task_id, []).append(artifact)

    _TASKS[task_id]["status"] = {"state": "completed"}
    _TASKS[task_id]["history"] = _TASK_MESSAGES[task_id]

    result = {
        "id": task_id,
        "status": _TASKS[task_id]["status"],
        "history": _TASK_MESSAGES[task_id],
    }
    if artifact:
        result["artifacts"] = [artifact]
    if context_id:
        result["contextId"] = context_id

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _handle_get_task(req_id: Any, params: Dict) -> Dict[str, Any]:
    task_id = params.get("id", "")
    task = _TASKS.get(task_id)
    if not task:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32001, "message": f"Task not found: {task_id}"},
        }
    result = {
        "id": task_id,
        "status": task["status"],
        "history": task.get("history", []),
    }
    if task_id in _TASK_ARTIFACTS:
        result["artifacts"] = _TASK_ARTIFACTS[task_id]
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _handle_cancel_task(req_id: Any, params: Dict) -> Dict[str, Any]:
    task_id = params.get("id", "")
    task = _TASKS.get(task_id)
    if not task:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32001, "message": f"Task not found: {task_id}"},
        }
    task["status"] = {"state": "canceled"}
    return {"jsonrpc": "2.0", "id": req_id, "result": {"id": task_id, "status": task["status"]}}


def _handle_list_tasks(req_id: Any, params: Dict) -> Dict[str, Any]:
    tasks = []
    for tid, task in _TASKS.items():
        tasks.append({"id": tid, "status": task["status"]})
    return {"jsonrpc": "2.0", "id": req_id, "result": {"tasks": tasks}}


def _extract_text(message: Dict) -> str:
    parts = message.get("parts", [])
    texts = []
    for part in parts:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
    return " ".join(texts)


_SKILL_ROUTERS = {
    "prompt-defense": {
        "keywords": ["prompt", "injection", "注入", "攻击", "defense", "防御"],
        "handler": "_handle_prompt_defense",
    },
    "alignment-check": {
        "keywords": ["alignment", "对齐", "hijack", "劫持", "drift", "漂移"],
        "handler": "_handle_alignment_check",
    },
    "iam-check": {
        "keywords": ["permission", "权限", "trust", "信任", "iam", "capability", "能力"],
        "handler": "_handle_iam_check",
    },
    "credential-broker": {
        "keywords": ["credential", "凭证", "broker", "key", "密钥"],
        "handler": "_handle_credential_broker",
    },
    "revocation": {
        "keywords": ["revoke", "撤销", "cascade", "级联", "封禁"],
        "handler": "_handle_revocation",
    },
    "delegation": {
        "keywords": ["delegate", "委派", "chain", "链", "token"],
        "handler": "_handle_delegation",
    },
    "governance-audit": {
        "keywords": ["audit", "审计", "governance", "治理", "compliance", "合规"],
        "handler": "_handle_governance_audit",
    },
}


def _handle_prompt_defense(text: str, task_id: str) -> str:
    try:
        from agentpass_sdk.src.agentpass.prompt_defense import PromptDefenseEngine
        engine = PromptDefenseEngine()
        result = engine.analyze(text)
        return json.dumps({
            "service": "prompt-defense",
            "analysis": result,
            "task_id": task_id,
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"service": "prompt-defense", "error": str(e), "task_id": task_id})


def _handle_alignment_check(text: str, task_id: str) -> str:
    return json.dumps({
        "service": "alignment-check",
        "message": "Use defense.check_alignment MCP tool with original_message and agent_output parameters",
        "task_id": task_id,
    }, indent=2, ensure_ascii=False)


def _handle_iam_check(text: str, task_id: str) -> str:
    try:
        from app.delegation.engine import get_trust_score, CAPABILITY_AGENTS
        scores = {aid: get_trust_score(aid) for aid in CAPABILITY_AGENTS}
        return json.dumps({
            "service": "iam",
            "trust_scores": scores,
            "task_id": task_id,
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"service": "iam", "error": str(e), "task_id": task_id})


def _handle_credential_broker(text: str, task_id: str) -> str:
    return json.dumps({
        "service": "credential-broker",
        "message": "Use broker.request_access or broker.execute MCP tools",
        "task_id": task_id,
    }, indent=2, ensure_ascii=False)


def _handle_revocation(text: str, task_id: str) -> str:
    return json.dumps({
        "service": "4-level-revocation",
        "levels": {"L1": "token", "L2": "agent", "L3": "task", "L4": "chain-cascade"},
        "message": "Use iam.revoke MCP tool with cascade=true for L4",
        "task_id": task_id,
    }, indent=2, ensure_ascii=False)


def _handle_delegation(text: str, task_id: str) -> str:
    try:
        from app.delegation.engine import CAPABILITY_AGENTS, get_all_trust_scores
        trust_info = get_all_trust_scores()
        return json.dumps({
            "service": "delegation",
            "agents": list(CAPABILITY_AGENTS.keys()),
            "trust_info": trust_info,
            "task_id": task_id,
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"service": "delegation", "error": str(e), "task_id": task_id})


def _handle_governance_audit(text: str, task_id: str) -> str:
    try:
        from app import audit
        summary = audit.get_audit_summary()
        return json.dumps({
            "service": "governance-audit",
            "summary": summary,
            "task_id": task_id,
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"service": "governance-audit", "error": str(e), "task_id": task_id})


def _process_agent_message(text: str, task_id: str) -> str:
    text_lower = text.lower()

    for skill_id, router in _SKILL_ROUTERS.items():
        if any(kw in text_lower for kw in router["keywords"]):
            handler_name = router["handler"]
            handler = globals().get(handler_name)
            if handler:
                return handler(text, task_id)

    return json.dumps({
        "service": "agent-iam",
        "message": "Agent IAM Security Server ready. Available skills: " + ", ".join(_SKILL_ROUTERS.keys()),
        "task_id": task_id,
    }, indent=2, ensure_ascii=False)


def get_a2a_server_info() -> Dict[str, Any]:
    return {
        "protocol": "A2A",
        "version": "0.2.6",
        "server_name": "Agent IAM Security A2A Server",
        "skills_count": len(get_agent_card()["skills"]),
        "skills": [s["id"] for s in get_agent_card()["skills"]],
        "transport": "HTTP + JSON-RPC 2.0",
        "endpoints": {
            "agent_card": "/.well-known/agent.json",
            "a2a": "/api/protocols/a2a",
        },
        "task_states": TASK_STATES,
    }


def run_a2a_delegation_demo() -> Dict[str, Any]:
    steps = []

    steps.append({
        "step": 1,
        "action": "user → doc_agent: 用户发起财务查询请求",
        "a2a_method": "message/send",
        "request": {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "查询Q1财务数据"}],
                },
            },
        },
    })

    steps.append({
        "step": 2,
        "action": "doc_agent → IAM: 六层安全验证",
        "a2a_method": "message/send (skill: iam-check)",
        "request": {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "permission check for doc_agent read:feishu_table:finance"}],
                },
            },
        },
    })

    steps.append({
        "step": 3,
        "action": "doc_agent → data_agent: 委派数据查询（JWT委派链）",
        "a2a_method": "message/send (skill: delegation)",
        "delegation_token": "doc_agent签发JWT委派Token，chain=[user, doc_agent, data_agent]",
        "request": {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "delegate read:feishu_table:finance to data_agent"}],
                },
            },
        },
    })

    steps.append({
        "step": 4,
        "action": "data_agent → credential_broker: 请求凭证注入",
        "a2a_method": "message/send (skill: credential-broker)",
        "request": {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "agent",
                    "parts": [{"type": "text", "text": "credential broker access feishu_table:finance"}],
                },
            },
        },
    })

    steps.append({
        "step": 5,
        "action": "credential_broker → feishu API: 凭证注入后调用飞书API",
        "detail": "Broker持有真实凭证，Agent不接触凭证，仅获得查询结果",
    })

    steps.append({
        "step": 6,
        "action": "data_agent → doc_agent → user: 返回查询结果",
        "a2a_method": "tasks/get",
        "result": "查询结果沿委派链返回，每一步都有审计记录",
    })

    try:
        from app.delegation.engine import DelegationEngine
        engine = DelegationEngine()

        root_token = engine.issue_root_token(
            agent_id="doc_agent",
            delegated_user="user_001",
            capabilities=["read:feishu_table:finance", "delegate:data_agent"],
            expires_in_minutes=30,
        )
        root_decoded = engine.decode_delegation_token(root_token)

        delegation_result = engine.delegate(
            parent_token=root_token,
            target_agent="data_agent",
            action="read:feishu_table:finance",
            caller_agent="doc_agent",
        )

        chain_evidence = {
            "root_token_jti": root_decoded.get("jti", "")[:12],
            "root_chain": root_decoded.get("chain", []),
            "delegation_success": delegation_result.success,
        }

        if delegation_result.success and delegation_result.token:
            del_decoded = engine.decode_delegation_token(delegation_result.token)
            chain_evidence["delegated_token_jti"] = del_decoded.get("jti", "")[:12]
            chain_evidence["delegated_chain"] = del_decoded.get("chain", [])
            chain_evidence["delegated_capabilities"] = del_decoded.get("capabilities", [])
    except Exception as e:
        chain_evidence = {"error": str(e)[:100]}

    return {
        "title": "A2A 跨Agent委派调用演示",
        "description": "展示A2A协议下的跨Agent安全委派调用流程",
        "protocol": "A2A v0.2.6 (JSON-RPC 2.0)",
        "flow_steps": steps,
        "chain_evidence": chain_evidence,
        "key_points": [
            "每一步Agent间通信都通过A2A JSON-RPC 2.0协议",
            "委派必须携带JWT Token，验证签名和chain完整性",
            "Agent不直接持有凭证，通过Credential Broker注入",
            "所有操作记录在审计日志中，形成完整追踪链",
        ],
    }
