"""
MCP (Model Context Protocol) Server Adapter

Exposes our security capabilities as MCP Tools, allowing any MCP Client
(e.g., Claude Desktop, Cursor, VS Code Copilot) to discover and invoke
our IAM, prompt defense, alignment checks, and credential broker.

Protocol: JSON-RPC 2.0 over HTTP
Spec: https://modelcontextprotocol.io/specification/2025-11-25

MCP Tools exposed:
  - iam.check_permission   — Check if agent has permission for an action
  - iam.get_trust_score    — Get agent trust score
  - iam.delegate           — Create a delegation token
  - iam.revoke             — Revoke tokens (4-level)
  - defense.check_prompt   — Check prompt for injection attacks
  - defense.check_alignment — Check agent output alignment
  - broker.request_access  — Request credential access via broker
  - broker.execute         — Execute API call via credential broker
  - governance.get_events  — Get governance audit events
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("agent_system")


@dataclass
class MCPTool:
    name: str
    title: str
    description: str
    input_schema: Dict[str, Any]
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class MCPToolResult:
    content: List[Dict[str, Any]]
    is_error: bool = False


MCP_TOOLS: List[MCPTool] = [
    MCPTool(
        name="iam.check_permission",
        title="Check Agent Permission",
        description="Check if an agent has permission to perform a specific action. Returns allowed/denied with trust score and policy details.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent identifier (e.g., doc_agent, data_agent)"},
                "action": {"type": "string", "description": "Action to check (e.g., read:feishu_table:finance)"},
                "token": {"type": "string", "description": "Optional delegation token for the check"},
            },
            "required": ["agent_id", "action"],
        },
    ),
    MCPTool(
        name="iam.get_trust_score",
        title="Get Agent Trust Score",
        description="Get the current trust score for an agent. Scores range from 0.0 (untrusted) to 1.0 (fully trusted). Below 0.5 = denied, below 0.4 = auto-revoked.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Agent identifier"},
            },
            "required": ["agent_id"],
        },
    ),
    MCPTool(
        name="iam.delegate",
        title="Create Delegation Token",
        description="Create a delegation token allowing one agent to act on behalf of another. Supports capability scoping and chain depth limits.",
        input_schema={
            "type": "object",
            "properties": {
                "parent_token": {"type": "string", "description": "Parent delegation token"},
                "target_agent": {"type": "string", "description": "Agent to delegate to"},
                "action": {"type": "string", "description": "Capability to delegate"},
            },
            "required": ["parent_token", "target_agent", "action"],
        },
    ),
    MCPTool(
        name="iam.revoke",
        title="Revoke Tokens (4-Level)",
        description="Revoke tokens at 4 levels: L1 (single token), L2 (agent), L3 (task), L4 (chain cascade). Set cascade=true for L4.",
        input_schema={
            "type": "object",
            "properties": {
                "jti": {"type": "string", "description": "Token JTI to revoke"},
                "agent_id": {"type": "string", "description": "Agent ID for L2 revocation"},
                "task_id": {"type": "string", "description": "Task ID for L3 revocation"},
                "cascade": {"type": "boolean", "description": "Enable L4 cascade revocation", "default": False},
                "reason": {"type": "string", "description": "Reason for revocation"},
            },
        },
    ),
    MCPTool(
        name="defense.check_prompt",
        title="Check Prompt for Injection",
        description="Analyze a prompt for injection attacks using the 3-layer fusion engine (rules 55% + semantic 30% + behavioral 15%). Returns risk score and attack types.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Prompt text to analyze"},
            },
            "required": ["message"],
        },
    ),
    MCPTool(
        name="defense.check_alignment",
        title="Check Output Alignment",
        description="Check if agent output aligns with original user intent. Detects goal hijacking, indirect injection, and DLP leaks in outputs.",
        input_schema={
            "type": "object",
            "properties": {
                "original_message": {"type": "string", "description": "Original user message"},
                "agent_output": {"type": "string", "description": "Agent's output to check"},
            },
            "required": ["original_message", "agent_output"],
        },
    ),
    MCPTool(
        name="broker.request_access",
        title="Request Credential Access",
        description="Request access to a service credential via the Credential Broker. Agent never sees the actual credential value.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Requesting agent ID"},
                "service": {"type": "string", "description": "Service name (e.g., feishu, bitable)"},
                "operation": {"type": "string", "description": "Operation (e.g., read, write)"},
            },
            "required": ["agent_id", "service", "operation"],
        },
    ),
    MCPTool(
        name="broker.execute",
        title="Execute via Credential Broker",
        description="Execute an API call via the Credential Broker. Broker injects credentials, agent only receives the result.",
        input_schema={
            "type": "object",
            "properties": {
                "agent_id": {"type": "string", "description": "Requesting agent ID"},
                "service": {"type": "string", "description": "Service name"},
                "operation": {"type": "string", "description": "Operation to perform"},
                "params": {"type": "object", "description": "Additional parameters for the API call"},
            },
            "required": ["agent_id", "service", "operation"],
        },
    ),
    MCPTool(
        name="governance.get_events",
        title="Get Governance Events",
        description="Retrieve governance audit events including policy decisions, trust score changes, and security incidents.",
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum events to return", "default": 20},
                "platform": {"type": "string", "description": "Filter by platform"},
            },
        },
    ),
]


def handle_mcp_request(request: Dict[str, Any]) -> Dict[str, Any]:
    jsonrpc = request.get("jsonrpc", "2.0")
    req_id = request.get("id")
    method = request.get("method", "")
    params = request.get("params", {})

    if method == "initialize":
        return _handle_initialize(req_id, params)
    elif method == "tools/list":
        return _handle_tools_list(req_id, params)
    elif method == "tools/call":
        return _handle_tools_call(req_id, params)
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}
    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def _handle_initialize(req_id: Any, params: Dict) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "protocolVersion": "2025-11-25",
            "capabilities": {
                "tools": {"listChanged": True},
            },
            "serverInfo": {
                "name": "agent-iam-mcp-server",
                "version": "2.4.0",
                "title": "Agent IAM Security MCP Server",
                "description": "Exposes Agent IAM security capabilities (prompt defense, alignment checks, credential broker, 4-level revocation) as MCP tools.",
            },
        },
    }


def _handle_tools_list(req_id: Any, params: Dict) -> Dict[str, Any]:
    tools = []
    for t in MCP_TOOLS:
        tool_def = {
            "name": t.name,
            "title": t.title,
            "description": t.description,
            "inputSchema": t.input_schema,
        }
        if t.annotations:
            tool_def["annotations"] = t.annotations
        tools.append(tool_def)

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {"tools": tools},
    }


def _handle_tools_call(req_id: Any, params: Dict) -> Dict[str, Any]:
    tool_name = params.get("name", "")
    arguments = params.get("arguments", {})

    try:
        result = _execute_tool(tool_name, arguments)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": result.content,
                "isError": result.is_error,
            },
        }
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                "isError": True,
            },
        }


def _execute_tool(tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
    if tool_name == "iam.check_permission":
        return _tool_check_permission(arguments)
    elif tool_name == "iam.get_trust_score":
        return _tool_get_trust_score(arguments)
    elif tool_name == "iam.delegate":
        return _tool_delegate(arguments)
    elif tool_name == "iam.revoke":
        return _tool_revoke(arguments)
    elif tool_name == "defense.check_prompt":
        return _tool_check_prompt(arguments)
    elif tool_name == "defense.check_alignment":
        return _tool_check_alignment(arguments)
    elif tool_name == "broker.request_access":
        return _tool_broker_request(arguments)
    elif tool_name == "broker.execute":
        return _tool_broker_execute(arguments)
    elif tool_name == "governance.get_events":
        return _tool_governance_events(arguments)
    else:
        return MCPToolResult(
            content=[{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            is_error=True,
        )


def _tool_check_permission(args: Dict) -> MCPToolResult:
    from app.delegation.engine import CAPABILITY_AGENTS, get_trust_score
    agent_id = args.get("agent_id", "")
    action = args.get("action", "")
    trust = get_trust_score(agent_id)
    caps = CAPABILITY_AGENTS.get(agent_id, {}).get("capabilities", [])
    allowed = action in caps and trust >= 0.5
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "agent_id": agent_id,
            "action": action,
            "allowed": allowed,
            "trust_score": trust,
            "has_capability": action in caps,
            "trust_sufficient": trust >= 0.5,
        }, indent=2, ensure_ascii=False),
    }])


def _tool_get_trust_score(args: Dict) -> MCPToolResult:
    from app.delegation.engine import get_trust_score
    agent_id = args.get("agent_id", "")
    trust = get_trust_score(agent_id)
    status = "trusted" if trust >= 0.7 else ("degraded" if trust >= 0.5 else ("denied" if trust >= 0.4 else "auto_revoked"))
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({"agent_id": agent_id, "trust_score": trust, "status": status}, ensure_ascii=False),
    }])


def _tool_delegate(args: Dict) -> MCPToolResult:
    from app.delegation.engine import DelegationEngine
    engine = DelegationEngine()
    parent_token = args.get("parent_token", "")
    target_agent = args.get("target_agent", "")
    action = args.get("action", "")
    result = engine.delegate(parent_token=parent_token, target_agent=target_agent, action=action)
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "success": result.success,
            "reason": result.reason,
            "token_preview": result.token[:20] + "..." if result.success and result.token else None,
        }, ensure_ascii=False),
    }])


def _tool_revoke(args: Dict) -> MCPToolResult:
    from app.delegation.revocation import revoke_4level
    result = revoke_4level(
        jti=args.get("jti"),
        agent_id=args.get("agent_id"),
        task_id=args.get("task_id"),
        cascade=args.get("cascade", False),
        reason=args.get("reason", ""),
    )
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps(result, indent=2, ensure_ascii=False),
    }])


def _tool_check_prompt(args: Dict) -> MCPToolResult:
    from agentpass_sdk.src.agentpass.prompt_defense import PromptDefenseEngine
    engine = PromptDefenseEngine()
    message = args.get("message", "")
    result = engine.analyze(message)
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "risk_score": result.get("risk_score", 0),
            "is_attack": result.get("is_attack", False),
            "attack_types": result.get("attack_types", []),
            "action": result.get("action", "allow"),
        }, indent=2, ensure_ascii=False),
    }])


def _tool_check_alignment(args: Dict) -> MCPToolResult:
    from app.security.alignment import check_alignment
    result = check_alignment(
        original_message=args.get("original_message", ""),
        agent_output=args.get("agent_output", ""),
    )
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "aligned": result.aligned,
            "risk_score": result.risk_score,
            "action": result.action,
            "reasons": result.reasons,
        }, indent=2, ensure_ascii=False),
    }])


def _tool_broker_request(args: Dict) -> MCPToolResult:
    from app.security.credential_broker import request_access
    result = request_access(
        agent_id=args.get("agent_id", ""),
        service=args.get("service", ""),
        operation=args.get("operation", ""),
    )
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }, indent=2, ensure_ascii=False),
    }])


def _tool_broker_execute(args: Dict) -> MCPToolResult:
    from app.security.credential_broker import execute_via_broker
    result = execute_via_broker(
        agent_id=args.get("agent_id", ""),
        service=args.get("service", ""),
        operation=args.get("operation", ""),
        params=args.get("params"),
    )
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({
            "success": result.success,
            "data": result.data,
            "error": result.error,
        }, indent=2, ensure_ascii=False),
    }])


def _tool_governance_events(args: Dict) -> MCPToolResult:
    from app.routers.governance import _governance_events
    limit = args.get("limit", 20)
    events = list(_governance_events)[-limit:]
    return MCPToolResult(content=[{
        "type": "text",
        "text": json.dumps({"events": events, "count": len(events)}, indent=2, ensure_ascii=False),
    }])


def get_mcp_server_info() -> Dict[str, Any]:
    return {
        "protocol": "MCP",
        "version": "2025-11-25",
        "server_name": "agent-iam-mcp-server",
        "tools_count": len(MCP_TOOLS),
        "tools": [t.name for t in MCP_TOOLS],
        "capabilities": {
            "tools": {"listChanged": True},
        },
        "transport": "HTTP + JSON-RPC 2.0",
        "endpoint": "/api/protocols/mcp",
    }
