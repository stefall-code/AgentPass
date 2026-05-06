import os
import time
import json
import asyncio
import logging
import subprocess
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from app.feishu.iam_gateway import (
    callIAMCheck,
    logAudit,
    get_audit_log,
    get_gateway_stats,
    IAMCheckResult,
    Decision,
)
from app.delegation.engine import (
    DelegationEngine,
    get_trust_score,
    update_trust_score,
    CAPABILITY_AGENTS,
    is_agent_auto_revoked,
    AUTO_REVOKE_THRESHOLD,
)

logger = logging.getLogger(__name__)

_CLI_COMMAND_ACTION_MAP = {
    "calendar": {
        "+agenda": "read:calendar",
        "+create-event": "write:calendar",
        "calendars": "read:calendar",
        "events": "read:calendar",
    },
    "im": {
        "+messages-send": "write:feishu_message",
        "+chat-create": "write:feishu_message",
        "+chat-messages-list": "read:feishu_message",
        "+chat-search": "read:feishu_message",
        "+messages-search": "read:feishu_message",
        "+messages-reply": "write:feishu_message",
        "+messages-resources-download": "read:feishu_message",
    },
    "docs": {
        "+create": "write:doc",
        "+search": "read:doc",
    },
    "base": {
        "+table-list": "read:bitable",
        "+table-create": "write:bitable",
        "+record-list": "read:bitable",
        "+record-create": "write:bitable",
        "+record-update": "write:bitable",
        "+record-delete": "write:bitable",
        "+field-list": "read:bitable",
        "+field-create": "write:bitable",
        "+data-query": "read:bitable",
        "+dashboard-list": "read:bitable",
        "+workflow-list": "read:bitable",
    },
    "sheets": {
        "+read": "read:sheet",
        "+write": "write:sheet",
        "+create": "write:sheet",
        "+append": "write:sheet",
        "+export": "read:sheet",
    },
    "task": {
        "+get-my-tasks": "read:task",
        "+create": "write:task",
        "+complete": "write:task",
        "+search": "read:task",
        "+tasklist-search": "read:task",
    },
    "contact": {
        "+search-user": "read:contact",
    },
    "mail": {
        "+triage": "read:mail",
        "+send": "write:mail",
        "+message": "read:mail",
        "+reply": "write:mail",
    },
    "vc": {
        "+search": "read:vc",
        "+notes": "read:vc",
        "+recording": "read:vc",
    },
    "wiki": {
        "+node-create": "write:wiki",
        "+move": "write:wiki",
        "spaces": "read:wiki",
        "nodes": "read:wiki",
    },
    "drive": {
        "+search": "read:drive",
        "+upload": "write:drive",
    },
    "approval": {
        "instances": "read:approval",
        "tasks": "write:approval",
    },
    "attendance": {
        "user_tasks": "read:attendance",
    },
    "okr": {
        "+cycle-list": "read:okr",
        "+cycle-detail": "read:okr",
        "+progress-create": "write:okr",
    },
    "slides": {
        "+create": "write:slides",
    },
    "whiteboard": {
        "+query": "read:whiteboard",
        "+update": "write:whiteboard",
    },
    "minutes": {
        "+search": "read:minutes",
    },
    "markdown": {
        "+create": "write:doc",
        "+fetch": "read:doc",
    },
}

_DOMAIN_DEFAULT_ACTION = {
    "calendar": "read:calendar",
    "im": "read:feishu_message",
    "docs": "read:doc",
    "base": "read:bitable",
    "sheets": "read:sheet",
    "task": "read:task",
    "contact": "read:contact",
    "mail": "read:mail",
    "vc": "read:vc",
    "wiki": "read:wiki",
    "drive": "read:drive",
    "approval": "read:approval",
    "attendance": "read:attendance",
    "okr": "read:okr",
    "slides": "read:slides",
    "whiteboard": "read:whiteboard",
    "minutes": "read:minutes",
    "markdown": "read:doc",
}

_WRITE_KEYWORDS = {"create", "send", "update", "delete", "write", "append", "complete", "reply", "forward", "move", "upload"}

_SENSITIVE_DOMAINS = {"base", "sheets", "docs", "im", "mail", "approval", "okr"}
_HIGH_RISK_DOMAINS = {"base", "approval", "okr"}


def map_cli_command_to_action(domain: str, subcommand: str = "", method: str = "GET") -> str:
    if domain == "api":
        if subcommand:
            parts = subcommand.strip("/").split("/")
            if len(parts) >= 3:
                resource = parts[2]
                scope = "write" if method.upper() in ("POST", "PUT", "PATCH", "DELETE") else "read"
                return f"{scope}:{resource}"
        return f"{'write' if method.upper() in ('POST', 'PUT', 'PATCH', 'DELETE') else 'read'}:feishu_api"

    domain_map = _CLI_COMMAND_ACTION_MAP.get(domain, {})
    if subcommand in domain_map:
        return domain_map[subcommand]

    if subcommand.startswith("+"):
        cmd_name = subcommand[1:]
        for keyword in _WRITE_KEYWORDS:
            if keyword in cmd_name.lower():
                default = _DOMAIN_DEFAULT_ACTION.get(domain, f"write:{domain}")
                if default.startswith("read:"):
                    return default.replace("read:", "write:", 1)
                return default
        return _DOMAIN_DEFAULT_ACTION.get(domain, f"read:{domain}")

    if subcommand:
        for key, action in domain_map.items():
            if subcommand.startswith(key):
                return action

    return _DOMAIN_DEFAULT_ACTION.get(domain, f"read:{domain}")


def _assess_cli_risk(domain: str, subcommand: str, action: str) -> float:
    risk = 0.0
    if action.startswith("write:"):
        risk += 0.2
    if domain in _HIGH_RISK_DOMAINS:
        risk += 0.15
    if domain in _SENSITIVE_DOMAINS:
        risk += 0.1
    if "finance" in action or "hr" in action:
        risk += 0.3
    return min(1.0, risk)


@dataclass
class CLIExecutionResult:
    success: bool
    output: str = ""
    exit_code: int = 0
    iam_allowed: bool = True
    iam_decision: str = "allow"
    iam_reason: str = ""
    iam_blocked_at: str = ""
    trust_score: Optional[float] = None
    risk_score: Optional[float] = None
    agent_id: str = ""
    action: str = ""
    domain: str = ""
    subcommand: str = ""
    latency_ms: float = 0.0
    six_layer: Optional[Dict[str, Any]] = None
    auto_revoked: bool = False
    timestamp: float = field(default_factory=time.time)


_cli_audit_log: List[Dict[str, Any]] = []
_MAX_CLI_AUDIT = 500

_cli_stats = {
    "total_commands": 0,
    "allowed_commands": 0,
    "blocked_commands": 0,
    "auto_revoked_commands": 0,
    "by_domain": {},
}


def _log_cli_audit(result: CLIExecutionResult) -> None:
    entry = {
        "timestamp": result.timestamp,
        "agent_id": result.agent_id,
        "domain": result.domain,
        "subcommand": result.subcommand,
        "action": result.action,
        "iam_decision": result.iam_decision,
        "iam_reason": result.iam_reason,
        "iam_blocked_at": result.iam_blocked_at,
        "trust_score": result.trust_score,
        "risk_score": result.risk_score,
        "success": result.success,
        "auto_revoked": result.auto_revoked,
        "latency_ms": round(result.latency_ms, 2),
    }
    _cli_audit_log.append(entry)
    if len(_cli_audit_log) > _MAX_CLI_AUDIT:
        _cli_audit_log.pop(0)

    _cli_stats["total_commands"] += 1
    if result.iam_allowed:
        _cli_stats["allowed_commands"] += 1
    else:
        _cli_stats["blocked_commands"] += 1
    if result.auto_revoked:
        _cli_stats["auto_revoked_commands"] += 1
    domain = result.domain
    if domain not in _cli_stats["by_domain"]:
        _cli_stats["by_domain"][domain] = {"total": 0, "allowed": 0, "blocked": 0}
    _cli_stats["by_domain"][domain]["total"] += 1
    if result.iam_allowed:
        _cli_stats["by_domain"][domain]["allowed"] += 1
    else:
        _cli_stats["by_domain"][domain]["blocked"] += 1

    logAudit(
        agent_id=result.agent_id,
        action=result.action,
        decision=result.iam_decision,
        reason=result.iam_reason,
        latency_ms=result.latency_ms,
        trust_score=result.trust_score,
        risk_score=result.risk_score,
        blocked_at=result.iam_blocked_at,
        auto_revoked=result.auto_revoked,
        path=f"/cli/{result.domain}/{result.subcommand}",
        method="CLI",
        six_layer=result.six_layer,
    )

    status = "ALLOW" if result.iam_allowed else ("REVOKE" if result.auto_revoked else "DENY")
    logger.info("CLI IAM %s: agent=%s domain=%s action=%s blocked_at=%s",
                status, result.agent_id, result.domain, result.action, result.iam_blocked_at)


def get_cli_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    return _cli_audit_log[-limit:]


def get_cli_stats() -> Dict[str, Any]:
    return dict(_cli_stats)


async def execute_cli_command(
    domain: str,
    subcommand: str = "",
    args: Optional[List[str]] = None,
    agent_id: str = "doc_agent",
    method: str = "GET",
    dry_run: bool = False,
) -> CLIExecutionResult:
    start = time.time()
    args = args or []

    action = map_cli_command_to_action(domain, subcommand, method)
    risk_score = _assess_cli_risk(domain, subcommand, action)

    iam_result = _cli_iam_check(agent_id, action)

    latency_iam = (time.time() - start) * 1000

    if not iam_result.allowed:
        result = CLIExecutionResult(
            success=False,
            output="",
            exit_code=403,
            iam_allowed=False,
            iam_decision=iam_result.decision.value,
            iam_reason=iam_result.reason,
            iam_blocked_at=iam_result.blocked_at,
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_iam,
            six_layer=iam_result.six_layer,
            auto_revoked=iam_result.auto_revoked,
        )
        _log_cli_audit(result)
        return result

    if dry_run:
        result = CLIExecutionResult(
            success=True,
            output="[DRY RUN] IAM check passed — command would be executed",
            exit_code=0,
            iam_allowed=True,
            iam_decision="allow",
            iam_reason="IAM check passed (dry run)",
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_iam,
            six_layer=iam_result.six_layer,
        )
        _log_cli_audit(result)
        return result

    cmd_parts = ["lark-cli.cmd" if os.name == "nt" else "lark-cli", domain]
    if subcommand:
        cmd_parts.append(subcommand)
    cmd_parts.extend(args)

    cmd_str = " ".join(cmd_parts)
    logger.info("CLI executing: %s (agent=%s action=%s)", cmd_str, agent_id, action)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        output = stdout.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        if stderr:
            err_output = stderr.decode("utf-8", errors="replace")
            if err_output.strip():
                output += "\n" + err_output

        latency_total = (time.time() - start) * 1000

        result = CLIExecutionResult(
            success=exit_code == 0,
            output=output,
            exit_code=exit_code,
            iam_allowed=True,
            iam_decision="allow",
            iam_reason="IAM check passed",
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_total,
            six_layer=iam_result.six_layer,
        )
        _log_cli_audit(result)
        return result

    except asyncio.TimeoutError:
        latency_total = (time.time() - start) * 1000
        result = CLIExecutionResult(
            success=False,
            output="CLI command timed out (30s)",
            exit_code=-1,
            iam_allowed=True,
            iam_decision="allow",
            iam_reason="IAM check passed but execution timed out",
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_total,
            six_layer=iam_result.six_layer,
        )
        _log_cli_audit(result)
        return result

    except FileNotFoundError:
        latency_total = (time.time() - start) * 1000
        result = CLIExecutionResult(
            success=False,
            output="lark-cli not found — install with: npm install -g @larksuite/cli",
            exit_code=-1,
            iam_allowed=True,
            iam_decision="allow",
            iam_reason="IAM check passed but CLI not installed",
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_total,
            six_layer=iam_result.six_layer,
        )
        _log_cli_audit(result)
        return result

    except Exception as e:
        latency_total = (time.time() - start) * 1000
        result = CLIExecutionResult(
            success=False,
            output=f"CLI execution error: {str(e)}",
            exit_code=-1,
            iam_allowed=True,
            iam_decision="allow",
            iam_reason="IAM check passed but execution failed",
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score or risk_score,
            agent_id=agent_id,
            action=action,
            domain=domain,
            subcommand=subcommand,
            latency_ms=latency_total,
            six_layer=iam_result.six_layer,
        )
        _log_cli_audit(result)
        return result


def get_cli_domain_capabilities() -> Dict[str, Any]:
    domains = {}
    for domain, commands in _CLI_COMMAND_ACTION_MAP.items():
        domain_info = {
            "domain": domain,
            "default_action": _DOMAIN_DEFAULT_ACTION.get(domain, f"read:{domain}"),
            "is_sensitive": domain in _SENSITIVE_DOMAINS,
            "is_high_risk": domain in _HIGH_RISK_DOMAINS,
            "commands": [],
        }
        for cmd, action in commands.items():
            domain_info["commands"].append({
                "command": cmd,
                "action": action,
                "is_write": action.startswith("write:"),
            })
        domains[domain] = domain_info
    return domains


def _cli_iam_check(agent_id: str, action: str) -> IAMCheckResult:
    start = time.time()
    try:
        auto_revoked, auto_revoke_reason = is_agent_auto_revoked(agent_id)
        if auto_revoked:
            latency_ms = (time.time() - start) * 1000
            return IAMCheckResult(
                allowed=False,
                decision=Decision.DENY,
                reason=f"Agent auto-revoked: {auto_revoke_reason}",
                agent_id=agent_id,
                action=action,
                latency_ms=latency_ms,
                blocked_at="auto_revocation",
                auto_revoked=True,
            )

        trust_score = get_trust_score(agent_id)
        if trust_score < 0.5:
            latency_ms = (time.time() - start) * 1000
            return IAMCheckResult(
                allowed=False,
                decision=Decision.DENY,
                reason=f"Low trust score: agent '{agent_id}' trust={trust_score:.2f} < threshold=0.5",
                agent_id=agent_id,
                action=action,
                latency_ms=latency_ms,
                trust_score=trust_score,
                risk_score=0.85,
                blocked_at="trust_check",
            )

        agent_caps = CAPABILITY_AGENTS.get(agent_id, {}).get("capabilities", [])
        if not agent_caps:
            latency_ms = (time.time() - start) * 1000
            return IAMCheckResult(
                allowed=False,
                decision=Decision.DENY,
                reason=f"Unknown agent '{agent_id}' — no capabilities defined",
                agent_id=agent_id,
                action=action,
                latency_ms=latency_ms,
                trust_score=trust_score,
                risk_score=0.9,
                blocked_at="identity_check",
            )

        engine = DelegationEngine()
        cap_ok, cap_reason = engine._check_capability(action, agent_caps)
        if not cap_ok:
            update_trust_score(agent_id, -0.05)
            trust_after = get_trust_score(agent_id)
            latency_ms = (time.time() - start) * 1000

            blocked_at = "capability_check"
            auto_rev = False
            if trust_after < AUTO_REVOKE_THRESHOLD:
                from app.delegation.engine import auto_revoke_agent
                auto_revoke_agent(agent_id, reason=f"Auto revoke: CLI escalation, trust={trust_after:.2f}")
                blocked_at = "auto_revocation"
                auto_rev = True

            return IAMCheckResult(
                allowed=False,
                decision=Decision.DENY,
                reason=f"Capability denied: '{action}' not in {agent_caps}",
                agent_id=agent_id,
                action=action,
                latency_ms=latency_ms,
                trust_score=trust_after,
                risk_score=0.7,
                blocked_at=blocked_at,
                auto_revoked=auto_rev,
            )

        update_trust_score(agent_id, 0.01)
        trust_after = get_trust_score(agent_id)
        latency_ms = (time.time() - start) * 1000

        six_layer_result = None
        try:
            from app.security.six_layer_verify import verify_six_layers
            six_layer_result = verify_six_layers(
                agent_id=agent_id,
                action=action,
                input_text="",
                trust_score=trust_after,
                risk_score=0.1,
                blocked_at="",
                auto_revoked=False,
                allowed=True,
                reason="CLI IAM check passed",
            )
        except Exception:
            pass

        return IAMCheckResult(
            allowed=True,
            decision=Decision.ALLOW,
            reason=f"Action '{action}' allowed for agent '{agent_id}'",
            agent_id=agent_id,
            action=action,
            latency_ms=latency_ms,
            trust_score=trust_after,
            risk_score=0.1,
            six_layer=six_layer_result.to_dict() if six_layer_result else None,
        )

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error("CLI IAM check error: %s", e)
        return IAMCheckResult(
            allowed=False,
            decision=Decision.ERROR,
            reason=f"IAM check error: {str(e)}",
            agent_id=agent_id,
            action=action,
            latency_ms=latency_ms,
        )


def check_cli_permission(agent_id: str, domain: str, subcommand: str = "", method: str = "GET") -> Dict[str, Any]:
    action = map_cli_command_to_action(domain, subcommand, method)
    risk_score = _assess_cli_risk(domain, subcommand, action)
    iam_result = _cli_iam_check(agent_id, action)

    return {
        "agent_id": agent_id,
        "domain": domain,
        "subcommand": subcommand,
        "action": action,
        "allowed": iam_result.allowed,
        "decision": iam_result.decision.value,
        "reason": iam_result.reason,
        "blocked_at": iam_result.blocked_at,
        "trust_score": iam_result.trust_score,
        "risk_score": iam_result.risk_score or risk_score,
        "auto_revoked": iam_result.auto_revoked,
        "six_layer": iam_result.six_layer,
    }
