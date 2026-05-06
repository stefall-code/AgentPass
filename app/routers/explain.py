from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.explainer import explain_decision
from app.delegation.engine import (
    DelegationEngine,
    get_trust_score,
    is_agent_auto_revoked,
    CAPABILITY_AGENTS,
    TRUST_THRESHOLD,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["Explainable IAM"])

_engine = DelegationEngine()


class ExplainRequest(BaseModel):
    token: str = Field(default="", description="Delegation token")
    action: str = Field(default="", description="Action to explain")
    agent_id: str = Field(default="", description="Agent ID (alternative to token)")
    message: str = Field(default="", description="Feishu message (alternative to token+action)")


class ExplainByResultRequest(BaseModel):
    agent_id: str = Field(default="")
    action: str = Field(default="")
    decision: str = Field(default="deny")
    reason: str = Field(default="")
    trust_score: Optional[float] = Field(default=None)
    risk_score: float = Field(default=0.0)
    chain_detail: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    blocked_at: str = Field(default="")
    auto_revoked: bool = Field(default=False)
    prompt_risk_score: Optional[float] = Field(default=None)
    attack_types: List[str] = Field(default_factory=list)
    attack_intent: str = Field(default="")
    severity: str = Field(default="")


@router.post("")
def explain_endpoint(req: ExplainRequest) -> Dict[str, Any]:
    if req.token and req.action:
        return _explain_by_token(req.token, req.action)
    if req.agent_id and req.action:
        return _explain_by_agent(req.agent_id, req.action)
    if req.message:
        return _explain_by_message(req.message)
    return {"summary": "⚠️ 缺少参数", "steps": [], "risk_analysis": "", "trust_analysis": "", "final_reason": "请提供 token+action 或 agent_id+action 或 message", "suggestion": ""}


@router.post("/result")
def explain_by_result_endpoint(req: ExplainByResultRequest) -> Dict[str, Any]:
    context = {
        "agent_id": req.agent_id,
        "action": req.action,
        "decision": req.decision,
        "reason": req.reason,
        "trust_score": req.trust_score,
        "risk_score": req.risk_score,
        "chain_detail": req.chain_detail,
        "capabilities": req.capabilities,
        "blocked_at": req.blocked_at,
        "auto_revoked": req.auto_revoked,
        "prompt_risk_score": req.prompt_risk_score,
        "attack_types": req.attack_types,
        "attack_intent": req.attack_intent,
        "severity": req.severity,
    }
    return explain_decision(context)


def _explain_by_token(token: str, action: str) -> Dict[str, Any]:
    check_result = _engine.check(token=token, action=action)
    trust = get_trust_score(check_result.agent_id or "unknown")
    context = {
        "agent_id": check_result.agent_id or "unknown",
        "action": action,
        "decision": "allow" if check_result.allowed else "deny",
        "reason": check_result.reason or "",
        "trust_score": trust,
        "risk_score": check_result.risk_score or 0.0,
        "chain_detail": check_result.chain or [],
        "capabilities": list(check_result.capabilities or []),
        "blocked_at": "",
        "auto_revoked": check_result.auto_revoked or False,
    }
    if not check_result.allowed:
        reason_lower = (check_result.reason or "").lower()
        if "auto-revoked" in reason_lower:
            context["blocked_at"] = "auto_revoke"
        elif "replay" in reason_lower:
            context["blocked_at"] = "replay"
        elif "escalation" in reason_lower or "capability" in reason_lower:
            context["blocked_at"] = "check"
        elif "dynamic policy" in reason_lower:
            context["blocked_at"] = "dynamic_policy"
        elif "low trust" in reason_lower:
            context["blocked_at"] = "trust"
        elif "revoked" in reason_lower:
            context["blocked_at"] = "revoke"
        elif "expired" in reason_lower or "invalid" in reason_lower:
            context["blocked_at"] = "token"
        else:
            context["blocked_at"] = "check"

    return explain_decision(context)


def _explain_by_agent(agent_id: str, action: str) -> Dict[str, Any]:
    trust = get_trust_score(agent_id)
    auto_revoked, _ = is_agent_auto_revoked(agent_id)
    agent_caps = CAPABILITY_AGENTS.get(agent_id, {}).get("capabilities", [])
    has_cap = action in agent_caps
    decision = "allow" if (has_cap and trust >= TRUST_THRESHOLD and not auto_revoked) else "deny"

    blocked_at = ""
    if auto_revoked:
        blocked_at = "auto_revoke"
    elif not has_cap:
        blocked_at = "check"
    elif trust < TRUST_THRESHOLD:
        blocked_at = "trust"

    context = {
        "agent_id": agent_id,
        "action": action,
        "decision": decision,
        "reason": f"Agent {agent_id} {'has' if has_cap else 'lacks'} capability {action}",
        "trust_score": trust,
        "risk_score": 0.5 if not has_cap else (0.1 if trust >= 0.7 else 0.7),
        "chain_detail": ["user", agent_id],
        "capabilities": agent_caps,
        "blocked_at": blocked_at,
        "auto_revoked": auto_revoked,
    }
    return explain_decision(context)


def _explain_by_message(message: str) -> Dict[str, Any]:
    from app.orchestrator.orchestrator import run_task
    from app.platform import PlatformRequest

    p_req = PlatformRequest(platform="web", user_id="explain_user", message=message)
    result = run_task(platform_request=p_req)

    context = {
        "agent_id": (result.get("chain") or [""])[-1] if result.get("chain") else "unknown",
        "action": result.get("capability", ""),
        "decision": "allow" if result.get("status") == "success" else "deny",
        "reason": result.get("reason", ""),
        "trust_score": result.get("trust_score"),
        "risk_score": result.get("platform_risk", 0.0),
        "chain_detail": result.get("chain", []),
        "capabilities": [],
        "blocked_at": result.get("blocked_at", ""),
        "auto_revoked": result.get("auto_revoked", False),
        "prompt_risk_score": result.get("prompt_risk_score"),
        "attack_types": result.get("attack_types", []),
        "attack_intent": result.get("attack_intent", ""),
        "severity": result.get("severity", ""),
    }
    return explain_decision(context)
