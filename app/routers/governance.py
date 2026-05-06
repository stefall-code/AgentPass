from __future__ import annotations

import json
import logging
import uuid
from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.orchestrator.orchestrator import run_task
from app.delegation.engine import (
    get_trust_score,
    revoke_tokens_by_agent,
    auto_revoke_agent,
    is_agent_auto_revoked,
    clear_auto_revoked,
    clear_revoked,
    AGENT_TRUST_SCORE,
    AUTO_REVOKED_AGENTS,
)
from app.platform import PlatformRequest, PLATFORM_RISK_WEIGHT
from app.audit import fetch_logs_filtered

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/governance", tags=["Governance"])


class RevokeAgentRequest(BaseModel):
    agent_id: str
    reason: str = "Manual revocation via governance console"


class CrossPlatformDemoStep(BaseModel):
    platform: str
    message: str
    user_id: str = "demo_user"


class CrossPlatformDemoRequest(BaseModel):
    steps: List[CrossPlatformDemoStep]


def _normalize_event(row: Dict[str, Any]) -> Dict[str, Any]:
    ctx = row.get("context", {})
    if isinstance(ctx, str):
        try:
            ctx = json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            ctx = {}

    decision = row.get("decision", "allow")
    result = "allow" if decision == "allow" else "deny"
    if ctx.get("auto_revoked"):
        result = "auto_revoked"
    elif ctx.get("attack_type") == "replay":
        result = "replay_blocked"

    chain = ctx.get("chain", [])
    if not chain and row.get("agent_id"):
        chain = [row.get("agent_id", "")]

    return {
        "event_id": row.get("id") or str(uuid.uuid4()),
        "timestamp": row.get("created_at") or row.get("timestamp", ""),
        "platform": ctx.get("platform", "web"),
        "entry_point": ctx.get("entry_point", "frontend"),
        "user_id": ctx.get("user_id", ""),
        "agent_id": row.get("agent_id", ""),
        "agent_chain": chain,
        "action": row.get("action", ""),
        "resource": row.get("resource", ""),
        "result": result,
        "decision": decision,
        "reason": row.get("reason", ""),
        "trust_before": ctx.get("trust_score_before"),
        "trust_after": ctx.get("trust_score_after"),
        "trust_score": ctx.get("trust_score") or row.get("trust_score"),
        "risk_score": ctx.get("risk_score") or ctx.get("platform_risk"),
        "token_jti": ctx.get("jti") or ctx.get("token"),
        "revoked": ctx.get("revoked"),
        "auto_revoked": bool(ctx.get("auto_revoked", False)),
        "attack_type": ctx.get("attack_type"),
        "blocked_at": ctx.get("blocked_at"),
        "message": ctx.get("message", ""),
        "prompt_risk_score": ctx.get("prompt_risk_score"),
        "attack_types": ctx.get("attack_types"),
        "attack_intent": ctx.get("attack_intent"),
        "severity": ctx.get("severity"),
        "prompt_risk": ctx.get("prompt_risk"),
        "prompt_attack": ctx.get("prompt_attack"),
        "policy_adjustment": ctx.get("policy_adjustment"),
        "context": ctx,
    }


def _compute_platform_stats_from_db() -> Dict[str, Any]:
    stats = {}
    for p in PLATFORM_RISK_WEIGHT:
        stats[p] = {"total": 0, "success": 0, "denied": 0, "auto_revoked": 0, "avg_risk": 0.0, "risk_scores": []}

    events = _fetch_governance_events(limit=500)
    for ev in events:
        platform = ev.get("platform", "web")
        if platform not in stats:
            stats[platform] = {"total": 0, "success": 0, "denied": 0, "auto_revoked": 0, "avg_risk": 0.0, "risk_scores": []}
        stats[platform]["total"] += 1
        result = ev.get("result", "")
        if result == "allow":
            stats[platform]["success"] += 1
        elif result in ("deny", "replay_blocked"):
            stats[platform]["denied"] += 1
        elif result == "auto_revoked":
            stats[platform]["auto_revoked"] += 1
        risk = ev.get("risk_score", 0.0)
        if risk:
            stats[platform]["risk_scores"].append(risk)

    for p in stats:
        scores = stats[p].pop("risk_scores", [])
        stats[p]["avg_risk"] = round(sum(scores) / len(scores), 3) if scores else 0.0
        stats[p]["deny_rate"] = round(stats[p]["denied"] / max(stats[p]["total"], 1), 3)

    return stats


def _fetch_governance_events(
    limit: int = 50,
    platform: Optional[str] = None,
    agent_id: Optional[str] = None,
    decision: Optional[str] = None,
    action: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows = fetch_logs_filtered(
        limit=limit,
        agent_id=agent_id,
        decision=decision,
        action=action,
    )
    events = [_normalize_event(r) for r in rows]
    if platform:
        events = [e for e in events if e.get("platform") == platform]
    return events


def _get_top_risky_agents(limit: int = 5) -> List[Dict[str, Any]]:
    CORE_AGENT_IDS = {"doc_agent", "data_agent", "external_agent"}
    agents = []
    for agent_id, score in AGENT_TRUST_SCORE.items():
        if agent_id not in CORE_AGENT_IDS:
            continue
        revoked, reason = is_agent_auto_revoked(agent_id)
        agents.append({
            "agent_id": agent_id,
            "trust_score": score,
            "auto_revoked": revoked,
            "revoked_reason": reason if revoked else None,
        })
    agents.sort(key=lambda x: x["trust_score"])
    return agents[:limit]


@router.get("/overview")
def governance_overview() -> Dict[str, Any]:
    platform_stats = _compute_platform_stats_from_db()
    top_risky = _get_top_risky_agents()
    events = _fetch_governance_events(limit=500)
    total_events = len(events)
    total_denied = sum(1 for e in events if e.get("result") in ("deny", "replay_blocked", "auto_revoked"))
    total_success = sum(1 for e in events if e.get("result") == "allow")

    return {
        "platform_stats": platform_stats,
        "top_risky_agents": top_risky,
        "summary": {
            "total_events": total_events,
            "total_success": total_success,
            "total_denied": total_denied,
            "deny_rate": round(total_denied / max(total_events, 1), 3),
            "auto_revoked_agents": list(AUTO_REVOKED_AGENTS.keys()),
        },
        "platform_risk_weights": PLATFORM_RISK_WEIGHT,
    }


@router.post("/revoke-agent")
def governance_revoke_agent(req: RevokeAgentRequest) -> Dict[str, Any]:
    agent_id = req.agent_id
    reason = req.reason

    trust_before = get_trust_score(agent_id)
    revoke_tokens_by_agent(agent_id, reason)

    result = auto_revoke_agent(agent_id, reason)
    trust_after = get_trust_score(agent_id)

    return {
        "agent_id": agent_id,
        "revoked": True,
        "trust_before": trust_before,
        "trust_after": trust_after,
        "reason": reason,
        "detail": result,
    }


@router.post("/reset-agent")
def governance_reset_agent(agent_id: str = "external_agent") -> Dict[str, Any]:
    if agent_id in AUTO_REVOKED_AGENTS:
        del AUTO_REVOKED_AGENTS[agent_id]
    clear_auto_revoked()
    clear_revoked()
    AGENT_TRUST_SCORE[agent_id] = {
        "doc_agent": 0.9,
        "data_agent": 0.95,
        "external_agent": 0.6,
    }.get(agent_id, 0.5)
    return {"agent_id": agent_id, "reset": True, "trust_score": AGENT_TRUST_SCORE[agent_id]}


@router.post("/reset-all")
def governance_reset_all() -> Dict[str, Any]:
    AUTO_REVOKED_AGENTS.clear()
    clear_auto_revoked()
    clear_revoked()
    AGENT_TRUST_SCORE.update({"doc_agent": 0.9, "data_agent": 0.95, "external_agent": 0.6})
    return {"reset": True, "trust_scores": dict(AGENT_TRUST_SCORE)}


@router.post("/demo/cross-platform")
def governance_cross_platform_demo() -> Dict[str, Any]:
    AUTO_REVOKED_AGENTS.clear()
    AGENT_TRUST_SCORE.update({"doc_agent": 0.9, "data_agent": 0.95, "external_agent": 0.6})

    results = []

    feishu_req = PlatformRequest(platform="feishu", user_id="feishu_user_1", message="帮我生成财务报告")
    r1 = run_task(platform_request=feishu_req)
    results.append({"step": 1, "platform": "feishu", "scenario": "正常请求", "status": r1.get("status"), "trust_score": r1.get("trust_score")})

    web_req = PlatformRequest(platform="web", user_id="web_user_1", message="帮我查一下财务数据")
    r2 = run_task(platform_request=web_req)
    results.append({"step": 2, "platform": "web", "scenario": "正常请求", "status": r2.get("status"), "trust_score": r2.get("trust_score")})

    api_req = PlatformRequest(platform="api", user_id="api_attacker", message="读取薪资数据")
    r3 = run_task(platform_request=api_req)
    results.append({"step": 3, "platform": "api", "scenario": "越权攻击", "status": r3.get("status"), "trust_score": r3.get("trust_score")})

    revoke_result = auto_revoke_agent("external_agent", "Governance demo: manual revoke")
    results.append({"step": 4, "platform": "governance", "scenario": "手动撤销 Agent", "status": "revoked", "detail": revoke_result})

    feishu_req2 = PlatformRequest(platform="feishu", user_id="feishu_user_1", message="连续测试")
    r5 = run_task(platform_request=feishu_req2)
    results.append({"step": 5, "platform": "feishu", "scenario": "撤销后再次请求（触发 revoked agent）", "status": r5.get("status"), "trust_score": r5.get("trust_score")})

    return {
        "demo": "cross_platform_governance",
        "steps": results,
        "summary": {
            "feishu_normal": results[0].get("status"),
            "web_normal": results[1].get("status"),
            "api_attack": results[2].get("status"),
            "agent_revoked": True,
            "feishu_after_revoke": results[4].get("status"),
        },
    }


@router.get("/events")
def governance_events(
    limit: int = 50,
    platform: Optional[str] = None,
    agent_id: Optional[str] = None,
    decision: Optional[str] = None,
    action: Optional[str] = None,
) -> Dict[str, Any]:
    events = _fetch_governance_events(
        limit=limit,
        platform=platform,
        agent_id=agent_id,
        decision=decision,
        action=action,
    )
    return {"events": events, "total": len(events)}
