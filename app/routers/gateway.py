from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from fastapi import APIRouter

from ..feishu.iam_gateway import (
    mapRequestToAction,
    callIAMCheck,
    logAudit,
    get_audit_log,
    get_gateway_stats,
    IAMCheckResult,
    Decision,
)

router = APIRouter(prefix="/gateway", tags=["IAM Gateway"])


class GatewayCheckRequest(BaseModel):
    agent_id: str = "doc_agent"
    action: str = ""
    path: Optional[str] = None
    method: Optional[str] = "GET"


class GatewayMapRequest(BaseModel):
    path: str
    method: Optional[str] = "GET"


@router.get("/stats")
def gateway_stats() -> Dict[str, Any]:
    stats = get_gateway_stats()
    return {"gateway": "feishu_iam", "status": "active", **stats}


@router.get("/audit")
def gateway_audit(limit: int = 50) -> List[Dict[str, Any]]:
    return get_audit_log(limit=limit)


@router.post("/check")
def gateway_check(req: GatewayCheckRequest) -> Dict[str, Any]:
    action = req.action
    if not action and req.path:
        action = mapRequestToAction(req.path, req.method or "GET")

    if not action:
        return {"allowed": False, "reason": "No action or path provided", "decision": "deny"}

    result = callIAMCheck(req.agent_id, action)

    logAudit(
        agent_id=result.agent_id,
        action=result.action,
        decision="allow" if result.allowed else ("error" if result.decision == Decision.ERROR else "deny"),
        reason=result.reason,
        latency_ms=result.latency_ms,
        trust_score=result.trust_score,
        risk_score=result.risk_score,
        blocked_at=result.blocked_at,
        auto_revoked=result.auto_revoked,
        path=req.path or "",
        method=req.method or "",
    )

    resp: Dict[str, Any] = {
        "allowed": result.allowed,
        "decision": result.decision.value,
        "reason": result.reason,
        "agent_id": result.agent_id,
        "action": result.action,
        "latency_ms": round(result.latency_ms, 2),
    }
    if result.trust_score is not None:
        resp["trust_score"] = result.trust_score
    if result.risk_score is not None:
        resp["risk_score"] = result.risk_score
    if result.blocked_at:
        resp["blocked_at"] = result.blocked_at
    if result.auto_revoked:
        resp["auto_revoked"] = True

    return resp


@router.post("/map-action")
def gateway_map_action(req: GatewayMapRequest) -> Dict[str, Any]:
    action = mapRequestToAction(req.path, req.method or "GET")
    return {"path": req.path, "method": req.method, "action": action}


@router.post("/demo/escalation")
def gateway_demo_escalation() -> Dict[str, Any]:
    from ..delegation.engine import clear_used_tokens
    clear_used_tokens()

    results = []

    r1 = callIAMCheck("data_agent", "read:feishu_table")
    results.append(_format_result("data_agent", "read:feishu_table", r1))

    r2 = callIAMCheck("data_agent", "read:feishu_table:finance")
    results.append(_format_result("data_agent", "read:feishu_table:finance", r2))

    r3 = callIAMCheck("doc_agent", "read:feishu_table:finance")
    results.append(_format_result("doc_agent", "read:feishu_table:finance", r3))

    r4 = callIAMCheck("external_agent", "read:feishu_table:finance")
    results.append(_format_result("external_agent", "read:feishu_table:finance", r4))

    return {
        "demo": "escalation",
        "description": "Escalation attack: agents try to access finance data with different privilege levels",
        "results": results,
        "summary": f"{sum(1 for r in results if r['allowed'])} allowed, {sum(1 for r in results if not r['allowed'])} blocked",
    }


@router.post("/demo/bypass-attempt")
def gateway_demo_bypass() -> Dict[str, Any]:
    from ..delegation.engine import clear_used_tokens
    clear_used_tokens()

    results = []

    paths = [
        ("/im/v1/messages", "POST", "write:feishu_message"),
        ("/docx/v1/documents", "POST", "write:doc"),
        ("/bitable/v1/apps/xxx/tables", "GET", "read:bitable"),
        ("/calendar/v4/calendars", "POST", "write:calendar"),
        ("/drive/v1/files/secret", "GET", "read:drive"),
    ]

    for path, method, action in paths:
        r = callIAMCheck("external_agent", action)
        results.append({
            "path": path,
            "method": method,
            "mapped_action": action,
            **_format_result("external_agent", action, r, path=path, method=method),
        })

    return {
        "demo": "bypass_attempt",
        "description": "external_agent attempts to bypass IAM via different API paths",
        "results": results,
        "summary": f"All {len(results)} requests processed through IAM Gateway — {sum(1 for r in results if not r['allowed'])} blocked",
    }


def _format_result(agent_id: str, action: str, result: IAMCheckResult, path: str = "", method: str = "") -> Dict[str, Any]:
    logAudit(
        agent_id=agent_id,
        action=action,
        decision="allow" if result.allowed else ("error" if result.decision == Decision.ERROR else "deny"),
        reason=result.reason,
        latency_ms=result.latency_ms,
        trust_score=result.trust_score,
        risk_score=result.risk_score,
        blocked_at=result.blocked_at,
        auto_revoked=result.auto_revoked,
        path=path,
        method=method,
    )
    resp: Dict[str, Any] = {
        "agent_id": agent_id,
        "action": action,
        "allowed": result.allowed,
        "decision": result.decision.value,
        "reason": result.reason,
        "latency_ms": round(result.latency_ms, 2),
    }
    if result.trust_score is not None:
        resp["trust_score"] = result.trust_score
    if result.risk_score is not None:
        resp["risk_score"] = result.risk_score
    if result.blocked_at:
        resp["blocked_at"] = result.blocked_at
    if result.auto_revoked:
        resp["auto_revoked"] = True
    return resp
