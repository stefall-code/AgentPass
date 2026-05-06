from __future__ import annotations

import logging
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.config import settings
from app import database

logger = logging.getLogger("agent_system")
router = APIRouter()


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "index.html")


@router.get("/chain", include_in_schema=False)
def chain_viewer() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "chain.html")


@router.get("/audit", include_in_schema=False)
def audit_center() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "audit.html")


@router.get("/trust", include_in_schema=False)
def trust_dashboard() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "trust.html")


@router.get("/risk", include_in_schema=False)
def risk_dashboard() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "risk.html")


@router.get("/feishu", include_in_schema=False)
def serve_feishu() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "feishu.html")


@router.get("/governance", include_in_schema=False)
def serve_governance() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "governance.html")


@router.get("/gateway", include_in_schema=False)
def serve_gateway() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "gateway.html")


@router.get("/demo", include_in_schema=False)
def serve_demo() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "demo.html")


@router.get("/api/overview")
def api_overview() -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    from app import audit as _audit
    audit_summary = _audit.get_audit_summary()
    return {
        "system": "Agent Identity & Permission System",
        "version": "v2.6",
        "features": [
            "agent registration",
            "JWT authentication",
            "refresh token",
            "RBAC permission control",
            "ABAC attribute policy",
            "time-based access policy",
            "policy engine with priority rules",
            "audit log with hash chain",
            "token IP binding",
            "token usage limit",
            "automatic risk lock on repeated denials",
            "web security console",
            "token introspection and revoke",
            "one-click demo reset",
            "batch login",
            "real-time audit WebSocket",
            "audit log export (JSON/CSV)",
            "agent CRUD management",
            "background token cleanup",
            "rate limiting",
            "request tracing",
            "policy decision trace",
            "delegation chain graph",
            "risk dashboard",
            "permission diff",
            "prompt injection defense (7 types)",
            "OpenClaw integration API",
            "OpenClaw request persistence",
            "daily statistics",
            "WebSocket long-connection (Feishu)",
        ],
        "policies": [
            {"name": "RBAC", "description": "基于角色的访问控制，admin/operator/viewer 三级权限"},
            {"name": "ABAC", "description": "基于属性的访问控制，资源敏感度 + 请求上下文"},
            {"name": "Time-Based", "description": "时间窗口访问策略，非工作时段自动降级"},
            {"name": "Delegation", "description": "委派链追踪，防止权限越级传递"},
            {"name": "Risk Lock", "description": "连续拒绝自动锁定，防止暴力破解"},
        ],
        "demo_agents": [
            {"agent_id": item["agent_id"], "role": item["role"], "name": item["name"]}
            for item in settings.DEMO_AGENTS
        ],
        "demo_documents": database.list_documents(),
        "ngrok_url": None,
        "ws_mode": True,
        "stats": {
            "health": "OK",
            "active_tokens": snapshot.get("tokens", {}).get("active", 0),
            "denied_requests": audit_summary.get("deny", 0),
            "suspended_agents": snapshot["agents"]["by_status"].get("suspended", 0),
        },
    }


@router.get("/api/health")
def api_health() -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    from app import audit as _audit
    audit_summary = _audit.get_audit_summary()
    return {
        "system": "Agent Identity & Permission System",
        "version": "v2.6",
        "ngrok_url": None,
        "stats": {
            "active_tokens": snapshot.get("tokens", {}).get("active", 0),
            "denied_requests": audit_summary.get("deny", 0),
            "suspended_agents": snapshot["agents"]["by_status"].get("suspended", 0),
        },
    }


@router.get("/api/admin/role-matrix")
def role_matrix() -> Dict[str, Any]:
    from app import permission as perm
    roles = list(perm.ROLE_PERMISSIONS.keys())
    all_actions = sorted(set(a for actions in perm.ROLE_PERMISSIONS.values() for a in actions))
    matrix = {role: {action: action in perm.ROLE_PERMISSIONS[role] for action in all_actions} for role in roles}
    return {"roles": roles, "permissions": all_actions, "matrix": matrix}


@router.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "database": str(settings.DATABASE_PATH), "version": "v2.6"}


@router.get("/api/monitor/alerts")
def monitor_alerts() -> Dict[str, Any]:
    alerts = []
    now = time.time()

    try:
        from app.delegation.engine import AUTO_REVOKED_AGENTS, AGENT_TRUST_SCORE, AUTO_REVOKE_THRESHOLD
        for agent_id, info in AUTO_REVOKED_AGENTS.items():
            alerts.append({
                "level": "CRITICAL",
                "type": "auto_revoke",
                "agent_id": agent_id,
                "message": f"Agent '{agent_id}' auto-revoked (trust={info.get('trust_score_at_revoke', 'N/A')})",
                "timestamp": info.get("revoked_at", ""),
            })
        for agent_id, score in AGENT_TRUST_SCORE.items():
            if agent_id not in AUTO_REVOKED_AGENTS and score <= AUTO_REVOKE_THRESHOLD + 0.1:
                alerts.append({
                    "level": "WARNING",
                    "type": "trust_low",
                    "agent_id": agent_id,
                    "message": f"Agent '{agent_id}' trust score critically low ({score:.2f})",
                    "timestamp": "",
                })
    except Exception:
        pass

    try:
        from app.feishu.ws_client import is_ws_connected
        if not is_ws_connected():
            alerts.append({
                "level": "WARNING",
                "type": "ws_disconnected",
                "agent_id": "feishu_bridge",
                "message": "Feishu WebSocket long-connection is not active",
                "timestamp": "",
            })
    except Exception:
        pass

    try:
        from app import audit as _audit
        summary = _audit.get_audit_summary()
        deny_count = summary.get("deny", 0)
        total_count = sum(summary.values())
        if total_count > 0 and deny_count / total_count > 0.3:
            alerts.append({
                "level": "WARNING",
                "type": "high_deny_rate",
                "agent_id": "system",
                "message": f"High denial rate: {deny_count}/{total_count} ({deny_count/total_count*100:.1f}%)",
                "timestamp": "",
            })
    except Exception:
        pass

    try:
        snapshot = database.get_system_snapshot()
        suspended = snapshot.get("agents", {}).get("by_status", {}).get("suspended", 0)
        if suspended > 0:
            alerts.append({
                "level": "WARNING",
                "type": "suspended_agents",
                "agent_id": "system",
                "message": f"{suspended} agent(s) currently suspended",
                "timestamp": "",
            })
    except Exception:
        pass

    try:
        chain_ok = True
        chain_msg = "OK"
        from app import audit as _audit
        result = _audit.verify_chain_integrity()
        if not result.get("valid", True):
            chain_ok = False
            chain_msg = f"Hash chain integrity FAILED: {result.get('reason', 'unknown')}"
            alerts.append({
                "level": "CRITICAL",
                "type": "chain_tampered",
                "agent_id": "audit_system",
                "message": chain_msg,
                "timestamp": "",
            })
    except Exception:
        chain_ok = True
        chain_msg = "N/A (no logs)"

    return {
        "status": "CRITICAL" if any(a["level"] == "CRITICAL" for a in alerts) else ("WARNING" if alerts else "OK"),
        "alert_count": len(alerts),
        "alerts": alerts,
        "checks": {
            "auto_revoked_agents": len(AUTO_REVOKED_AGENTS) if 'AUTO_REVOKED_AGENTS' in dir() else 0,
            "ws_connection": "connected" if 'is_ws_connected' in dir() and is_ws_connected() else "disconnected",
            "audit_chain_integrity": chain_msg,
            "high_deny_rate": any(a["type"] == "high_deny_rate" for a in alerts),
        },
    }


@router.post("/api/feishu/approval-callback", tags=["Feishu"])
def feishu_approval_callback(
    approval_id: int = Query(...),
    decision: str = Query(..., pattern="^(approved|denied)$"),
    operator: str = Query(default="feishu_user"),
) -> Dict[str, Any]:
    from app.db import SessionLocal
    from app.models import ApprovalRequest
    from sqlalchemy import select

    with SessionLocal() as db:
        row = db.execute(
            select(ApprovalRequest).where(ApprovalRequest.id == approval_id)
        ).scalar_one_or_none()
        if not row:
            raise HTTPException(status_code=404, detail="Approval request not found")
        if row.status != "pending":
            return {"message": f"Already {row.status}", "approval_id": approval_id}

        now = database.utc_now()
        row.status = decision
        row.decided_at = now
        row.decided_by = f"feishu:{operator}"
        row.reason = "Decided via Feishu card"
        db.commit()

    from app import audit as _audit
    _audit.log_event(
        agent_id=row.agent_id,
        action=f"approval_{decision}",
        resource=row.resource,
        decision="allow" if decision == "approved" else "deny",
        reason=f"Feishu callback: {decision} by {operator}",
    )
    return {"message": f"Approval {decision}", "approval_id": approval_id}
