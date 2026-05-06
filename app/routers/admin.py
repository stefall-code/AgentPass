from __future__ import annotations

import csv
import io
import json
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app import audit, auth, database, identity, permission, schemas
from app.dependencies import require_admin_permission

admin_router = APIRouter(prefix="/admin", tags=["Administration"])


@admin_router.get("/agents", response_model=List[schemas.AgentSummaryResponse])
def list_agents(
    context: auth.AuthContext = Depends(require_admin_permission("manage_agents", "admin:agents")),
) -> List[schemas.AgentSummaryResponse]:
    return [schemas.AgentSummaryResponse(**agent) for agent in identity.list_agents()]


@admin_router.get("/dashboard")
def admin_dashboard(
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:dashboard")),
) -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    audit_summary = audit.get_audit_summary()
    return {
        "snapshot": snapshot,
        "audit": audit_summary,
        "summary": {
            "total_agents": snapshot["agents"]["total"],
            "active_tokens": snapshot["tokens"]["active"],
            "total_audit_logs": audit_summary["total"],
            "total_documents": snapshot["documents"]["total"],
            "allow_count": audit_summary["allow"],
            "deny_count": audit_summary["deny"],
            "role_distribution": snapshot["agents"]["by_role"],
            "sensitivity_distribution": snapshot["documents"]["by_sensitivity"],
            "top_actions": [a["action"] for a in audit_summary["top_actions"]],
            "recent_denials": audit_summary["recent_denials"],
        },
        "role_permissions": permission.ROLE_PERMISSIONS,
        "demo_agents": [
            {"agent_id": item["agent_id"], "name": item["name"], "role": item["role"]}
            for item in settings.DEMO_AGENTS
        ],
        "policy_notes": [
            "Confidential resources require admin role.",
            "Allowed resource allowlist is enforced per agent.",
            "Repeated denied requests trigger automatic suspension for non-admin agents.",
            "Tokens can be bound to IP and limited by usage count.",
            "Time-based access policies are enforced (business hours for sensitive ops).",
            "ABAC attribute conditions can restrict cross-department access.",
        ],
    }


@admin_router.post("/agents/{agent_id}/status", response_model=schemas.AgentStatusUpdateResponse)
def update_agent_status(
    agent_id: str,
    payload: schemas.AgentStatusUpdateRequest,
    context: auth.AuthContext = Depends(require_admin_permission("manage_agents", "admin:agents")),
) -> schemas.AgentStatusUpdateResponse:
    try:
        updated = identity.update_status(agent_id=agent_id, status=payload.status, reason=payload.reason)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return schemas.AgentStatusUpdateResponse(**updated)


@admin_router.put("/agents/{agent_id}", response_model=schemas.AgentUpdateResponse)
def update_agent(
    agent_id: str,
    payload: schemas.AgentUpdateRequest,
    context: auth.AuthContext = Depends(require_admin_permission("manage_agents", "admin:agents")),
) -> schemas.AgentUpdateResponse:
    try:
        updated = identity.update_agent(
            agent_id=agent_id,
            name=payload.name,
            attributes=payload.attributes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return schemas.AgentUpdateResponse(**updated)


@admin_router.delete("/agents/{agent_id}", response_model=schemas.ActionMessageResponse)
def delete_agent(
    agent_id: str,
    context: auth.AuthContext = Depends(require_admin_permission("manage_agents", "admin:agents")),
) -> schemas.ActionMessageResponse:
    try:
        deleted = identity.delete_agent(agent_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    audit.log_event(
        agent_id=context.agent["agent_id"],
        action="delete_agent",
        resource=f"admin:agents:{agent_id}",
        decision="allow",
        reason=f"Agent {agent_id} deleted.",
        ip_address=context.request_ip,
        token_id=context.token_id,
    )
    return schemas.ActionMessageResponse(
        message=f"Agent {agent_id} has been deleted.",
        details={"agent_id": agent_id, "deleted": deleted},
    )


@admin_router.post("/demo/reset", response_model=schemas.ActionMessageResponse)
def reset_demo_state(
    context: auth.AuthContext = Depends(require_admin_permission("manage_agents", "admin:demo_reset")),
) -> schemas.ActionMessageResponse:
    cleared_tokens = database.clear_tokens()
    cleared_logs = audit.clear_logs()
    documents = database.reset_documents()
    identity.sync_demo_agents(reset_state=True)
    audit.log_event(
        agent_id=context.agent["agent_id"],
        action="reset_demo_state",
        resource="admin:demo_reset",
        decision="allow",
        reason="Demo state reset by administrator.",
        ip_address=context.request_ip,
        token_id=context.token_id,
        context={"cleared_tokens": cleared_tokens, "cleared_logs": cleared_logs},
    )
    return schemas.ActionMessageResponse(
        message="Demo state has been reset. All existing tokens are now invalid; please log in again.",
        details={
            "cleared_tokens": cleared_tokens,
            "cleared_logs": cleared_logs,
            "documents_restored": len(documents),
        },
    )


@admin_router.get("/audit/logs", response_model=List[schemas.AuditEventResponse])
def get_audit_logs(
    limit: int = Query(default=20, ge=1, le=100),
    agent_id: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:audit")),
) -> List[schemas.AuditEventResponse]:
    return [
        schemas.AuditEventResponse(**item)
        for item in audit.fetch_logs_filtered(limit=limit, agent_id=agent_id, decision=decision, action=action)
    ]


@admin_router.get("/audit/export")
def export_audit_logs(
    format: str = Query(default="json", pattern="^(json|csv)$"),
    agent_id: Optional[str] = Query(default=None),
    decision: Optional[str] = Query(default=None),
    action: Optional[str] = Query(default=None),
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:audit_export")),
):
    from fastapi.responses import Response

    logs = audit.fetch_logs_filtered(limit=2000, agent_id=agent_id, decision=decision, action=action)

    if format == "csv":
        output = io.StringIO()
        if logs:
            writer = csv.DictWriter(output, fieldnames=logs[0].keys())
            writer.writeheader()
            writer.writerows(logs)
        return Response(
            content=output.getvalue().encode("utf-8"),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit_logs.csv"},
        )

    return Response(
        content=json.dumps(logs, ensure_ascii=False, indent=2).encode("utf-8"),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=audit_logs.json"},
    )


@admin_router.get("/activity/timeline")
def get_activity_timeline(
    limit: int = Query(default=50, ge=1, le=200),
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:activity")),
) -> Dict[str, Any]:
    logs = audit.fetch_logs_filtered(limit=limit)
    timeline = []
    for log in logs:
        timeline.append({
            "id": log["id"],
            "timestamp": log["created_at"],
            "agent_id": log.get("agent_id"),
            "action": log["action"],
            "resource": log["resource"],
            "decision": log["decision"],
            "reason": log["reason"],
            "ip_address": log.get("ip_address"),
        })
    return {"timeline": timeline, "count": len(timeline)}


@admin_router.get("/audit/verify-integrity")
def verify_audit_integrity(
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:audit")),
) -> Dict[str, Any]:
    """验证审计日志哈希链完整性"""
    return audit.verify_chain_integrity()


@admin_router.get("/stats/realtime")
def get_realtime_stats(
    context: auth.AuthContext = Depends(require_admin_permission("view_audit", "admin:stats")),
) -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    summary = audit.get_audit_summary()
    return {
        "agents": snapshot["agents"],
        "tokens": snapshot["tokens"],
        "audit_summary": {
            "total": summary["total"],
            "allow": summary["allow"],
            "deny": summary["deny"],
        },
        "timestamp": database.utc_now(),
    }
