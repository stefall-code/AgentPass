from __future__ import annotations

import hashlib
import json
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app import identity, permission, policy
from app.auth import AuthContext, get_auth_context
from app.config import settings
from app.dependencies import require_admin_permission

insight_router = APIRouter(prefix="/insights", tags=["Insights"])


class PolicyTraceRequest(BaseModel):
    agent_id: str
    action: str
    resource: str
    resource_sensitivity: str = Field(default="public")


class PolicyTraceStepResponse(BaseModel):
    step_id: str
    name: str
    passed: bool
    reason: str
    detail: str


class PolicyTraceResponse(BaseModel):
    allowed: bool
    reason: str
    rule_id: str
    agent_id: str
    action: str
    resource: str
    trace: List[PolicyTraceStepResponse]


@insight_router.post("/policy-trace", response_model=PolicyTraceResponse)
def policy_trace(
    payload: PolicyTraceRequest,
    context: AuthContext = Depends(get_auth_context),
) -> PolicyTraceResponse:
    agent = identity.get_agent(payload.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found.")
    decision = policy.evaluate(
        agent=agent,
        action=payload.action,
        resource=payload.resource,
        resource_meta={"sensitivity": payload.resource_sensitivity},
    )
    return PolicyTraceResponse(
        allowed=decision.allowed,
        reason=decision.reason,
        rule_id=decision.rule_id,
        agent_id=payload.agent_id,
        action=payload.action,
        resource=payload.resource,
        trace=[
            PolicyTraceStepResponse(
                step_id=s.step_id,
                name=s.name,
                passed=s.passed,
                reason=s.reason,
                detail=s.detail,
            )
            for s in decision.trace
        ],
    )


class DelegationEdgeResponse(BaseModel):
    source: str
    target: str
    count: int
    success_count: int
    fail_count: int


class DelegationNodeResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str


class DelegationGraphResponse(BaseModel):
    nodes: List[DelegationNodeResponse]
    edges: List[DelegationEdgeResponse]


@insight_router.get("/delegation-graph", response_model=DelegationGraphResponse)
def delegation_graph(
    context: AuthContext = Depends(require_admin_permission("view_audit", "insights:delegation")),
) -> DelegationGraphResponse:
    from app.db import SessionLocal
    from app.models import AuditLogRow
    from sqlalchemy import select, func

    agents = identity.list_agents()
    nodes = [
        DelegationNodeResponse(
            agent_id=a["agent_id"],
            name=a["name"],
            role=a["role"],
            status=a["status"],
        )
        for a in agents
    ]

    with SessionLocal() as db:
        rows = db.execute(
            select(AuditLogRow.agent_id, AuditLogRow.resource, AuditLogRow.decision, func.count())
            .where(AuditLogRow.action == "delegate_task")
            .group_by(AuditLogRow.agent_id, AuditLogRow.resource, AuditLogRow.decision)
        ).all()

    edge_map: Dict[tuple, Dict[str, int]] = {}
    for agent_id, resource, decision, count in rows:
        target_id = resource.replace("agent:", "")
        key = (agent_id, target_id)
        if key not in edge_map:
            edge_map[key] = {"count": 0, "success_count": 0, "fail_count": 0}
        edge_map[key]["count"] += count
        if decision == "allow":
            edge_map[key]["success_count"] += count
        else:
            edge_map[key]["fail_count"] += count

    edges = [
        DelegationEdgeResponse(
            source=k[0], target=k[1],
            count=v["count"], success_count=v["success_count"], fail_count=v["fail_count"],
        )
        for k, v in edge_map.items()
    ]

    return DelegationGraphResponse(nodes=nodes, edges=edges)


class AgentRiskGaugeResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str
    risk_score: float
    denial_count: int
    total_requests: int
    risk_level: str
    recent_denial_rate: float


class RiskDashboardResponse(BaseModel):
    agents: List[AgentRiskGaugeResponse]
    system_risk_level: str


@insight_router.get("/risk-dashboard", response_model=RiskDashboardResponse)
def risk_dashboard(
    context: AuthContext = Depends(require_admin_permission("view_audit", "insights:risk")),
) -> RiskDashboardResponse:
    from app.db import SessionLocal
    from app.models import AuditLogRow
    from sqlalchemy import select, func, and_
    from datetime import timedelta, timezone, datetime

    agents = identity.list_agents()
    agent_risks = []
    now = datetime.now(timezone.utc)
    window = (now - timedelta(minutes=settings.DENIAL_WINDOW_MINUTES)).isoformat()

    with SessionLocal() as db:
        for a in agents:
            total = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(AuditLogRow.agent_id == a["agent_id"])
            ).scalar() or 0

            denials = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(and_(AuditLogRow.agent_id == a["agent_id"], AuditLogRow.decision == "deny"))
            ).scalar() or 0

            recent_denials = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(and_(
                    AuditLogRow.agent_id == a["agent_id"],
                    AuditLogRow.decision == "deny",
                    AuditLogRow.created_at >= window,
                ))
            ).scalar() or 0

            denial_rate = (denials / total * 100) if total > 0 else 0
            recent_rate = (recent_denials / max(total, 1) * 100)

            if a["status"] == "suspended":
                risk_score = 100.0
                risk_level = "critical"
            elif recent_denials >= 2:
                risk_score = min(80 + recent_denials * 5, 99)
                risk_level = "high"
            elif denial_rate > 30:
                risk_score = min(50 + denial_rate, 79)
                risk_level = "medium"
            elif denial_rate > 10:
                risk_score = min(20 + denial_rate, 49)
                risk_level = "low"
            else:
                risk_score = denial_rate
                risk_level = "safe"

            agent_risks.append(AgentRiskGaugeResponse(
                agent_id=a["agent_id"],
                name=a["name"],
                role=a["role"],
                status=a["status"],
                risk_score=round(risk_score, 1),
                denial_count=denials,
                total_requests=total,
                risk_level=risk_level,
                recent_denial_rate=round(recent_rate, 1),
            ))

    critical_count = sum(1 for r in agent_risks if r.risk_level in ("critical", "high"))
    system_risk = "critical" if critical_count >= 2 else "high" if critical_count >= 1 else "medium" if any(r.risk_level == "medium" for r in agent_risks) else "safe"

    return RiskDashboardResponse(agents=agent_risks, system_risk_level=system_risk)


class AuditIntegrityResponse(BaseModel):
    valid: bool
    total_logs: int
    tampered_ids: List[int]
    message: str


@insight_router.get("/audit-integrity", response_model=AuditIntegrityResponse)
def audit_integrity(
    context: AuthContext = Depends(require_admin_permission("view_audit", "insights:integrity")),
) -> AuditIntegrityResponse:
    from app.db import SessionLocal
    from app.models import AuditLogRow
    from sqlalchemy import select

    with SessionLocal() as db:
        rows = db.execute(
            select(AuditLogRow).order_by(AuditLogRow.id)
        ).scalars().all()

    if not rows:
        return AuditIntegrityResponse(valid=True, total_logs=0, tampered_ids=[], message="No audit logs to verify.")

    tampered = []
    prev_hash = "genesis"

    for row in rows:
        payload = json.dumps({
            "id": row.id,
            "agent_id": row.agent_id,
            "action": row.action,
            "resource": row.resource,
            "decision": row.decision,
            "reason": row.reason,
            "created_at": row.created_at,
            "prev_hash": prev_hash,
        }, sort_keys=True, ensure_ascii=False)
        expected_hash = hashlib.sha256(payload.encode()).hexdigest()

        stored_context = {}
        try:
            stored_context = json.loads(row.context_json or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        stored_hash = stored_context.get("_chain_hash")
        if stored_hash and stored_hash != expected_hash:
            tampered.append(row.id)

        prev_hash = stored_hash or expected_hash

    return AuditIntegrityResponse(
        valid=len(tampered) == 0,
        total_logs=len(rows),
        tampered_ids=tampered,
        message="All audit logs are intact." if not tampered else f"Found {len(tampered)} tampered log entries.",
    )


class PermissionDiffResponse(BaseModel):
    agent_a: str
    agent_b: str
    a_only: List[str]
    b_only: List[str]
    common: List[str]
    a_resources: List[str]
    b_resources: List[str]
    a_only_resources: List[str]
    b_only_resources: List[str]


@insight_router.get("/permission-diff", response_model=PermissionDiffResponse)
def permission_diff(
    agent_a: str = Query(...),
    agent_b: str = Query(...),
    context: AuthContext = Depends(get_auth_context),
) -> PermissionDiffResponse:
    a = identity.get_agent(agent_a)
    b = identity.get_agent(agent_b)
    if not a or not b:
        raise HTTPException(status_code=404, detail="One or both agents not found.")

    a_perms = set(permission.list_permissions(a["role"]))
    b_perms = set(permission.list_permissions(b["role"]))
    a_resources = set(a.get("attributes", {}).get("allowed_resources", []))
    b_resources = set(b.get("attributes", {}).get("allowed_resources", []))

    return PermissionDiffResponse(
        agent_a=agent_a,
        agent_b=agent_b,
        a_only=sorted(a_perms - b_perms),
        b_only=sorted(b_perms - a_perms),
        common=sorted(a_perms & b_perms),
        a_resources=sorted(a_resources),
        b_resources=sorted(b_resources),
        a_only_resources=sorted(a_resources - b_resources),
        b_only_resources=sorted(b_resources - a_resources),
    )


class ReputationResponse(BaseModel):
    agent_id: str
    score: float
    allow_rate: float
    denial_streak: int
    suspicious_pattern_count: int
    consistency_bonus: float
    trend: str
    last_computed_at: str
    history: list = Field(default_factory=list)


class ReputationRankingItem(BaseModel):
    agent_id: str
    score: float
    trend: str


@insight_router.get("/reputation/ranking", response_model=List[ReputationRankingItem])
def reputation_ranking(
    context: AuthContext = Depends(get_auth_context),
) -> List[ReputationRankingItem]:
    from app.services.reputation_service import ReputationEngine
    engine = ReputationEngine()
    ranking = engine.get_ranking()
    CORE_AGENT_IDS = {"doc_agent", "data_agent", "external_agent"}
    return [r for r in ranking if r["agent_id"] in CORE_AGENT_IDS]


@insight_router.get("/reputation/{agent_id}", response_model=ReputationResponse)
def get_reputation(
    agent_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> ReputationResponse:
    from app.services.reputation_service import ReputationEngine
    engine = ReputationEngine()
    rep = engine.get_reputation(agent_id)
    if not rep:
        rep = engine.compute_score(agent_id)
    return ReputationResponse(**rep)


class PermissionSuggestionItem(BaseModel):
    agent_id: str
    name: str
    role: str
    accessed_resources: List[str]
    unused_resources: List[str]
    suggestion: str


class PermissionSuggestionsResponse(BaseModel):
    agents: List[PermissionSuggestionItem]


@insight_router.get("/permission-suggestions", response_model=PermissionSuggestionsResponse)
def permission_suggestions(
    context: AuthContext = Depends(get_auth_context),
) -> PermissionSuggestionsResponse:
    from app.db import SessionLocal
    from app.models import AuditLogRow
    from sqlalchemy import select

    agents = identity.list_agents()
    items = []

    with SessionLocal() as db:
        for a in agents:
            allowed = set(a.get("attributes", {}).get("allowed_resources", []))
            accessed_rows = db.execute(
                select(AuditLogRow.resource)
                .where(AuditLogRow.agent_id == a["agent_id"], AuditLogRow.decision == "allow")
                .distinct()
            ).scalars().all()
            accessed = set(accessed_rows)
            unused = sorted(allowed - accessed)

            suggestion = ""
            if len(unused) > 3:
                suggestion = f"建议回收 {len(unused)} 个未使用资源的访问权限，遵循最小权限原则"
            elif len(unused) > 0:
                suggestion = f"有 {len(unused)} 个资源从未被访问，可考虑收回权限"

            items.append(PermissionSuggestionItem(
                agent_id=a["agent_id"],
                name=a["name"],
                role=a["role"],
                accessed_resources=sorted(accessed & allowed)[:10],
                unused_resources=unused,
                suggestion=suggestion,
            ))

    return PermissionSuggestionsResponse(agents=items)


class HeatmapDayItem(BaseModel):
    date: str
    total: int
    allowed: int
    denied: int


class AccessHeatmapResponse(BaseModel):
    days: List[HeatmapDayItem]


@insight_router.get("/access-heatmap", response_model=AccessHeatmapResponse)
def access_heatmap(
    days: int = Query(default=30, ge=1, le=90),
    context: AuthContext = Depends(get_auth_context),
) -> AccessHeatmapResponse:
    from datetime import datetime, timedelta, timezone
    from app.db import SessionLocal
    from app.models import AuditLogRow
    from sqlalchemy import select, func, and_

    result = []
    now = datetime.now(timezone.utc)

    with SessionLocal() as db:
        for i in range(days):
            day_start = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            day_end = (now - timedelta(days=i - 1)).strftime("%Y-%m-%d")
            total = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(and_(AuditLogRow.created_at >= day_start, AuditLogRow.created_at < day_end))
            ).scalar() or 0
            allowed = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(and_(AuditLogRow.created_at >= day_start, AuditLogRow.created_at < day_end, AuditLogRow.decision == "allow"))
            ).scalar() or 0
            denied = db.execute(
                select(func.count()).select_from(AuditLogRow)
                .where(and_(AuditLogRow.created_at >= day_start, AuditLogRow.created_at < day_end, AuditLogRow.decision == "deny"))
            ).scalar() or 0
            result.append(HeatmapDayItem(date=day_start, total=total, allowed=allowed, denied=denied))

    return AccessHeatmapResponse(days=result)
