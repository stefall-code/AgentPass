from __future__ import annotations

import hashlib
import logging
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select, func, Integer

from app import database, identity
from app.schemas_prompt import (
    OpenClawCheckRequest,
    OpenClawCheckResponse,
)

logger = logging.getLogger("agent_system")
router = APIRouter()

_guard_ref = None


def set_guard(guard):
    global _guard_ref
    _guard_ref = guard


def _log_openclaw_audit(agent_id: str, user: str, action: str, resource: str,
                        decision: str, reason: str, risk_score: float = 0.0,
                        prompt_hash: str | None = None):
    try:
        from app import audit
        from app.db import SessionLocal
        from app.models import OpenClawRequest

        audit.log_event(
            action="openclaw_check",
            agent_id=agent_id,
            resource=resource,
            decision=decision,
            reason=reason,
            context={"user": user, "source": "openclaw", "risk_score": risk_score}
        )

        with SessionLocal() as db:
            db.add(OpenClawRequest(
                agent_id=agent_id,
                user=user,
                action=action,
                resource=resource,
                prompt_hash=prompt_hash,
                allowed=1 if decision == "allow" else 0,
                risk_score=risk_score,
                reason=reason,
                created_at=database.utc_now(),
            ))
            db.commit()
    except Exception as e:
        logger.error("Failed to write OpenClaw audit log: %s", e)


@router.post("/openclaw/check", response_model=OpenClawCheckResponse, tags=["OpenClaw"])
def openclaw_check(request: OpenClawCheckRequest) -> OpenClawCheckResponse:
    if not _guard_ref:
        raise HTTPException(status_code=503, detail="AgentPass Guard not available")

    agent_obj = identity.get_agent(request.agent_id)
    agent_role = agent_obj["role"] if agent_obj else "basic"
    context = {"user": request.user, "source": "openclaw", "role": agent_role}
    prompt_hash = None

    if request.prompt:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()
        prompt_result = _guard_ref.prompt_defense.analyze(request.prompt)

        from app.services.context_guard import ContextGuard
        cg = ContextGuard()
        leak_result = cg.scan_cross_agent_leak(request.prompt, request.agent_id)

        if not prompt_result.is_safe or leak_result["leaked"]:
            final_risk = max(prompt_result.risk_score, leak_result["risk_score"])
            _log_openclaw_audit(
                agent_id=request.agent_id,
                user=request.user,
                action=request.action,
                resource=request.resource,
                decision="deny",
                reason=f"Blocked: {'cross-agent info leak' if leak_result['leaked'] else 'prompt injection'}: {prompt_result.reason}",
                risk_score=final_risk,
                prompt_hash=prompt_hash,
            )
            return OpenClawCheckResponse(
                allowed=False,
                risk_score=final_risk,
                reason=f"Blocked: {'cross-agent info leak detected' if leak_result['leaked'] else prompt_result.reason}"
            )
        context["prompt_risk_score"] = prompt_result.risk_score

    try:
        check_result = _guard_ref.check_with_context(
            agent_id=request.agent_id,
            action=request.action,
            resource=request.resource,
            context=context
        )

        allowed = check_result.get("allowed", False)
        risk_score = check_result.get("risk_score", 0.0)
        reason = check_result.get("reason", "No reason provided")

        _log_openclaw_audit(
            agent_id=request.agent_id,
            user=request.user,
            action=request.action,
            resource=request.resource,
            decision="allow" if allowed else "deny",
            reason=reason,
            risk_score=risk_score,
            prompt_hash=prompt_hash,
        )

        return OpenClawCheckResponse(
            allowed=allowed,
            risk_score=risk_score,
            reason=reason
        )

    except Exception as e:
        logger.error("OpenClaw check error: %s", e)
        _log_openclaw_audit(
            agent_id=request.agent_id,
            user=request.user,
            action=request.action,
            resource=request.resource,
            decision="deny",
            reason=f"Internal error: {str(e)}",
            risk_score=1.0,
            prompt_hash=prompt_hash,
        )
        raise HTTPException(status_code=500, detail="Internal processing error")


@router.get("/openclaw/stats", tags=["OpenClaw"])
def openclaw_stats(days: int = Query(default=7, ge=1, le=90)) -> Dict[str, Any]:
    from datetime import datetime, timedelta, timezone
    from app.db import SessionLocal
    from app.models import OpenClawRequest

    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")

    with SessionLocal() as db:
        daily = db.execute(
            select(
                func.substr(OpenClawRequest.created_at, 1, 10).label("date"),
                func.count().label("total"),
                func.sum(func.cast(OpenClawRequest.allowed == 1, type_=Integer)).label("allowed"),
                func.sum(func.cast(OpenClawRequest.allowed == 0, type_=Integer)).label("denied"),
                func.avg(OpenClawRequest.risk_score).label("avg_risk"),
                func.sum(func.cast(OpenClawRequest.risk_score > 0.7, type_=Integer)).label("high_risk"),
            )
            .where(OpenClawRequest.created_at >= since)
            .group_by("date")
            .order_by("date")
        ).all()

    return {
        "days": days,
        "daily": [
            {
                "date": r[0],
                "total": r[1],
                "allowed": int(r[2] or 0),
                "denied": int(r[3] or 0),
                "avg_risk": round(float(r[4] or 0), 3),
                "high_risk": int(r[5] or 0),
            }
            for r in daily
        ],
    }


@router.get("/openclaw/export", tags=["OpenClaw"])
def openclaw_export(
    format: str = Query(default="csv", pattern="^(json|csv)$"),
    start: str = Query(default=""),
    end: str = Query(default=""),
):
    import csv
    import io
    import json as _json
    from app.db import SessionLocal
    from app.models import OpenClawRequest

    with SessionLocal() as db:
        q = select(OpenClawRequest).order_by(OpenClawRequest.id.desc())
        if start:
            q = q.where(OpenClawRequest.created_at >= start)
        if end:
            q = q.where(OpenClawRequest.created_at <= end)
        rows = db.execute(q.limit(5000)).scalars().all()

    records = [
        {
            "id": r.id,
            "agent_id": r.agent_id,
            "user": r.user,
            "action": r.action,
            "resource": r.resource,
            "prompt_hash": r.prompt_hash,
            "allowed": bool(r.allowed),
            "risk_score": r.risk_score,
            "reason": r.reason,
            "created_at": r.created_at,
        }
        for r in rows
    ]

    if format == "csv":
        output = io.StringIO()
        if records:
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=openclaw_requests.csv"},
        )

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(_json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=openclaw_requests.json"},
    )


@router.get("/openclaw/high-risk", tags=["OpenClaw"])
def openclaw_high_risk(limit: int = Query(default=50, ge=1, le=200)) -> List[Dict[str, Any]]:
    from app.db import SessionLocal
    from app.models import OpenClawRequest

    with SessionLocal() as db:
        rows = db.execute(
            select(OpenClawRequest)
            .where(OpenClawRequest.risk_score > 0.7)
            .order_by(OpenClawRequest.id.desc())
            .limit(limit)
        ).scalars().all()

    return [
        {
            "id": r.id,
            "agent_id": r.agent_id,
            "user": r.user,
            "action": r.action,
            "resource": r.resource,
            "risk_score": r.risk_score,
            "reason": r.reason,
            "created_at": r.created_at,
        }
        for r in rows
    ]
