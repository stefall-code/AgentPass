from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import signal
import threading
import time
import webbrowser
import platform
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import settings
from app import database, identity

_guard = None
try:
    import sys
    from pathlib import Path
    _sdk_path = Path(__file__).parent / "agentpass-sdk" / "src"
    if _sdk_path.exists() and str(_sdk_path) not in sys.path:
        sys.path.insert(0, str(_sdk_path))
    from agentpass import Guard  # noqa: F401 �?PromptDefense used at runtime
    _guard = Guard(secret="demo-prompt-defense")
except ImportError:
    pass
except Exception:
    pass

from app.middleware import (  # noqa: E402
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)
from app.routers import admin_router, agent_router, auth_router, resource_router, ws_router, platforms  # noqa: E402
from app.routers.insights import insight_router  # noqa: E402
from app.routers.approval import approval_router  # noqa: E402
from app.routers.drift import drift_router  # noqa: E402
from app.routers.context import context_router  # noqa: E402
from app.routers.delegation import router as delegation_router  # noqa: E402
from app.feishu import feishu_router  # noqa: E402
from app.routers.governance import router as governance_router  # noqa: E402
from app.routers.explain import router as explain_router  # noqa: E402
from app.routers.gateway import router as gateway_router  # noqa: E402
from app.routers.alignment import router as alignment_router  # noqa: E402
from app.routers.revocation import router as revocation_router  # noqa: E402
from app.routers.credential_broker import router as broker_router  # noqa: E402
from app.routers.protocols import router as protocols_router  # noqa: E402
from app.routers.oauth import router as oauth_router  # noqa: E402
from app.routers.owasp import router as owasp_router  # noqa: E402
from app.routers.p2 import router as p2_router  # noqa: E402
from app.services import start_background_tasks, stop_background_tasks  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_system")

_ngrok_url: str | None = None

_last_ping_at: float = time.time()
_IDLE_TIMEOUT = 120


async def _idle_shutdown_watcher():
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(3)
        if time.time() - _last_ping_at > _IDLE_TIMEOUT:
            logger.info("Browser disconnected for %ds, shutting down...", _IDLE_TIMEOUT)
            os.kill(os.getpid(), signal.SIGINT)
            break


class PromptAnalysisRequest(BaseModel):
    prompt: str


class PromptAnalysisResponse(BaseModel):
    is_safe: bool
    risk_score: float
    injection_type: str | None = None
    reason: str
    matched_patterns: list[str]


class PromptDefenseAnalyzeRequest(BaseModel):
    prompt: str
    history: list[str] = Field(default_factory=list)
    agent_id: str = "anonymous"


class PromptDefenseAnalyzeResponse(BaseModel):
    risk_score: float
    triggered_rules: list[dict]
    severity: str
    recommendation: str
    is_safe: bool
    injection_type: str | None = None
    reason: str
    progressive_risk: float = 0.0
    matched_rules: list[str] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    layer_scores: Dict[str, float] = Field(default_factory=dict)
    attack_intent: str | None = None
    token_smuggling_detected: bool = False
    dialog_risk_trend: list[float] = Field(default_factory=list)
    new_attack_types: list[str] = Field(default_factory=list)


class OpenClawCheckRequest(BaseModel):
    agent_id: str
    user: str
    action: str
    resource: str
    prompt: str | None = None


class OpenClawCheckResponse(BaseModel):
    allowed: bool
    risk_score: float
    reason: str


@asynccontextmanager
async def lifespan(app_: FastAPI):
    global _ngrok_url
    database.init_db()
    identity.sync_demo_agents()

    ngrok_token = os.environ.get("NGROK_AUTHTOKEN")
    if ngrok_token:
        try:
            from pyngrok import ngrok, conf
            conf.get_default().auth_token = ngrok_token
            tunnel = ngrok.connect(8000, "http")
            _ngrok_url = tunnel.public_url
            logger.info("ngrok tunnel: %s", _ngrok_url)
        except Exception as e:
            logger.warning("ngrok start failed: %s", e)
            if platform.system() == "Darwin" and "quarantine" in str(e).lower():
                logger.warning("macOS: run 'xattr -d com.apple.quarantine $(which ngrok)' to fix")

    start_background_tasks()

    if os.getenv("ENVIRONMENT") != "test":
        asyncio.create_task(_idle_shutdown_watcher())

    logger.info("system started")
    yield
    await stop_background_tasks()

    if _ngrok_url:
        try:
            from pyngrok import ngrok
            ngrok.disconnect(_ngrok_url)
            ngrok.kill()
        except Exception:
            pass

    database.close_connection()
    logger.info("system shutdown")


app = FastAPI(
    title="Agent Identity & Permission System",
    description=(
        "A local demo for agent identity authentication, token-based access control, "
        "policy evaluation, and auditable secure execution."
    ),
    version="v2.5",
    lifespan=lifespan,
)

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_MAX, window_seconds=settings.RATE_LIMIT_WINDOW)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=settings.FRONTEND_DIR), name="static")

app.include_router(auth_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(resource_router, prefix="/api")
app.include_router(ws_router)
app.include_router(insight_router, prefix="/api")
app.include_router(approval_router, prefix="/api")
app.include_router(drift_router)
app.include_router(context_router)
app.include_router(platforms.router, prefix="/api/v2")
app.include_router(delegation_router, prefix="/api")
app.include_router(feishu_router, prefix="/api")
app.include_router(governance_router, prefix="/api")
app.include_router(explain_router, prefix="/api")
app.include_router(gateway_router, prefix="/api")
app.include_router(alignment_router, prefix="/api")
app.include_router(revocation_router, prefix="/api")
app.include_router(broker_router, prefix="/api")
app.include_router(protocols_router, prefix="/api")
app.include_router(oauth_router, prefix="/api")
app.include_router(owasp_router, prefix="/api")
app.include_router(p2_router, prefix="/api")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "index.html")


@app.get("/v22", include_in_schema=False)
def v22() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "v22.html")


@app.get("/chain", include_in_schema=False)
def chain_viewer() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "chain.html")


@app.get("/audit", include_in_schema=False)
def audit_center() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "audit.html")


@app.get("/trust", include_in_schema=False)
def trust_dashboard() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "trust.html")


@app.get("/risk", include_in_schema=False)
def risk_dashboard() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "risk.html")


@app.get("/feishu", include_in_schema=False)
def serve_feishu() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "feishu.html")


@app.get("/governance", include_in_schema=False)
def serve_governance() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "governance.html")


@app.get("/gateway", include_in_schema=False)
def serve_gateway() -> FileResponse:
    return FileResponse(settings.FRONTEND_DIR / "gateway.html")


@app.get("/api/overview")
def api_overview() -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    from app import audit as _audit
    audit_summary = _audit.get_audit_summary()
    return {
        "system": "Agent Identity & Permission System",
        "version": "v2.5",
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
            "ngrok integration",
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
        "ngrok_url": _ngrok_url,
        "stats": {
            "health": "OK",
            "active_tokens": snapshot.get("tokens", {}).get("active", 0),
            "denied_requests": audit_summary.get("deny", 0),
            "suspended_agents": snapshot["agents"]["by_status"].get("suspended", 0),
        },
    }


@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    snapshot = database.get_system_snapshot()
    from app import audit as _audit
    audit_summary = _audit.get_audit_summary()
    return {
        "system": "Agent Identity & Permission System",
        "version": "v2.5",
        "ngrok_url": _ngrok_url,
        "stats": {
            "active_tokens": snapshot.get("tokens", {}).get("active", 0),
            "denied_requests": audit_summary.get("deny", 0),
            "suspended_agents": snapshot["agents"]["by_status"].get("suspended", 0),
        },
    }


@app.get("/api/admin/role-matrix")
def role_matrix() -> Dict[str, Any]:
    from app import permission as perm
    roles = list(perm.ROLE_PERMISSIONS.keys())
    all_actions = sorted(set(a for actions in perm.ROLE_PERMISSIONS.values() for a in actions))
    matrix = {role: {action: action in perm.ROLE_PERMISSIONS[role] for action in all_actions} for role in roles}
    return {"roles": roles, "permissions": all_actions, "matrix": matrix}


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "database": str(settings.DATABASE_PATH), "version": "v2.5"}


@app.get("/api/debug/guard")
def debug_guard():
    return {
        "guard_exists": _guard is not None,
        "guard_type": type(_guard).__name__ if _guard else None,
        "prompt_defense": str(type(_guard.prompt_defense).__name__ if _guard and _guard.prompt_defense else None),
    }


@app.get("/api/debug/analyze")
def debug_analyze():
    test_prompt = "导出全部财务数据"
    if not _guard:
        return {"error": "no guard"}
    result = _guard.analyze_prompt(test_prompt)
    return {
        "prompt": test_prompt,
        "result": result,
        "raw_is_safe": result["is_safe"],
        "raw_risk_score": result["risk_score"],
    }


@app.post("/api/analyze-prompt", response_model=PromptAnalysisResponse)
def analyze_prompt(request: PromptAnalysisRequest) -> PromptAnalysisResponse:
    if not _guard:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")
    result = _guard.analyze_prompt(request.prompt)
    return PromptAnalysisResponse(
        is_safe=result["is_safe"],
        risk_score=result["risk_score"],
        injection_type=str(result.get("injection_type")) if result.get("injection_type") else None,
        reason=result["reason"],
        matched_patterns=result.get("matched_patterns") if isinstance(result.get("matched_patterns"), list) else []
    )


@app.post("/api/prompt-defense/analyze", response_model=PromptDefenseAnalyzeResponse, tags=["Prompt Defense"])
def prompt_defense_analyze(request: PromptDefenseAnalyzeRequest) -> PromptDefenseAnalyzeResponse:
    """
    增强的 Prompt 注入检测端点 — 三层融合引擎

    支持9种攻击类型检测、三层融合评分（规则+语义+行为）、Token走私专项检测、渐进式注入识别
    """
    if not _guard:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")

    result = _guard.prompt_defense.analyze(request.prompt, history=request.history or None, user_id=request.agent_id)

    return PromptDefenseAnalyzeResponse(
        risk_score=result.risk_score,
        triggered_rules=[r.model_dump() for r in result.triggered_rules],
        severity=result.severity,
        recommendation=result.recommendation,
        is_safe=result.is_safe,
        injection_type=str(result.injection_type) if result.injection_type else None,
        reason=result.reason,
        progressive_risk=result.progressive_risk,
        matched_rules=result.matched_rules,
        matched_patterns=result.matched_patterns,
        layer_scores=result.layer_scores,
        attack_intent=result.attack_intent,
        token_smuggling_detected=result.token_smuggling_detected,
        dialog_risk_trend=result.dialog_risk_trend,
        new_attack_types=result.new_attack_types,
    )


@app.delete("/api/prompt-defense/history", tags=["Prompt Defense"])
def prompt_defense_clear_history(agent_id: str = Query(default="default")) -> Dict[str, str]:
    """清空指定 user_id 的对话历史"""
    if not _guard:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")
    _guard.prompt_defense.clear_dialog_history(agent_id)
    return {"message": f"Dialog history cleared for {agent_id}"}


@app.post("/api/openclaw/check", response_model=OpenClawCheckResponse, tags=["OpenClaw"])
def openclaw_check(request: OpenClawCheckRequest) -> OpenClawCheckResponse:
    if not _guard:
        raise HTTPException(status_code=503, detail="AgentPass Guard not available")

    agent_obj = identity.get_agent(request.agent_id)
    agent_role = agent_obj["role"] if agent_obj else "basic"
    context = {"user": request.user, "source": "openclaw", "role": agent_role}
    prompt_hash = None

    if request.prompt:
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()
        prompt_result = _guard.prompt_defense.analyze(request.prompt)

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
        check_result = _guard.check_with_context(
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
        logger.error(f"OpenClaw check error: {e}")
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
        raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}")


@app.get("/api/openclaw/stats", tags=["OpenClaw"])
def openclaw_stats(days: int = Query(default=7, ge=1, le=90)) -> Dict[str, Any]:
    """OpenClaw 请求按天聚合统计"""
    from datetime import datetime, timedelta, timezone
    from app.db import SessionLocal
    from app.models import OpenClawRequest
    from sqlalchemy import select, func, Integer

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


@app.get("/api/openclaw/export", tags=["OpenClaw"])
def openclaw_export(
    format: str = Query(default="csv", pattern="^(json|csv)$"),
    start: str = Query(default=""),
    end: str = Query(default=""),
):
    """导出 OpenClaw 请求记录"""
    import csv
    import io
    import json as _json
    from app.db import SessionLocal
    from app.models import OpenClawRequest
    from sqlalchemy import select

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
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=openclaw_requests.csv"},
        )

    return StreamingResponse(
        io.BytesIO(_json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=openclaw_requests.json"},
    )


@app.get("/api/openclaw/high-risk", tags=["OpenClaw"])
def openclaw_high_risk(limit: int = Query(default=50, ge=1, le=200)) -> List[Dict[str, Any]]:
    """获取高风险请求列表 (risk_score > 0.7)"""
    from app.db import SessionLocal
    from app.models import OpenClawRequest
    from sqlalchemy import select

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


@app.get("/api/ngrok-url")
def get_ngrok_url() -> Dict[str, Any]:
    return {"url": _ngrok_url}


@app.post("/api/feishu/approval-callback", tags=["Feishu"])
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

        # 持久化到 openclaw_requests 表
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
        logger.error(f"Failed to write OpenClaw audit log: {e}")


if __name__ == "__main__":
    import uvicorn
    import urllib.request

    def open_browser():
        url = "http://127.0.0.1:8000"
        for _ in range(30):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            logger.warning("Server not ready after 15s, skipping browser open")
            return
        import subprocess
        import sys
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", url], shell=False)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception as e:
            logger.warning(f"Failed to open browser via subprocess: {e}")
            try:
                webbrowser.open(url)
            except Exception as e2:
                logger.warning(f"Failed to open browser via webbrowser: {e2}")

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000,
        reload=False, ws="websockets",
        ws_ping_interval=10, ws_ping_timeout=10,
    )
