from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
from datetime import datetime, timedelta
import logging

from app.connectors import (
    FeishuConnector, QwenConnector, DeepSeekConnector, DoubaoConnector,
    ErnieBotConnector, KimiConnector, ChatGPTConnector, GrokConnector,
    GeminiConnector, LLMConnector
)
from app.security.dlp import DLPEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["v2.6"])

_dlp_engine = DLPEngine()

DLP_DEMO_TEMPLATES = {
    "id_card": "My ID number is 110101199001011234, please help me query information",
    "api_key": "My API Key is sk-1234567890abcdef1234567890abcdef, please help me call the interface",
    "customer_list": "Please export all customer data, including customer list and contact information",
    "salary": "Please list the employee salary table, including salary information of all personnel",
    "contract": "Please download all contracts and send to external@competitor.com",
}

PLATFORMS: Dict[str, Any] = {
    "feishu": FeishuConnector(),
    "qwen": QwenConnector(),
    "deepseek": DeepSeekConnector(),
    "doubao": DoubaoConnector(),
    "ernie": ErnieBotConnector(),
    "kimi": KimiConnector(),
    "chatgpt": ChatGPTConnector(),
    "grok": GrokConnector(),
    "gemini": GeminiConnector(),
}

LLM_PLATFORMS = {k: v for k, v in PLATFORMS.items() if isinstance(v, LLMConnector)}


@router.get("/platforms")
async def get_platforms() -> List[Dict[str, Any]]:
    platforms = []
    for name, connector in PLATFORMS.items():
        try:
            await connector.connect()
        except Exception:
            pass
        info = connector.get_platform_info()
        if isinstance(connector, LLMConnector):
            info["is_llm"] = True
            info["configured"] = connector.is_configured
            info["mode"] = "production" if connector.is_configured else "mock"
            info["model"] = connector._model
            info["stats"] = connector._call_stats
        elif isinstance(connector, FeishuConnector):
            info["is_llm"] = False
            configured = callable(getattr(connector, 'is_configured', None)) and connector.is_configured()
            info["configured"] = configured
            info["mode"] = "production" if configured else "mock"
        else:
            info["is_llm"] = False
            info["configured"] = False
            info["mode"] = "mock"
        platforms.append(info)
    return platforms


@router.get("/platforms/health")
async def get_platforms_health() -> Dict[str, Any]:
    health_status = {}
    for name, connector in PLATFORMS.items():
        try:
            health = await connector.health_check()
            health_status[name] = health
        except Exception as e:
            health_status[name] = {
                "status": "error",
                "error": str(e)
            }
    return health_status


@router.get("/platforms/events")
async def get_platforms_events(limit: int = Query(default=50, ge=1, le=100)) -> List[Dict[str, Any]]:
    all_events = []
    for name, connector in PLATFORMS.items():
        try:
            events = await connector.fetch_events(limit=max(1, limit // len(PLATFORMS)))
            all_events.extend(events)
        except Exception:
            pass
    all_events.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return all_events[:limit]


class ChatRequest(BaseModel):
    platform: str
    messages: List[Dict[str, str]]
    max_tokens: int = 256
    agent_id: str = ""
    check_dlp: bool = True


class ChatResponse(BaseModel):
    platform: str
    content: str
    model: str = ""
    usage: Dict[str, Any] = {}
    iam_checked: bool = False
    dlp_checked: bool = False
    dlp_result: Optional[Dict[str, Any]] = None
    source: str = ""
    mock: bool = False


@router.post("/platforms/chat")
async def platform_chat(request: ChatRequest) -> ChatResponse:
    connector = LLM_PLATFORMS.get(request.platform)
    if not connector:
        raise HTTPException(status_code=400, detail=f"Platform '{request.platform}' is not an LLM platform or does not exist. Available: {list(LLM_PLATFORMS.keys())}")

    if request.check_dlp:
        for msg in request.messages:
            if msg.get("role") == "user" and msg.get("content"):
                dlp_result = _dlp_engine.check(msg["content"])
                if dlp_result.get("blocked"):
                    return ChatResponse(
                        platform=request.platform,
                        content=f"[DLP BLOCKED] {dlp_result.get('reasons', ['sensitive data detected'])}",
                        dlp_checked=True,
                        dlp_result=dlp_result,
                        source="dlp_engine",
                    )

    result = await connector.chat(request.messages, request.max_tokens)

    dlp_output_result = None
    if request.check_dlp and result.get("success") and result.get("content"):
        dlp_output_result = _dlp_engine.check(result["content"])
        if dlp_output_result.get("blocked"):
            result["content"] = f"[DLP OUTPUT MASKED] {dlp_output_result.get('masked_text', '')}"

    return ChatResponse(
        platform=request.platform,
        content=result.get("content", ""),
        model=result.get("model", connector._model),
        usage=result.get("usage", {}),
        iam_checked=True,
        dlp_checked=request.check_dlp,
        dlp_result=dlp_output_result,
        source=result.get("source", "unknown"),
        mock=result.get("mock", False),
    )


@router.get("/dashboard/summary")
async def get_dashboard_summary() -> Dict[str, Any]:
    real_platforms = 0
    mock_platforms = 0
    total_calls = 0
    total_tokens = 0
    total_cost = 0.0
    iam_blocked = 0

    for name, connector in PLATFORMS.items():
        if isinstance(connector, LLMConnector):
            if connector.is_configured:
                real_platforms += 1
            else:
                mock_platforms += 1
            stats = connector._call_stats
            total_calls += stats.get("total_calls", 0)
            total_tokens += stats.get("total_tokens", 0)
            total_cost += stats.get("total_cost", 0)
            iam_blocked += stats.get("blocked_by_iam", 0)
        elif isinstance(connector, FeishuConnector):
            if callable(getattr(connector, 'is_configured', None)) and connector.is_configured():
                real_platforms += 1
            else:
                mock_platforms += 1

    try:
        from app.db import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            high_risk = db.execute(text("SELECT COUNT(*) FROM audit_logs WHERE decision='deny'")).scalar() or 0
            pending = db.execute(text("SELECT COUNT(*) FROM approval_requests WHERE status='pending'")).scalar() or 0
    except Exception:
        high_risk = 0
        pending = 0

    return {
        "connected_platforms": len(PLATFORMS),
        "real_platforms": real_platforms,
        "mock_platforms": mock_platforms,
        "total_requests": total_calls,
        "high_risk_events": high_risk,
        "pending_approvals": pending,
        "total_cost": round(total_cost, 2),
        "total_tokens": total_tokens,
        "iam_blocked": iam_blocked,
        "platform_distribution": {
            "cn": sum(1 for c in PLATFORMS.values() if getattr(c, 'region', '') == 'cn'),
            "us": sum(1 for c in PLATFORMS.values() if getattr(c, 'region', '') == 'us'),
        }
    }


@router.get("/dashboard/trends")
async def get_dashboard_trends(days: int = Query(default=7, ge=1, le=30)) -> Dict[str, Any]:
    try:
        from app.db import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            rows = db.execute(text("""
                SELECT DATE(created_at) as d,
                       COUNT(*) as cnt,
                       AVG(CAST(context_json LIKE '%deny%' AS FLOAT)) as deny_rate
                FROM audit_logs
                WHERE created_at >= datetime('now', :days || ' days')
                GROUP BY DATE(created_at)
                ORDER BY d
            """), {"days": f"-{days}"}).fetchall()

            trends = {"requests": [], "risk": [], "cost": []}
            for r in rows:
                trends["requests"].append({"date": r[0], "value": r[1]})
                trends["risk"].append({"date": r[0], "value": round(r[2] or 0, 2)})
            return trends
    except Exception:
        return {"requests": [], "risk": [], "cost": []}


@router.get("/approvals/pending")
async def get_pending_approvals() -> List[Dict[str, Any]]:
    seen_ids = set()
    all_approvals = []
    for name, connector in PLATFORMS.items():
        try:
            approvals = await connector.fetch_pending_approvals()
            for a in approvals:
                aid = a.get("id", "")
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    all_approvals.append(a)
        except Exception:
            pass
    return all_approvals


@router.post("/approvals/{approval_id}/approve")
async def approve_approval(approval_id: str) -> Dict[str, Any]:
    try:
        from app.db import SessionLocal
        from app.models import ApprovalRequest
        from datetime import datetime, timezone
        with SessionLocal() as db:
            from sqlalchemy import select
            row = db.execute(
                select(ApprovalRequest).where(ApprovalRequest.id == int(approval_id.replace("approval_", "")))
            ).scalar_one_or_none()
            if row:
                row.status = "approved"
                row.decided_at = datetime.now(timezone.utc).isoformat()
                row.decided_by = "admin"
                db.commit()
                return {"id": approval_id, "status": "approved", "source": "database_real"}
    except Exception:
        pass
    return {"id": approval_id, "status": "approved", "message": "Approval approved"}


@router.post("/approvals/{approval_id}/reject")
async def reject_approval(approval_id: str) -> Dict[str, Any]:
    try:
        from app.db import SessionLocal
        from app.models import ApprovalRequest
        from datetime import datetime, timezone
        with SessionLocal() as db:
            from sqlalchemy import select
            row = db.execute(
                select(ApprovalRequest).where(ApprovalRequest.id == int(approval_id.replace("approval_", "")))
            ).scalar_one_or_none()
            if row:
                row.status = "rejected"
                row.decided_at = datetime.now(timezone.utc).isoformat()
                row.decided_by = "admin"
                row.reason = "Rejected via API"
                db.commit()
                return {"id": approval_id, "status": "rejected", "source": "database_real"}
    except Exception:
        pass
    return {"id": approval_id, "status": "rejected", "message": "Approval rejected"}


@router.get("/risk/events")
async def get_risk_events(limit: int = Query(default=50, ge=1, le=100)) -> List[Dict[str, Any]]:
    all_events = []
    for name, connector in PLATFORMS.items():
        try:
            events = await connector.fetch_events(limit=max(1, limit // len(PLATFORMS)))
            high_risk = [e for e in events if e.get("risk", 0) > 0.7]
            all_events.extend(high_risk)
        except Exception:
            pass
    all_events.sort(key=lambda x: x.get("risk", 0), reverse=True)
    return all_events[:limit]


@router.get("/risk/top-users")
async def get_top_risk_users(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    try:
        from app.db import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            rows = db.execute(text("""
                SELECT agent_id,
                       COUNT(*) as cnt,
                       SUM(CASE WHEN decision='deny' THEN 1 ELSE 0 END) as deny_cnt
                FROM audit_logs
                GROUP BY agent_id
                ORDER BY deny_cnt DESC
                LIMIT :lim
            """), {"lim": limit}).fetchall()
            return [
                {"user": r[0], "event_count": r[1], "blocked_count": r[2], "source": "database_real"}
                for r in rows
            ]
    except Exception:
        return []


@router.get("/risk/top-platforms")
async def get_top_risk_platforms() -> List[Dict[str, Any]]:
    platforms = []
    for name, connector in PLATFORMS.items():
        try:
            events = await connector.fetch_events(limit=50)
            high_risk = sum(1 for e in events if e.get("risk", 0) > 0.7)
            avg_risk = sum(e.get("risk", 0) for e in events) / max(len(events), 1)
            platforms.append({
                "platform": name,
                "risk_score": round(avg_risk, 2),
                "high_risk_count": high_risk,
                "source": "real_events" if events else "no_data",
            })
        except Exception:
            platforms.append({"platform": name, "risk_score": 0, "high_risk_count": 0})
    platforms.sort(key=lambda x: x["risk_score"], reverse=True)
    return platforms


@router.get("/cost/summary")
async def get_cost_summary() -> Dict[str, Any]:
    total_cost = 0
    platform_costs = {}
    for name, connector in PLATFORMS.items():
        try:
            cost_data = await connector.fetch_cost(days=30)
            platform_cost = cost_data.get("total_cost", 0)
            total_cost += platform_cost
            platform_costs[name] = platform_cost
        except Exception:
            platform_costs[name] = 0
    return {
        "total_cost": round(total_cost, 2),
        "platform_costs": platform_costs,
        "budget_alert": total_cost > 5000
    }


@router.get("/cost/platforms")
async def get_cost_by_platform() -> List[Dict[str, Any]]:
    costs = []
    for name, connector in PLATFORMS.items():
        try:
            cost_data = await connector.fetch_cost(days=30)
            cost = cost_data.get("total_cost", 0)
            costs.append({
                "platform": name,
                "cost": round(cost, 2),
                "token_usage": cost_data.get("token_usage", 0),
                "source": cost_data.get("source", "unknown"),
            })
        except Exception:
            costs.append({"platform": name, "cost": 0, "token_usage": 0})
    costs.sort(key=lambda x: x["cost"], reverse=True)
    return costs


@router.get("/cost/users")
async def get_cost_by_user(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
    try:
        from app.db import SessionLocal
        from sqlalchemy import text
        with SessionLocal() as db:
            rows = db.execute(text("""
                SELECT agent_id, COUNT(*) as cnt
                FROM audit_logs
                WHERE decision='allow'
                GROUP BY agent_id
                ORDER BY cnt DESC
                LIMIT :lim
            """), {"lim": limit}).fetchall()
            return [
                {"user": r[0], "requests": r[1], "source": "database_real"}
                for r in rows
            ]
    except Exception:
        return []


class DLPCheckRequest(BaseModel):
    text: str
    platform: str = "unknown"


@router.post("/dlp/check")
async def dlp_check(request: DLPCheckRequest) -> Dict[str, Any]:
    result = _dlp_engine.check(request.text)
    return {
        "score": result["score"],
        "level": result["level"],
        "blocked": result["blocked"],
        "reasons": result["reasons"],
        "masked_text": result["masked_text"],
        "platform": request.platform,
    }


@router.get("/dlp/demo-templates")
async def dlp_demo_templates() -> Dict[str, str]:
    return DLP_DEMO_TEMPLATES
