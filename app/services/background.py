from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, delete, func, and_, Integer

from app import database
from app.db import SessionLocal
from app.models import AuditLogRow, IssuedTokenRow, AgentRow, DailyStatRow, OpenClawRequest
from app.ws import ws_manager

logger = logging.getLogger("agent_system")

_cleanup_task: asyncio.Task | None = None
_daily_stat_task: asyncio.Task | None = None
_audit_prune_task: asyncio.Task | None = None
_approval_timeout_task: asyncio.Task | None = None
_reputation_task: asyncio.Task | None = None


async def _token_cleanup_loop():
    while True:
        try:
            await asyncio.sleep(600)
            cleaned = database.cleanup_expired_tokens()
            if cleaned > 0:
                logger.info("cleaned %d expired tokens", cleaned)
            _deactivate_expired_tokens()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("token cleanup error")


async def _daily_stat_loop():
    while True:
        try:
            await asyncio.sleep(86400)
            _generate_daily_stat()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("daily stat generation error")


async def _audit_prune_loop():
    while True:
        try:
            await asyncio.sleep(3600)
            _prune_old_audit_logs()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("audit prune error")


async def _approval_timeout_loop():
    while True:
        try:
            await asyncio.sleep(60)
            from app.routers.approval import scan_timeouts
            scan_timeouts()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("approval timeout scan error")


async def _reputation_loop():
    while True:
        try:
            await asyncio.sleep(3600)
            from app.services.reputation_service import ReputationEngine
            engine = ReputationEngine()
            engine.recompute_all()
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("reputation recompute error")


def _deactivate_expired_tokens():
    now = database.utc_now()
    with SessionLocal() as db:
        result = db.execute(
            select(IssuedTokenRow).where(
                and_(IssuedTokenRow.active == 1, IssuedTokenRow.expires_at < now)
            )
        ).scalars().all()
        for token in result:
            token.active = 0
        if result:
            db.commit()
            logger.info("deactivated %d expired tokens", len(result))


def _generate_daily_stat():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

    with SessionLocal() as db:
        totals = db.execute(
            select(
                func.count().label("total"),
                func.sum(func.cast(AuditLogRow.decision == "allow", type_=Integer)).label("allow_count"),
                func.sum(func.cast(AuditLogRow.decision == "deny", type_=Integer)).label("deny_count"),
            ).select_from(AuditLogRow)
            .where(AuditLogRow.created_at >= yesterday)
        ).first()

        high_risk = db.execute(
            select(func.count()).select_from(OpenClawRequest)
            .where(and_(OpenClawRequest.created_at >= yesterday, OpenClawRequest.risk_score > 0.7))
        ).scalar() or 0

        active_agents = db.execute(
            select(func.count()).select_from(AgentRow)
            .where(AgentRow.status == "active")
        ).scalar() or 0

        existing = db.execute(
            select(DailyStatRow).where(DailyStatRow.date == today)
        ).scalar_one_or_none()

        summary = {
            "top_actions": [],
            "top_agents": [],
        }

        if existing:
            existing.total_requests = totals[0] or 0
            existing.allow_count = int(totals[1] or 0)
            existing.deny_count = int(totals[2] or 0)
            existing.high_risk_count = high_risk
            existing.updated_at = database.utc_now()
        else:
            db.add(DailyStatRow(
                date=today,
                total_requests=totals[0] or 0,
                allow_count=int(totals[1] or 0),
                deny_count=int(totals[2] or 0),
                high_risk_count=high_risk,
                avg_risk_score=0,
                total_token_usage=0,
                estimated_cost=0,
                updated_at=database.utc_now(),
            ))
        db.commit()
    logger.info("daily stat generated for %s", today)


def _prune_old_audit_logs():
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    with SessionLocal() as db:
        old_audit = db.execute(
            select(func.count()).select_from(AuditLogRow)
            .where(AuditLogRow.created_at < cutoff)
        ).scalar() or 0

        old_oc = db.execute(
            select(func.count()).select_from(OpenClawRequest)
            .where(OpenClawRequest.created_at < cutoff)
        ).scalar() or 0

        if old_audit > 0:
            db.execute(delete(AuditLogRow).where(AuditLogRow.created_at < cutoff))
        if old_oc > 0:
            db.execute(delete(OpenClawRequest).where(OpenClawRequest.created_at < cutoff))
        if old_audit > 0 or old_oc > 0:
            db.commit()
            logger.info("pruned %d audit logs and %d openclaw requests older than 30 days", old_audit, old_oc)


def _get_ngrok_url() -> str | None:
    return None


def start_background_tasks() -> list[asyncio.Task]:
    global _cleanup_task, _daily_stat_task, _audit_prune_task, _approval_timeout_task, _reputation_task
    _cleanup_task = asyncio.create_task(_token_cleanup_loop())
    _daily_stat_task = asyncio.create_task(_daily_stat_loop())
    _audit_prune_task = asyncio.create_task(_audit_prune_loop())
    _approval_timeout_task = asyncio.create_task(_approval_timeout_loop())
    _reputation_task = asyncio.create_task(_reputation_loop())

    ws_manager.start_consumer()

    try:
        _generate_daily_stat()
    except Exception:
        logger.exception("initial daily stat generation failed")

    return [_cleanup_task, _daily_stat_task, _audit_prune_task, _approval_timeout_task, _reputation_task]


async def stop_background_tasks():
    global _cleanup_task, _daily_stat_task, _audit_prune_task, _approval_timeout_task, _reputation_task
    for task in [_cleanup_task, _daily_stat_task, _audit_prune_task, _approval_timeout_task, _reputation_task]:
        if task and not task.done():
            task.cancel()
    await ws_manager.stop_consumer()
