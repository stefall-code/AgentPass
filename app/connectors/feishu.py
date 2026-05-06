import logging
import time
from typing import List, Dict, Any
from datetime import datetime

from .base import BaseConnector

logger = logging.getLogger(__name__)


class FeishuConnector(BaseConnector):

    def __init__(self):
        super().__init__("feishu", "cn", mock=False)
        self._client = None
        self._connected = False

    def is_configured(self) -> bool:
        client = self._get_client()
        return client is not None and client.is_configured()

    def _get_client(self):
        if self._client is None:
            try:
                from app.feishu.client import get_feishu_client
                self._client = get_feishu_client()
            except Exception as e:
                logger.warning("FeishuConnector: cannot get FeishuClient: %s", e)
                self._client = None
        return self._client

    async def connect(self) -> bool:
        client = self._get_client()
        if client and client.is_configured():
            try:
                token = await client.get_tenant_access_token()
                self._connected = bool(token)
            except Exception as e:
                logger.warning("FeishuConnector connect failed: %s", e)
                self._connected = False
        else:
            self._connected = False
        self.connected = self._connected
        return self._connected

    async def fetch_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        client = self._get_client()
        if not client or not client.is_configured():
            return self._generate_fallback_events(limit)

        try:
            from app.feishu.iam_gateway import get_audit_log
            audit_records = get_audit_log(limit=limit)
            events = []
            for r in audit_records:
                risk_score = r.get("risk_score") or 0.0
                risk_level = "low"
                if risk_score >= 0.9:
                    risk_level = "critical"
                elif risk_score >= 0.7:
                    risk_level = "high"
                elif risk_score >= 0.3:
                    risk_level = "medium"

                events.append({
                    "id": f"feishu_iam_{r.get('timestamp', time.time()):.0f}",
                    "timestamp": datetime.fromtimestamp(r.get("timestamp", time.time())).isoformat(),
                    "platform": "feishu",
                    "region": "cn",
                    "user": r.get("agent_id", "unknown"),
                    "team": "iam",
                    "action": r.get("action", "unknown"),
                    "resource": r.get("path", ""),
                    "prompt": "",
                    "output": r.get("reason", ""),
                    "risk": risk_score,
                    "risk_level": risk_level,
                    "approval_required": risk_score > 0.7,
                    "approval_status": "pending" if risk_score > 0.7 else "none",
                    "cost": 0,
                    "token_usage": 0,
                    "blocked": r.get("decision") == "deny",
                    "reason": r.get("reason", ""),
                    "latency_ms": r.get("latency_ms", 0),
                    "trust_score": r.get("trust_score"),
                    "auto_revoked": r.get("auto_revoked", False),
                    "blocked_at": r.get("blocked_at", ""),
                    "six_layer": r.get("six_layer"),
                    "source": "iam_gateway_real",
                })
            return events
        except Exception as e:
            logger.error("FeishuConnector fetch_events error: %s", e)
            return self._generate_fallback_events(limit)

    async def fetch_cost(self, days: int = 7) -> Dict[str, Any]:
        from app.cost.engine import CostEngine
        cost_engine = CostEngine()
        try:
            feishu_factor = cost_engine.PLATFORM_COST_FACTORS.get("feishu", {})
            return {
                "platform": "feishu",
                "region": "cn",
                "days": days,
                "total_cost": 0,
                "token_usage": 0,
                "request_count": 0,
                "daily_cost": [],
                "cost_factor": feishu_factor,
                "source": "cost_engine_real",
            }
        except Exception:
            return {
                "platform": "feishu",
                "region": "cn",
                "days": days,
                "total_cost": 0,
                "token_usage": 0,
                "request_count": 0,
                "daily_cost": [],
            }

    async def fetch_pending_approvals(self) -> List[Dict[str, Any]]:
        try:
            from app.db import SessionLocal
            from app.models import ApprovalRequest
            from sqlalchemy import select

            with SessionLocal() as db:
                rows = db.execute(
                    select(ApprovalRequest)
                    .where(ApprovalRequest.status == "pending")
                    .order_by(ApprovalRequest.requested_at.desc())
                    .limit(20)
                ).scalars().all()

                return [
                    {
                        "id": f"approval_{r.id}",
                        "platform": "feishu",
                        "region": "cn",
                        "user": r.agent_id,
                        "action": r.action,
                        "resource": r.resource,
                        "risk_score": r.risk_score or 0,
                        "created_at": r.requested_at,
                        "status": r.status,
                        "source": "database_real",
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error("FeishuConnector fetch_pending_approvals error: %s", e)
            return []

    async def health_check(self) -> Dict[str, Any]:
        client = self._get_client()
        configured = client is not None and client.is_configured()

        if configured:
            try:
                token = await client.get_tenant_access_token()
                return {
                    "platform": "feishu",
                    "region": "cn",
                    "status": "online" if token else "auth_failed",
                    "latency": 0,
                    "last_connected": datetime.now().isoformat(),
                    "mode": "production",
                    "source": "real_health_check",
                }
            except Exception as e:
                return {
                    "platform": "feishu",
                    "region": "cn",
                    "status": "error",
                    "error": str(e),
                    "mode": "production",
                }

        return {
            "platform": "feishu",
            "region": "cn",
            "status": "mock",
            "latency": 0,
            "last_connected": datetime.now().isoformat(),
            "mode": "mock",
            "source": "fallback",
        }

    def _generate_fallback_events(self, limit: int) -> List[Dict[str, Any]]:
        try:
            from app.feishu.iam_gateway import get_audit_log
            audit_records = get_audit_log(limit=limit)
            if audit_records:
                events = []
                for r in audit_records:
                    risk_score = r.get("risk_score") or 0.0
                    events.append({
                        "id": f"feishu_iam_{r.get('timestamp', time.time()):.0f}",
                        "timestamp": datetime.fromtimestamp(r.get("timestamp", time.time())).isoformat(),
                        "platform": "feishu",
                        "region": "cn",
                        "user": r.get("agent_id", "unknown"),
                        "action": r.get("action", "unknown"),
                        "resource": r.get("path", ""),
                        "risk": risk_score,
                        "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low",
                        "blocked": r.get("decision") == "deny",
                        "reason": r.get("reason", ""),
                        "source": "iam_gateway_cache",
                    })
                return events
        except Exception:
            pass

        return []
