import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base import BaseConnector

logger = logging.getLogger(__name__)


class LLMConnector(BaseConnector):

    def __init__(
        self,
        platform: str,
        region: str,
        api_key: str = "",
        base_url: str = "",
        model: str = "",
    ):
        super().__init__(platform, region, mock=not bool(api_key))
        self._api_key = api_key
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._model = model
        self._connected = False
        self._last_health_at: float = 0
        self._health_cache: Optional[Dict[str, Any]] = None
        self._call_stats: Dict[str, Any] = {
            "total_calls": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "blocked_by_iam": 0,
            "last_call_at": None,
        }

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key and self._base_url)

    async def connect(self) -> bool:
        if not self.is_configured:
            self._connected = False
            self.mock = True
            return False
        try:
            health = await self.health_check()
            self._connected = health.get("status") == "online"
            self.mock = not self._connected
            return self._connected
        except Exception as e:
            logger.warning("%s connect failed: %s", self.platform, e)
            self._connected = False
            self.mock = True
            return False

    def _get_headers(self) -> Dict[str, str]:
        return {}

    async def _real_chat(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> Dict[str, Any]:
        if not self.is_configured:
            return {"error": "not configured", "mock": True}

        try:
            import httpx
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"
            payload = {
                "model": self._model,
                "messages": messages,
                "max_tokens": max_tokens,
            }

            url = f"{self._base_url}/chat/completions"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                data = resp.json()

                if resp.status_code == 200:
                    usage = data.get("usage", {})
                    self._call_stats["total_calls"] += 1
                    self._call_stats["total_tokens"] += usage.get("total_tokens", 0)
                    self._call_stats["last_call_at"] = datetime.now().isoformat()
                    return {
                        "success": True,
                        "content": data["choices"][0]["message"]["content"],
                        "model": data.get("model", self._model),
                        "usage": usage,
                        "source": "real_api",
                    }
                else:
                    return {
                        "success": False,
                        "error": data.get("error", {}).get("message", str(data)),
                        "status_code": resp.status_code,
                    }
        except Exception as e:
            logger.error("%s _real_chat error: %s", self.platform, e)
            return {"success": False, "error": str(e)}

    async def fetch_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        real_events = self._fetch_iam_audit_events(limit)
        if real_events:
            return real_events
        return self._generate_fallback_events(limit)

    def _fetch_iam_audit_events(self, limit: int) -> List[Dict[str, Any]]:
        try:
            from app.feishu.iam_gateway import get_audit_log
            audit_records = get_audit_log(limit=limit)
            events = []
            for r in audit_records:
                risk_score = r.get("risk_score") or 0.0
                events.append({
                    "id": f"{self.platform}_iam_{r.get('timestamp', time.time()):.0f}",
                    "timestamp": datetime.fromtimestamp(r.get("timestamp", time.time())).isoformat(),
                    "platform": self.platform,
                    "region": self.region,
                    "user": r.get("agent_id", "unknown"),
                    "action": r.get("action", "unknown"),
                    "resource": r.get("path", ""),
                    "risk": risk_score,
                    "risk_level": "high" if risk_score > 0.7 else "medium" if risk_score > 0.3 else "low",
                    "blocked": r.get("decision") == "deny",
                    "reason": r.get("reason", ""),
                    "source": "iam_audit_real",
                })
            return events
        except Exception:
            return []

    def _generate_fallback_events(self, limit: int) -> List[Dict[str, Any]]:
        return []

    async def fetch_cost(self, days: int = 7) -> Dict[str, Any]:
        from app.cost.engine import CostEngine
        cost_engine = CostEngine()
        factor = cost_engine.PLATFORM_COST_FACTORS.get(self.platform, {})
        return {
            "platform": self.platform,
            "region": self.region,
            "days": days,
            "total_cost": self._call_stats.get("total_cost", 0),
            "token_usage": self._call_stats.get("total_tokens", 0),
            "request_count": self._call_stats.get("total_calls", 0),
            "daily_cost": [],
            "cost_factor": factor,
            "source": "real_stats" if self._call_stats["total_calls"] > 0 else "no_data",
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
                        "platform": self.platform,
                        "region": self.region,
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
            logger.error("%s fetch_pending_approvals error: %s", self.platform, e)
            return []

    async def health_check(self) -> Dict[str, Any]:
        now = time.time()
        if self._health_cache and (now - self._last_health_at) < 60:
            return self._health_cache

        if not self.is_configured:
            result = {
                "platform": self.platform,
                "region": self.region,
                "status": "not_configured",
                "latency": 0,
                "last_connected": datetime.now().isoformat(),
                "mode": "mock",
                "source": "no_api_key",
            }
            self._health_cache = result
            self._last_health_at = now
            return result

        try:
            import httpx
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"
            payload = {
                "model": self._model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
            url = f"{self._base_url}/chat/completions"
            start = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
            latency = round((time.time() - start) * 1000, 1)

            if resp.status_code == 200:
                result = {
                    "platform": self.platform,
                    "region": self.region,
                    "status": "online",
                    "latency": latency,
                    "last_connected": datetime.now().isoformat(),
                    "mode": "production",
                    "model": self._model,
                    "source": "real_health_check",
                }
            else:
                result = {
                    "platform": self.platform,
                    "region": self.region,
                    "status": "auth_failed",
                    "latency": latency,
                    "last_connected": datetime.now().isoformat(),
                    "mode": "degraded",
                    "error": resp.status_code,
                    "source": "real_health_check",
                }
        except Exception as e:
            result = {
                "platform": self.platform,
                "region": self.region,
                "status": "offline",
                "latency": 0,
                "last_connected": datetime.now().isoformat(),
                "mode": "degraded",
                "error": str(e),
                "source": "real_health_check",
            }

        self._health_cache = result
        self._last_health_at = now
        return result

    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> Dict[str, Any]:
        if not self.is_configured:
            return {
                "success": False,
                "content": f"[{self.platform}] API not configured — running in mock mode",
                "mock": True,
                "mode": "degraded_with_iam",
                "source": "no_api_key",
            }

        iam_allowed = self._check_iam("chat_completion")
        if not iam_allowed:
            self._call_stats["blocked_by_iam"] += 1
            return {
                "success": False,
                "content": f"[{self.platform}] IAM Gateway: Request blocked",
                "iam_blocked": True,
                "source": "iam_gateway",
            }

        result = await self._real_chat(messages, max_tokens)
        if result.get("success"):
            self._log_audit("chat_completion", "allow", "real_api_call")
        else:
            self._log_audit("chat_completion", "error", result.get("error", "unknown"))
        return result

    def _check_iam(self, action: str) -> bool:
        try:
            from app.feishu.iam_gateway import callIAMCheck, logAudit
            result = callIAMCheck(self.platform + "_agent", action)
            if not result.allowed:
                logAudit(
                    agent_id=self.platform + "_agent",
                    action=action,
                    decision="deny",
                    reason=result.reason,
                    latency_ms=result.latency_ms,
                    trust_score=result.trust_score,
                    risk_score=result.risk_score,
                    path=f"/{self.platform}/chat/completions",
                    method="POST",
                )
                return False
            logAudit(
                agent_id=self.platform + "_agent",
                action=action,
                decision="allow",
                reason=result.reason,
                latency_ms=result.latency_ms,
                trust_score=result.trust_score,
                risk_score=result.risk_score,
                path=f"/{self.platform}/chat/completions",
                method="POST",
            )
            return True
        except Exception as e:
            self.logger.warning("IAM check failed (fail-closed): %s", e)
            return False

    def _log_audit(self, action: str, decision: str, reason: str = ""):
        try:
            from app import audit
            audit.log_event(
                agent_id=self.platform + "_agent",
                action=action,
                resource=f"/{self.platform}/chat/completions",
                decision=decision,
                reason=reason,
            )
        except Exception:
            pass
