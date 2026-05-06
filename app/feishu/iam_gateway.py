import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.delegation.engine import DelegationEngine, get_trust_score

logger = logging.getLogger(__name__)

_SIX_LAYER_ENABLED = True

_delegation_engine: Optional[DelegationEngine] = None


def _get_engine() -> DelegationEngine:
    global _delegation_engine
    if _delegation_engine is None:
        _delegation_engine = DelegationEngine()
    return _delegation_engine


class Decision(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ERROR = "error"


@dataclass
class IAMCheckResult:
    allowed: bool
    decision: Decision
    reason: str = ""
    trust_score: Optional[float] = None
    risk_score: Optional[float] = None
    agent_id: str = ""
    action: str = ""
    blocked_at: str = ""
    auto_revoked: bool = False
    latency_ms: float = 0.0
    six_layer: Optional[Dict[str, Any]] = None


@dataclass
class AuditRecord:
    agent_id: str
    action: str
    decision: str
    reason: str
    latency_ms: float
    trust_score: Optional[float] = None
    risk_score: Optional[float] = None
    blocked_at: str = ""
    auto_revoked: bool = False
    timestamp: float = field(default_factory=time.time)
    path: str = ""
    method: str = ""
    six_layer: Optional[Dict[str, Any]] = None


_PATH_ACTION_MAP = {
    "/open-apis/im/v1/messages": "write:feishu_message",
    "/open-apis/docx/v1/documents": "write:doc",
    "/open-apis/sheets/v3/spreadsheets": "write:sheet",
    "/open-apis/calendar/v4/calendars": "write:calendar",
    "/open-apis/drive/v1/files": "read:drive",
    "/open-apis/bitable/v1/apps": "read:bitable",
    "/open-apis/wiki/v2/spaces": "read:wiki",
    "/open-apis/approval/v4/approvals": "read:approval",
    "/open-apis/contact/v3/users": "read:contact",
    "/open-apis/auth/v3/tenant_access_token": "auth:token",
    "/im/v1/messages": "write:feishu_message",
    "/im/v1/messages/": "write:feishu_message",
    "/docx/v1/documents": "write:doc",
    "/docx/v1/documents/": "write:doc",
    "/sheets/v3/spreadsheets": "write:sheet",
    "/calendar/v4/calendars": "write:calendar",
    "/calendar/v4/calendars/": "read:calendar",
    "/drive/v1/files": "read:drive",
    "/drive/v1/files/": "write:drive",
    "/bitable/v1/apps": "read:bitable",
    "/bitable/v1/apps/": "write:bitable",
    "/wiki/v2/spaces": "read:wiki",
    "/wiki/v2/spaces/": "read:wiki",
    "/approval/v4/approvals": "read:approval",
    "/contact/v3/users": "read:contact",
    "/contact/v3/users/": "read:contact",
    "/mail/v1/usermailgroups": "read:mail",
    "/auth/v3/tenant_access_token/internal": "auth:token",
}

_METHOD_OVERRIDE = {
    "GET": "read",
    "POST": "write",
    "PUT": "write",
    "PATCH": "write",
    "DELETE": "write",
}


def mapRequestToAction(path: str, method: str = "GET") -> str:
    if not path:
        return "read:unknown"

    for mapped_path, action in _PATH_ACTION_MAP.items():
        if path.startswith(mapped_path):
            if action == "auth:token":
                return action
            return action

    base = path.rstrip("/")
    parts = base.split("/")
    if len(parts) >= 3:
        resource = parts[2] if len(parts) > 2 else "unknown"
        scope = _METHOD_OVERRIDE.get(method.upper(), "read")
        return f"{scope}:{resource}"

    return f"{_METHOD_OVERRIDE.get(method.upper(), 'read')}:feishu_api"


def _issue_root_token(agent_id: str) -> Optional[str]:
    try:
        from app.delegation.engine import CAPABILITY_AGENTS
        agent_caps = CAPABILITY_AGENTS.get(agent_id, {})
        capabilities = agent_caps.get("capabilities", ["read:public"])
        engine = _get_engine()
        token = engine.issue_root_token(
            agent_id=agent_id,
            delegated_user="admin",
            capabilities=capabilities,
            expires_in_minutes=60,
        )
        return token
    except Exception as e:
        logger.error("IAM Gateway: Failed to issue root token for %s: %s", agent_id, e)
    return None


def callIAMCheck(agent_id: str, action: str, token: Optional[str] = None) -> IAMCheckResult:
    start = time.time()
    try:
        if not token:
            token = _issue_root_token(agent_id)
            if not token:
                latency_ms = (time.time() - start) * 1000
                return IAMCheckResult(
                    allowed=False,
                    decision=Decision.DENY,
                    reason=f"Cannot issue token for agent '{agent_id}' — agent may not exist or is revoked",
                    agent_id=agent_id,
                    action=action,
                    latency_ms=latency_ms,
                )

        engine = _get_engine()
        result = engine.check(token=token, action=action)

        latency_ms = (time.time() - start) * 1000

        allowed = result.allowed
        reason = result.reason
        risk_score = result.risk_score
        auto_revoked = result.auto_revoked
        blocked_at = ""
        if not allowed:
            if "capability" in reason.lower() or "scope" in reason.lower():
                blocked_at = "capability_check"
            elif "revoked" in reason.lower():
                blocked_at = "token_revocation"
            elif "trust" in reason.lower():
                blocked_at = "trust_check"
            elif "replay" in reason.lower():
                blocked_at = "replay_check"
            else:
                blocked_at = "iam_check"

        trust_score = get_trust_score(agent_id)

        decision = Decision.ALLOW if allowed else Decision.DENY

        if _SIX_LAYER_ENABLED:
            try:
                from app.security.six_layer_verify import verify_six_layers
                _six_layer_result = verify_six_layers(
                    agent_id=agent_id,
                    action=action,
                    input_text="",
                    trust_score=trust_score or 0.5,
                    risk_score=risk_score or 0.0,
                    blocked_at="" if allowed else blocked_at,
                    auto_revoked=auto_revoked,
                    allowed=allowed,
                    reason=reason,
                )
            except Exception:
                _six_layer_result = None
        else:
            _six_layer_result = None

        return IAMCheckResult(
            allowed=allowed,
            decision=decision,
            reason=reason,
            trust_score=trust_score,
            risk_score=risk_score,
            agent_id=agent_id,
            action=action,
            blocked_at=blocked_at,
            auto_revoked=auto_revoked,
            latency_ms=latency_ms,
            six_layer=_six_layer_result.to_dict() if _six_layer_result else None,
        )

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error("IAM Gateway check failed: %s", e)
        return IAMCheckResult(
            allowed=False,
            decision=Decision.ERROR,
            reason=f"IAM check error: {str(e)}",
            agent_id=agent_id,
            action=action,
            latency_ms=latency_ms,
        )


_audit_log: List[AuditRecord] = []
_MAX_AUDIT_LOG = 500


def logAudit(
    agent_id: str,
    action: str,
    decision: str,
    reason: str,
    latency_ms: float,
    trust_score: Optional[float] = None,
    risk_score: Optional[float] = None,
    blocked_at: str = "",
    auto_revoked: bool = False,
    path: str = "",
    method: str = "",
    six_layer: Optional[Dict[str, Any]] = None,
) -> AuditRecord:
    record = AuditRecord(
        agent_id=agent_id,
        action=action,
        decision=decision,
        reason=reason,
        latency_ms=latency_ms,
        trust_score=trust_score,
        risk_score=risk_score,
        blocked_at=blocked_at,
        auto_revoked=auto_revoked,
        path=path,
        method=method,
        six_layer=six_layer,
    )
    _audit_log.append(record)
    if len(_audit_log) > _MAX_AUDIT_LOG:
        _audit_log.pop(0)

    try:
        from app.audit import log_event as audit_log_event
        audit_log_event(
            action=action,
            resource=path or "iam_gateway",
            decision=decision,
            reason=reason[:200],
            agent_id=agent_id,
            context={
                "trust_score": trust_score,
                "risk_score": risk_score,
                "blocked_at": blocked_at,
                "auto_revoked": auto_revoked,
                "latency_ms": latency_ms,
                "path": path,
                "method": method,
                "source": "iam_gateway",
            },
        )
    except Exception as e:
        logger.warning("IAM Gateway audit log_event failed: %s", e)

    status_icon = "✅" if decision == "allow" else ("🔥" if auto_revoked else "❌")
    logger.info(
        "IAM Gateway %s agent=%s action=%s decision=%s reason=%s latency=%.1fms path=%s",
        status_icon, agent_id, action, decision, reason[:60], latency_ms, path,
    )
    return record


def get_audit_log(limit: int = 50) -> List[Dict[str, Any]]:
    records = _audit_log[-limit:]
    return [
        {
            "agent_id": r.agent_id,
            "action": r.action,
            "decision": r.decision,
            "reason": r.reason,
            "latency_ms": round(r.latency_ms, 2),
            "trust_score": r.trust_score,
            "risk_score": r.risk_score,
            "blocked_at": r.blocked_at,
            "auto_revoked": r.auto_revoked,
            "path": r.path,
            "method": r.method,
            "timestamp": r.timestamp,
            "six_layer": r.six_layer,
        }
        for r in records
    ]


def get_gateway_stats() -> Dict[str, Any]:
    if not _audit_log:
        return {
            "total_requests": 0,
            "allowed": 0,
            "denied": 0,
            "auto_revoked": 0,
            "errors": 0,
            "avg_latency_ms": 0,
            "deny_rate": "0%",
        }

    total = len(_audit_log)
    allowed = sum(1 for r in _audit_log if r.decision == "allow")
    denied = sum(1 for r in _audit_log if r.decision == "deny")
    auto_revoked = sum(1 for r in _audit_log if r.auto_revoked)
    errors = sum(1 for r in _audit_log if r.decision == "error")
    avg_latency = sum(r.latency_ms for r in _audit_log) / total
    deny_rate = f"{(denied / total * 100):.1f}%" if total > 0 else "0%"

    return {
        "total_requests": total,
        "allowed": allowed,
        "denied": denied,
        "auto_revoked": auto_revoked,
        "errors": errors,
        "avg_latency_ms": round(avg_latency, 2),
        "deny_rate": deny_rate,
    }


class IAMTransport(httpx.AsyncBaseTransport):
    """
    Security Gateway Transport — all Feishu API requests MUST pass through IAM.

    Flow: FeishuClient → IAMTransport → IAM Check → Decision → Feishu API
    """

    def __init__(
        self,
        agent_id: str = "doc_agent",
        bypass_paths: Optional[List[str]] = None,
    ):
        self._inner = httpx.AsyncHTTPTransport()
        self.agent_id = agent_id
        self._bypass_paths = set(bypass_paths or ["/auth/v3/tenant_access_token/internal"])

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method.upper()

        action = mapRequestToAction(path, method)

        if action == "auth:token" or path in self._bypass_paths:
            logger.debug("IAM Gateway: bypassing auth path %s", path)
            response = await self._inner.handle_async_request(request)
            return response

        iam_result = await self._async_iam_check(self.agent_id, action)

        if not iam_result.allowed:
            logAudit(
                agent_id=iam_result.agent_id,
                action=iam_result.action,
                decision="deny" if iam_result.decision != Decision.ERROR else "error",
                reason=iam_result.reason,
                latency_ms=iam_result.latency_ms,
                trust_score=iam_result.trust_score,
                risk_score=iam_result.risk_score,
                blocked_at=iam_result.blocked_at,
                auto_revoked=iam_result.auto_revoked,
                path=path,
                method=method,
                six_layer=iam_result.six_layer,
            )

            deny_reason = iam_result.reason
            if iam_result.auto_revoked:
                deny_reason = f"🔥 Agent {self.agent_id} has been AUTO-REVOKED — all tokens invalidated"
            elif "trust" in deny_reason.lower():
                deny_reason = f"⚠️ Agent {self.agent_id} trust score too low — {deny_reason}"
            elif "capability" in deny_reason.lower() or "scope" in deny_reason.lower():
                deny_reason = f"❌ Agent {self.agent_id} lacks capability '{action}' — {deny_reason}"
            elif "revoked" in deny_reason.lower():
                deny_reason = f"🔥 Agent {self.agent_id} token revoked — {deny_reason}"

            logger.warning("IAM Gateway BLOCKED: path=%s action=%s agent=%s reason=%s", path, action, self.agent_id, deny_reason)

            return httpx.Response(
                status_code=403,
                json={
                    "code": -1,
                    "msg": f"IAM Gateway: Request blocked — {deny_reason}",
                    "iam_blocked": True,
                    "agent_id": self.agent_id,
                    "action": action,
                    "reason": deny_reason,
                    "auto_revoked": iam_result.auto_revoked,
                    "trust_score": iam_result.trust_score,
                    "risk_score": iam_result.risk_score,
                },
                request=request,
            )

        if iam_result.trust_score is not None:
            request.headers["X-Agent-ID"] = self.agent_id
            request.headers["X-Trust-Score"] = f"{iam_result.trust_score:.4f}"
        if iam_result.risk_score is not None:
            request.headers["X-Risk-Score"] = f"{iam_result.risk_score:.4f}"

        start = time.time()
        try:
            response = await self._inner.handle_async_request(request)
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logAudit(
                agent_id=self.agent_id,
                action=action,
                decision="error",
                reason=f"Upstream error: {str(e)}",
                latency_ms=latency_ms,
                trust_score=iam_result.trust_score,
                risk_score=iam_result.risk_score,
                path=path,
                method=method,
                six_layer=iam_result.six_layer,
            )
            raise

        latency_ms = (time.time() - start) * 1000

        logAudit(
            agent_id=self.agent_id,
            action=action,
            decision="allow",
            reason="IAM check passed",
            latency_ms=iam_result.latency_ms + latency_ms,
            trust_score=iam_result.trust_score,
            risk_score=iam_result.risk_score,
            path=path,
            method=method,
            six_layer=iam_result.six_layer,
        )

        return response

    async def _async_iam_check(self, agent_id: str, action: str) -> IAMCheckResult:
        start = time.time()
        try:
            token = _issue_root_token(agent_id)
            if not token:
                latency_ms = (time.time() - start) * 1000
                return IAMCheckResult(
                    allowed=False,
                    decision=Decision.DENY,
                    reason=f"Cannot issue token for agent '{agent_id}' — agent may not exist or is revoked",
                    agent_id=agent_id,
                    action=action,
                    latency_ms=latency_ms,
                )

            engine = _get_engine()
            result = engine.check(token=token, action=action)

            latency_ms = (time.time() - start) * 1000

            allowed = result.allowed
            reason = result.reason
            risk_score = result.risk_score
            auto_revoked = result.auto_revoked
            blocked_at = ""
            if not allowed:
                if "capability" in reason.lower() or "scope" in reason.lower():
                    blocked_at = "capability_check"
                elif "revoked" in reason.lower():
                    blocked_at = "token_revocation"
                elif "trust" in reason.lower():
                    blocked_at = "trust_check"
                elif "replay" in reason.lower():
                    blocked_at = "replay_check"
                else:
                    blocked_at = "iam_check"

            trust_score = get_trust_score(agent_id)

            return IAMCheckResult(
                allowed=allowed,
                decision=Decision.ALLOW if allowed else Decision.DENY,
                reason=reason,
                trust_score=trust_score,
                risk_score=risk_score,
                agent_id=agent_id,
                action=action,
                blocked_at=blocked_at,
                auto_revoked=auto_revoked,
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            logger.error("IAM Gateway async check failed: %s", e)
            return IAMCheckResult(
                allowed=False,
                decision=Decision.ERROR,
                reason=f"IAM check error: {str(e)}",
                agent_id=agent_id,
                action=action,
                latency_ms=latency_ms,
            )

    async def aclose(self) -> None:
        await self._inner.aclose()
