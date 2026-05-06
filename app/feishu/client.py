import os
import json
import logging
import time
import httpx
from typing import Optional, Dict, Any

from .iam_gateway import IAMTransport, callIAMCheck, logAudit
from app.config import settings

logger = logging.getLogger(__name__)

IAM_GATEWAY_ENABLED = os.getenv("IAM_GATEWAY_ENABLED", "true").lower() in ("true", "1", "yes")
IAM_DEFAULT_AGENT = os.getenv("IAM_DEFAULT_AGENT", "doc_agent")

_tenant_access_token: Optional[str] = None
_token_expires_at: float = 0


class FeishuClient:
    _tenant_domain: Optional[str] = None

    def __init__(self, agent_id: Optional[str] = None, iam_enabled: bool = IAM_GATEWAY_ENABLED):
        self.app_id = settings.FEISHU_APP_ID
        self.app_secret = settings.FEISHU_APP_SECRET
        self.verification_token = settings.FEISHU_VERIFICATION_TOKEN
        self.encrypt_key = settings.FEISHU_ENCRYPT_KEY
        self.base_url = "https://open.feishu.cn/open-apis"
        self.agent_id = agent_id or IAM_DEFAULT_AGENT
        self.iam_enabled = iam_enabled

        if self.iam_enabled:
            self._iam_transport = IAMTransport(
                agent_id=self.agent_id,
                bypass_paths=["/auth/v3/tenant_access_token/internal"],
            )
            logger.info("FeishuClient initialized with IAM Gateway (agent=%s)", self.agent_id)
        else:
            self._iam_transport = None
            logger.info("FeishuClient initialized WITHOUT IAM Gateway")

    def _get_client(self, timeout: float = 10) -> httpx.AsyncClient:
        if self.iam_enabled and self._iam_transport:
            return httpx.AsyncClient(
                timeout=timeout,
                transport=self._iam_transport,
            )
        return httpx.AsyncClient(timeout=timeout)

    def is_configured(self) -> bool:
        return bool(self.app_id and self.app_secret)

    async def _get_tenant_domain(self) -> str:
        if FeishuClient._tenant_domain:
            return FeishuClient._tenant_domain
        try:
            token = await self.get_tenant_access_token()
            if not token:
                return "feishu.cn"
            url = f"{self.base_url}/tenant/v2/tenant/query"
            headers = {"Authorization": f"Bearer {token}"}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url, headers=headers)
                data = resp.json()
            if data.get("code") == 0:
                domain = data.get("data", {}).get("tenant", {}).get("domain", "")
                if domain:
                    FeishuClient._tenant_domain = domain
                    logger.info("Feishu tenant domain: %s", domain)
                    return domain
        except Exception as e:
            logger.warning("Failed to get tenant domain: %s", e)
        return "feishu.cn"

    async def get_tenant_access_token(self) -> str:
        global _tenant_access_token, _token_expires_at

        if _tenant_access_token and time.time() < _token_expires_at - 60:
            return _tenant_access_token

        if not self.is_configured():
            logger.error("Feishu credentials not configured")
            return ""

        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        payload = {
            "app_id": self.app_id,
            "app_secret": self.app_secret,
        }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

        if data.get("code") != 0:
            logger.error("Failed to get tenant_access_token: %s", data)
            raise Exception(f"Feishu auth failed: {data.get('msg')}")

        _tenant_access_token = data["tenant_access_token"]
        _token_expires_at = time.time() + data.get("expire", 7200)
        logger.info("Got tenant_access_token, expires in %ds", data.get("expire", 7200))
        return _tenant_access_token

    async def send_message(self, receive_id: str, content: str, msg_type: str = "text", receive_id_type: str = "open_id") -> Dict[str, Any]:
        if not self.is_configured():
            action = "write:feishu_message"
            iam_result = None
            if self.iam_enabled:
                iam_result = callIAMCheck(self.agent_id, action)
                logAudit(
                    agent_id=self.agent_id,
                    action=action,
                    decision="allow" if iam_result.allowed else "deny",
                    reason=iam_result.reason,
                    latency_ms=iam_result.latency_ms,
                    trust_score=iam_result.trust_score,
                    risk_score=iam_result.risk_score,
                    blocked_at=iam_result.blocked_at,
                    auto_revoked=iam_result.auto_revoked,
                    path="/im/v1/messages",
                    method="POST",
                )
                if not iam_result.allowed:
                    return {
                        "code": -1,
                        "msg": f"IAM Gateway: Request blocked — {iam_result.reason}",
                        "iam_blocked": True,
                        "agent_id": self.agent_id,
                        "action": action,
                    }
            logger.warning("send_message skipped: Feishu not configured")
            return {"code": -1, "msg": "Feishu not configured: APP_ID/APP_SECRET required", "iam_checked": self.iam_enabled, "mode": "not_configured"}

        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/im/v1/messages"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        body_content = content
        if msg_type == "text":
            body_content = json.dumps({"text": content})

        payload = {
            "receive_id": receive_id,
            "msg_type": msg_type,
            "content": body_content,
        }

        params = {"receive_id_type": receive_id_type}

        async with self._get_client() as client:
            resp = await client.post(url, json=payload, headers=headers, params=params)
            if resp.status_code == 403:
                try:
                    return resp.json()
                except Exception:
                    return {"code": -1, "msg": "IAM Gateway blocked request", "iam_blocked": True}
            data = resp.json()

        if data.get("code") != 0:
            logger.error("Failed to send message: %s", data)
            return data

        logger.info("Message sent to %s", receive_id)
        return data

    async def create_doc(self, title: str, content: str) -> Dict[str, Any]:
        if not self.is_configured():
            action = "write:doc"
            iam_result = None
            if self.iam_enabled:
                iam_result = callIAMCheck(self.agent_id, action)
                logAudit(
                    agent_id=self.agent_id,
                    action=action,
                    decision="allow" if iam_result.allowed else "deny",
                    reason=iam_result.reason,
                    latency_ms=iam_result.latency_ms,
                    trust_score=iam_result.trust_score,
                    risk_score=iam_result.risk_score,
                    blocked_at=iam_result.blocked_at,
                    auto_revoked=iam_result.auto_revoked,
                    path="/docx/v1/documents",
                    method="POST",
                )
                if not iam_result.allowed:
                    return {
                        "code": -1,
                        "msg": f"IAM Gateway: Request blocked — {iam_result.reason}",
                        "iam_blocked": True,
                        "agent_id": self.agent_id,
                        "action": action,
                    }
            logger.warning("create_doc skipped: Feishu not configured")
            return {
                "code": -1,
                "msg": "Feishu not configured: cannot create real document",
                "iam_checked": self.iam_enabled,
                "mode": "not_configured",
            }

        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/docx/v1/documents"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        payload = {"title": title, "folder_token": ""}

        async with self._get_client() as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 403:
                try:
                    return resp.json()
                except Exception:
                    return {"code": -1, "msg": "IAM Gateway blocked request", "iam_blocked": True}
            data = resp.json()

        if data.get("code") != 0:
            logger.error("Failed to create doc: %s", data)
            return data

        doc_id = data.get("data", {}).get("document", {}).get("document_id", "")
        tenant_domain = await self._get_tenant_domain()
        doc_url = f"https://{tenant_domain}/docx/{doc_id}"

        if doc_id and content:
            block_url = f"{self.base_url}/docx/v1/documents/{doc_id}/blocks/{doc_id}/children"
            paragraphs = content.split("\n")
            children = []
            for para in paragraphs:
                children.append({
                    "block_type": 2,
                    "text": {
                        "elements": [{"text_run": {"content": para}}],
                        "style": {},
                    },
                })
            text_content = {"children": children}
            async with self._get_client() as client:
                block_resp = await client.post(block_url, json=text_content, headers=headers)
                block_data = block_resp.json()
                if block_data.get("code") != 0:
                    logger.warning("Failed to write doc content: code=%s msg=%s", block_data.get("code"), block_data.get("msg", ""))
                else:
                    logger.info("Doc content written: %d paragraphs", len(paragraphs))

        logger.info("Doc created: %s (%s)", title, doc_url)
        return {"code": 0, "data": {"document": {"document_id": doc_id, "title": title, "url": doc_url}}}

    async def query_bitable(self, app_token: str, table_id: str, view_id: str = None, page_size: int = 20) -> Dict[str, Any]:
        if not self.is_configured():
            action = "read:bitable"
            iam_result = None
            if self.iam_enabled:
                iam_result = callIAMCheck(self.agent_id, action)
                logAudit(
                    agent_id=self.agent_id,
                    action=action,
                    decision="allow" if iam_result.allowed else "deny",
                    reason=iam_result.reason,
                    latency_ms=iam_result.latency_ms,
                    trust_score=iam_result.trust_score,
                    risk_score=iam_result.risk_score,
                    blocked_at=iam_result.blocked_at,
                    auto_revoked=iam_result.auto_revoked,
                    path=f"/bitable/v1/apps/{app_token}/tables/{table_id}/records",
                    method="GET",
                )
                if not iam_result.allowed:
                    return {
                        "code": -1,
                        "msg": f"IAM Gateway: Request blocked — {iam_result.reason}",
                        "iam_blocked": True,
                        "agent_id": self.agent_id,
                        "action": action,
                    }
            logger.warning("query_bitable skipped: Feishu not configured")
            return {
                "code": -1,
                "msg": "Feishu not configured: cannot query real bitable data",
                "iam_checked": self.iam_enabled,
                "mode": "not_configured",
            }

        token = await self.get_tenant_access_token()
        url = f"{self.base_url}/bitable/v1/apps/{app_token}/tables/{table_id}/records"

        headers = {
            "Authorization": f"Bearer {token}",
        }

        params = {"page_size": page_size}
        if view_id:
            params["view_id"] = view_id

        async with self._get_client() as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 403:
                try:
                    return resp.json()
                except Exception:
                    return {"code": -1, "msg": "IAM Gateway blocked request", "iam_blocked": True}
            data = resp.json()

        if data.get("code") != 0:
            logger.error("Failed to query bitable: %s", data)
            return data

        items = data.get("data", {}).get("items", [])
        logger.info("Bitable query returned %d records", len(items))
        return {"code": 0, "msg": "ok", "data": {"items": items, "total": data.get("data", {}).get("total", len(items))}}

    def get_bitable_records(self, app_token: str, table_id: str, view_id: str = None, page_size: int = 20) -> Dict[str, Any]:
        logger.info("get_bitable_records: app_token=%s table_id=%s", app_token, table_id)

        if not self.is_configured():
            logger.error("Feishu credentials not configured — cannot make real Bitable request")
            return {"code": -1, "msg": "Feishu not configured: FEISHU_APP_ID and FEISHU_APP_SECRET required"}

        tenant_token = self._get_tenant_token_sync()
        if not tenant_token:
            return {"code": -1, "msg": "Failed to obtain tenant_access_token"}

        url = f"{self.base_url}/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        headers = {"Authorization": f"Bearer {tenant_token}"}
        params = {"page_size": page_size}
        if view_id:
            params["view_id"] = view_id

        logger.info("Bitable API request: GET %s", url)

        resp = httpx.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()

        if data.get("code") != 0:
            logger.error("Bitable API error: code=%s msg=%s", data.get("code"), data.get("msg"))
            return data

        items = data.get("data", {}).get("items", [])
        total = data.get("data", {}).get("total", len(items))
        logger.info("Bitable API success: %d records returned (total=%s)", len(items), total)
        return {"code": 0, "msg": "ok", "data": {"items": items, "total": total}}

    def _get_tenant_token_sync(self) -> Optional[str]:
        global _tenant_access_token, _token_expires_at

        if _tenant_access_token and time.time() < _token_expires_at - 60:
            return _tenant_access_token

        if not self.is_configured():
            return None

        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        resp = httpx.post(url, json={"app_id": self.app_id, "app_secret": self.app_secret}, timeout=10)
        data = resp.json()

        if data.get("code") != 0:
            logger.error("Failed to get tenant_access_token: code=%s msg=%s", data.get("code"), data.get("msg"))
            return None

        _tenant_access_token = data["tenant_access_token"]
        _token_expires_at = time.time() + data.get("expire", 7200) - 60
        logger.info("Got tenant_access_token (sync), expires in %ds", data.get("expire", 7200))
        return _tenant_access_token

    def verify_event(self, body: Dict[str, Any]) -> bool:
        if not self.verification_token:
            return True

        token = body.get("token", "")
        if not token:
            return True

        if token != self.verification_token:
            logger.warning("Event verification failed: token mismatch (expected=%s got=%s)", self.verification_token[:8] + "...", token[:8] + "...")
            return False

        return True

    def parse_message_event(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        event = body.get("event", {})
        msg_type = event.get("message", {}).get("message_type", "")

        if msg_type != "text":
            logger.info("Ignoring non-text message: type=%s", msg_type)
            return None

        sender = event.get("sender", {})
        sender_id = sender.get("sender_id", {})
        user_id = sender_id.get("open_id", "") or sender_id.get("user_id", "") or sender_id.get("union_id", "")

        message = event.get("message", {})
        content_str = message.get("content", "{}")

        try:
            content_data = json.loads(content_str)
            text = content_data.get("text", "")
        except (json.JSONDecodeError, TypeError):
            text = content_str if isinstance(content_str, str) else ""

        text = text.strip()
        if not text:
            return None

        chat_id = message.get("chat_id", "")
        message_id = message.get("message_id", "")

        return {
            "user_id": user_id,
            "message": text,
            "chat_id": chat_id,
            "message_id": message_id,
            "msg_type": msg_type,
        }

    async def _api_get(self, path: str, params: Dict = None) -> Dict[str, Any]:
        if not self.is_configured():
            return {"code": -1, "msg": "Feishu not configured", "mode": "not_configured"}
        token = await self.get_tenant_access_token()
        if not token:
            return {"code": -1, "msg": "no token"}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params or {})
            return resp.json()

    async def _api_post(self, path: str, body: Dict = None) -> Dict[str, Any]:
        if not self.is_configured():
            return {"code": -1, "msg": "Feishu not configured", "mode": "not_configured"}
        token = await self.get_tenant_access_token()
        if not token:
            return {"code": -1, "msg": "no token"}
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=body or {})
            return resp.json()

    async def get_bot_info(self) -> Dict[str, Any]:
        return await self._api_get("/bot/v3/info/")

    async def get_calendar_primary(self) -> Dict[str, Any]:
        return await self._api_get("/calendar/v4/calendars/primary")

    async def get_calendar_events(self, start_time: str = "", end_time: str = "", page_size: int = 10) -> Dict[str, Any]:
        import time as _t
        if not start_time:
            start_time = str(int(_t.time()))
        if not end_time:
            end_time = str(int(_t.time()) + 86400 * 7)
        return await self._api_get("/calendar/v4/calendars/primary/events", {
            "start_time": start_time, "end_time": end_time, "page_size": page_size,
        })

    async def get_contact_users(self, page_size: int = 10) -> Dict[str, Any]:
        return await self._api_get("/contact/v3/users", {"page_size": page_size})

    async def get_task_list(self, page_size: int = 10) -> Dict[str, Any]:
        return await self._api_get("/task/v2/tasks", {"page_size": page_size})

    async def get_drive_root(self) -> Dict[str, Any]:
        return await self._api_get("/drive/explorer/v2/root_folder/meta")

    async def search_docs(self, search_key: str, page_size: int = 5) -> Dict[str, Any]:
        return await self._api_post("/suite/docs-api/search/object", {
            "search_key": search_key, "page_size": page_size,
        })

    async def get_vc_rooms(self, page_size: int = 10) -> Dict[str, Any]:
        return await self._api_get("/vc/v1/rooms", {"page_size": page_size})

    async def get_attendance_shifts(self, page_size: int = 10) -> Dict[str, Any]:
        return await self._api_get("/attendance/v1/shifts", {"page_size": page_size})

    async def get_im_chats(self, page_size: int = 10) -> Dict[str, Any]:
        return await self._api_get("/im/v1/chats", {"page_size": page_size})

    async def get_available_capabilities(self) -> Dict[str, Any]:
        if not self.is_configured():
            return {"available": [], "unavailable": [], "mode": "not_configured"}

        capabilities = [
            ("bot_info", "Bot Info", "/bot/v3/info/"),
            ("calendar", "Calendar Primary", "/calendar/v4/calendars/primary"),
            ("calendar_events", "Calendar Events", "/calendar/v4/calendars/primary/events"),
            ("contact_users", "Contact Users", "/contact/v3/users"),
            ("task_list", "Task List", "/task/v2/tasks"),
            ("drive_root", "Drive Root", "/drive/explorer/v2/root_folder/meta"),
            ("doc_search", "Doc Search", "/suite/docs-api/search/object"),
            ("vc_rooms", "VC Rooms", "/vc/v1/rooms"),
            ("attendance", "Attendance Shifts", "/attendance/v1/shifts"),
            ("im_chats", "IM Chats", "/im/v1/chats"),
            ("bitable", "Bitable Query", "/bitable/v1/apps"),
            ("doc_create", "Doc Create", "/docx/v1/documents"),
            ("im_send", "IM Send Message", "/im/v1/messages"),
        ]

        available = []
        unavailable = []
        token = await self.get_tenant_access_token()
        if not token:
            return {"available": [], "unavailable": capabilities, "mode": "no_token"}

        headers = {"Authorization": f"Bearer {token}"}
        for cap_id, cap_name, path in capabilities:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    if cap_id in ("doc_search",):
                        resp = await client.post(
                            f"{self.base_url}{path}",
                            headers=headers,
                            json={"search_key": "test", "page_size": 1},
                        )
                    else:
                        resp = await client.get(
                            f"{self.base_url}{path}",
                            headers=headers,
                            params={"page_size": 1},
                        )
                    data = resp.json()
                    if data.get("code") == 0:
                        available.append({"id": cap_id, "name": cap_name, "path": path})
                    else:
                        unavailable.append({"id": cap_id, "name": cap_name, "reason": data.get("msg", "")[:40]})
            except Exception:
                unavailable.append({"id": cap_id, "name": cap_name, "reason": "timeout"})

        return {
            "available": available,
            "unavailable": unavailable,
            "total_available": len(available),
            "total_capabilities": len(capabilities),
            "mode": "production",
        }


_feishu_client: Optional[FeishuClient] = None


def get_feishu_client(agent_id: Optional[str] = None) -> FeishuClient:
    global _feishu_client
    if _feishu_client is None:
        _feishu_client = FeishuClient(agent_id=agent_id)
    return _feishu_client


def set_feishu_agent(agent_id: str) -> FeishuClient:
    global _feishu_client
    _feishu_client = FeishuClient(agent_id=agent_id)
    return _feishu_client
