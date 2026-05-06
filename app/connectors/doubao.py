import logging
import time
from typing import Dict, List, Any

from .llm_base import LLMConnector

logger = logging.getLogger(__name__)


class DoubaoConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="doubao",
                region="cn",
                api_key=settings.DOUBAO_API_KEY,
                base_url=settings.DOUBAO_BASE_URL,
                model=settings.DOUBAO_MODEL,
            )
        except Exception:
            super().__init__(platform="doubao", region="cn")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    async def _real_chat(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> Dict[str, Any]:
        if not self.is_configured:
            return {"error": "not configured", "mock": True}

        try:
            import httpx
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"

            input_items = []
            for m in messages:
                if m.get("role") == "system":
                    input_items.append({
                        "role": "system",
                        "content": [{"type": "input_text", "text": m["content"]}],
                    })
                else:
                    input_items.append({
                        "role": m.get("role", "user"),
                        "content": [{"type": "input_text", "text": m.get("content", "")}],
                    })

            payload = {
                "model": self._model,
                "input": input_items,
            }

            url = f"{self._base_url}/responses"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                data = resp.json()

                if resp.status_code == 200:
                    output_items = data.get("output", [])
                    content_text = ""
                    for item in output_items:
                        if item.get("type") == "message":
                            for c in item.get("content", []):
                                if c.get("type") == "output_text":
                                    content_text += c.get("text", "")

                    usage = data.get("usage", {})
                    self._call_stats["total_calls"] += 1
                    self._call_stats["total_tokens"] += usage.get("total_tokens", 0)
                    self._call_stats["last_call_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                    return {
                        "success": True,
                        "content": content_text,
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
            logger.error("Doubao _real_chat error: %s", e)
            return {"success": False, "error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        import time as _t
        now = _t.time()
        if self._health_cache and (now - self._last_health_at) < 60:
            return self._health_cache

        if not self.is_configured:
            result = {
                "platform": self.platform,
                "region": self.region,
                "status": "not_configured",
                "latency": 0,
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
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "ping"}]}],
            }
            url = f"{self._base_url}/responses"
            start = _t.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
            latency = round((_t.time() - start) * 1000, 1)

            if resp.status_code == 200:
                result = {
                    "platform": self.platform,
                    "region": self.region,
                    "status": "online",
                    "latency": latency,
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
                "mode": "degraded",
                "error": str(e),
                "source": "real_health_check",
            }

        self._health_cache = result
        self._last_health_at = now
        return result
