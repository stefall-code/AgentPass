import logging
import time
from typing import Dict, List, Any

from .llm_base import LLMConnector

logger = logging.getLogger(__name__)


class ClaudeConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="claude",
                region="us",
                api_key=settings.ANTHROPIC_API_KEY,
                base_url=settings.ANTHROPIC_BASE_URL,
                model=settings.ANTHROPIC_MODEL,
            )
        except Exception:
            super().__init__(platform="claude", region="us")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

    async def _real_chat(self, messages: List[Dict[str, str]], max_tokens: int = 256) -> Dict[str, Any]:
        if not self.is_configured:
            return {"error": "not configured", "mock": True}

        try:
            import httpx
            headers = self._get_headers()
            headers["Content-Type"] = "application/json"

            system_msg = ""
            user_messages = []
            for m in messages:
                if m.get("role") == "system":
                    system_msg = m.get("content", "")
                else:
                    user_messages.append(m)

            payload = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": user_messages,
            }
            if system_msg:
                payload["system"] = system_msg

            url = f"{self._base_url}/v1/messages"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                data = resp.json()

                if resp.status_code == 200:
                    content_blocks = data.get("content", [])
                    content_text = "".join(
                        b.get("text", "") for b in content_blocks if b.get("type") == "text"
                    )
                    usage = data.get("usage", {})
                    self._call_stats["total_calls"] += 1
                    self._call_stats["total_tokens"] += usage.get("output_tokens", 0)
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
            logger.error("Claude _real_chat error: %s", e)
            return {"success": False, "error": str(e)}
