from typing import Dict

from .llm_base import LLMConnector


class QwenConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="qwen",
                region="cn",
                api_key=settings.QWEN_API_KEY,
                base_url=settings.QWEN_BASE_URL,
                model=settings.QWEN_MODEL,
            )
        except Exception:
            super().__init__(platform="qwen", region="cn")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
