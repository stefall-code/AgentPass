from typing import Dict, List

from .llm_base import LLMConnector


class GeminiConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="gemini",
                region="us",
                api_key=settings.GEMINI_API_KEY,
                base_url=settings.GEMINI_BASE_URL,
                model=settings.GEMINI_MODEL,
            )
        except Exception:
            super().__init__(platform="gemini", region="us")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
