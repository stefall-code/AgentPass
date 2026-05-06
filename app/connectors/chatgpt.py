from typing import Dict, List

from .llm_base import LLMConnector


class ChatGPTConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="chatgpt",
                region="us",
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model=settings.OPENAI_MODEL,
            )
        except Exception:
            super().__init__(platform="chatgpt", region="us")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
