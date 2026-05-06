from typing import Dict, List

from .llm_base import LLMConnector


class DeepSeekConnector(LLMConnector):

    def __init__(self):
        try:
            from app.config import settings
            super().__init__(
                platform="deepseek",
                region="cn",
                api_key=settings.DEEPSEEK_API_KEY,
                base_url=settings.DEEPSEEK_BASE_URL,
                model=settings.DEEPSEEK_MODEL,
            )
        except Exception:
            super().__init__(platform="deepseek", region="cn")

    def _get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
