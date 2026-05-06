from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BASE_DIR: Path = _PROJECT_ROOT
    DATABASE_PATH: Path = _PROJECT_ROOT / "agent_identity.db"
    DATABASE_URL: str = ""
    FRONTEND_DIR: Path = _PROJECT_ROOT / "frontend"

    JWT_SECRET: str = "dev-secret-change-me-at-least-32-bytes"
    JWT_ALGORITHM: str = "HS256"
    TOKEN_EXPIRE_MINUTES: int = 60
    DEFAULT_USAGE_LIMIT: int = 1000

    _INSECURE_DEFAULTS = {
        "JWT_SECRET": "dev-secret-change-me-at-least-32-bytes",
    }

    def validate_security(self) -> list[str]:
        warnings = []
        for key, default in self._INSECURE_DEFAULTS.items():
            if getattr(self, key) == default:
                warnings.append(key)
        return warnings

    DENIAL_WINDOW_MINUTES: int = 10
    DENIAL_LOCK_THRESHOLD: int = 10

    CORS_ORIGINS: list[str] = os.environ.get(
        "CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")

    RATE_LIMIT_MAX: int = 30
    RATE_LIMIT_WINDOW: int = 60

    APPROVAL_TIMEOUT_MINUTES: int = 10
    HITL_RISK_THRESHOLD: float = 0.6
    HITL_CRITICAL_ACTIONS: list[str] = ["delete_resource", "write_confidential", "batch_execute"]

    FEISHU_WEBHOOK_URL: str = ""
    FEISHU_APP_ID: str = ""
    FEISHU_APP_SECRET: str = ""
    FEISHU_VERIFICATION_TOKEN: str = ""
    FEISHU_ENCRYPT_KEY: str = ""
    FEISHU_PUBLIC_URL: str = ""

    BITABLE_FINANCE_APP_TOKEN: str = ""
    BITABLE_FINANCE_TABLE_ID: str = ""
    BITABLE_HR_APP_TOKEN: str = ""
    BITABLE_HR_TABLE_ID: str = ""
    BITABLE_SALES_APP_TOKEN: str = ""
    BITABLE_SALES_TABLE_ID: str = ""

    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    GEMINI_API_KEY: str = ""
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    GEMINI_MODEL: str = "gemini-2.0-flash"

    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    QWEN_API_KEY: str = ""
    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL: str = "qwen-plus"

    DOUBAO_API_KEY: str = ""
    DOUBAO_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/v3"
    DOUBAO_MODEL: str = "doubao-pro-32k-241215"

    BCRYPT_ROUNDS: int = 12
    HOST: str = os.environ.get("HOST", "127.0.0.1")
    PORT: int = int(os.environ.get("PORT", "8000"))

    @property
    def effective_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"sqlite:///{self.DATABASE_PATH.as_posix()}"

    DEFAULT_DOCS: list[dict] = [
        {
            "doc_id": "public_brief",
            "content": "Open project brief. Any authenticated agent with read permission can access it.",
            "sensitivity": "public",
        },
        {
            "doc_id": "team_notes",
            "content": "Internal collaboration notes intended for trusted agents only.",
            "sensitivity": "internal",
        },
        {
            "doc_id": "admin_playbook",
            "content": "Confidential runbook. Only admin agents should be allowed to read or modify this file.",
            "sensitivity": "confidential",
        },
    ]

    DEMO_AGENTS: list[dict] = [
        {
            "agent_id": "doc_agent",
            "name": "Document Agent",
            "role": "coordinator",
            "api_key": "doc-agent-key",
            "attributes": {"allowed_resources": ["read:doc", "write:doc:public", "delegate:data_agent", "delegate:external_agent"]},
        },
        {
            "agent_id": "data_agent",
            "name": "Data Agent",
            "role": "data_reader",
            "api_key": "data-agent-key",
            "attributes": {"allowed_resources": ["read:feishu_table:finance", "read:feishu_table:hr", "write:feishu_table:finance"]},
        },
        {
            "agent_id": "external_agent",
            "name": "External Agent",
            "role": "web_reader",
            "api_key": "external-agent-key",
            "attributes": {"allowed_resources": ["read:web"]},
        },
        {
            "agent_id": "agent_admin_demo",
            "name": "Admin Demo Agent",
            "role": "admin",
            "api_key": "admin-demo-key",
            "attributes": {"allowed_resources": ["*"]},
        },
        {
            "agent_id": "agent_operator_demo",
            "name": "Operator Demo Agent",
            "role": "operator",
            "api_key": "operator-demo-key",
            "attributes": {
                "allowed_resources": [
                    "doc:public_brief",
                    "doc:team_notes",
                    "task:sandbox",
                    "api:knowledge_base",
                    "agent:agent_editor_demo",
                    "agent:agent_operator_peer_demo",
                    "agent:agent_operator_demo",
                ]
            },
        },
        {
            "agent_id": "agent_operator_peer_demo",
            "name": "Operator Peer Demo Agent",
            "role": "operator",
            "api_key": "operator-peer-demo-key",
            "attributes": {
                "allowed_resources": [
                    "doc:public_brief",
                    "doc:team_notes",
                    "task:sandbox",
                    "api:knowledge_base",
                    "agent:agent_operator_demo",
                ]
            },
        },
        {
            "agent_id": "agent_editor_demo",
            "name": "Editor Demo Agent",
            "role": "editor",
            "api_key": "editor-demo-key",
            "attributes": {
                "allowed_resources": [
                    "doc:public_brief",
                    "doc:team_notes",
                    "doc:daily_report",
                ]
            },
        },
        {
            "agent_id": "agent_basic_demo",
            "name": "Basic Demo Agent",
            "role": "basic",
            "api_key": "basic-demo-key",
            "attributes": {"allowed_resources": ["doc:public_brief"]},
        },
    ]


settings = Settings()
