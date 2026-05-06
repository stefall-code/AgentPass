from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy import Column, Integer, Float, String, Text, ForeignKey
from app.db import Base


# SQLAlchemy 模型
class AgentRow(Base):
    __tablename__ = "agents"
    agent_id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False, default="basic")
    api_key_hash = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")
    status_reason = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True, default="{}")
    attributes_json = Column(Text, nullable=True, default="{}")
    created_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)
    last_login_at = Column(String, nullable=True)


class IssuedTokenRow(Base):
    __tablename__ = "issued_tokens"
    jti = Column(String, primary_key=True, index=True)
    agent_id = Column(String, ForeignKey("agents.agent_id"), index=True)
    issued_at = Column(String, nullable=False)
    expires_at = Column(String, nullable=False)
    active = Column(Integer, nullable=False, default=1)
    bound_ip = Column(String, nullable=True)
    usage_limit = Column(Integer, nullable=True)
    usage_count = Column(Integer, nullable=False, default=0)
    refresh_jti = Column(String, nullable=True, index=True)


class DocumentRow(Base):
    __tablename__ = "documents"
    doc_id = Column(String, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    sensitivity = Column(String, nullable=False, default="public")
    updated_by = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)


class AuditLogRow(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    agent_id = Column(String, nullable=True, index=True)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    decision = Column(String, nullable=False)
    reason = Column(Text, nullable=False)
    ip_address = Column(String, nullable=True)
    token_id = Column(String, nullable=True)
    created_at = Column(String, nullable=False)
    context_json = Column(Text, nullable=True)


class DailyStatRow(Base):
    __tablename__ = "daily_stats"
    date = Column(String, primary_key=True)
    total_requests = Column(Integer, nullable=False, default=0)
    allow_count = Column(Integer, nullable=False, default=0)
    deny_count = Column(Integer, nullable=False, default=0)
    high_risk_count = Column(Integer, nullable=False, default=0)
    avg_risk_score = Column(Float, nullable=False, default=0.0)
    total_token_usage = Column(Integer, nullable=False, default=0)
    estimated_cost = Column(Float, nullable=False, default=0.0)
    updated_at = Column(String, nullable=False)


class OpenClawRequest(Base):
    __tablename__ = "openclaw_requests"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    agent_id = Column(String, nullable=True, index=True)
    user = Column(String, nullable=False)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    prompt_hash = Column(String, nullable=True)
    allowed = Column(Integer, nullable=False, default=0)
    risk_score = Column(Float, nullable=False, default=0.0)
    reason = Column(Text, nullable=False)
    created_at = Column(String, nullable=False)


class ApprovalRequest(Base):
    __tablename__ = "approval_requests"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    agent_id = Column(String, nullable=False, index=True)
    action = Column(String, nullable=False)
    resource = Column(String, nullable=False)
    risk_score = Column(Float, nullable=False, default=0.0)
    payload_json = Column(Text, nullable=True)
    status = Column(String, nullable=False, default="pending")
    requested_at = Column(String, nullable=False)
    decided_at = Column(String, nullable=True)
    decided_by = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    timeout_at = Column(String, nullable=False)


class AgentReputationRow(Base):
    __tablename__ = "agent_reputations"
    agent_id = Column(String, primary_key=True, index=True)
    score = Column(Float, nullable=False, default=70.0)
    allow_rate = Column(Float, nullable=False, default=0.5)
    denial_streak = Column(Integer, nullable=False, default=0)
    suspicious_pattern_count = Column(Integer, nullable=False, default=0)
    consistency_bonus = Column(Float, nullable=False, default=0.0)
    trend = Column(String, nullable=False, default="stable")
    last_computed_at = Column(String, nullable=False)
    history_json = Column(Text, nullable=True, default="[]")


class Ed25519KeyRow(Base):
    __tablename__ = "ed25519_keys"
    agent_id = Column(String, primary_key=True, index=True)
    public_key_b64 = Column(String, nullable=False)
    public_key_fingerprint = Column(String, nullable=False)
    registered_at = Column(String, nullable=False)
    auth_count = Column(Integer, nullable=False, default=0)
    last_auth = Column(String, nullable=True)


class TokenRevocationRow(Base):
    __tablename__ = "token_revocations"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    revoke_type = Column(String, nullable=False, index=True)
    revoke_key = Column(String, nullable=False, index=True)
    reason = Column(Text, nullable=True)
    created_at = Column(String, nullable=False)


# Pydantic 模型
class OpenClawRequestRow(BaseModel):
    id: int = Field(default=None)
    agent_id: str
    user: str
    action: str
    resource: str
    prompt_hash: Optional[str] = None
    allowed: bool
    risk_score: float
    reason: str
    created_at: str


class ApprovalRequestRow(BaseModel):
    id: int = Field(default=None)
    agent_id: str
    action: str
    resource: str
    risk_score: float
    status: str
    created_at: str
    decided_at: Optional[str] = None
    decided_by: Optional[str] = None
    reason: Optional[str] = None


class UnifiedEvent(BaseModel):
    """统一事件模型"""
    id: str
    timestamp: str
    platform: str
    region: str
    user: str
    team: str
    action: str
    resource: str
    prompt: str
    output: str
    risk: float
    risk_level: str
    approval_required: bool
    approval_status: str
    cost: float
    token_usage: int
    blocked: bool
    reason: str
