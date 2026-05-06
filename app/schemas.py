from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RegisterAgentRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    role: str = Field(default="basic")
    metadata: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class RegisterAgentResponse(BaseModel):
    message: str
    agent_id: str
    name: str
    role: str
    api_key: str


class LoginRequest(BaseModel):
    agent_id: str
    api_key: str
    bound_ip: str | None = None
    usage_limit: int = Field(default=1000, ge=1, le=1000)
    expires_in_minutes: int = Field(default=60, ge=5, le=1440)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str
    expires_at: str
    jti: str
    usage_limit: int
    bound_ip: str | None = None
    role: str | None = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str
    bound_ip: str | None = None
    usage_limit: int | None = Field(default=None, ge=1, le=1000)
    expires_in_minutes: int | None = Field(default=None, ge=5, le=1440)


class BatchLoginItemRequest(BaseModel):
    agent_id: str
    api_key: str
    bound_ip: str | None = None
    usage_limit: int = Field(default=1000, ge=1, le=1000)
    expires_in_minutes: int = Field(default=60, ge=5, le=1440)


class BatchLoginRequest(BaseModel):
    admin_agent_id: str
    admin_api_key: str
    agents: list[BatchLoginItemRequest] = Field(..., min_length=1, max_length=20)


class BatchLoginItemResponse(BaseModel):
    agent_id: str
    success: bool
    error: str | None = None
    token: Optional[TokenResponse] = None


class TokenIntrospectionResponse(BaseModel):
    jti: str
    agent_id: str
    role: str
    active: bool
    issued_at: str
    expires_at: str
    bound_ip: str | None = None
    usage_limit: int
    usage_count: int
    usage_remaining: int


class AgentProfileResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str
    permissions: list[str]
    metadata: dict[str, Any]
    attributes: dict[str, Any]
    created_at: str
    last_login_at: str | None = None


class DocumentWriteRequest(BaseModel):
    content: str = Field(..., min_length=1)
    sensitivity: str = Field(default="internal")


class DocumentResponse(BaseModel):
    doc_id: str
    content: str
    sensitivity: str
    updated_by: str | None = None
    updated_at: str


class TaskExecutionRequest(BaseModel):
    task_name: str = Field(..., min_length=2, max_length=128)
    resource: str = Field(default="sandbox", min_length=2, max_length=64)
    parameters: dict[str, Any] = Field(default_factory=dict)


class TaskExecutionResponse(BaseModel):
    execution_id: str
    status: str
    task_name: str
    resource: str
    result_preview: str


class IntegrationCallRequest(BaseModel):
    service_name: str = Field(..., min_length=2, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class IntegrationCallResponse(BaseModel):
    service_name: str
    status: str
    echoed_payload: dict[str, Any]
    message: str


class DelegateTaskRequest(BaseModel):
    target_agent_id: str
    task_name: str = Field(..., min_length=2, max_length=128)
    resource: str = Field(default="sandbox", min_length=2, max_length=64)


class DelegationResponse(BaseModel):
    delegation_id: str
    source_agent_id: str
    target_agent_id: str
    task_name: str
    status: str


class AgentSummaryResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str
    status_reason: str | None = None
    created_at: str
    last_login_at: str | None = None


class AgentStatusUpdateRequest(BaseModel):
    status: str
    reason: str = Field(default="Updated by administrator.", min_length=4, max_length=255)


class AgentStatusUpdateResponse(BaseModel):
    agent_id: str
    status: str
    status_reason: str | None = None
    updated_at: str


class AgentUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=2, max_length=64)
    attributes: dict[str, Any] | None = None


class AgentUpdateResponse(BaseModel):
    agent_id: str
    name: str
    attributes: dict[str, Any]
    updated_at: str


class ActionMessageResponse(BaseModel):
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class AuditEventResponse(BaseModel):
    id: int
    agent_id: str | None = None
    action: str
    resource: str
    decision: str
    reason: str
    ip_address: str | None = None
    token_id: str | None = None
    created_at: str
    context: dict[str, Any]


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str
    created_at: str
    last_login_at: str | None = None


class AgentDetailResponse(BaseModel):
    agent_id: str
    name: str
    role: str
    status: str
    status_reason: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    last_login_at: str | None = None


class AgentCreateRequest(BaseModel):
    agent_id: str = Field(..., min_length=2, max_length=128)
    name: str = Field(..., min_length=2, max_length=64)
    role: str = Field(default="basic")
    api_key: str = Field(..., min_length=4, max_length=256)
    attributes: dict[str, Any] = Field(default_factory=dict)


class InteractionRequest(BaseModel):
    interaction_type: str = Field(..., min_length=2, max_length=64)


class InteractionResponse(BaseModel):
    source_agent_id: str
    target_agent_id: str
    interaction_type: str
    status: str
    message: str
