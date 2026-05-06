"""
OAuth 2.0 / OIDC Delegation Extension API Router
"""
from __future__ import annotations

import json
from typing import Optional, List
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.security.oauth_delegation import (
    authorize_agent, exchange_code, token_exchange, validate_access_token,
    get_oidc_discovery, get_oauth_stats,
    AGENT_OIDC_CLAIMS_NAMESPACE, NL_PERMISSION_MAP,
)

router = APIRouter(prefix="/oauth", tags=["OAuth 2.0 / OIDC"])


class AuthorizeRequest(BaseModel):
    user_id: str
    agent_id: str
    scopes: List[str]
    redirect_uri: Optional[str] = None
    nl_permission: Optional[str] = None


class TokenRequest(BaseModel):
    grant_type: str = Field(default="authorization_code")
    code: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None


class TokenExchangeRequest(BaseModel):
    subject_token: str
    subject_token_type: str = Field(default="urn:ietf:params:oauth:token-type:jwt")
    requested_token_type: str = Field(default="urn:ietf:params:oauth:token-type:access_token")
    requested_scopes: Optional[List[str]] = None


class ValidateTokenRequest(BaseModel):
    access_token: str


@router.post("/authorize")
async def authorize_endpoint(req: AuthorizeRequest):
    return authorize_agent(req.user_id, req.agent_id, req.scopes, req.redirect_uri, req.nl_permission)


@router.post("/token")
async def token_endpoint(req: TokenRequest):
    if req.grant_type == "authorization_code":
        if not req.code:
            return JSONResponse(content={"error": "invalid_request", "error_description": "Missing code"}, status_code=400)
        try:
            token_set = exchange_code(req.code, req.client_id or "", req.client_secret or "")
            return {
                "access_token": token_set.access_token,
                "id_token": token_set.id_token,
                "token_type": token_set.token_type,
                "expires_in": token_set.expires_in,
                "scope": token_set.scope,
                "refresh_token": token_set.refresh_token,
            }
        except ValueError as e:
            return JSONResponse(content={"error": "invalid_grant", "error_description": str(e)}, status_code=400)
    elif req.grant_type == "urn:ietf:params:oauth:grant-type:token-exchange":
        return JSONResponse(content={"error": "invalid_request", "error_description": "Use /oauth/exchange endpoint"}, status_code=400)
    else:
        return JSONResponse(content={"error": "unsupported_grant_type"}, status_code=400)


@router.post("/exchange")
async def token_exchange_endpoint(req: TokenExchangeRequest):
    result = token_exchange(req.subject_token, req.subject_token_type, req.requested_token_type, req.requested_scopes)
    if "error" in result:
        return JSONResponse(content=result, status_code=400)
    return result


@router.post("/validate")
async def validate_endpoint(req: ValidateTokenRequest):
    return validate_access_token(req.access_token)


@router.get("/.well-known/openid-configuration")
async def oidc_discovery_endpoint():
    return JSONResponse(content=get_oidc_discovery())


@router.get("/stats")
async def stats_endpoint():
    return get_oauth_stats()


@router.post("/demo")
async def oauth_demo():
    steps = []

    steps.append({
        "step": 1,
        "action": "用户授权 Agent — OAuth 2.0 Authorization Request",
        "flow": "User → Agent IAM → 授权码 (Authorization Code)",
        "detail": "用户明确授权 Agent 可代表自己执行特定操作",
        "level": "authorize",
    })

    auth_result = authorize_agent(
        user_id="user_alice",
        agent_id="data_agent",
        requested_scopes=["read:feishu_table:finance", "read:feishu_table:sales"],
    )
    steps.append({
        "step": 2,
        "action": "授权结果 — Scope 精细化控制",
        "agent_id": "data_agent",
        "authorized": auth_result.get("authorized", False),
        "granted_scopes": auth_result.get("granted_scopes", []),
        "denied_scopes": auth_result.get("denied_scopes", []),
        "trust_score": auth_result.get("trust_score"),
        "key_point": "data_agent 有 finance 读取权限但无 sales 权限，OAuth scope 精确控制",
        "level": "authorize_result",
    })

    if not auth_result.get("authorized"):
        steps.append({
            "step": 3,
            "action": "授权被拒绝",
            "reason": auth_result.get("reason"),
            "level": "error",
        })
    else:
        code = auth_result["code"]
        try:
            token_set = exchange_code(code)
            steps.append({
                "step": 3,
                "action": "Token Exchange — 授权码换取 Access Token + ID Token",
                "flow": "Authorization Code → Access Token + ID Token",
                "token_type": token_set.token_type,
                "expires_in": token_set.expires_in,
                "scope": token_set.scope,
                "key_point": "ID Token 包含 Agent 特有的 OIDC Claims（agent_id, trust_score, capabilities, delegation_chain）",
                "level": "token_exchange",
            })

            access_claims = validate_access_token(token_set.access_token)
            agent_claims = access_claims.get(AGENT_OIDC_CLAIMS_NAMESPACE, {})
            steps.append({
                "step": 4,
                "action": "Access Token 验证 — 解码 Agent OIDC Claims",
                "valid": access_claims.get("valid"),
                "sub": access_claims.get("sub"),
                "agent_id": agent_claims.get("agent_id"),
                "trust_score": agent_claims.get("trust_score"),
                "capabilities": agent_claims.get("capabilities"),
                "delegation_chain": agent_claims.get("delegation_chain"),
                "agent_type": agent_claims.get("agent_type"),
                "key_point": "企业 IdP（Azure AD/Okta）可以通过自定义 Claims 读取这些 Agent 元数据",
                "level": "validate",
            })

            id_token_parts = token_set.id_token.split(".")
            import base64
            try:
                id_payload = json.loads(base64.urlsafe_b64decode(id_token_parts[1] + "=="))
            except Exception:
                id_payload = {}
            id_agent_claims = id_payload.get(AGENT_OIDC_CLAIMS_NAMESPACE, {})
            steps.append({
                "step": 5,
                "action": "ID Token — OIDC Agent 扩展 Claims",
                "acr": id_payload.get("acr"),
                "amr": id_payload.get("amr"),
                "agent_id": id_agent_claims.get("agent_id"),
                "delegated_by": id_agent_claims.get("delegated_by"),
                "is_autonomous": id_agent_claims.get("is_autonomous"),
                "permission_source": id_agent_claims.get("permission_source"),
                "audit_trail": id_agent_claims.get("audit_trail"),
                "key_point": "OIDC 标准字段（acr/amr）+ Agent 扩展 Claims 构成完整的 Agent 身份凭证",
                "level": "id_token",
            })

            from app.delegation.engine import DelegationEngine, CAPABILITY_AGENTS
            engine = DelegationEngine()
            root_token = engine.issue_root_token(
                agent_id="data_agent",
                delegated_user="user_alice",
                capabilities=CAPABILITY_AGENTS["data_agent"]["capabilities"],
            )
            exchange_result = token_exchange(
                subject_token=root_token,
                subject_token_type="urn:ietf:params:oauth:token-type:jwt",
                requested_scopes=["read:feishu_table:finance"],
            )
            steps.append({
                "step": 6,
                "action": "RFC 8693 Token Exchange — 自定义 JWT → OAuth Access Token",
                "flow": "Custom Delegation JWT → OAuth 2.0 Access Token",
                "exchange_success": "error" not in exchange_result,
                "new_scope": exchange_result.get("scope"),
                "delegation_chain": exchange_result.get("delegation_chain"),
                "key_point": "RFC 8693 Token Exchange 让我们的自定义委派 JWT 可以无缝转换为标准 OAuth Token",
                "level": "token_exchange_rfc8693",
            })

        except ValueError as e:
            steps.append({
                "step": 3,
                "action": "Token Exchange 失败",
                "error": str(e),
                "level": "error",
            })

    nl_result = authorize_agent(
        user_id="user_bob",
        agent_id="doc_agent",
        requested_scopes=[],
        nl_permission="读取财务数据",
    )
    steps.append({
        "step": 7,
        "action": "自然语言权限翻译 — NL → OAuth Scopes",
        "nl_input": "读取财务数据",
        "translated_scopes": NL_PERMISSION_MAP.get("读取财务数据", []),
        "authorized": nl_result.get("authorized", False),
        "granted_scopes": nl_result.get("granted_scopes", []),
        "denied_scopes": nl_result.get("denied_scopes", []),
        "key_point": "用户用自然语言描述权限，系统自动翻译成可审计的 OAuth Scopes",
        "level": "nl_translation",
    })

    discovery = get_oidc_discovery()
    steps.append({
        "step": 8,
        "action": "OIDC Discovery — 标准化端点发布",
        "issuer": discovery["issuer"],
        "grant_types": discovery["grant_types_supported"],
        "agent_claims": [c.split("/")[-1] for c in discovery["claims_supported"] if AGENT_OIDC_CLAIMS_NAMESPACE in c],
        "key_point": "OIDC Discovery 让企业 IdP 自动发现我们的 Agent 扩展端点和 Claims",
        "level": "discovery",
    })

    return {
        "title": "OAuth 2.0 / OIDC 委派扩展演示",
        "paper": "ICML 2025: Authenticated Delegation and Authorized AI Agents",
        "steps": steps,
        "key_insight": "OAuth 2.0 / OIDC 委派扩展的核心价值："
                       "1) 兼容企业现有 IdP（Azure AD、Okta、Keycloak），无需定制集成；"
                       "2) Agent 特有的 OIDC Claims（trust_score, capabilities, delegation_chain）让资源服务器能做出更智能的授权决策；"
                       "3) RFC 8693 Token Exchange 让自定义委派 JWT 和标准 OAuth Token 无缝互转；"
                       "4) 自然语言权限翻译让非技术用户也能精确控制 Agent 权限。",
        "comparison": {
            "自定义 JWT": "企业 IdP 不识别 ✅ | 标准兼容 ❌ | Token 交换 ❌ | NL 权限 ❌",
            "OAuth 2.0 + OIDC": "企业 IdP 识别 ✅ | 标准兼容 ✅ | RFC 8693 ✅ | NL→Scopes ✅",
        },
        "idp_compatibility": {
            "Azure AD": "通过 optionalClaims 注入 agent claims",
            "Okta": "通过自定义属性添加到 ID Token",
            "Keycloak": "通过 Protocol Mapper 映射 agent claims",
            "Auth0": "通过 Rules/Hooks 注入 agent claims",
        },
    }
