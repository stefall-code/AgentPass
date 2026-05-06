"""
OAuth 2.0 / OIDC Delegation Extension for AI Agents

Inspired by: ICML 2025 "Authenticated Delegation and Authorized AI Agents"
             (Chan, Mahari, South, Pentland, Hardjono, Marro, Whitney, Greenwood)

Key Extensions over standard OAuth 2.0 / OIDC:
  1. Agent-Specific Credentials: ID Tokens with agent claims (agent_id, capabilities, trust_score)
  2. Token Exchange (RFC 8693): Convert between custom JWT delegation and OAuth access tokens
  3. Agent Delegation Scope: OAuth scopes mapped to agent capabilities
  4. Chain of Accountability: Delegation chain embedded in token claims
  5. Natural Language Permission Translation: NL permissions -> OAuth scopes
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import jwt

from app.config import settings

logger = logging.getLogger("agent_system")


AGENT_OIDC_CLAIMS_NAMESPACE = "https://agent-iam.io/claims/"

AGENT_SCOPES = {
    "doc_agent": ["read:documents", "write:documents", "read:feishu_table:finance", "read:feishu_table:hr"],
    "data_agent": ["read:feishu_table:finance", "read:feishu_table:hr", "read:feishu_table:sales", "read:web"],
    "external_agent": [],
    "admin_agent": ["read:documents", "write:documents", "read:feishu_table:finance", "read:feishu_table:hr",
                     "read:feishu_table:sales", "write:feishu_table", "admin:all"],
    "operator_agent": ["read:documents", "read:feishu_table:finance", "read:feishu_table:hr"],
    "editor_agent": ["read:documents", "write:documents"],
}

SCOPE_TO_CAPABILITY = {
    "read:documents": "read:doc",
    "write:documents": "write:doc",
    "read:feishu_table:finance": "read:feishu_table:finance",
    "read:feishu_table:hr": "read:feishu_table:hr",
    "read:feishu_table:sales": "read:feishu_table:sales",
    "write:feishu_table": "write:feishu_table",
    "read:web": "read:web",
    "admin:all": "admin:all",
}

NL_PERMISSION_MAP = {
    "读取财务数据": ["read:feishu_table:finance"],
    "查看HR信息": ["read:feishu_table:hr"],
    "读取销售数据": ["read:feishu_table:sales"],
    "编辑文档": ["write:documents"],
    "查看文档": ["read:documents"],
    "管理所有": ["admin:all"],
    "read financial data": ["read:feishu_table:finance"],
    "view HR info": ["read:feishu_table:hr"],
    "edit documents": ["write:documents"],
    "view documents": ["read:documents"],
    "manage everything": ["admin:all"],
}

_AUTH_CODES: Dict[str, Dict[str, Any]] = {}
_ISSUED_OAUTH_TOKENS: Dict[str, Dict[str, Any]] = {}
_OAUTH_LOG: List[Dict[str, Any]] = []


@dataclass
class AgentTokenSet:
    access_token: str
    id_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str = ""
    refresh_token: Optional[str] = None


def authorize_agent(
    user_id: str,
    agent_id: str,
    requested_scopes: List[str],
    redirect_uri: Optional[str] = None,
    nl_permission: Optional[str] = None,
) -> Dict[str, Any]:
    if nl_permission:
        mapped = _translate_nl_permission(nl_permission)
        if mapped:
            requested_scopes = list(set(requested_scopes + mapped))

    allowed_scopes = AGENT_SCOPES.get(agent_id, [])
    granted_scopes = [s for s in requested_scopes if s in allowed_scopes]
    denied_scopes = [s for s in requested_scopes if s not in allowed_scopes]

    if not granted_scopes:
        _log_oauth("authorize_denied", user_id, agent_id, requested_scopes, "no_valid_scopes")
        return {
            "authorized": False,
            "reason": f"Agent '{agent_id}' has no matching scopes for: {', '.join(requested_scopes)}",
            "denied_scopes": denied_scopes,
        }

    from app.delegation.engine import get_trust_score
    trust_score = get_trust_score(agent_id)

    if trust_score < 0.5:
        _log_oauth("authorize_denied", user_id, agent_id, requested_scopes, f"trust_too_low:{trust_score:.2f}")
        return {
            "authorized": False,
            "reason": f"Agent trust score too low: {trust_score:.2f} (minimum: 0.50)",
            "trust_score": trust_score,
        }

    code = f"auth_code_{uuid.uuid4().hex[:16]}"
    _AUTH_CODES[code] = {
        "user_id": user_id,
        "agent_id": agent_id,
        "granted_scopes": granted_scopes,
        "denied_scopes": denied_scopes,
        "trust_score": trust_score,
        "created_at": time.time(),
        "expires_at": time.time() + 600,
        "redirect_uri": redirect_uri,
        "nl_permission": nl_permission,
    }

    _log_oauth("authorize_granted", user_id, agent_id, granted_scopes, f"code={code[:16]}")

    return {
        "authorized": True,
        "code": code,
        "granted_scopes": granted_scopes,
        "denied_scopes": denied_scopes,
        "trust_score": trust_score,
        "expires_in": 600,
    }


def exchange_code(code: str, client_id: str = "", client_secret: str = "") -> AgentTokenSet:
    auth_data = _AUTH_CODES.get(code)
    if not auth_data:
        raise ValueError("Invalid authorization code")

    if time.time() > auth_data["expires_at"]:
        del _AUTH_CODES[code]
        raise ValueError("Authorization code expired")

    del _AUTH_CODES[code]

    user_id = auth_data["user_id"]
    agent_id = auth_data["agent_id"]
    scopes = auth_data["granted_scopes"]
    trust_score = auth_data["trust_score"]

    access_token = _issue_access_token(user_id, agent_id, scopes, trust_score)
    id_token = _issue_id_token(user_id, agent_id, scopes, trust_score)
    refresh_token = f"rt_{uuid.uuid4().hex[:24]}"

    _ISSUED_OAUTH_TOKENS[access_token[:16]] = {
        "user_id": user_id,
        "agent_id": agent_id,
        "scopes": scopes,
        "trust_score": trust_score,
        "issued_at": time.time(),
        "refresh_token": refresh_token,
    }

    _log_oauth("token_issued", user_id, agent_id, scopes, f"access={access_token[:12]}...")

    return AgentTokenSet(
        access_token=access_token,
        id_token=id_token,
        token_type="Bearer",
        expires_in=3600,
        scope=" ".join(scopes),
        refresh_token=refresh_token,
    )


def token_exchange(
    subject_token: str,
    subject_token_type: str = "urn:ietf:params:oauth:token-type:jwt",
    requested_token_type: str = "urn:ietf:params:oauth:token-type:access_token",
    requested_scopes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    try:
        claims = jwt.decode(subject_token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return {"error": "invalid_grant", "error_description": "Subject token expired"}
    except jwt.InvalidTokenError as e:
        return {"error": "invalid_grant", "error_description": f"Invalid subject token: {e}"}

    agent_id = claims.get("agent_id", claims.get("sub", ""))
    delegation_chain = claims.get("chain", [])
    original_scopes = claims.get("capabilities", [])
    trust_score = claims.get("trust_score", 1.0)

    if requested_scopes:
        allowed = AGENT_SCOPES.get(agent_id, [])
        granted = [s for s in requested_scopes if s in allowed]
    else:
        granted = _capabilities_to_scopes(original_scopes)

    if not granted:
        return {"error": "invalid_scope", "error_description": "No valid scopes for agent"}

    user_id = claims.get("delegated_user", claims.get("sub", ""))

    new_access_token = _issue_access_token(user_id, agent_id, granted, trust_score)
    new_id_token = _issue_id_token(user_id, agent_id, granted, trust_score)

    _log_oauth("token_exchanged", user_id, agent_id, granted,
               f"from={subject_token_type} to={requested_token_type}")

    return {
        "access_token": new_access_token,
        "id_token": new_id_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(granted),
        "issued_token_type": requested_token_type,
        "delegation_chain": delegation_chain,
    }


def validate_access_token(access_token: str) -> Dict[str, Any]:
    try:
        claims = jwt.decode(access_token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except jwt.InvalidTokenError as e:
        return {"valid": False, "error": str(e)}

    agent_claims = claims.get(AGENT_OIDC_CLAIMS_NAMESPACE, {})
    return {
        "valid": True,
        "sub": claims.get("sub"),
        "agent_id": agent_claims.get("agent_id"),
        "scopes": claims.get("scope", "").split(),
        "trust_score": agent_claims.get("trust_score"),
        "capabilities": agent_claims.get("capabilities"),
        "delegation_chain": agent_claims.get("delegation_chain"),
        "expires_at": claims.get("exp"),
        "issuer": claims.get("iss"),
    }


def _issue_access_token(user_id: str, agent_id: str, scopes: List[str], trust_score: float) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iss": f"https://agent-iam.io/oauth/{agent_id}",
        "sub": f"agent:{agent_id}",
        "aud": "https://agent-iam.io/resources",
        "exp": int((now + timedelta(hours=1)).timestamp()),
        "iat": int(now.timestamp()),
        "jti": uuid.uuid4().hex,
        "scope": " ".join(scopes),
        "client_id": agent_id,
        AGENT_OIDC_CLAIMS_NAMESPACE: {
            "agent_id": agent_id,
            "delegated_by": user_id,
            "trust_score": trust_score,
            "capabilities": scopes,
            "delegation_chain": [user_id, agent_id],
            "agent_type": "AI",
            "auth_method": "oauth2_delegation",
        },
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def _issue_id_token(user_id: str, agent_id: str, scopes: List[str], trust_score: float) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "iss": "https://agent-iam.io",
        "sub": f"agent:{agent_id}",
        "aud": agent_id,
        "exp": int((now + timedelta(hours=1)).timestamp()),
        "iat": int(now.timestamp()),
        "auth_time": int(now.timestamp()),
        "nonce": uuid.uuid4().hex[:16],
        "acr": "urn:agent-iam:acr:delegated-agent",
        "amr": ["oauth2", "delegation"],
        AGENT_OIDC_CLAIMS_NAMESPACE: {
            "agent_id": agent_id,
            "delegated_by": user_id,
            "trust_score": trust_score,
            "capabilities": scopes,
            "delegation_chain": [user_id, agent_id],
            "agent_type": "AI",
            "is_autonomous": True,
            "permission_source": "delegation",
            "audit_trail": f"https://agent-iam.io/audit/{agent_id}",
        },
        "name": f"Agent {agent_id}",
        "type": "ai_agent",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def _translate_nl_permission(nl_text: str) -> List[str]:
    scopes = []
    nl_lower = nl_text.lower()
    for pattern, mapped_scopes in NL_PERMISSION_MAP.items():
        if pattern.lower() in nl_lower:
            scopes.extend(mapped_scopes)
    return list(set(scopes))


def _capabilities_to_scopes(capabilities: List[str]) -> List[str]:
    scopes = []
    for cap in capabilities:
        if cap in SCOPE_TO_CAPABILITY.values():
            for scope, capability in SCOPE_TO_CAPABILITY.items():
                if capability == cap:
                    scopes.append(scope)
                    break
        else:
            scopes.append(cap)
    return list(set(scopes))


def get_oidc_discovery() -> Dict[str, Any]:
    base = "https://agent-iam.io"
    return {
        "issuer": base,
        "authorization_endpoint": f"{base}/api/oauth/authorize",
        "token_endpoint": f"{base}/api/oauth/token",
        "userinfo_endpoint": f"{base}/api/oauth/userinfo",
        "jwks_uri": f"{base}/.well-known/jwks.json",
        "registration_endpoint": f"{base}/api/oauth/register",
        "scopes_supported": list(SCOPE_TO_CAPABILITY.keys()),
        "response_types_supported": ["code", "token", "id_token", "code id_token"],
        "grant_types_supported": ["authorization_code", "urn:ietf:params:oauth:grant-type:token-exchange"],
        "subject_types_supported": ["public", "pairwise"],
        "id_token_signing_alg_values_supported": ["RS256", "HS256"],
        "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post", "private_key_jwt"],
        "claims_supported": [
            "sub", "iss", "aud", "exp", "iat", "nonce",
            f"{AGENT_OIDC_CLAIMS_NAMESPACE}agent_id",
            f"{AGENT_OIDC_CLAIMS_NAMESPACE}trust_score",
            f"{AGENT_OIDC_CLAIMS_NAMESPACE}capabilities",
            f"{AGENT_OIDC_CLAIMS_NAMESPACE}delegation_chain",
            f"{AGENT_OIDC_CLAIMS_NAMESPACE}agent_type",
        ],
        "agent_claims_namespace": AGENT_OIDC_CLAIMS_NAMESPACE,
    }


def get_oauth_stats() -> Dict[str, Any]:
    return {
        "auth_codes_issued": len(_AUTH_CODES),
        "oauth_tokens_issued": len(_ISSUED_OAUTH_TOKENS),
        "oauth_log_entries": len(_OAUTH_LOG),
        "supported_agents": list(AGENT_SCOPES.keys()),
        "supported_scopes": list(SCOPE_TO_CAPABILITY.keys()),
        "nl_permission_patterns": len(NL_PERMISSION_MAP),
        "oidc_discovery_available": True,
    }


def _log_oauth(action: str, user_id: str, agent_id: str, scopes: List[str], detail: str) -> None:
    _OAUTH_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "user_id": user_id,
        "agent_id": agent_id,
        "scopes": scopes,
        "detail": detail,
    })
    if len(_OAUTH_LOG) > 200:
        _OAUTH_LOG.pop(0)
