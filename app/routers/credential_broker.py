"""
Credential Broker API Router
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.security.credential_broker import (
    request_access, execute_via_broker, register_credential,
    revoke_lease, list_leases,
    get_agent_scopes, set_agent_scope,
    list_vault_entries, get_access_log, get_broker_stats,
)

router = APIRouter(prefix="/broker", tags=["Credential Broker"])


class RequestAccessRequest(BaseModel):
    agent_id: str
    service: str
    operation: str
    context: Optional[Dict[str, Any]] = None


class ExecuteRequest(BaseModel):
    agent_id: str
    service: str
    operation: str
    params: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class RegisterCredentialRequest(BaseModel):
    service: str
    key: str
    value: str
    cred_type: str = Field(default="secret")


class SetScopeRequest(BaseModel):
    agent_id: str
    scopes: List[str]


class RevokeLeaseRequest(BaseModel):
    lease_id: str


@router.post("/request")
async def request_access_endpoint(req: RequestAccessRequest):
    result = request_access(req.agent_id, req.service, req.operation, req.context)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "service": result.service,
        "agent_id": result.agent_id,
        "audit_ref": result.audit_ref,
    }


@router.post("/execute")
async def execute_endpoint(req: ExecuteRequest):
    result = execute_via_broker(req.agent_id, req.service, req.operation, req.params, req.context)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error,
        "service": result.service,
        "agent_id": result.agent_id,
        "audit_ref": result.audit_ref,
    }


@router.post("/register")
async def register_credential_endpoint(req: RegisterCredentialRequest):
    ref_id = register_credential(req.service, req.key, req.value, req.cred_type)
    return {"ref_id": ref_id, "service": req.service, "key": req.key}


@router.get("/vault")
async def list_vault_endpoint():
    return {"entries": list_vault_entries()}


@router.get("/stats")
async def stats_endpoint():
    return get_broker_stats()


@router.get("/scopes")
async def scopes_endpoint(agent_id: Optional[str] = None):
    return get_agent_scopes(agent_id)


@router.post("/scopes")
async def set_scopes_endpoint(req: SetScopeRequest):
    return set_agent_scope(req.agent_id, req.scopes)


@router.get("/leases")
async def leases_endpoint(agent_id: Optional[str] = None):
    return {"leases": list_leases(agent_id)}


@router.post("/leases/revoke")
async def revoke_lease_endpoint(req: RevokeLeaseRequest):
    ok = revoke_lease(req.lease_id)
    return {"revoked": ok, "lease_id": req.lease_id}


@router.get("/access-log")
async def access_log_endpoint(limit: int = 50):
    return {"logs": get_access_log(limit)}


@router.post("/demo")
async def broker_demo():
    steps = []

    steps.append({
        "step": 1,
        "action": "展示传统模式：Agent 直接读取 .env 中的 API Key",
        "traditional_flow": "Agent → .env → FEISHU_APP_SECRET → 直接调用飞书 API",
        "risk": "Agent 可以看到、记录、泄露 app_secret",
        "level": "problem",
    })

    steps.append({
        "step": 2,
        "action": "展示 Broker 模式：Agent 通过 Broker 间接访问",
        "broker_flow": "Agent → Broker.request(agent_id, service, op) → Broker 注入凭证 → 调用 API → 返回结果",
        "benefit": "Agent 只拿到结果，永远看不到 app_secret",
        "level": "solution",
    })

    r1 = request_access("data_agent", "feishu", "read")
    steps.append({
        "step": 3,
        "action": "data_agent 请求访问 feishu:read（有权限）",
        "agent_id": "data_agent",
        "service": "feishu",
        "operation": "read",
        "success": r1.success,
        "lease_id": r1.data.get("lease_id", "")[:16] if r1.success else None,
        "credential_keys": r1.data.get("credential_keys", []) if r1.success else [],
        "agent_saw_credentials": False,
        "key_insight": "Agent 只拿到 lease_id，看不到 app_id/app_secret 的值",
        "level": "L1_grant",
    })

    r2 = request_access("external_agent", "feishu", "read")
    steps.append({
        "step": 4,
        "action": "external_agent 请求访问 feishu:read（无权限）",
        "agent_id": "external_agent",
        "service": "feishu",
        "operation": "read",
        "success": r2.success,
        "error": r2.error,
        "key_insight": "外部 Agent 没有被授权任何服务范围，请求被拒绝",
        "level": "L1_deny",
    })

    r3 = request_access("data_agent", "bitable", "finance:read")
    steps.append({
        "step": 5,
        "action": "data_agent 请求访问 bitable:finance:read（有权限）",
        "agent_id": "data_agent",
        "service": "bitable",
        "operation": "finance:read",
        "success": r3.success,
        "lease_id": r3.data.get("lease_id", "")[:16] if r3.success else None,
        "key_insight": "细粒度权限控制：data_agent 可以访问 finance 和 hr，但不能访问 sales",
        "level": "L2_scope",
    })

    r4 = request_access("doc_agent", "bitable", "sales:read")
    steps.append({
        "step": 6,
        "action": "doc_agent 请求访问 bitable:sales:read（无权限）",
        "agent_id": "doc_agent",
        "service": "bitable",
        "operation": "sales:read",
        "success": r4.success,
        "error": r4.error,
        "key_insight": "doc_agent 只有 finance 和 hr 的读取权限，sales 被拒绝",
        "level": "L2_scope_deny",
    })

    if r1.success and r1.data:
        lease_id = r1.data.get("lease_id", "")
        if lease_id:
            revoke_ok = revoke_lease(lease_id)
            steps.append({
                "step": 7,
                "action": "撤销 lease（凭证租约）",
                "lease_id": lease_id[:16],
                "revoked": revoke_ok,
                "key_insight": "租约可随时撤销，撤销后 Agent 即使有 lease_id 也无法继续访问",
                "level": "L3_lease",
            })

    vault_entries = list_vault_entries()
    masked_entries = [{"service": e["service"], "key": e["key"], "masked": e["masked"]} for e in vault_entries[:6]]

    steps.append({
        "step": 8,
        "action": "Vault 中的凭证全部脱敏存储",
        "vault_sample": masked_entries,
        "key_insight": "即使 Vault 被访问，也只能看到脱敏值（如 cl****3x），看不到明文",
        "level": "L4_vault",
    })

    stats = get_broker_stats()

    return {
        "title": "凭证经纪人模式演示 (Credential Broker)",
        "architecture": {
            "传统模式": "Agent → .env → API Key → 直接调用第三方 API",
            "Broker 模式": "Agent → Broker → 注入凭证 → 调用 API → 返回结果（Agent 只拿到结果）",
        },
        "steps": steps,
        "stats": stats,
        "key_insight": "凭证经纪人模式的核心价值：Agent 永远不接触真实凭证。"
                       "1) 凭证存储在加密 Vault 中，Agent 无法直接读取；"
                       "2) 通过 scope 控制每个 Agent 可访问的服务和操作；"
                       "3) 凭证访问通过 lease（租约）机制，可随时撤销；"
                       "4) 所有凭证访问都有审计日志。",
        "comparison": {
            "传统 .env": "Agent 可读 ✅ | 权限控制 ❌ | 租约撤销 ❌ | 审计追踪 ❌",
            "Credential Broker": "Agent 不可读 ✅ | Scope 控制 ✅ | 租约撤销 ✅ | 审计追踪 ✅",
        },
    }
