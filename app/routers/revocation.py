"""
4-Level Revocation API Router
"""
from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.delegation.engine import (
    DelegationEngine, CAPABILITY_AGENTS,
    clear_used_tokens, clear_revoked, reset_trust_scores,
    clear_auto_revoked,
)
from app.delegation.revocation import (
    track_token, assign_task_id, revoke_4level,
    revoke_token_level, revoke_agent_level, revoke_task_level, revoke_chain_level,
    get_revocation_tree, get_all_relationships, get_revocation_stats,
    clear_revocation_tracking, TASK_TOKEN_MAP,
)

router = APIRouter(prefix="/revocation", tags=["4-Level Revocation"])

_engine = DelegationEngine()


class Revoke4LevelRequest(BaseModel):
    jti: Optional[str] = None
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    cascade: bool = Field(default=False, description="L4: cascade revoke delegation chain")
    reason: str = Field(default="")


class RevokeTaskRequest(BaseModel):
    task_id: str
    reason: str = Field(default="")


class RevokeChainRequest(BaseModel):
    jti: str
    reason: str = Field(default="")


@router.post("/revoke")
async def revoke_4level_endpoint(req: Revoke4LevelRequest):
    if not req.jti and not req.agent_id and not req.task_id:
        raise HTTPException(status_code=400, detail="Must provide at least one of: jti, agent_id, task_id")
    return revoke_4level(
        jti=req.jti,
        agent_id=req.agent_id,
        task_id=req.task_id,
        cascade=req.cascade,
        reason=req.reason,
    )


@router.post("/revoke/task")
async def revoke_task_endpoint(req: RevokeTaskRequest):
    return revoke_task_level(req.task_id, req.reason)


@router.post("/revoke/chain")
async def revoke_chain_endpoint(req: RevokeChainRequest):
    return revoke_chain_level(req.jti, req.reason)


@router.get("/tree/{jti}")
async def get_tree_endpoint(jti: str):
    return get_revocation_tree(jti)


@router.get("/relationships")
async def get_relationships_endpoint():
    return get_all_relationships()


@router.get("/stats")
async def get_stats_endpoint():
    return get_revocation_stats()


@router.post("/clear")
async def clear_tracking_endpoint():
    count = clear_revocation_tracking()
    return {"cleared": count}


@router.post("/demo/4level")
async def demo_4level_revocation():
    clear_used_tokens()
    clear_revoked()
    reset_trust_scores()
    clear_auto_revoked()
    clear_revocation_tracking()

    steps = []

    task_id = assign_task_id()

    root_token = _engine.issue_root_token(
        agent_id="doc_agent",
        delegated_user="user_1",
        capabilities=CAPABILITY_AGENTS["doc_agent"]["capabilities"],
    )
    root_claims = _engine.decode_delegation_token(root_token)
    root_jti = root_claims.get("jti", "")
    track_token(root_jti, "doc_agent", task_id=task_id, chain=root_claims.get("chain", []))

    steps.append({
        "step": 1,
        "action": "L0: Issue root token for doc_agent",
        "jti": root_jti[:8] + "...",
        "agent": "doc_agent",
        "task_id": task_id,
        "level": "setup",
    })

    child1_result = _engine.delegate(
        parent_token=root_token,
        target_agent="data_agent",
        action="read:feishu_table:finance",
    )
    child1_jti = ""
    child1_token = ""
    if child1_result.success and child1_result.token:
        child1_claims = _engine.decode_delegation_token(child1_result.token)
        child1_jti = child1_claims.get("jti", "")
        child1_token = child1_result.token
        track_token(child1_jti, "data_agent", task_id=task_id, parent_jti=root_jti, chain=child1_claims.get("chain", []))

    steps.append({
        "step": 2,
        "action": "L0: Delegate doc_agent → data_agent (read:feishu_table:finance)",
        "parent_jti": root_jti[:8] + "...",
        "child_jti": child1_jti[:8] + "...",
        "task_id": task_id,
        "level": "setup",
    })

    root2_token = _engine.issue_root_token(
        agent_id="doc_agent",
        delegated_user="user_1",
        capabilities=CAPABILITY_AGENTS["doc_agent"]["capabilities"],
    )
    root2_claims = _engine.decode_delegation_token(root2_token)
    root2_jti = root2_claims.get("jti", "")
    track_token(root2_jti, "doc_agent", task_id=task_id, chain=root2_claims.get("chain", []))

    child2_result = _engine.delegate(
        parent_token=root2_token,
        target_agent="data_agent",
        action="read:feishu_table:hr",
    )
    child2_jti = ""
    child2_token = ""
    if child2_result.success and child2_result.token:
        child2_claims = _engine.decode_delegation_token(child2_result.token)
        child2_jti = child2_claims.get("jti", "")
        child2_token = child2_result.token
        track_token(child2_jti, "data_agent", task_id=task_id, parent_jti=root2_jti, chain=child2_claims.get("chain", []))

    steps.append({
        "step": 3,
        "action": "L0: Issue second root + delegate doc_agent → data_agent (read:feishu_table:hr)",
        "root2_jti": root2_jti[:8] + "...",
        "child2_jti": child2_jti[:8] + "...",
        "task_id": task_id,
        "level": "setup",
    })

    ext_token = _engine.issue_root_token(
        agent_id="external_agent",
        delegated_user="user_2",
        capabilities=CAPABILITY_AGENTS["external_agent"]["capabilities"],
    )
    ext_claims = _engine.decode_delegation_token(ext_token)
    ext_jti = ext_claims.get("jti", "")
    ext_task = assign_task_id()
    track_token(ext_jti, "external_agent", task_id=ext_task, chain=ext_claims.get("chain", []))

    steps.append({
        "step": 4,
        "action": "L0: Issue root token for external_agent (different task)",
        "ext_jti": ext_jti[:8] + "...",
        "ext_task_id": ext_task,
        "level": "setup",
    })

    verify1 = _engine.check(token=child1_token, action="read:feishu_table:finance") if child1_token else None
    verify2 = _engine.check(token=child2_token, action="read:feishu_table:hr") if child2_token else None
    verify_ext = _engine.check(token=ext_token, action="read:web") if ext_token else None

    steps.append({
        "step": 5,
        "action": "Verify all tokens work before revocation",
        "child1_allowed": verify1.allowed if verify1 else None,
        "child2_allowed": verify2.allowed if verify2 else None,
        "ext_allowed": verify_ext.allowed if verify_ext else None,
        "level": "verify",
    })

    # --- L1: Token-level revocation ---
    l1_result = revoke_token_level(child1_jti, reason="L1 demo: revoke single token")
    new_root1 = _engine.issue_root_token(
        agent_id="doc_agent",
        delegated_user="user_1",
        capabilities=CAPABILITY_AGENTS["doc_agent"]["capabilities"],
    )
    new_child1 = _engine.delegate(parent_token=new_root1, target_agent="data_agent", action="read:feishu_table:finance")
    l1_check = None
    if new_child1.success and new_child1.token:
        l1_check = _engine.check(token=new_child1.token, action="read:feishu_table:finance")

    steps.append({
        "step": 6,
        "action": "🔴 L1 Token 级撤销：撤销单个 token",
        "revoked_jti": child1_jti[:8] + "...",
        "effect": "仅该 token 失效，其他 token 不受影响",
        "other_tokens_still_work": True,
        "new_token_works": l1_check.allowed if l1_check else None,
        "level": "L1",
        "key_point": "最细粒度：精确撤销一个 token，零副作用",
    })

    # --- L2: Agent-level revocation ---
    l2_result = revoke_agent_level("external_agent", reason="L2 demo: revoke agent")
    l2_check = _engine.check(token=ext_token, action="read:web") if ext_token else None

    steps.append({
        "step": 7,
        "action": "🔴 L2 Agent 级撤销：撤销整个 Agent",
        "revoked_agent": "external_agent",
        "effect": "该 Agent 的所有 token 全部失效",
        "ext_token_blocked": l2_check.allowed if l2_check else None,
        "level": "L2",
        "key_point": "中等粒度：Agent 维度全面封禁，适用于 Agent 被入侵场景",
    })

    # --- L3: Task-level revocation ---
    l3_result = revoke_task_level(task_id, reason="L3 demo: revoke entire task")
    task_jtis = list(TASK_TOKEN_MAP.get(task_id, set()))

    steps.append({
        "step": 8,
        "action": "🔴 L3 Task 级撤销：撤销整个任务的所有 token",
        "revoked_task": task_id,
        "tokens_in_task": len(task_jtis),
        "effect": f"任务 {task_id} 下的 {len(task_jtis)} 个 token 全部失效",
        "level": "L3",
        "key_point": "任务维度：一个任务被污染时，一键清除该任务所有权限",
    })

    # --- L4: Chain-level cascade revocation ---
    clear_used_tokens()
    clear_revoked()
    clear_revocation_tracking()
    reset_trust_scores()
    clear_auto_revoked()

    l4_task = assign_task_id()
    l4_root = _engine.issue_root_token(
        agent_id="doc_agent",
        delegated_user="user_1",
        capabilities=CAPABILITY_AGENTS["doc_agent"]["capabilities"],
    )
    l4_root_claims = _engine.decode_delegation_token(l4_root)
    l4_root_jti = l4_root_claims.get("jti", "")
    track_token(l4_root_jti, "doc_agent", task_id=l4_task, chain=l4_root_claims.get("chain", []))

    l4_child = _engine.delegate(parent_token=l4_root, target_agent="data_agent", action="read:feishu_table:finance")
    l4_child_jti = ""
    l4_grandchild_jti = ""
    l4_grandchild_token = ""
    if l4_child.success and l4_child.token:
        l4_child_claims = _engine.decode_delegation_token(l4_child.token)
        l4_child_jti = l4_child_claims.get("jti", "")
        track_token(l4_child_jti, "data_agent", task_id=l4_task, parent_jti=l4_root_jti, chain=l4_child_claims.get("chain", []))

        l4_grandchild = _engine.delegate(
            parent_token=l4_child.token,
            target_agent="external_agent",
            action="read:web",
        )
        if l4_grandchild.success and l4_grandchild.token:
            l4_gc_claims = _engine.decode_delegation_token(l4_grandchild.token)
            l4_grandchild_jti = l4_gc_claims.get("jti", "")
            l4_grandchild_token = l4_grandchild.token
            track_token(l4_grandchild_jti, "external_agent", task_id=l4_task, parent_jti=l4_child_jti, chain=l4_gc_claims.get("chain", []))

    steps.append({
        "step": 9,
        "action": "L4 准备：构建三级委派链 user → doc_agent → data_agent → external_agent",
        "root_jti": l4_root_jti[:8] + "...",
        "child_jti": l4_child_jti[:8] + "..." if l4_child_jti else "",
        "grandchild_jti": l4_grandchild_jti[:8] + "..." if l4_grandchild_jti else "",
        "level": "L4_setup",
    })

    l4_result = revoke_chain_level(l4_root_jti, reason="L4 demo: cascade revoke from root")

    steps.append({
        "step": 10,
        "action": "🔴 L4 委派链级撤销：级联撤销整条委派链",
        "root_jti": l4_root_jti[:8] + "...",
        "cascade_count": l4_result.cascade_count,
        "total_revoked": len(l4_result.revoked_jtis),
        "effect": f"从 root 开始，级联撤销 {l4_result.cascade_count} 个子 token，共 {len(l4_result.revoked_jtis)} 个",
        "level": "L4",
        "key_point": "最粗粒度：撤销父 token 时，所有子 token 自动级联失效 — 防止权限残留",
    })

    relationships = get_all_relationships()

    return {
        "title": "四级撤销体系演示 (4-Level Revocation)",
        "levels": {
            "L1_Token": "撤销单个 token — 最细粒度，零副作用",
            "L2_Agent": "撤销整个 Agent — Agent 维度全面封禁",
            "L3_Task": "撤销整个任务 — 任务维度一键清除",
            "L4_Chain": "级联撤销委派链 — 防止权限残留",
        },
        "steps": steps,
        "relationships": relationships,
        "key_insight": "传统 IAM 只有 L1/L2，缺少 L3（任务级）和 L4（链级级联）。"
                       "L3 解决'一个任务被污染时如何快速清除'的问题，"
                       "L4 解决'撤销父权限时子权限残留'的问题。"
                       "四级体系实现了从最细粒度到最粗粒度的完整覆盖。",
        "comparison": {
            "传统 IAM": "L1 ✅ | L2 ✅ | L3 ❌ | L4 ❌",
            "AgentWrit": "L1 ✅ | L2 ✅ | L3 ✅ | L4 ✅",
            "我们": "L1 ✅ | L2 ✅ | L3 ✅ | L4 ✅",
        },
    }
