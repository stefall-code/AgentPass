"""
4-Level Revocation Engine — Inspired by AgentWrit

Levels:
  L1 Token    — Revoke a single token by jti
  L2 Agent    — Revoke all tokens for an agent (auto-revoke)
  L3 Task     — Revoke all tokens belonging to a task_id
  L4 Chain    — Cascade revoke: revoke a token + all its delegation children

Token Relationship Tracking:
  - parent_jti → child_jti (delegation chain)
  - task_id → [jti, ...] (task grouping)
  - agent_id → [jti, ...] (agent grouping)
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field

from app.delegation.engine import (
    revoke_token_by_jti,
    revoke_tokens_by_agent,
    REVOKED_TOKENS,
    REVOKED_AGENTS,
    REVOKED_USERS,
)

logger = logging.getLogger("agent_system")


PARENT_CHILD_MAP: Dict[str, Set[str]] = {}
TASK_TOKEN_MAP: Dict[str, Set[str]] = {}
TOKEN_TASK_MAP: Dict[str, str] = {}
TOKEN_AGENT_MAP: Dict[str, str] = {}
TOKEN_CHAIN_MAP: Dict[str, List[str]] = {}

REVOCATION_LOG: List[Dict[str, Any]] = []


def track_token(
    jti: str,
    agent_id: str,
    task_id: Optional[str] = None,
    parent_jti: Optional[str] = None,
    chain: Optional[List[str]] = None,
) -> None:
    TOKEN_AGENT_MAP[jti] = agent_id

    if chain:
        TOKEN_CHAIN_MAP[jti] = list(chain)

    if task_id:
        TOKEN_TASK_MAP[jti] = task_id
        if task_id not in TASK_TOKEN_MAP:
            TASK_TOKEN_MAP[task_id] = set()
        TASK_TOKEN_MAP[task_id].add(jti)

    if parent_jti:
        if parent_jti not in PARENT_CHILD_MAP:
            PARENT_CHILD_MAP[parent_jti] = set()
        PARENT_CHILD_MAP[parent_jti].add(jti)


def assign_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:12]}"


def get_children(jti: str) -> Set[str]:
    return PARENT_CHILD_MAP.get(jti, set())


def get_all_descendants(jti: str) -> Set[str]:
    visited: Set[str] = set()
    queue = [jti]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        children = get_children(current)
        queue.extend(children)
    visited.discard(jti)
    return visited


def get_task_tokens(task_id: str) -> Set[str]:
    return TASK_TOKEN_MAP.get(task_id, set())


@dataclass
class RevocationResult:
    level: str
    revoked_jtis: List[str] = field(default_factory=list)
    revoked_agents: List[str] = field(default_factory=list)
    cascade_count: int = 0
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


def revoke_token_level(jti: str, reason: str = "") -> RevocationResult:
    revoke_token_by_jti(jti)
    _log_revocation("L1_token", [jti], [], reason)
    return RevocationResult(
        level="L1_token",
        revoked_jtis=[jti],
        reason=reason or f"Token {jti[:8]}... revoked",
    )


def revoke_agent_level(agent_id: str, reason: str = "") -> RevocationResult:
    revoke_tokens_by_agent(agent_id, reason)
    agent_jtis = [j for j, a in TOKEN_AGENT_MAP.items() if a == agent_id]
    for jti in agent_jtis:
        revoke_token_by_jti(jti)
    _log_revocation("L2_agent", agent_jtis, [agent_id], reason)
    return RevocationResult(
        level="L2_agent",
        revoked_jtis=agent_jtis,
        revoked_agents=[agent_id],
        reason=reason or f"Agent {agent_id} revoked",
    )


def revoke_task_level(task_id: str, reason: str = "") -> RevocationResult:
    token_jtis = list(TASK_TOKEN_MAP.get(task_id, set()))
    for jti in token_jtis:
        revoke_token_by_jti(jti)
    _log_revocation("L3_task", token_jtis, [], reason, task_id=task_id)
    return RevocationResult(
        level="L3_task",
        revoked_jtis=token_jtis,
        reason=reason or f"Task {task_id} revoked ({len(token_jtis)} tokens)",
        details={"task_id": task_id, "token_count": len(token_jtis)},
    )


def revoke_chain_level(jti: str, reason: str = "") -> RevocationResult:
    descendants = get_all_descendants(jti)
    all_jtis = descendants | {jti}

    for jti_to_revoke in all_jtis:
        revoke_token_by_jti(jti_to_revoke)

    _log_revocation("L4_chain", list(all_jtis), [], reason, root_jti=jti)

    return RevocationResult(
        level="L4_chain",
        revoked_jtis=sorted(all_jtis),
        cascade_count=len(descendants),
        reason=reason or f"Chain revoked: root={jti[:8]}... + {len(descendants)} descendants",
        details={"root_jti": jti, "descendant_count": len(descendants)},
    )


def revoke_4level(
    jti: Optional[str] = None,
    agent_id: Optional[str] = None,
    task_id: Optional[str] = None,
    cascade: bool = False,
    reason: str = "",
) -> Dict[str, Any]:
    results: List[RevocationResult] = []

    if jti:
        if cascade:
            result = revoke_chain_level(jti, reason)
        else:
            result = revoke_token_level(jti, reason)
        results.append(result)

    if agent_id:
        result = revoke_agent_level(agent_id, reason)
        results.append(result)

    if task_id:
        result = revoke_task_level(task_id, reason)
        results.append(result)

    all_revoked_jtis = []
    all_revoked_agents = []
    for r in results:
        all_revoked_jtis.extend(r.revoked_jtis)
        all_revoked_agents.extend(r.revoked_agents)

    return {
        "revoked": True,
        "levels_triggered": [r.level for r in results],
        "total_tokens_revoked": len(set(all_revoked_jtis)),
        "agents_revoked": list(set(all_revoked_agents)),
        "results": [
            {
                "level": r.level,
                "revoked_jtis": [j[:8] + "..." for j in r.revoked_jtis],
                "cascade_count": r.cascade_count,
                "reason": r.reason,
                "details": r.details,
            }
            for r in results
        ],
    }


def get_revocation_tree(jti: str) -> Dict[str, Any]:
    descendants = get_all_descendants(jti)
    tree = {
        "root": jti,
        "agent": TOKEN_AGENT_MAP.get(jti, "unknown"),
        "task": TOKEN_TASK_MAP.get(jti),
        "chain": TOKEN_CHAIN_MAP.get(jti, []),
        "children": list(get_children(jti)),
        "descendants": sorted(descendants),
        "total_in_chain": len(descendants) + 1,
        "is_revoked": jti in REVOKED_TOKENS,
    }
    for child_jti in sorted(descendants):
        tree[f"child_{child_jti[:8]}"] = {
            "jti": child_jti,
            "agent": TOKEN_AGENT_MAP.get(child_jti, "unknown"),
            "task": TOKEN_TASK_MAP.get(child_jti),
            "is_revoked": child_jti in REVOKED_TOKENS,
            "children": list(get_children(child_jti)),
        }
    return tree


def get_all_relationships() -> Dict[str, Any]:
    return {
        "parent_child_map": {k: list(v) for k, v in PARENT_CHILD_MAP.items()},
        "task_token_map": {k: list(v) for k, v in TASK_TOKEN_MAP.items()},
        "token_agent_map": dict(TOKEN_AGENT_MAP),
        "token_task_map": dict(TOKEN_TASK_MAP),
        "total_tracked_tokens": len(TOKEN_AGENT_MAP),
        "total_tasks": len(TASK_TOKEN_MAP),
        "total_delegation_edges": sum(len(v) for v in PARENT_CHILD_MAP.values()),
    }


def get_revocation_stats() -> Dict[str, Any]:
    return {
        "L1_token_revocations": len([l for l in REVOCATION_LOG if l["level"] == "L1_token"]),
        "L2_agent_revocations": len([l for l in REVOCATION_LOG if l["level"] == "L2_agent"]),
        "L3_task_revocations": len([l for l in REVOCATION_LOG if l["level"] == "L3_task"]),
        "L4_chain_revocations": len([l for l in REVOCATION_LOG if l["level"] == "L4_chain"]),
        "total_tokens_revoked": len(REVOKED_TOKENS),
        "total_agents_revoked": len(REVOKED_AGENTS),
        "total_users_revoked": len(REVOKED_USERS),
        "tracked_tokens": len(TOKEN_AGENT_MAP),
        "tracked_tasks": len(TASK_TOKEN_MAP),
        "delegation_edges": sum(len(v) for v in PARENT_CHILD_MAP.values()),
        "recent_log": REVOCATION_LOG[-10:],
    }


def clear_revocation_tracking() -> int:
    count = len(TOKEN_AGENT_MAP)
    PARENT_CHILD_MAP.clear()
    TASK_TOKEN_MAP.clear()
    TOKEN_TASK_MAP.clear()
    TOKEN_AGENT_MAP.clear()
    TOKEN_CHAIN_MAP.clear()
    REVOCATION_LOG.clear()
    return count


def _log_revocation(
    level: str,
    jtis: List[str],
    agents: List[str],
    reason: str,
    root_jti: Optional[str] = None,
    task_id: Optional[str] = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "revoked_jtis": [j[:8] + "..." for j in jtis],
        "revoked_agents": agents,
        "reason": reason,
        "jti_count": len(jtis),
    }
    if root_jti:
        entry["root_jti"] = root_jti[:8] + "..."
    if task_id:
        entry["task_id"] = task_id
    REVOCATION_LOG.append(entry)
    if len(REVOCATION_LOG) > 100:
        REVOCATION_LOG.pop(0)
    logger.info("Revocation %s: %d tokens, reason=%s", level, len(jtis), reason)
