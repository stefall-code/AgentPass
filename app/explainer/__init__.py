from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from app.delegation.engine import (
    get_trust_score,
    is_agent_auto_revoked,
    is_token_revoked,
    is_token_used,
    AGENT_TRUST_SCORE,
    TRUST_THRESHOLD,
    AUTO_REVOKE_THRESHOLD,
    TRUST_PENALTY_ESCALATION,
    TRUST_PENALTY_DENY,
    TRUST_REWARD,
    CAPABILITY_AGENTS,
)

logger = logging.getLogger(__name__)


@dataclass
class ExplainStep:
    step: str
    result: str
    detail: str = ""
    icon: str = ""


def explain_decision(context: Dict[str, Any]) -> Dict[str, Any]:
    agent_id = context.get("agent_id", "unknown")
    action = context.get("action", "")
    chain_detail = context.get("chain_detail", [])
    decision = context.get("decision", "unknown")
    risk_score = context.get("risk_score", 0.0)
    trust_score = context.get("trust_score")
    reason = context.get("reason", "")
    auto_revoked = context.get("auto_revoked", False)
    capabilities = context.get("capabilities", [])
    blocked_at = context.get("blocked_at", "")

    steps: List[Dict[str, Any]] = []
    risk_analysis = ""
    trust_analysis = ""
    final_reason = ""
    suggestion = ""

    steps.append(_explain_token_validation(context))
    steps.append(_explain_prompt_defense(context))
    steps.append(_explain_auto_revoke_check(agent_id))
    steps.append(_explain_token_revocation(context))
    steps.append(_explain_replay_check(context))
    steps.append(_explain_chain_validation(context))
    steps.append(_explain_capability_check(agent_id, action, capabilities, decision))
    steps.append(_explain_dynamic_policy(context, decision))
    steps.append(_explain_trust_check(agent_id, trust_score, decision))

    risk_analysis = _build_risk_analysis(agent_id, action, risk_score, decision)
    trust_analysis = _build_trust_analysis(agent_id, trust_score, decision, auto_revoked)
    final_reason = _build_final_reason(decision, reason, blocked_at, auto_revoked)
    suggestion = _build_suggestion(decision, agent_id, action, trust_score, auto_revoked, capabilities)

    summary = _build_summary(decision, agent_id, action, blocked_at, auto_revoked)

    return {
        "summary": summary,
        "steps": steps,
        "risk_analysis": risk_analysis,
        "trust_analysis": trust_analysis,
        "final_reason": final_reason,
        "suggestion": suggestion,
        "decision": decision,
        "agent_id": agent_id,
        "action": action,
        "auto_revoked": auto_revoked,
        "blocked_at": blocked_at,
    }


def _explain_token_validation(context: Dict[str, Any]) -> Dict[str, Any]:
    reason = context.get("reason", "")
    if "expired" in reason.lower():
        return {"step": "Token 验证", "result": "fail", "detail": "Token 已过期，需要重新签发", "icon": "❌"}
    if "invalid" in reason.lower():
        return {"step": "Token 验证", "result": "fail", "detail": "Token 无效，签名校验失败", "icon": "❌"}
    if "not a delegation" in reason.lower():
        return {"step": "Token 验证", "result": "fail", "detail": "Token 类型不正确，需要 Delegation Token", "icon": "❌"}
    return {"step": "Token 验证", "result": "pass", "detail": "Token 签名有效，类型正确", "icon": "✔️"}


def _explain_prompt_defense(context: Dict[str, Any]) -> Dict[str, Any]:
    blocked_at = context.get("blocked_at", "")
    prompt_risk_score = context.get("prompt_risk_score")
    attack_types = context.get("attack_types", [])
    attack_intent = context.get("attack_intent", "")
    severity = context.get("severity", "")

    if blocked_at == "prompt_defense":
        detail_parts = ["三层融合引擎检测到 Prompt 注入攻击"]
        if prompt_risk_score is not None:
            detail_parts.append(f"风险评分 {prompt_risk_score:.2f}")
        if attack_types:
            detail_parts.append(f"攻击类型：{', '.join(attack_types[:3])}")
        if attack_intent:
            detail_parts.append(f"攻击意图：{attack_intent}")
        if severity:
            detail_parts.append(f"严重度：{severity}")
        return {"step": "🛡️ Prompt 防御", "result": "fail", "detail": "，".join(detail_parts), "icon": "🛡️"}

    if prompt_risk_score is not None and prompt_risk_score > 0:
        return {"step": "🛡️ Prompt 防御", "result": "pass", "detail": f"三层融合引擎检测通过，风险评分 {prompt_risk_score:.2f}（阈值 0.35）", "icon": "🛡️"}

    return {"step": "🛡️ Prompt 防御", "result": "pass", "detail": "三层融合引擎检测通过，未检测到注入攻击", "icon": "🛡️"}


def _explain_auto_revoke_check(agent_id: str) -> Dict[str, Any]:
    auto_revoked, reason = is_agent_auto_revoked(agent_id)
    if auto_revoked:
        return {"step": "Auto-Revoke 检查", "result": "fail", "detail": f"Agent {agent_id} 已被自动封禁：{reason}", "icon": "🔥"}
    return {"step": "Auto-Revoke 检查", "result": "pass", "detail": f"Agent {agent_id} 未被封禁", "icon": "✔️"}


def _explain_token_revocation(context: Dict[str, Any]) -> Dict[str, Any]:
    reason = context.get("reason", "")
    if "revoked" in reason.lower() and "auto" not in reason.lower():
        return {"step": "Token 撤销检查", "result": "fail", "detail": "Token 已被手动撤销", "icon": "❌"}
    return {"step": "Token 撤销检查", "result": "pass", "detail": "Token 未被撤销", "icon": "✔️"}


def _explain_replay_check(context: Dict[str, Any]) -> Dict[str, Any]:
    reason = context.get("reason", "")
    if "replay" in reason.lower():
        return {"step": "重放攻击检测", "result": "fail", "detail": "检测到 Token 重放，同一 Token 被重复使用", "icon": "🔁"}
    return {"step": "重放攻击检测", "result": "pass", "detail": "Token 使用记录正常", "icon": "✔️"}


def _explain_chain_validation(context: Dict[str, Any]) -> Dict[str, Any]:
    chain = context.get("chain_detail", [])
    reason = context.get("reason", "")
    if "chain" in reason.lower() and "invalid" in reason.lower():
        return {"step": "调用链验证", "result": "fail", "detail": f"调用链不合法：{reason}", "icon": "❌"}
    chain_str = " → ".join(chain) if chain else "direct"
    return {"step": "调用链验证", "result": "pass", "detail": f"调用链合法：{chain_str}", "icon": "✔️"}


def _explain_capability_check(agent_id: str, action: str, capabilities: List[str], decision: str) -> Dict[str, Any]:
    if decision == "deny" and action:
        agent_caps = CAPABILITY_AGENTS.get(agent_id, {}).get("capabilities", [])
        if action not in agent_caps and action not in capabilities:
            return {"step": "Capability 检查", "result": "fail", "detail": f"Agent {agent_id} 不具备能力 {action}，当前能力：{', '.join(agent_caps[:5])}", "icon": "❌"}
    if action in capabilities:
        return {"step": "Capability 检查", "result": "pass", "detail": f"Agent {agent_id} 具备能力 {action}", "icon": "✔️"}
    return {"step": "Capability 检查", "result": "pass", "detail": "能力检查通过", "icon": "✔️"}


def _explain_dynamic_policy(context: Dict[str, Any], decision: str) -> Dict[str, Any]:
    reason = context.get("reason", "")
    if "dynamic policy" in reason.lower():
        return {"step": "动态策略检查", "result": "fail", "detail": f"动态策略拒绝：{reason}", "icon": "⚠️"}
    return {"step": "动态策略检查", "result": "pass", "detail": "动态策略检查通过", "icon": "✔️"}


def _explain_trust_check(agent_id: str, trust_score: Optional[float], decision: str) -> Dict[str, Any]:
    if trust_score is None:
        trust_score = get_trust_score(agent_id)
    if trust_score < AUTO_REVOKE_THRESHOLD:
        return {"step": "信任评分检查", "result": "fail", "detail": f"信任评分 {trust_score:.2f} 低于自动封禁阈值 {AUTO_REVOKE_THRESHOLD}", "icon": "🔥"}
    if trust_score < TRUST_THRESHOLD:
        return {"step": "信任评分检查", "result": "fail", "detail": f"信任评分 {trust_score:.2f} 低于安全阈值 {TRUST_THRESHOLD}", "icon": "⚠️"}
    return {"step": "信任评分检查", "result": "pass", "detail": f"信任评分 {trust_score:.2f} 在安全范围内", "icon": "✔️"}


def _build_risk_analysis(agent_id: str, action: str, risk_score: float, decision: str) -> str:
    if risk_score >= 0.9:
        return f"风险评分 {risk_score:.2f} — 极高风险。Agent {agent_id} 请求 {action} 被判定为严重安全威胁，可能涉及越权访问或攻击行为。"
    if risk_score >= 0.7:
        return f"风险评分 {risk_score:.2f} — 高风险。Agent {agent_id} 请求 {action} 存在显著安全风险，需要额外审查。"
    if risk_score >= 0.5:
        return f"风险评分 {risk_score:.2f} — 中等风险。Agent {agent_id} 请求 {action} 存在一定风险，已被策略引擎评估。"
    return f"风险评分 {risk_score:.2f} — 低风险。Agent {agent_id} 请求 {action} 在安全范围内。"


def _build_trust_analysis(agent_id: str, trust_score: Optional[float], decision: str, auto_revoked: bool) -> str:
    if trust_score is None:
        trust_score = get_trust_score(agent_id)
    if auto_revoked:
        return f"信任评分 {trust_score:.2f} — Agent {agent_id} 已被系统自动封禁。信任评分降至危险线以下，所有后续请求将被拒绝，直到管理员手动重置。"
    if trust_score < AUTO_REVOKE_THRESHOLD:
        return f"信任评分 {trust_score:.2f} — 严重不足。Agent {agent_id} 的信任评分已降至自动封禁阈值以下，系统将自动撤销其所有 Token。"
    if trust_score < TRUST_THRESHOLD:
        return f"信任评分 {trust_score:.2f} — 不足。Agent {agent_id} 的信任评分低于安全阈值（{TRUST_THRESHOLD}），请求被拒绝。建议检查该 Agent 的近期行为。"
    if trust_score >= 0.8:
        return f"信任评分 {trust_score:.2f} — 优秀。Agent {agent_id} 的信任评分处于健康水平，表明其历史行为良好。"
    return f"信任评分 {trust_score:.2f} — 正常。Agent {agent_id} 的信任评分在安全范围内。"


def _build_final_reason(decision: str, reason: str, blocked_at: str, auto_revoked: bool) -> str:
    if decision == "allow":
        return "✅ 请求通过 — 所有安全检查均已通过，Agent 被授权执行该操作。"
    if auto_revoked:
        return "🔥 请求被拒绝 — Agent 已被系统自动封禁，所有操作被禁止。"
    if blocked_at:
        block_names = {"delegate": "委派阶段", "check": "权限检查阶段", "prompt_defense": "🛡️ Prompt 防御阶段", "dynamic_policy": "动态策略阶段", "trust": "信任评分阶段"}
        block_name = block_names.get(blocked_at, blocked_at)
        return f"❌ 请求被拒绝 — 在{block_name}被阻断。原因：{reason}"
    return f"❌ 请求被拒绝 — 原因：{reason}"


def _build_suggestion(decision: str, agent_id: str, action: str, trust_score: Optional[float], auto_revoked: bool, capabilities: List[str]) -> str:
    if decision == "allow":
        return "无需操作。该请求已通过所有安全检查。"
    if auto_revoked:
        return f"Agent {agent_id} 已被自动封禁。如需恢复，请在治理控制台手动重置该 Agent 的信任评分。"
    if trust_score is not None and trust_score < TRUST_THRESHOLD:
        return f"Agent {agent_id} 信任评分不足。建议：1) 检查该 Agent 的近期行为日志；2) 确认是否存在异常操作；3) 如确认安全，可在治理控制台重置信任评分。"
    if action and action not in capabilities:
        agent_caps = CAPABILITY_AGENTS.get(agent_id, {}).get("capabilities", [])
        return f"Agent {agent_id} 不具备能力 {action}。当前能力：{', '.join(agent_caps[:5])}。如需授权，请更新 Agent 的 Capability 配置。"
    return "建议检查请求参数和 Agent 配置，确保符合安全策略要求。"


def _build_summary(decision: str, agent_id: str, action: str, blocked_at: str, auto_revoked: bool) -> str:
    if decision == "allow":
        return f"✅ ALLOW — Agent {agent_id} 请求 {action} 被授权"
    if auto_revoked:
        return f"🔥 AUTO-REVOKED — Agent {agent_id} 已被自动封禁"
    if blocked_at:
        return f"❌ DENY — Agent {agent_id} 请求 {action} 在 {blocked_at} 阶段被拒绝"
    return f"❌ DENY — Agent {agent_id} 请求 {action} 被拒绝"
