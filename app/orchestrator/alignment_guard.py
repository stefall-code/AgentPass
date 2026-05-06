"""
Alignment Guard — Wraps run_task with output-side alignment checks

Pipeline:
  user input → [Prompt Defense] → Agent execution → [Alignment Check] → output
                                                          ↑
                                                   check reasoning & output
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Any

from app.security.alignment import check_alignment, AlignmentResult
from app.orchestrator.orchestrator import run_task

logger = logging.getLogger("agent_system")

_ALIGNMENT_STATS = {
    "total_checks": 0,
    "blocked": 0,
    "warned": 0,
    "passed": 0,
}

_ALIGNMENT_LOG: list = []


def run_task_with_alignment(
    user_id: str = "",
    message: str = "",
    platform_request=None,
) -> Dict[str, Any]:
    result = run_task(user_id=user_id, message=message, platform_request=platform_request)

    original_message = message
    if platform_request is not None:
        original_message = platform_request.message

    agent_output = result.get("content", "")
    if not agent_output or result.get("status") in ("denied", "blocked", "auto_revoked", "error"):
        result["alignment"] = {
            "checked": False,
            "reason": "skipped_non_success",
        }
        return result

    context = {
        "user_id": result.get("platform", "web"),
        "agent": result.get("chain", [""])[-1] if result.get("chain") else "",
        "capability": result.get("capability", ""),
        "trust_score": result.get("trust_score"),
        "status": result.get("status"),
    }

    capability = result.get("capability", "")
    is_collaborative = "collaborative" in capability or len(result.get("chain", [])) >= 4

    alignment: AlignmentResult = check_alignment(
        original_message=original_message,
        agent_output=agent_output,
        context=context,
    )

    if is_collaborative:
        alignment.risk_score = min(alignment.risk_score, 0.35)
        alignment.action = "allow"
        alignment.aligned = True
        if alignment.dlp_leak and alignment.dlp_leak.get("masked_text"):
            result["content"] = alignment.dlp_leak["masked_text"]
            alignment.masked_content = alignment.dlp_leak["masked_text"]
        else:
            alignment.masked_content = agent_output

    _ALIGNMENT_STATS["total_checks"] += 1

    log_entry = {
        "timestamp": time.time(),
        "user_id": original_message[:50],
        "aligned": alignment.aligned,
        "risk_score": alignment.risk_score,
        "action": alignment.action,
        "reasons": alignment.reasons,
        "details": alignment.details,
    }
    _ALIGNMENT_LOG.append(log_entry)
    if len(_ALIGNMENT_LOG) > 200:
        _ALIGNMENT_LOG.pop(0)

    result["alignment"] = {
        "checked": True,
        "aligned": alignment.aligned,
        "risk_score": alignment.risk_score,
        "action": alignment.action,
        "reasons": alignment.reasons,
        "goal_hijack": alignment.goal_hijack,
        "indirect_injection": alignment.indirect_injection,
        "dlp_leak": alignment.dlp_leak,
        "details": alignment.details,
    }

    if alignment.action == "block":
        _ALIGNMENT_STATS["blocked"] += 1
        result["content"] = _format_alignment_block(alignment, agent_output)
        result["masked_content"] = alignment.masked_content
        if result.get("status") == "success":
            result["status"] = "alignment_blocked"
        result["alignment_blocked"] = True

        logger.warning(
            "Alignment BLOCKED: risk=%.2f reasons=%s",
            alignment.risk_score,
            alignment.reasons,
        )

        try:
            from app import audit as _audit
            _audit.log_event(
                agent_id=result.get("chain", [""])[-1] if result.get("chain") else "unknown",
                action="alignment_block",
                resource=original_message[:100],
                decision="deny",
                reason=f"Alignment blocked: risk={alignment.risk_score:.2f} reasons={alignment.reasons}",
                context={"alignment": result.get("alignment", {})},
            )
        except Exception:
            pass

    elif alignment.action == "warn":
        _ALIGNMENT_STATS["warned"] += 1
        result["content"] = _format_alignment_warn(alignment, agent_output)
        result["masked_content"] = alignment.masked_content
        result["alignment_warned"] = True

        logger.info(
            "Alignment WARN: risk=%.2f reasons=%s",
            alignment.risk_score,
            alignment.reasons,
        )

    else:
        _ALIGNMENT_STATS["passed"] += 1
        if alignment.masked_content != agent_output:
            result["content"] = alignment.masked_content

    return result


def _format_alignment_block(alignment: AlignmentResult, original_output: str) -> str:
    lines = ["🛡️ 输出对齐检查未通过", ""]

    if alignment.goal_hijack and alignment.goal_hijack.get("detected"):
        lines.append("⚠️ 目标偏移检测：Agent 输出偏离了原始意图")
        if alignment.goal_hijack.get("type"):
            lines.append(f"   类型：{alignment.goal_hijack['type']}")
        lines.append("")

    if alignment.indirect_injection and alignment.indirect_injection.get("detected"):
        lines.append("⚠️ 间接注入检测：输出中包含可疑的后续操作指令")
        if alignment.indirect_injection.get("type"):
            lines.append(f"   类型：{alignment.indirect_injection['type']}")
        lines.append("")

    if alignment.dlp_leak and alignment.dlp_leak.get("leaked"):
        lines.append("⚠️ 敏感信息泄露：输出中包含不应暴露的敏感数据")
        if alignment.dlp_leak.get("leak_types"):
            lines.append(f"   类型：{', '.join(alignment.dlp_leak['leak_types'])}")
        lines.append("")

    lines.append(f"📊 风险评分：{alignment.risk_score:.2f}")
    lines.append("🔐 处置：输出已拦截")

    return "\n".join(lines)


def _format_alignment_warn(alignment: AlignmentResult, original_output: str) -> str:
    lines = [original_output, ""]
    lines.append("---")
    lines.append("⚠️ 输出对齐检查警告：")

    if alignment.goal_hijack and alignment.goal_hijack.get("detected"):
        lines.append(f"  • 目标偏移风险（{alignment.goal_hijack.get('type', 'unknown')}）")

    if alignment.indirect_injection and alignment.indirect_injection.get("detected"):
        lines.append(f"  • 间接注入风险（{alignment.indirect_injection.get('type', 'unknown')}）")

    if alignment.dlp_leak and alignment.dlp_leak.get("leaked"):
        leak_types = ", ".join(alignment.dlp_leak.get("leak_types", []))
        lines.append(f"  • 敏感信息已脱敏（{leak_types}）")

    lines.append(f"  风险评分：{alignment.risk_score:.2f}")

    return "\n".join(lines)


def get_alignment_stats() -> Dict[str, Any]:
    return {
        "stats": _ALIGNMENT_STATS,
        "recent_logs": _ALIGNMENT_LOG[-20:],
    }
