import logging
import time
from typing import Dict, Any, Optional, List, Union
from app.delegation.engine import DelegationEngine, get_trust_score
from app.platform import PlatformRequest, calculate_platform_risk
from app.config import settings

_engine_instance: Optional[DelegationEngine] = None

logger = logging.getLogger(__name__)

AGENT_CAPABILITIES = {
    "doc_agent": ["read:doc", "write:doc:public", "delegate:data_agent"],
    "data_agent": ["read:feishu_table", "read:feishu_table:finance", "read:feishu_table:hr"],
    "external_agent": ["write:doc:public"],
}

MOCK_DATA = {
    "finance": {
        "Q1营收": "¥12,580,000",
        "Q1利润": "¥3,150,000",
        "同比增长": "+18.5%",
        "利润率": "25.0%",
    },
    "hr": {
        "在职人数": "156",
        "本月入职": "8",
        "本月离职": "3",
        "平均司龄": "2.8年",
    },
    "sales": {
        "本月订单": "342",
        "成交额": "¥5,670,000",
        "转化率": "23.5%",
        "客单价": "¥16,578",
    },
}

EVENT_LOG: list = []


def _get_engine() -> DelegationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DelegationEngine()
    return _engine_instance


def _log_event(user_id: str, message: str, agent: str, action: str, result: str, trust_score: float = None, extra: dict = None, platform: str = "web"):
    entry = {
        "user_id": user_id,
        "message": message[:100],
        "agent": agent,
        "action": action,
        "result": result,
        "trust_score": trust_score,
        "timestamp": time.time(),
        "platform": platform,
    }
    if extra:
        entry.update(extra)
    EVENT_LOG.append(entry)
    if len(EVENT_LOG) > 500:
        EVENT_LOG.pop(0)


def get_event_log(limit: int = 50) -> list:
    return EVENT_LOG[-limit:]


def secure_agent_call(
    engine: DelegationEngine,
    token: str,
    caller_agent: str,
    target_agent: str,
    action: str,
    platform: str = "web",
) -> Dict[str, Any]:
    delegation_result = engine.delegate(
        parent_token=token,
        target_agent=target_agent,
        action=action,
        caller_agent=caller_agent,
    )

    if not delegation_result.success:
        logger.warning("secure_agent_call DELEGATE FAILED: %s -> %s action=%s reason=%s", caller_agent, target_agent, action, delegation_result.reason)
        return {
            "allowed": False,
            "blocked_at": "delegate",
            "reason": delegation_result.reason or "Delegation failed",
            "human_reason": _humanize_block("delegate", delegation_result.reason, target_agent),
            "chain": [],
            "capability": action,
            "trust_score": get_trust_score(caller_agent),
        }

    delegated_token = delegation_result.token

    check_result = engine.check(token=delegated_token, action=action)

    if not check_result.allowed:
        logger.warning("secure_agent_call CHECK FAILED: %s -> %s action=%s reason=%s auto_revoked=%s", caller_agent, target_agent, action, check_result.reason, check_result.auto_revoked)
        return {
            "allowed": False,
            "blocked_at": "check",
            "reason": check_result.reason,
            "human_reason": _humanize_block("check", check_result.reason, target_agent, check_result.auto_revoked),
            "chain": check_result.chain,
            "auto_revoked": check_result.auto_revoked,
            "risk_score": check_result.risk_score,
            "capability": action,
            "trust_score": get_trust_score(target_agent),
        }

    trust_info = engine.get_trust_scores() if hasattr(engine, 'get_trust_scores') else {}
    agent_trust = trust_info.get(target_agent, {}).get("trust_score") if isinstance(trust_info, dict) else None
    if agent_trust is None:
        agent_trust = get_trust_score(target_agent)

    execution_result = _execute_agent(target_agent, action)

    return {
        "allowed": True,
        "blocked_at": None,
        "result": execution_result,
        "chain": check_result.chain,
        "delegated_token": delegated_token,
        "risk_score": check_result.risk_score,
        "capability": action,
        "trust_score": agent_trust,
    }


def _execute_agent(agent_id: str, action: str) -> Dict[str, Any]:
    logger.info("Executing agent=%s action=%s", agent_id, action)
    if agent_id == "data_agent":
        return _execute_data_agent(action)
    elif agent_id == "doc_agent":
        return _execute_doc_agent(action)
    elif agent_id == "external_agent":
        return _execute_external_agent(action)
    else:
        return {"content": f"Unknown agent: {agent_id}", "status": "error"}


def _execute_data_agent(action: str) -> Dict[str, Any]:
    if "finance" in action:
        mock_data = MOCK_DATA["finance"]
        title = "财务数据"
        app_token = settings.BITABLE_FINANCE_APP_TOKEN
        table_id = settings.BITABLE_FINANCE_TABLE_ID
    elif "hr" in action:
        mock_data = MOCK_DATA["hr"]
        title = "HR数据"
        app_token = settings.BITABLE_HR_APP_TOKEN
        table_id = settings.BITABLE_HR_TABLE_ID
    else:
        mock_data = MOCK_DATA["sales"]
        title = "销售数据"
        app_token = settings.BITABLE_SALES_APP_TOKEN
        table_id = settings.BITABLE_SALES_TABLE_ID

    real_data = None
    if app_token and table_id:
        try:
            from app.feishu.client import get_feishu_client
            client = get_feishu_client()
            if client.is_configured():
                result = client.get_bitable_records(app_token, table_id)
                if result.get("code") == 0:
                    items = result.get("data", {}).get("items", [])
                    if items:
                        first_record = items[0].get("fields", {})
                        real_data = {}
                        for k, v in first_record.items():
                            if isinstance(v, dict) and "text" in v:
                                real_data[k] = v["text"]
                            elif isinstance(v, dict) and "link" in v:
                                real_data[k] = v["link"]
                            elif isinstance(v, list):
                                texts = []
                                for item in v:
                                    if isinstance(item, dict) and "text" in item:
                                        texts.append(item["text"])
                                    else:
                                        texts.append(str(item))
                                real_data[k] = ", ".join(texts) if texts else str(v)
                            else:
                                real_data[k] = str(v)
                        logger.info("Bitable real data fetched: %d fields from %s", len(real_data), title)
                    else:
                        logger.warning("Bitable returned 0 records for %s", title)
                else:
                    logger.warning("Bitable API error for %s: %s", title, result.get("msg", "unknown"))
        except Exception as e:
            logger.error("Failed to fetch Bitable data for %s: %s", title, e)

    data = real_data if real_data else mock_data
    source = "飞书多维表格" if real_data else "Mock数据"

    content_lines = [f"📊 {title}查询结果（{source}）：\n"]
    for k, v in data.items():
        content_lines.append(f"  • {k}: {v}")

    return {"status": "success", "content": "\n".join(content_lines), "data": data, "agent": "data_agent", "action": action}


def _execute_doc_agent(action: str) -> Dict[str, Any]:
    return {"status": "success", "content": "📄 文档操作完成", "agent": "doc_agent", "action": action, "doc_url": None}


def _execute_external_agent(action: str) -> Dict[str, Any]:
    return {"status": "success", "content": "🌐 外部Agent操作完成", "agent": "external_agent", "action": action}


def _humanize_block(stage: str, reason: Optional[str], target_agent: str, auto_revoked: bool = False) -> str:
    if auto_revoked:
        return f"🔥 当前 Agent（{target_agent}）已被系统封禁\n原因：异常行为触发自动撤销，所有 Token 已失效"
    if reason and "replay" in reason.lower():
        return f"🔁 检测到异常请求（Token 重放），已阻断\n详情：{reason}"
    if reason and "revoked" in reason.lower():
        return f"🔥 当前 Agent（{target_agent}）已被系统封禁\n原因：{reason}"
    if reason and "trust" in reason.lower():
        return f"⚠️ 当前 Agent 信任评分过低，已拒绝执行\n详情：{reason}"
    if reason and "capability" in reason.lower():
        return f"❌ 请求被拒绝\n原因：当前 Agent 无权限访问该资源\n详情：{reason}"
    return f"❌ 请求被拒绝\n原因：{reason or '权限不足（IAM拦截）'}"


def _format_success(user_id: str, content: str, chain: list, capability: str, trust_score: float = None, data: dict = None) -> str:
    chain_str = " → ".join(chain) if chain else "direct"
    trust_str = f"{trust_score:.2f}" if trust_score is not None else "—"
    lines = [
        f"✅ 操作成功",
        content,
        "",
        f"🤖 Agent路径：{chain_str}",
        f"🔐 使用能力：{capability}",
        f"🏆 信任评分：{trust_str}",
    ]
    return "\n".join(lines)


def _format_denied(human_reason: str, chain: list, capability: str, trust_score: float = None) -> str:
    chain_str = " → ".join(chain) if chain else "direct"
    trust_str = f"{trust_score:.2f}" if trust_score is not None else "—"
    lines = [
        human_reason,
        "",
        f"🔐 缺失能力：{capability}",
        f"🏆 当前信任：{trust_str}",
        f"⚠️ 已记录审计日志",
    ]
    return "\n".join(lines)


def run_task(user_id: str = "", message: str = "", platform_request: PlatformRequest = None) -> Dict[str, Any]:
    engine = _get_engine()

    if platform_request is not None:
        p_req = platform_request
        user_id = p_req.user_id
        message = p_req.message
        platform = p_req.platform
        entry_point = p_req.entry_point
        risk_context = p_req.risk_context
    else:
        platform = "web"
        entry_point = "frontend"
        risk_context = {"time": time.time(), "platform": "web", "platform_risk": 0.3}

    task_type, target_action, target_agent, is_attack = _parse_intent(message)

    logger.info("run_task: user=%s message='%s' platform=%s task_type=%s target_agent=%s action=%s attack=%s", user_id, message[:50], platform, task_type, target_agent, target_action, is_attack)

    platform_risk = calculate_platform_risk(platform, target_action)
    risk_context["platform_risk_adjusted"] = platform_risk

    try:
        if is_attack == "replay":
            return _handle_replay_attack(user_id, message, engine, platform)

        if is_attack == "auto_revoke":
            return _handle_auto_revoke_attack(user_id, message, engine, platform)

        if is_attack == "escalation":
            return _handle_escalation_attack(user_id, message, engine, target_action, platform)
    except Exception as e:
        logger.error("Attack handler error: %s", e, exc_info=True)
        return {"status": "error", "content": _format_denied(f"❌ 系统处理异常\n原因：{str(e)}", ["user:" + user_id], target_action), "reason": str(e), "chain": ["user:" + user_id], "capability": target_action, "trust_score": None, "platform": platform}

    capabilities = AGENT_CAPABILITIES.get("doc_agent", ["read:doc", "write:doc:public", "delegate:data_agent"])

    try:
        root_token = engine.issue_root_token(
            agent_id="doc_agent",
            delegated_user=user_id,
            capabilities=capabilities,
            metadata={
                "platform": platform,
                "source": entry_point,
                "risk_context": risk_context,
            },
        )
    except Exception as e:
        logger.error("Failed to issue root token: %s", e)
        return {"status": "error", "content": _format_denied("❌ 系统错误：无法签发安全令牌", ["user:" + user_id], target_action), "reason": str(e), "chain": [], "capability": target_action, "trust_score": None, "platform": platform}

    chain = ["user:" + user_id, "doc_agent"]

    try:
        if target_agent == "data_agent":
            agent_result = secure_agent_call(engine=engine, token=root_token, caller_agent="doc_agent", target_agent="data_agent", action=target_action, platform=platform)

            if not agent_result.get("allowed"):
                trust = agent_result.get("trust_score")
                chain_with_target = chain + ["data_agent"]
                formatted = _format_denied(agent_result.get("human_reason", "❌ 请求被拒绝"), chain_with_target, target_action, trust)
                _log_event(user_id, message, "data_agent", target_action, "denied", trust, {"blocked_at": agent_result.get("blocked_at"), "auto_revoked": agent_result.get("auto_revoked", False), "chain": chain_with_target, "platform_risk": platform_risk}, platform=platform)
                return {"status": "denied", "content": formatted, "reason": agent_result.get("reason"), "chain": chain_with_target, "blocked_at": agent_result.get("blocked_at"), "auto_revoked": agent_result.get("auto_revoked", False), "capability": target_action, "trust_score": trust, "platform": platform, "platform_risk": platform_risk}

            chain.append("data_agent")
            data_result = agent_result.get("result", {})
            trust = agent_result.get("trust_score")

            if task_type == "report":
                report_content = _generate_report(user_id, data_result)
                formatted = _format_success(user_id, report_content, chain, target_action, trust, data_result.get("data"))
            else:
                formatted = _format_success(user_id, data_result.get("content", "查询完成"), chain, target_action, trust, data_result.get("data"))

            _log_event(user_id, message, "data_agent", target_action, "success", trust, {"chain": chain, "data": data_result.get("data"), "platform_risk": platform_risk}, platform=platform)
            return {"status": "success", "content": formatted, "chain": chain, "data": data_result.get("data"), "capability": target_action, "trust_score": trust, "platform": platform, "platform_risk": platform_risk}

        elif target_agent == "doc_agent":
            check_result = engine.check(token=root_token, action=target_action, caller_agent="doc_agent")
            if not check_result.allowed:
                doc_trust = get_trust_score("doc_agent")
                formatted = _format_denied(_humanize_block("check", check_result.reason, "doc_agent", check_result.auto_revoked), chain, target_action, doc_trust)
                _log_event(user_id, message, "doc_agent", target_action, "denied", doc_trust, {"auto_revoked": check_result.auto_revoked, "platform_risk": platform_risk}, platform=platform)
                return {"status": "denied", "content": formatted, "reason": check_result.reason, "chain": chain, "auto_revoked": check_result.auto_revoked, "capability": target_action, "trust_score": doc_trust, "platform": platform, "platform_risk": platform_risk}

            doc_result = _execute_doc_agent(target_action)
            doc_trust = get_trust_score("doc_agent")
            formatted = _format_success(user_id, doc_result.get("content", "文档操作完成"), chain, target_action, doc_trust)
            _log_event(user_id, message, "doc_agent", target_action, "success", doc_trust, {"platform_risk": platform_risk}, platform=platform)
            return {"status": "success", "content": formatted, "chain": chain, "capability": target_action, "trust_score": doc_trust, "platform": platform, "platform_risk": platform_risk}

        else:
            agent_result = secure_agent_call(engine=engine, token=root_token, caller_agent="doc_agent", target_agent=target_agent, action=target_action, platform=platform)

            if not agent_result.get("allowed"):
                trust = agent_result.get("trust_score")
                chain_with_target = chain + [target_agent]
                formatted = _format_denied(agent_result.get("human_reason", "❌ 请求被拒绝"), chain_with_target, target_action, trust)
                _log_event(user_id, message, target_agent, target_action, "denied", trust, {"auto_revoked": agent_result.get("auto_revoked", False), "platform_risk": platform_risk}, platform=platform)
                return {"status": "denied", "content": formatted, "reason": agent_result.get("reason"), "chain": chain_with_target, "auto_revoked": agent_result.get("auto_revoked", False), "capability": target_action, "trust_score": trust, "platform": platform, "platform_risk": platform_risk}

            formatted = _format_success(user_id, agent_result.get("result", {}).get("content", "操作完成"), chain + [target_agent], target_action, agent_result.get("trust_score"))
            _log_event(user_id, message, target_agent, target_action, "success", agent_result.get("trust_score"), {"platform_risk": platform_risk}, platform=platform)
            return {"status": "success", "content": formatted, "chain": chain + [target_agent], "capability": target_action, "trust_score": agent_result.get("trust_score"), "platform": platform, "platform_risk": platform_risk}

    except Exception as e:
        logger.error("run_task execution error: %s", e, exc_info=True)
        human_msg = _humanize_block("execute", str(e), target_agent)
        return {"status": "error", "content": _format_denied(human_msg, chain, target_action), "reason": str(e), "chain": chain, "capability": target_action, "trust_score": None, "platform": platform}


def _handle_escalation_attack(user_id: str, message: str, engine: DelegationEngine, target_action: str, platform: str = "web") -> Dict[str, Any]:
    capabilities = AGENT_CAPABILITIES.get("doc_agent", ["read:doc", "write:doc:public", "delegate:data_agent"])
    root_token = engine.issue_root_token(agent_id="doc_agent", delegated_user=user_id, capabilities=capabilities)

    result = secure_agent_call(engine=engine, token=root_token, caller_agent="doc_agent", target_agent="data_agent", action=target_action, platform=platform)

    chain = ["user:" + user_id, "doc_agent", "data_agent"]
    platform_risk = calculate_platform_risk(platform, target_action)

    if not result.get("allowed"):
        trust = result.get("trust_score")
        human_reason = result.get("human_reason", "❌ 请求被拒绝\n原因：无权限访问该数据")
        formatted = _format_denied(human_reason, chain, target_action, trust)
        _log_event(user_id, message, "data_agent", target_action, "denied", trust, {"attack_type": "escalation", "blocked_at": result.get("blocked_at")}, platform=platform)
        return {"status": "denied", "content": formatted, "chain": chain, "capability": target_action, "trust_score": trust, "attack_type": "escalation", "blocked_at": result.get("blocked_at"), "platform": platform, "platform_risk": platform_risk}

    _log_event(user_id, message, "data_agent", target_action, "success", result.get("trust_score"), {"attack_type": "escalation"}, platform=platform)
    return {"status": "success", "content": "⚠️ 越权攻击未被拦截（data_agent 拥有该能力）", "chain": chain, "capability": target_action, "trust_score": result.get("trust_score"), "attack_type": "escalation", "platform": platform, "platform_risk": platform_risk}


def _handle_replay_attack(user_id: str, message: str, engine: DelegationEngine, platform: str = "web") -> Dict[str, Any]:
    capabilities = AGENT_CAPABILITIES.get("doc_agent", ["read:doc", "write:doc:public", "delegate:data_agent"])
    root_token = engine.issue_root_token(agent_id="doc_agent", delegated_user=user_id, capabilities=capabilities)

    first = secure_agent_call(engine=engine, token=root_token, caller_agent="doc_agent", target_agent="data_agent", action="read:feishu_table", platform=platform)
    replay = secure_agent_call(engine=engine, token=root_token, caller_agent="doc_agent", target_agent="data_agent", action="read:feishu_table", platform=platform)

    chain = ["user:" + user_id, "doc_agent", "data_agent"]
    platform_risk = calculate_platform_risk(platform, "read:feishu_table")

    if not replay.get("allowed"):
        formatted = _format_denied(replay.get("human_reason", "🔁 Token 重放已阻断"), chain, "read:feishu_table", replay.get("trust_score"))
        _log_event(user_id, message, "data_agent", "read:feishu_table", "replay_blocked", replay.get("trust_score"), {"attack_type": "replay", "first_allowed": first.get("allowed")}, platform=platform)
        return {"status": "denied", "content": formatted, "chain": chain, "capability": "read:feishu_table", "trust_score": replay.get("trust_score"), "attack_type": "replay", "first_allowed": first.get("allowed"), "replay_allowed": False, "platform": platform, "platform_risk": platform_risk}

    _log_event(user_id, message, "data_agent", "read:feishu_table", "success", replay.get("trust_score"), {"attack_type": "replay"}, platform=platform)
    return {"status": "success", "content": "⚠️ 重放攻击未被检测（Token 未被标记为已使用）", "chain": chain, "capability": "read:feishu_table", "trust_score": replay.get("trust_score"), "attack_type": "replay", "platform": platform, "platform_risk": platform_risk}


def _handle_auto_revoke_attack(user_id: str, message: str, engine: DelegationEngine, platform: str = "web") -> Dict[str, Any]:
    root_token = engine.issue_root_token(agent_id="external_agent", delegated_user=user_id, capabilities=["write:doc:public"])

    steps = []
    actions = [
        ("write:doc:public", "正常写入文档"),
        ("read:feishu_table:finance", "越权读取财务数据"),
        ("read:feishu_table:hr", "越权读取HR数据"),
        ("read:feishu_table:salary", "越权读取薪资数据"),
        ("read:feishu_table:sales", "越权读取销售数据"),
    ]

    for action, desc in actions:
        check_result = engine.check(token=root_token, action=action, caller_agent="external_agent")
        steps.append({
            "action": action,
            "description": desc,
            "allowed": check_result.allowed,
            "reason": check_result.reason,
            "auto_revoked": check_result.auto_revoked,
            "trust_score": get_trust_score("external_agent"),
        })

    chain = ["user:" + user_id, "external_agent"]
    last_step = steps[-1]
    any_auto_revoked = any(s.get("auto_revoked") for s in steps)

    if any_auto_revoked:
        formatted = "🔥 Agent 已被系统自动封禁\n\n"
        for s in steps:
            if s["auto_revoked"]:
                formatted += f"🔥 {s['description']} → AUTO-REVOKED\n"
            elif s["allowed"]:
                formatted += f"✅ {s['description']} → ALLOWED\n"
            else:
                formatted += f"❌ {s['description']} → DENIED\n"
        formatted += f"\n🔐 安全链路：{' → '.join(chain)}\n🏆 信任评分已降至危险线以下\n⚠️ 已记录审计日志"

        _log_event(user_id, message, "external_agent", "auto_revoke", "auto_revoked", last_step.get("trust_score"), {"attack_type": "auto_revoke", "steps": steps}, platform=platform)
        return {"status": "auto_revoked", "content": formatted, "chain": chain, "steps": steps, "capability": "multiple", "trust_score": last_step.get("trust_score"), "attack_type": "auto_revoke", "auto_revoked": True, "platform": platform}

    formatted = "⚠️ 连续攻击演示结果：\n\n"
    for s in steps:
        if s["allowed"]:
            formatted += f"✅ {s['description']} → ALLOWED\n"
        else:
            formatted += f"❌ {s['description']} → DENIED\n"

    _log_event(user_id, message, "external_agent", "multi_attack", "denied", None, {"attack_type": "auto_revoke", "steps": steps}, platform=platform)
    return {"status": "denied", "content": formatted, "chain": chain, "steps": steps, "capability": "multiple", "trust_score": None, "attack_type": "auto_revoke", "platform": platform}


def _parse_intent(message: str) -> tuple:
    msg = message.lower()

    if any(kw in msg for kw in ["重复请求", "replay", "重放"]):
        return "replay_attack", "read:feishu_table", "data_agent", "replay"

    if any(kw in msg for kw in ["连续测试", "连续攻击", "auto.revoke", "auto revoke", "暴力"]):
        return "auto_revoke_attack", "write:doc:public", "external_agent", "auto_revoke"

    if any(kw in msg for kw in ["薪资", "salary", "工资"]):
        return "escalation", "read:feishu_table:salary", "data_agent", "escalation"

    if any(kw in msg for kw in ["报告", "报表", "总结", "汇总", "report"]):
        if any(kw in msg for kw in ["财务", "finance", "营收", "利润"]):
            return "report", "read:feishu_table:finance", "data_agent", None
        elif any(kw in msg for kw in ["hr", "人事", "员工", "离职"]):
            return "report", "read:feishu_table:hr", "data_agent", None
        else:
            return "report", "read:feishu_table:finance", "data_agent", None

    elif any(kw in msg for kw in ["数据", "查询", "查一下", "data", "query"]):
        if any(kw in msg for kw in ["财务", "finance", "营收"]):
            return "data_query", "read:feishu_table:finance", "data_agent", None
        elif any(kw in msg for kw in ["hr", "人事", "员工"]):
            return "data_query", "read:feishu_table:hr", "data_agent", None
        else:
            return "data_query", "read:feishu_table", "data_agent", None

    elif any(kw in msg for kw in ["文档", "写入", "创建", "doc", "write"]):
        return "doc_write", "write:doc:public", "doc_agent", None

    elif any(kw in msg for kw in ["外部", "第三方", "external"]):
        return "external", "write:doc:public", "external_agent", None

    else:
        return "data_query", "read:feishu_table", "data_agent", None


def _generate_report(user_id: str, data_result: Dict[str, Any]) -> str:
    data = data_result.get("data", {})
    if not data:
        return "📊 报告生成失败：无数据"

    lines = ["📋 业务数据报告", "", f"👤 请求人: {user_id}", f"🤖 执行Agent: doc_agent → data_agent", f"🔐 安全链路: IAM校验通过 ✓", "", "---", ""]
    for k, v in data.items():
        lines.append(f"**{k}**: {v}")
    lines.extend(["", "---", "", "✅ 报告已生成", "🔐 全链路 IAM 审计记录已保存"])
    return "\n".join(lines)
