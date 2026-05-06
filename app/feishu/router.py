import json
import time
import logging
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from .client import get_feishu_client
from app.orchestrator.orchestrator import secure_agent_call
from app.orchestrator.alignment_guard import run_task_with_alignment
from app.delegation.engine import DelegationEngine
from app.platform import PlatformRequest

_logger = logging.getLogger(__name__)

_engine_instance = None

def _get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DelegationEngine()
    return _engine_instance

router = APIRouter(prefix="/feishu", tags=["Feishu Integration"])


class FeishuWebhookBody(BaseModel):
    challenge: Optional[str] = None
    token: Optional[str] = None
    type: Optional[str] = None
    event: Optional[Dict[str, Any]] = None
    header: Optional[Dict[str, Any]] = None
    schema_: Optional[str] = None

    class Config:
        extra = "allow"


class FeishuTestRequest(BaseModel):
    user_id: str = "test_user_001"
    message: str = "帮我查一下财务数据"
    platform: str = "feishu"


class FeishuSendMessageRequest(BaseModel):
    user_id: str
    content: str
    msg_type: str = "text"


class FeishuCreateDocRequest(BaseModel):
    title: str
    content: str


_feishu_event_log: list = []


def _log_feishu_event(event_type: str, user_id: str, message: str, result: Dict[str, Any]):
    import time
    entry = {
        "timestamp": time.time(),
        "event_type": event_type,
        "user_id": user_id,
        "message": message[:100],
        "agent": result.get("chain", [""])[-1] if result.get("chain") else "",
        "action": result.get("capability", ""),
        "result": result.get("status", "unknown"),
        "trust_score": result.get("trust_score"),
        "chain": result.get("chain", []),
        "blocked_at": result.get("blocked_at"),
        "auto_revoked": result.get("auto_revoked", False),
        "attack_type": result.get("attack_type"),
        "capability": result.get("capability", ""),
    }
    _feishu_event_log.append(entry)
    if len(_feishu_event_log) > 200:
        _feishu_event_log.pop(0)


_processed_messages: Dict[str, float] = {}
_MESSAGE_TTL = 300
_message_lock = asyncio.Lock()


async def _process_feishu_message(user_id: str, message: str, chat_id: str = "", message_id: str = ""):
    global _processed_messages
    now = time.time()
    _processed_messages = {k: v for k, v in _processed_messages.items() if now - v < _MESSAGE_TTL}

    async with _message_lock:
        dedup_key = message_id if message_id else f"{user_id}:{message[:100]}:{int(now / 30)}"
        if dedup_key in _processed_messages:
            _logger.info("Skipping duplicate message: %s", dedup_key)
            return {"status": "skipped", "content": "duplicate", "reason": "already processed"}

        _processed_messages[dedup_key] = now

    client = get_feishu_client()

    if message.strip().startswith("/cli"):
        return await _process_cli_command(user_id, message, chat_id, client)

    try:
        p_req = PlatformRequest(platform="feishu", user_id=user_id, message=message)
        result = run_task_with_alignment(platform_request=p_req)
        _log_feishu_event("message", user_id, message, result)

        reply_content = _format_feishu_reply(result)

        if chat_id:
            await client.send_message(chat_id, reply_content, receive_id_type="chat_id")
        else:
            await client.send_message(user_id, reply_content)

        _logger.info("Feishu message processed: user=%s status=%s", user_id, result.get("status"))
        return result

    except Exception as e:
        _logger.error("Error processing feishu message: %s", e, exc_info=True)

        try:
            from app.audit import log_event as audit_log_event
            audit_log_event(
                action="feishu:message:error",
                resource="feishu:webhook",
                decision="deny",
                reason=f"feishu message processing error: {str(e)[:200]}",
                agent_id="system",
                context={
                    "user_id": user_id,
                    "message": message[:100],
                    "platform": "feishu",
                    "error": str(e)[:200],
                    "source": "feishu_webhook_error",
                },
            )
        except Exception:
            pass

        error_msg = f"❌ 系统处理异常\n原因：{str(e)}"
        try:
            if chat_id:
                await client.send_message(chat_id, error_msg, receive_id_type="chat_id")
            else:
                await client.send_message(user_id, error_msg)
        except Exception:
            pass
        return {"status": "error", "content": error_msg, "reason": str(e)}


async def _process_cli_command(user_id: str, message: str, chat_id: str, client) -> Dict[str, Any]:
    from .cli_proxy import execute_cli_command, check_cli_permission

    parts = message.strip().split()
    if len(parts) < 2:
        help_text = (
            "🖥️ CLI IAM Gateway 使用说明\n\n"
            "格式：/cli <domain> [subcommand] [args...]\n\n"
            "示例：\n"
            "  /cli calendar +agenda\n"
            "  /cli base +table-list --base-token xxx\n"
            "  /cli im +chat-search --query test\n"
            "  /cli task +get-my-tasks\n"
            "  /cli wiki spaces list\n\n"
            "🔍 检查权限（不执行）：\n"
            "  /cli check calendar +agenda\n\n"
            "📊 查看CLI域：\n"
            "  /cli domains\n\n"
            "🔐 所有CLI命令均经过IAM网关管控"
        )
        if chat_id:
            await client.send_message(chat_id, help_text, receive_id_type="chat_id")
        else:
            await client.send_message(user_id, help_text)
        return {"status": "success", "content": help_text}

    if parts[1] == "domains":
        from .cli_proxy import get_cli_domain_capabilities
        caps = get_cli_domain_capabilities()
        lines = ["📋 CLI IAM Domain Capabilities\n"]
        for k, v in caps.items():
            risk = "🔴HIGH" if v["is_high_risk"] else ("🟡MED" if v["is_sensitive"] else "🟢LOW")
            lines.append(f"  {k} ({risk}): {len(v['commands'])} commands")
        reply = "\n".join(lines)
        if chat_id:
            await client.send_message(chat_id, reply, receive_id_type="chat_id")
        else:
            await client.send_message(user_id, reply)
        return {"status": "success", "content": reply}

    if parts[1] == "check":
        if len(parts) < 3:
            reply = "格式：/cli check <domain> [subcommand]"
            if chat_id:
                await client.send_message(chat_id, reply, receive_id_type="chat_id")
            else:
                await client.send_message(user_id, reply)
            return {"status": "success", "content": reply}
        domain = parts[2]
        subcommand = parts[3] if len(parts) > 3 else ""
        perm = check_cli_permission("doc_agent", domain, subcommand)
        icon = "✅" if perm["allowed"] else "❌"
        reply = f"{icon} CLI IAM Check\n  domain: {domain}\n  subcommand: {subcommand}\n  action: {perm['action']}\n  allowed: {perm['allowed']}\n  trust: {perm.get('trust_score', 'N/A')}\n  blocked_at: {perm.get('blocked_at', '-')}"
        if chat_id:
            await client.send_message(chat_id, reply, receive_id_type="chat_id")
        else:
            await client.send_message(user_id, reply)
        return {"status": "success", "content": reply}

    domain = parts[1]
    subcommand = parts[2] if len(parts) > 2 else ""
    args = parts[3:] if len(parts) > 3 else []

    result = await execute_cli_command(
        domain=domain,
        subcommand=subcommand,
        args=args,
        agent_id="doc_agent",
    )

    if result.iam_allowed:
        icon = "✅"
        reply_lines = [
            f"{icon} CLI IAM ALLOWED",
            f"  domain: {result.domain}",
            f"  subcommand: {result.subcommand}",
            f"  action: {result.action}",
            f"  trust: {result.trust_score:.2f}" if result.trust_score else "  trust: N/A",
            "",
            "--- CLI Output ---",
        ]
        output = result.output.strip()
        if output:
            try:
                parsed = json.loads(output)
                reply_lines.append(json.dumps(parsed, indent=2, ensure_ascii=False)[:1500])
            except (json.JSONDecodeError, TypeError):
                reply_lines.append(output[:1500])
        else:
            reply_lines.append("(no output)")
    else:
        icon = "❌"
        reply_lines = [
            f"{icon} CLI IAM BLOCKED",
            f"  domain: {result.domain}",
            f"  subcommand: {result.subcommand}",
            f"  action: {result.action}",
            f"  blocked_at: {result.iam_blocked_at}",
            f"  reason: {result.iam_reason[:100]}",
            f"  trust: {result.trust_score:.2f}" if result.trust_score else "  trust: N/A",
        ]

    reply = "\n".join(reply_lines)
    _log_feishu_event("cli_command", user_id, message, {
        "status": "success" if result.iam_allowed else "denied",
        "domain": result.domain,
        "action": result.action,
        "iam_allowed": result.iam_allowed,
        "iam_blocked_at": result.iam_blocked_at,
    })

    if chat_id:
        await client.send_message(chat_id, reply, receive_id_type="chat_id")
    else:
        await client.send_message(user_id, reply)

    return {"status": "success" if result.iam_allowed else "denied", "content": reply}


def _format_feishu_reply(result: Dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    content = result.get("content", "")
    chain = result.get("chain", [])
    capability = result.get("capability", "")
    trust_score = result.get("trust_score")
    blocked_at = result.get("blocked_at", "")
    auto_revoked = result.get("auto_revoked", False)
    attack_type = result.get("attack_type")
    prompt_risk = result.get("prompt_risk", {})
    alignment = result.get("alignment", {})
    six_layer = result.get("six_layer")
    platform_risk = result.get("platform_risk", 0)
    data = result.get("data")

    lines = []

    if status == "success":
        lines.append("✅ 操作成功")
    elif status == "denied":
        lines.append("❌ 请求被拒绝")
    elif status == "auto_revoked":
        lines.append("🔥 Agent 已被自动封禁")
    elif status == "hitl_pending":
        lines.append("⏳ 人工审批中")
    elif status == "alignment_blocked":
        lines.append("🛡️ 输出对齐检查未通过")
    else:
        lines.append(f"⚠️ {status}")

    if content:
        clean = content.replace("✅ 操作成功", "").replace("❌ 请求被拒绝", "").strip()
        if clean:
            lines.append("")
            lines.append(clean)

    lines.append("")
    lines.append("--- 安全链路证据 ---")

    if chain:
        lines.append(f"🔗 Chain: {' → '.join(chain)}")

    if capability:
        lines.append(f"🔐 Capability: {capability}")

    if trust_score is not None:
        trust_emoji = "🟢" if trust_score >= 0.7 else ("🟡" if trust_score >= 0.5 else "🔴")
        trust_line = f"🏆 Trust: {trust_emoji} {trust_score:.2f}"
        trust_before = result.get("trust_before")
        trust_after = result.get("trust_after")
        if trust_before is not None and trust_after is not None and trust_before != trust_after:
            trust_line += f" ({trust_before:.2f}→{trust_after:.2f})"
        lines.append(trust_line)

    if blocked_at:
        lines.append(f"🚫 Blocked@: {blocked_at}")

    if auto_revoked:
        lines.append("🔥 Auto-Revoked: YES")

    if attack_type:
        lines.append(f"⚔️ Attack: {attack_type}")

    if prompt_risk and isinstance(prompt_risk, dict):
        risk_score = prompt_risk.get("risk_score", 0)
        threats = prompt_risk.get("threats", [])
        if risk_score > 0:
            lines.append(f"🧠 Prompt Risk: {risk_score:.2f}")
        if threats:
            lines.append(f"⚠️ Threats: {', '.join(threats[:3])}")

    if platform_risk > 0:
        lines.append(f"📱 Platform Risk: {platform_risk:.2f}")

    if alignment and alignment.get("checked"):
        al_action = alignment.get("action", "pass")
        al_risk = alignment.get("risk_score", 0)
        al_icon = "🛡️" if al_action == "block" else ("⚠️" if al_action == "warn" else "✅")
        lines.append(f"{al_icon} Alignment: {al_action.upper()} (risk: {al_risk:.2f})")

    if six_layer and isinstance(six_layer, dict):
        layers = six_layer.get("layers", {})
        if layers and isinstance(layers, dict):
            layer_summary = []
            for lid, ldata in layers.items():
                if isinstance(ldata, dict):
                    lst = ldata.get("status", "pass")
                else:
                    lst = str(ldata)
                icon = "✔" if lst == "pass" else ("⚠" if lst == "warn" else "✘")
                layer_summary.append(f"{icon}{lid}")
            lines.append(f"🧱 6-Layer: {' '.join(layer_summary)}")

    if data and isinstance(data, dict):
        source = data.get("source", "")
        if source:
            lines.append(f"📊 Data Source: {source}")

    lines.append("🔐 Audit: logged")

    return "\n".join(lines)


@router.post("/webhook")
async def feishu_webhook(body: Dict[str, Any], background_tasks: BackgroundTasks):
    if body.get("type") == "url_verification":
        challenge = body.get("challenge", "")
        return {"challenge": challenge}

    client = get_feishu_client()

    if not client.verify_event(body):
        _logger.warning("Feishu event verification failed")
        return {"code": -1, "msg": "verification failed"}

    parsed = client.parse_message_event(body)
    if parsed is None:
        return {"code": 0, "msg": "ignored"}

    user_id = parsed["user_id"]
    message = parsed["message"]
    chat_id = parsed.get("chat_id", "")
    message_id = parsed.get("message_id", "")

    async with _message_lock:
        now = time.time()
        dedup_key = message_id if message_id else f"{user_id}:{message[:100]}:{int(now / 30)}"
        if dedup_key in _processed_messages:
            _logger.info("Webhook duplicate skipped: %s", dedup_key)
            return {"code": 0, "msg": "duplicate"}
        _processed_messages[dedup_key] = now

    _logger.info("Feishu webhook: user=%s message='%s'", user_id, message[:50])

    background_tasks.add_task(_process_feishu_message_sync, user_id, message, chat_id, message_id)

    return {"code": 0, "msg": "ok"}


@router.post("/test")
async def feishu_test_endpoint(req: FeishuTestRequest):
    p_req = PlatformRequest(platform=req.platform, user_id=req.user_id, message=req.message)
    result = run_task_with_alignment(platform_request=p_req)
    _log_feishu_event("test", req.user_id, req.message, result)
    response = {
        "status": result.get("status", "unknown"),
        "content": result.get("content", "处理完成"),
        "chain": result.get("chain", []),
        "capability": result.get("capability", ""),
        "trust_score": result.get("trust_score"),
        "blocked_at": result.get("blocked_at"),
        "auto_revoked": result.get("auto_revoked", False),
        "attack_type": result.get("attack_type"),
        "reason": result.get("reason"),
        "data": result.get("data"),
        "platform": result.get("platform", req.platform),
        "platform_risk": result.get("platform_risk"),
    }
    if result.get("doc_url"):
        response["doc_url"] = result.get("doc_url")
    if result.get("trust_before") is not None:
        response["trust_before"] = result.get("trust_before")
    if result.get("trust_after") is not None:
        response["trust_after"] = result.get("trust_after")
    if result.get("steps"):
        response["steps"] = result.get("steps")
    return response


@router.post("/send")
async def feishu_send_message(req: FeishuSendMessageRequest):
    client = get_feishu_client()
    result = await client.send_message(req.user_id, req.content, req.msg_type)
    return result


@router.post("/create-doc")
async def feishu_create_doc(req: FeishuCreateDocRequest):
    client = get_feishu_client()
    result = await client.create_doc(req.title, req.content)
    return result


@router.get("/events")
async def get_feishu_events(limit: int = 50):
    events = _feishu_event_log[-limit:]
    return {"events": events, "total": len(_feishu_event_log)}


@router.get("/status")
async def feishu_status():
    client = get_feishu_client()
    ws_connected = False
    try:
        from app.feishu.ws_client import is_ws_connected
        ws_connected = is_ws_connected()
    except Exception:
        pass
    return {
        "configured": client.is_configured(),
        "app_id_set": bool(client.app_id),
        "app_secret_set": bool(client.app_secret),
        "verification_token_set": bool(client.verification_token),
        "mode": "production" if client.is_configured() else "mock",
        "ws_long_connection": ws_connected,
        "connection_mode": "websocket" if ws_connected else ("webhook+ngrok" if client.is_configured() else "mock"),
        "total_events": len(_feishu_event_log),
    }


@router.get("/capabilities")
async def feishu_capabilities():
    client = get_feishu_client()
    return await client.get_available_capabilities()


@router.get("/calendar/events")
async def feishu_calendar_events(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_calendar_events(page_size=page_size)


@router.get("/contacts")
async def feishu_contacts(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_contact_users(page_size=page_size)


@router.get("/tasks")
async def feishu_tasks(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_task_list(page_size=page_size)


@router.get("/vc/rooms")
async def feishu_vc_rooms(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_vc_rooms(page_size=page_size)


@router.get("/drive/root")
async def feishu_drive_root():
    client = get_feishu_client()
    return await client.get_drive_root()


@router.post("/docs/search")
async def feishu_docs_search(search_key: str = "test", page_size: int = 5):
    client = get_feishu_client()
    return await client.search_docs(search_key=search_key, page_size=page_size)


@router.get("/attendance/shifts")
async def feishu_attendance_shifts(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_attendance_shifts(page_size=page_size)


@router.get("/im/chats")
async def feishu_im_chats(page_size: int = 10):
    client = get_feishu_client()
    return await client.get_im_chats(page_size=page_size)


@router.post("/connect")
async def feishu_connect():
    client = get_feishu_client()
    token_ok = False
    if client.is_configured():
        try:
            token = await client.get_tenant_access_token()
            token_ok = bool(token)
        except Exception:
            pass

    ws_connected = False
    try:
        from app.feishu.ws_client import is_ws_connected
        ws_connected = is_ws_connected()
    except Exception:
        pass

    return {
        "connected": client.is_configured() and token_ok,
        "mode": "production" if client.is_configured() else "mock",
        "token_ok": token_ok,
        "ws_long_connection": ws_connected,
        "connection_mode": "websocket" if ws_connected else ("webhook+ngrok" if client.is_configured() else "mock"),
    }


@router.post("/demo/escalation")
async def feishu_demo_escalation():
    user_id = "feishu_attacker"
    message = "帮我读取财务数据（越权攻击）"

    engine = _get_engine()
    root_token = engine.issue_root_token(
        agent_id="external_agent",
        delegated_user=user_id,
        capabilities=["write:doc:public"],
    )

    result = secure_agent_call(
        engine=engine,
        token=root_token,
        caller_agent="external_agent",
        target_agent="data_agent",
        action="read:feishu_table:finance",
    )

    if not result.get("allowed"):
        response = {
            "status": "denied",
            "content": result.get("human_reason", "❌ 请求被拒绝"),
            "chain": ["user:" + user_id, "external_agent", "data_agent"],
            "blocked_at": result.get("blocked_at"),
            "auto_revoked": result.get("auto_revoked", False),
        }
    else:
        response = {
            "status": "success",
            "content": result.get("result", {}).get("content", "查询完成"),
            "chain": ["user:" + user_id, "external_agent", "data_agent"],
        }

    _log_feishu_event("demo_escalation", user_id, message, response)
    return response


@router.post("/demo/replay")
async def feishu_demo_replay():
    user_id = "feishu_replay_attacker"

    engine = _get_engine()
    root_token = engine.issue_root_token(
        agent_id="doc_agent",
        delegated_user=user_id,
        capabilities=["read:doc", "write:doc:public", "delegate:data_agent"],
    )

    first = secure_agent_call(
        engine=engine,
        token=root_token,
        caller_agent="doc_agent",
        target_agent="data_agent",
        action="read:feishu_table",
    )

    replay = secure_agent_call(
        engine=engine,
        token=root_token,
        caller_agent="doc_agent",
        target_agent="data_agent",
        action="read:feishu_table",
    )

    response = {
        "status": "denied" if not replay.get("allowed") else "success",
        "content": replay.get("human_reason", "处理完成"),
        "first_call": {"allowed": first.get("allowed"), "reason": first.get("reason")},
        "replay_call": {"allowed": replay.get("allowed"), "reason": replay.get("reason")},
        "chain": ["user:" + user_id, "doc_agent", "data_agent"],
    }

    _log_feishu_event("demo_replay", user_id, "replay attack", response)
    return response


@router.post("/demo/auto-revoke")
async def feishu_demo_auto_revoke():
    user_id = "feishu_abuser"

    engine = _get_engine()
    root_token = engine.issue_root_token(
        agent_id="external_agent",
        delegated_user=user_id,
        capabilities=["write:doc:public"],
    )

    steps = []
    actions = [
        ("write:doc:public", "正常写入"),
        ("read:feishu_table:finance", "越权读取财务"),
        ("read:feishu_table:hr", "越权读取HR"),
    ]

    for action, desc in actions:
        result = secure_agent_call(
            engine=engine,
            token=root_token,
            caller_agent="external_agent",
            target_agent="data_agent",
            action=action,
        )
        steps.append({
            "action": action,
            "description": desc,
            "allowed": result.get("allowed", False),
            "reason": result.get("reason"),
            "human_reason": result.get("human_reason"),
            "auto_revoked": result.get("auto_revoked", False),
            "blocked_at": result.get("blocked_at"),
        })

    response = {
        "status": "auto_revoked",
        "content": "🔥 当前 Agent 已被系统封禁（异常行为触发）",
        "steps": steps,
        "chain": ["user:" + user_id, "external_agent", "data_agent"],
    }

    _log_feishu_event("demo_auto_revoke", user_id, "auto revoke demo", response)
    return response


class CLIExecuteRequest(BaseModel):
    domain: str
    subcommand: str = ""
    args: list = []
    agent_id: str = "doc_agent"
    method: str = "GET"
    dry_run: bool = False


class CLICheckRequest(BaseModel):
    domain: str
    subcommand: str = ""
    agent_id: str = "doc_agent"
    method: str = "GET"


@router.get("/cli/domains")
async def cli_domains():
    from .cli_proxy import get_cli_domain_capabilities
    return {"domains": get_cli_domain_capabilities(), "title": "Feishu CLI IAM Domain Capabilities"}


@router.post("/cli/check")
async def cli_check(req: CLICheckRequest):
    from .cli_proxy import check_cli_permission
    return check_cli_permission(req.agent_id, req.domain, req.subcommand, req.method)


@router.post("/cli/execute")
async def cli_execute(req: CLIExecuteRequest):
    from .cli_proxy import execute_cli_command
    result = await execute_cli_command(
        domain=req.domain,
        subcommand=req.subcommand,
        args=req.args,
        agent_id=req.agent_id,
        method=req.method,
        dry_run=req.dry_run,
    )
    return {
        "success": result.success,
        "output": result.output,
        "exit_code": result.exit_code,
        "iam_allowed": result.iam_allowed,
        "iam_decision": result.iam_decision,
        "iam_reason": result.iam_reason,
        "iam_blocked_at": result.iam_blocked_at,
        "trust_score": result.trust_score,
        "risk_score": result.risk_score,
        "agent_id": result.agent_id,
        "action": result.action,
        "domain": result.domain,
        "subcommand": result.subcommand,
        "latency_ms": round(result.latency_ms, 2),
        "auto_revoked": result.auto_revoked,
        "six_layer": result.six_layer,
    }


@router.get("/cli/audit")
async def cli_audit(limit: int = 50):
    from .cli_proxy import get_cli_audit_log
    return {"audit": get_cli_audit_log(limit), "total": len(get_cli_audit_log(9999))}


@router.get("/cli/stats")
async def cli_stats():
    from .cli_proxy import get_cli_stats
    from .iam_gateway import get_gateway_stats
    return {
        "cli_stats": get_cli_stats(),
        "iam_gateway_stats": get_gateway_stats(),
    }


@router.post("/cli/demo/blocked")
async def cli_demo_blocked():
    from .cli_proxy import execute_cli_command
    result = await execute_cli_command(
        domain="base",
        subcommand="+record-list",
        args=["--base-token", "ZaJ3bqnLOaKW4QscFyRc001GnSf", "--table-id", "tblkLoMhXCeVtchv"],
        agent_id="external_agent",
    )
    return {
        "title": "CLI IAM Demo: External Agent Blocked",
        "title_cn": "CLI管控演示：外部Agent被拦截",
        "scenario": "external_agent tries to read finance table via CLI",
        "scenario_cn": "外部Agent通过CLI尝试读取财务数据",
        "result": {
            "success": result.success,
            "iam_allowed": result.iam_allowed,
            "iam_decision": result.iam_decision,
            "iam_reason": result.iam_reason,
            "iam_blocked_at": result.iam_blocked_at,
            "trust_score": result.trust_score,
            "risk_score": result.risk_score,
            "action": result.action,
            "auto_revoked": result.auto_revoked,
        },
    }


@router.post("/cli/demo/allowed")
async def cli_demo_allowed():
    from .cli_proxy import execute_cli_command
    result = await execute_cli_command(
        domain="calendar",
        subcommand="+agenda",
        agent_id="doc_agent",
    )
    return {
        "title": "CLI IAM Demo: Doc Agent Allowed",
        "title_cn": "CLI管控演示：文档Agent被放行",
        "scenario": "doc_agent reads calendar via CLI",
        "scenario_cn": "文档Agent通过CLI读取日历",
        "result": {
            "success": result.success,
            "iam_allowed": result.iam_allowed,
            "iam_decision": result.iam_decision,
            "iam_reason": result.iam_reason,
            "trust_score": result.trust_score,
            "risk_score": result.risk_score,
            "action": result.action,
        },
    }


@router.post("/cli/demo/escalation")
async def cli_demo_escalation():
    from .cli_proxy import execute_cli_command
    steps = []
    actions = [
        ("im", "+messages-send", "write:feishu_message", "doc_agent", "正常发送消息"),
        ("base", "+record-list", "read:bitable", "doc_agent", "读取多维表格"),
        ("base", "+record-list", "read:bitable", "external_agent", "外部Agent越权读取"),
        ("approval", "instances", "read:approval", "external_agent", "外部Agent越权审批"),
    ]
    for domain, subcmd, expected_action, agent, desc in actions:
        result = await execute_cli_command(
            domain=domain,
            subcommand=subcmd,
            agent_id=agent,
        )
        steps.append({
            "domain": domain,
            "subcommand": subcmd,
            "expected_action": expected_action,
            "agent_id": agent,
            "description": desc,
            "description_cn": desc,
            "iam_allowed": result.iam_allowed,
            "iam_decision": result.iam_decision,
            "iam_reason": result.iam_reason,
            "iam_blocked_at": result.iam_blocked_at,
            "trust_score": result.trust_score,
            "risk_score": result.risk_score,
            "auto_revoked": result.auto_revoked,
        })

    blocked = sum(1 for s in steps if not s["iam_allowed"])
    total = len(steps)
    return {
        "title": "CLI IAM Escalation Test",
        "title_cn": "CLI管控越权测试",
        "statement": "Every CLI command passes through IAM Gateway before execution",
        "statement_cn": "每条CLI命令执行前必须经过IAM网关检查",
        "total_steps": total,
        "blocked": blocked,
        "allowed": total - blocked,
        "pass_rate": f"{blocked}/{total}",
        "all_boundary_held": blocked > 0,
        "steps": steps,
    }


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _process_feishu_message_sync(user_id: str, message: str, chat_id: str = "", message_id: str = ""):
    _logger.info("_process_feishu_message_sync called: user=%s msg='%s' chat=%s", user_id, message[:50], chat_id)
    global _processed_messages
    now = time.time()
    _processed_messages = {k: v for k, v in _processed_messages.items() if now - v < _MESSAGE_TTL}

    try:
        client = get_feishu_client()
        if message.strip().startswith("/cli"):
            try:
                result = _run_async(_process_cli_command(user_id, message, chat_id, client))
                return result
            except Exception as e:
                _logger.error("CLI command error: %s", e, exc_info=True)
                return {"status": "error", "content": str(e)}

        p_req = PlatformRequest(platform="feishu", user_id=user_id, message=message)
        result = run_task_with_alignment(platform_request=p_req)
        _log_feishu_event("ws_message", user_id, message, result)

        reply_content = _format_feishu_reply(result)

        try:
            _run_async(client.send_message(chat_id or user_id, reply_content, receive_id_type="chat_id" if chat_id else "open_id"))
        except Exception as send_err:
            _logger.warning("Failed to send feishu reply: %s", send_err)

        _logger.info("Feishu WS message processed: user=%s status=%s", user_id, result.get("status"))
        return result

    except Exception as e:
        _logger.error("Error processing feishu WS message: %s", e, exc_info=True)
        try:
            error_msg = f"❌ 系统处理异常\n原因：{str(e)}"
            _run_async(client.send_message(chat_id or user_id, error_msg, receive_id_type="chat_id" if chat_id else "open_id"))
        except Exception:
            pass
        return {"status": "error", "content": str(e)}
