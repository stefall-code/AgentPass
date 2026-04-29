﻿﻿import json
import logging
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, BackgroundTasks
from pydantic import BaseModel

from .client import get_feishu_client
from app.orchestrator.orchestrator import run_task, secure_agent_call
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


async def _process_feishu_message(user_id: str, message: str, chat_id: str = ""):
    client = get_feishu_client()

    try:
        p_req = PlatformRequest(platform="feishu", user_id=user_id, message=message)
        result = run_task_with_alignment(platform_request=p_req)
        _log_feishu_event("message", user_id, message, result)

        reply_content = result.get("content", "处理完成")

        if result.get("status") == "success" and result.get("data"):
            if chat_id:
                await client.send_message(chat_id, reply_content, receive_id_type="chat_id")
            else:
                await client.send_message(user_id, reply_content)
        else:
            if chat_id:
                await client.send_message(chat_id, reply_content, receive_id_type="chat_id")
            else:
                await client.send_message(user_id, reply_content)

        _logger.info("Feishu message processed: user=%s status=%s", user_id, result.get("status"))
        return result

    except Exception as e:
        _logger.error("Error processing feishu message: %s", e, exc_info=True)
        error_msg = f"❌ 系统处理异常\n原因：{str(e)}"
        try:
            if chat_id:
                await client.send_message(chat_id, error_msg, receive_id_type="chat_id")
            else:
                await client.send_message(user_id, error_msg)
        except Exception:
            pass
        return {"status": "error", "content": error_msg, "reason": str(e)}


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

    _logger.info("Feishu webhook: user=%s message='%s'", user_id, message[:50])

    background_tasks.add_task(_process_feishu_message, user_id, message, chat_id)

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
    return {
        "configured": client.is_configured(),
        "app_id_set": bool(client.app_id),
        "app_secret_set": bool(client.app_secret),
        "verification_token_set": bool(client.verification_token),
        "mode": "production" if client.is_configured() else "mock",
        "total_events": len(_feishu_event_log),
    }


@router.post("/connect")
async def feishu_connect():
    import subprocess
    import httpx

    from main import _ngrok_url

    ngrok_url = _ngrok_url
    ngrok_started = False

    if not ngrok_url:
        try:
            ngrok_procs = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq ngrok.exe"],
                capture_output=True, text=True, timeout=5,
            )
            if "ngrok.exe" not in ngrok_procs.stdout:
                subprocess.Popen(
                    ["ngrok", "http", "8000"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=0x00000008 if os.name == "nt" else 0,
                )
                ngrok_started = True
                await asyncio.sleep(4)

            for _ in range(10):
                try:
                    resp = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: httpx.get("http://127.0.0.1:4040/api/tunnels", timeout=3),
                    )
                    data = resp.json()
                    for t in data.get("tunnels", []):
                        if t.get("proto") == "https":
                            ngrok_url = t["public_url"]
                            break
                    if ngrok_url:
                        break
                except Exception:
                    await asyncio.sleep(1)
        except Exception as e:
            _logger.warning("Ngrok start failed: %s", e)

    if ngrok_url:
        import main as main_mod
        main_mod._ngrok_url = ngrok_url
        os.environ["FEISHU_PUBLIC_URL"] = ngrok_url

    client = get_feishu_client()
    token_ok = False
    if client.is_configured():
        try:
            token = await client.get_tenant_access_token()
            token_ok = bool(token)
        except Exception:
            pass

    webhook_url = f"{ngrok_url}/api/feishu/webhook" if ngrok_url else ""

    return {
        "connected": client.is_configured() and token_ok,
        "mode": "production" if client.is_configured() else "mock",
        "token_ok": token_ok,
        "ngrok_url": ngrok_url,
        "ngrok_started": ngrok_started,
        "webhook_url": webhook_url,
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
