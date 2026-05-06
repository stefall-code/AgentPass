from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app import auth as auth_module
from app.ws import ws_manager

logger = logging.getLogger("agent_system")
ws_router = APIRouter(tags=["WebSocket"])


def _update_heartbeat():
    try:
        import main as _main
        _main._last_ping_at = time.time()
    except Exception:
        pass


@ws_router.websocket("/ws")
async def websocket_main(websocket: WebSocket):
    try:
        await websocket.accept()
        await ws_manager.add(websocket, None)
        _update_heartbeat()
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                ws_manager.touch()
                _update_heartbeat()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except asyncio.TimeoutError:
                break
            except Exception:
                break
    finally:
        await ws_manager.remove(websocket)


@ws_router.websocket("/ws/keepalive")
async def websocket_endpoint(websocket: WebSocket):
    agent_id = None
    token_param = websocket.query_params.get("token")
    if token_param:
        try:
            ctx = auth_module.resolve_token(token_param, "ws-client")
            agent_id = ctx.agent.get("agent_id")
        except Exception:
            pass

    try:
        await websocket.accept()
        await ws_manager.add(websocket, agent_id)
        _update_heartbeat()
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                ws_manager.touch()
                _update_heartbeat()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
            except asyncio.TimeoutError:
                break
            except Exception:
                break
    finally:
        await ws_manager.remove(websocket)


@ws_router.websocket("/ws/audit")
async def websocket_audit(websocket: WebSocket):
    agent_id = None
    token_param = websocket.query_params.get("token")
    if token_param:
        try:
            ctx = auth_module.resolve_token(token_param, "ws-client")
            agent_id = ctx.agent.get("agent_id")
        except Exception:
            pass

    try:
        await websocket.accept()
        await ws_manager.add(websocket, agent_id)
        ws_manager.start_consumer()
        _update_heartbeat()
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                ws_manager.touch()
                _update_heartbeat()
            except WebSocketDisconnect:
                break
            except asyncio.TimeoutError:
                break
            except Exception:
                break
    finally:
        await ws_manager.remove(websocket)
