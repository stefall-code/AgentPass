from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app import database, identity

_guard = None
try:
    import sys
    from pathlib import Path
    _sdk_path = Path(__file__).parent / "agentpass-sdk" / "src"
    if _sdk_path.exists() and str(_sdk_path) not in sys.path:
        sys.path.insert(0, str(_sdk_path))
    from agentpass import Guard
    _guard = Guard(secret="demo-prompt-defense")
except ImportError:
    pass
except Exception:
    pass

from app.middleware import (
    ErrorHandlerMiddleware,
    RateLimitMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)
from app.routers import admin_router, agent_router, auth_router, resource_router, ws_router, platforms
from app.routers.insights import insight_router
from app.routers.approval import approval_router
from app.routers.drift import drift_router
from app.routers.context import context_router
from app.routers.delegation import router as delegation_router
from app.feishu import feishu_router
from app.routers.governance import router as governance_router
from app.routers.explain import router as explain_router
from app.routers.gateway import router as gateway_router
from app.routers.alignment import router as alignment_router
from app.routers.revocation import router as revocation_router
from app.routers.credential_broker import router as broker_router
from app.routers.protocols import router as protocols_router
from app.routers.oauth import router as oauth_router
from app.routers.owasp import router as owasp_router
from app.routers.p2 import router as p2_router
from app.routers.openclaw import router as openclaw_router
from app.routers.prompt_defense import router as prompt_defense_router
from app.routers.debug import router as debug_router
from app.routers.pages import router as pages_router
from app.services import start_background_tasks, stop_background_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_system")

_last_ping_at: float = time.time()
_IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "0"))


async def _idle_shutdown_watcher():
    if _IDLE_TIMEOUT <= 0:
        return
    await asyncio.sleep(5)
    while True:
        await asyncio.sleep(30)
        if time.time() - _last_ping_at > _IDLE_TIMEOUT:
            logger.info("Browser disconnected for %ds, shutting down...", _IDLE_TIMEOUT)
            os.kill(os.getpid(), signal.SIGINT)
            break


@asynccontextmanager
async def lifespan(app_: FastAPI):
    security_warnings = settings.validate_security()
    if security_warnings:
        logger.warning("⚠️  INSECURE DEFAULTS DETECTED: %s", ", ".join(security_warnings))
        logger.warning("⚠️  Set proper values via .env or environment variables before production!")
        if os.getenv("ENVIRONMENT") == "production":
            for key in security_warnings:
                logger.error("❌ FATAL: %s uses insecure default in production!", key)
            raise RuntimeError(f"Production environment requires secure configuration for: {', '.join(security_warnings)}")

    database.init_db()
    identity.sync_demo_agents(reset_state=True)

    try:
        from app.security.owasp_shield import load_state
        load_state()
    except Exception:
        pass

    from app.routers import openclaw as _oc, prompt_defense as _pd, debug as _dbg
    _oc.set_guard(_guard)
    _pd.set_guard(_guard)
    _dbg.set_guard(_guard)

    start_background_tasks()

    try:
        from app.feishu.ws_client import start_ws_bridge
        start_ws_bridge()
        logger.info("Feishu WebSocket bridge process started")
    except Exception as e:
        logger.warning("Feishu WS bridge start failed (non-fatal): %s", e)

    asyncio.create_task(_idle_shutdown_watcher())

    logger.info("system started")
    yield
    await stop_background_tasks()

    try:
        from app.feishu.ws_client import stop_ws_bridge
        stop_ws_bridge()
        logger.info("Feishu WebSocket bridge stopped")
    except Exception:
        pass

    try:
        from app.security.owasp_shield import save_state
        save_state()
    except Exception:
        pass

    database.close_connection()
    logger.info("system shutdown")


app = FastAPI(
    title="Agent Identity & Permission System",
    description=(
        "A local demo for agent identity authentication, token-based access control, "
        "policy evaluation, and auditable secure execution."
    ),
    version="v2.6.1",
    lifespan=lifespan,
)

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_MAX, window_seconds=settings.RATE_LIMIT_WINDOW)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.mount("/static", StaticFiles(directory=settings.FRONTEND_DIR), name="static")

app.include_router(pages_router)
app.include_router(auth_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(resource_router, prefix="/api")
app.include_router(ws_router)
app.include_router(insight_router, prefix="/api")
app.include_router(approval_router, prefix="/api")
app.include_router(drift_router)
app.include_router(context_router)
app.include_router(platforms.router, prefix="/api/v2")
app.include_router(delegation_router, prefix="/api")
app.include_router(feishu_router, prefix="/api")
app.include_router(governance_router, prefix="/api")
app.include_router(explain_router, prefix="/api")
app.include_router(gateway_router, prefix="/api")
app.include_router(alignment_router, prefix="/api")
app.include_router(revocation_router, prefix="/api")
app.include_router(broker_router, prefix="/api")
app.include_router(protocols_router, prefix="/api")
app.include_router(oauth_router, prefix="/api")
app.include_router(owasp_router, prefix="/api")
app.include_router(p2_router, prefix="/api")
app.include_router(openclaw_router, prefix="/api")
app.include_router(prompt_defense_router, prefix="/api")
app.include_router(debug_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    import urllib.request

    def _kill_port_occupier(port=8000):
        import subprocess as _sp
        try:
            result = _sp.run(
                ["netstat", "-aon"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.split("\n"):
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
                    pid = int(parts[-1])
                    if pid != os.getpid() and pid != 0:
                        logger.info("Port %d occupied by PID %d, killing...", port, pid)
                        try:
                            os.kill(pid, signal.SIGTERM)
                        except OSError:
                            try:
                                _sp.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
                            except Exception:
                                pass
                        time.sleep(1)
        except Exception:
            pass

    _kill_port_occupier(8000)

    def open_browser():
        url = "http://127.0.0.1:8000"
        for _ in range(30):
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            logger.warning("Server not ready after 15s, skipping browser open")
            return
        import subprocess
        import sys
        try:
            if sys.platform == "win32":
                subprocess.Popen(["cmd", "/c", "start", url], shell=False)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception as e:
            logger.warning("Failed to open browser via subprocess: %s", e)
            try:
                webbrowser.open(url)
            except Exception as e2:
                logger.warning("Failed to open browser via webbrowser: %s", e2)

    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()

    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000,
        reload=False, ws="websockets",
        ws_ping_interval=10, ws_ping_timeout=10,
    )
