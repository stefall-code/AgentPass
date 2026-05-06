from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger("agent_system.feishu_ws")

_ws_process: Optional[subprocess.Popen] = None


def start_ws_bridge():
    global _ws_process

    if _ws_process is not None and _ws_process.poll() is None:
        logger.info("Feishu WS bridge: already running (pid=%d)", _ws_process.pid)
        return

    from app.config import settings

    app_id = settings.FEISHU_APP_ID
    app_secret = settings.FEISHU_APP_SECRET

    if not app_id or not app_secret or app_id == "your_app_id":
        logger.warning("Feishu WS bridge: APP_ID/SECRET not configured, skipping")
        return

    bridge_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "feishu_ws_bridge.py",
    )

    if not os.path.exists(bridge_script):
        logger.warning("Feishu WS bridge script not found: %s", bridge_script)
        return

    env = os.environ.copy()
    env["API_BASE"] = f"http://127.0.0.1:{settings.PORT}"

    _ws_process = subprocess.Popen(
        [sys.executable, bridge_script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(bridge_script),
    )

    logger.info("Feishu WS bridge: started subprocess pid=%d", _ws_process.pid)


def stop_ws_bridge():
    global _ws_process

    if _ws_process is not None:
        try:
            _ws_process.terminate()
            _ws_process.wait(timeout=5)
        except Exception:
            try:
                _ws_process.kill()
            except Exception:
                pass
        _ws_process = None
        logger.info("Feishu WS bridge: stopped")


def is_ws_connected():
    return _ws_process is not None and _ws_process.poll() is None
