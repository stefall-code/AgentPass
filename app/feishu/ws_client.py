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

    _kill_stale_bridges()

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

    log_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "feishu_ws_bridge.log",
    )

    _ws_process = subprocess.Popen(
        [sys.executable, bridge_script],
        env=env,
        stdout=open(log_path, "a", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(bridge_script),
    )

    logger.info("Feishu WS bridge: started subprocess pid=%d log=%s", _ws_process.pid, log_path)


def _kill_stale_bridges():
    import subprocess as _sp
    try:
        result = _sp.run(
            ["wmic", "process", "where", "commandline like '%feishu_ws_bridge%'", "get", "processid"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.isdigit() and int(line) != os.getpid():
                logger.info("Killing stale feishu_ws_bridge PID %s", line)
                try:
                    os.kill(int(line), 9)
                except OSError:
                    pass
    except Exception:
        pass


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
