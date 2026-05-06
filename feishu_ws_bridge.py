from __future__ import annotations

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("feishu_ws_bridge")


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from app.config import settings

    app_id = settings.FEISHU_APP_ID
    app_secret = settings.FEISHU_APP_SECRET
    verification_token = settings.FEISHU_VERIFICATION_TOKEN or ""

    if not app_id or not app_secret or app_id == "your_app_id":
        logger.error("FEISHU_APP_ID / FEISHU_APP_SECRET not configured in .env")
        sys.exit(1)

    import lark_oapi as lark

    def on_message_receive(data, **kwargs):
        try:
            event = data.event if hasattr(data, "event") else data
            message = event.message if hasattr(event, "message") else {}
            sender = event.sender if hasattr(event, "sender") else {}
            sender_id = sender.sender_id if hasattr(sender, "sender_id") else {}

            user_id = sender_id.get("user_id", "") or sender_id.get("open_id", "") or "unknown"
            message_content = message.get("content", "{}") if isinstance(message, dict) else "{}"
            chat_id = message.get("chat_id", "") if isinstance(message, dict) else ""
            message_id = message.get("message_id", "") if isinstance(message, dict) else ""

            try:
                content_obj = json.loads(message_content) if isinstance(message_content, str) else message_content
                text = content_obj.get("text", "") if isinstance(content_obj, dict) else str(message_content)
            except (json.JSONDecodeError, TypeError):
                text = str(message_content)

            if not text.strip():
                return

            logger.info("Received: user=%s message='%s'", user_id, text[:50])

            import requests
            api_base = os.getenv("API_BASE", "http://127.0.0.1:8000")
            r = requests.post(
                f"{api_base}/api/feishu/webhook",
                json={
                    "type": "event_callback",
                    "event": {
                        "message": {
                            "content": json.dumps({"text": text}),
                            "chat_id": chat_id,
                            "message_id": message_id,
                            "message_type": "text",
                        },
                        "sender": {
                            "sender_id": sender_id,
                            "type": "user",
                        },
                    },
                    "header": {
                        "event_id": message_id or f"ws-{int(time.time())}",
                        "event_type": "im.message.receive_v1",
                    },
                },
                timeout=30,
            )
            logger.info("Forwarded to API: status=%d", r.status_code)

        except Exception as e:
            logger.error("Message handler error: %s", e, exc_info=True)

    event_handler = (
        lark.EventDispatcherHandler
        .builder("", verification_token)
        .register_p2_im_message_receive_v1(on_message_receive)
        .build()
    )

    cli = lark.ws.Client(
        app_id=app_id,
        app_secret=app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )

    logger.info("Starting Feishu WebSocket long-connection bridge...")
    logger.info("App ID: %s", app_id[:8] + "...")
    logger.info("API Base: %s", os.getenv("API_BASE", "http://127.0.0.1:8000"))
    cli.start()


if __name__ == "__main__":
    main()
