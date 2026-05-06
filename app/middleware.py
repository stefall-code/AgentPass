from __future__ import annotations

import time
import uuid
from collections import defaultdict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

import logging

logger = logging.getLogger("agent_system")


import re

_REQUEST_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-_]{1,64}$')


class RequestIDMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        raw_id = request.headers.get("X-Request-ID", "")
        request_id = raw_id if _REQUEST_ID_PATTERN.match(raw_id) else str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """请求耗时测量与日志"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            import main as _main
            _main._last_ping_at = time.time()
        except Exception:
            pass

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        rid = getattr(request.state, "request_id", "-")
        logger.info(
            "%s %s %d %.1fms rid=%s",
            request.method, request.url.path, response.status_code, duration_ms, rid,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """令牌桶限流：保护认证端点免受暴力攻击"""

    def __init__(self, app: ASGIApp, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _is_limited(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self.window_seconds
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]
        if len(self._buckets[key]) >= self.max_requests:
            return True
        self._buckets[key].append(now)
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        sensitive_prefixes = ("/login", "/register", "/auth/")
        if not any(request.url.path.startswith(p) for p in sensitive_prefixes):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"rl:{client_ip}:{request.url.path}"

        if self._is_limited(key):
            logger.warning("rate_limited ip=%s path=%s", client_ip, request.url.path)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again later."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """全局异常兜底：未处理异常返回统一JSON"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except Exception:
            rid = getattr(request.state, "request_id", "-")
            logger.exception("unhandled_exception rid=%s path=%s", rid, request.url.path)
            return Response(
                content=f'{{"detail":"Internal server error.","request_id":"{rid}"}}',
                status_code=500,
                media_type="application/json",
            )
