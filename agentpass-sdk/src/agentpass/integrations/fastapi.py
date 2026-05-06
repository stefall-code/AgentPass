from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from typing import Optional, Callable, Dict, Any, List
from functools import wraps

from ..guard import Guard


class AuthenticatedRequest:
    def __init__(self, request: Request, user_payload: Dict[str, Any]):
        self.request = request
        self.user = user_payload
        self.user_id = user_payload.get("sub")
        self.role = user_payload.get("role", "user")
        self.is_authenticated = True

    def __getattr__(self, name: str):
        return getattr(self.request, name)


class GuardMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        secret: str,
        exclude_paths: Optional[List[str]] = None,
        auth_header: str = "Authorization",
        bearer_prefix: str = "Bearer",
        risk_threshold: float = 0.7,
        enable_audit: bool = True
    ):
        super().__init__(app)
        self.secret = secret
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
        self.auth_header = auth_header
        self.bearer_prefix = bearer_prefix
        self.risk_threshold = risk_threshold
        self.enable_audit = enable_audit

        self._setup_guard()

    def _setup_guard(self):
        self.guard = Guard(secret=self.secret)

    def _is_excluded_path(self, path: str) -> bool:
        for pattern in self.exclude_paths:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if path.startswith(prefix):
                    return True
            elif path == pattern or path.startswith(pattern.rstrip("/") + "/"):
                return True
        return False

    def _extract_token(self, request: Request) -> Optional[str]:
        auth_header_value = request.headers.get(self.auth_header)
        if not auth_header_value:
            return None

        parts = auth_header_value.split()
        if len(parts) != 2:
            return None

        scheme, token = parts
        if scheme.lower() == self.bearer_prefix.lower():
            return token

        return None

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        token = self._extract_token(request)

        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Missing or invalid Authorization header"
                }
            )

        payload = self.guard.authenticate(token)
        if not payload:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "detail": "Invalid or expired token"
                }
            )

        request.state.user = payload
        request.state.user_id = payload.get("sub")
        request.state.role = payload.get("role", "user")

        response = await call_next(request)
        return response

    def check_access(
        self,
        request: Request,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not hasattr(request.state, "user") or not request.state.user:
            return {
                "allowed": False,
                "error": "Unauthorized",
                "detail": "User not authenticated"
            }

        user = request.state.user
        context = context or {}
        context["user_id"] = user.get("sub")
        context["role"] = user.get("role", "user")
        context["ip_address"] = request.client.host if request.client else None

        result = self.guard.check(token=request.state.user.get("token"), action=action, resource=resource)

        if not result["allowed"]:
            return {
                "allowed": False,
                "error": "Forbidden",
                "detail": result.get("reason", "Access denied")
            }

        return {
            "allowed": True,
            "user_id": user.get("sub"),
            "role": user.get("role", "user")
        }


def require_auth(
    resource: str,
    action: str,
    context_fn: Optional[Callable[[Request], Dict[str, Any]]] = None
):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            auth_user = getattr(request.state, "user", None)
            if not auth_user:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "detail": "Authentication required"
                    }
                )

            token = None
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]

            guard = Guard(secret="")

            context = context_fn(request) if context_fn else {}
            context["user_id"] = auth_user.get("sub")
            context["role"] = auth_user.get("role", "user")
            context["ip_address"] = request.client.host if request.client else None

            result = guard.check(token=token, action=action, resource=resource)

            if not result["allowed"]:
                return JSONResponse(
                    status_code=403,
                    content={
                        "error": "Forbidden",
                        "detail": result.get("reason", "Access denied")
                    }
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator


class AgentPassAuth:
    def __init__(self, secret: str):
        self.secret = secret
        self.guard = Guard(secret=self.secret)

    def create_token(self, user_id: str, role: str = "user", **extra_claims) -> str:
        return self.guard.issue_token(user_id, role=role, **extra_claims)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        return self.guard.authenticate(token)

    def check_permission(
        self,
        token: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        context = context or {}
        result = self.guard.check(token=token, action=action, resource=resource)
        return result

    async def __call__(self, request: Request) -> Optional[AuthenticatedRequest]:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None

        token = auth_header[7:]
        payload = self.guard.authenticate(token)

        if not payload:
            return None

        payload["token"] = token
        return AuthenticatedRequest(request, payload)
