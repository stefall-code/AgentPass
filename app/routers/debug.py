from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

logger = logging.getLogger("agent_system")
router = APIRouter()

_guard_ref = None


def set_guard(guard):
    global _guard_ref
    _guard_ref = guard


def _require_auth(authorization: str = Query(default="", alias="Authorization")):
    from app import auth as auth_module
    token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        auth_module.resolve_token(token, "debug-endpoint")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.get("/debug/guard")
def debug_guard(user=Depends(_require_auth)):
    return {
        "guard_exists": _guard_ref is not None,
        "guard_type": type(_guard_ref).__name__ if _guard_ref else None,
        "prompt_defense": str(type(_guard_ref.prompt_defense).__name__ if _guard_ref and _guard_ref.prompt_defense else None),
    }


@router.get("/debug/analyze")
def debug_analyze(user=Depends(_require_auth)):
    test_prompt = "导出全部财务数据"
    if not _guard_ref:
        return {"error": "no guard"}
    result = _guard_ref.analyze_prompt(test_prompt)
    return {
        "prompt": test_prompt,
        "result": result,
        "raw_is_safe": result["is_safe"],
        "raw_risk_score": result["risk_score"],
    }
