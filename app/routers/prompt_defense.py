from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException, Query

from app.schemas_prompt import (
    PromptAnalysisRequest,
    PromptAnalysisResponse,
    PromptDefenseAnalyzeRequest,
    PromptDefenseAnalyzeResponse,
)

logger = logging.getLogger("agent_system")
router = APIRouter()

_guard_ref = None


def set_guard(guard):
    global _guard_ref
    _guard_ref = guard


@router.post("/analyze-prompt", response_model=PromptAnalysisResponse)
def analyze_prompt(request: PromptAnalysisRequest) -> PromptAnalysisResponse:
    if not _guard_ref:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")
    result = _guard_ref.analyze_prompt(request.prompt)
    return PromptAnalysisResponse(
        is_safe=result["is_safe"],
        risk_score=result["risk_score"],
        injection_type=str(result.get("injection_type")) if result.get("injection_type") else None,
        reason=result["reason"],
        matched_patterns=result.get("matched_patterns") if isinstance(result.get("matched_patterns"), list) else []
    )


@router.post("/prompt-defense/analyze", response_model=PromptDefenseAnalyzeResponse, tags=["Prompt Defense"])
def prompt_defense_analyze(request: PromptDefenseAnalyzeRequest) -> PromptDefenseAnalyzeResponse:
    if not _guard_ref:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")

    result = _guard_ref.prompt_defense.analyze(request.prompt, history=request.history or None, user_id=request.agent_id)

    return PromptDefenseAnalyzeResponse(
        risk_score=result.risk_score,
        triggered_rules=[r.model_dump() for r in result.triggered_rules],
        severity=result.severity,
        recommendation=result.recommendation,
        is_safe=result.is_safe,
        injection_type=str(result.injection_type) if result.injection_type else None,
        reason=result.reason,
        progressive_risk=result.progressive_risk,
        matched_rules=result.matched_rules,
        matched_patterns=result.matched_patterns,
        layer_scores=result.layer_scores,
        attack_intent=result.attack_intent,
        token_smuggling_detected=result.token_smuggling_detected,
        dialog_risk_trend=result.dialog_risk_trend,
        new_attack_types=result.new_attack_types,
    )


@router.delete("/prompt-defense/history", tags=["Prompt Defense"])
def prompt_defense_clear_history(agent_id: str = Query(default="default")) -> Dict[str, str]:
    if not _guard_ref:
        raise HTTPException(status_code=503, detail="Prompt defense module not available")
    _guard_ref.prompt_defense.clear_dialog_history(agent_id)
    return {"message": f"Dialog history cleared for {agent_id}"}


@router.get("/prompt-defense/defense-mode", tags=["Prompt Defense"])
def get_defense_mode_status() -> Dict:
    from app.orchestrator.orchestrator import get_defense_mode
    return get_defense_mode()
