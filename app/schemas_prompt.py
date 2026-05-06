from __future__ import annotations

from typing import Dict, Any, List

from pydantic import BaseModel, Field


class PromptAnalysisRequest(BaseModel):
    prompt: str


class PromptAnalysisResponse(BaseModel):
    is_safe: bool
    risk_score: float
    injection_type: str | None = None
    reason: str
    matched_patterns: list[str]


class PromptDefenseAnalyzeRequest(BaseModel):
    prompt: str
    history: list[str] = Field(default_factory=list)
    agent_id: str = "anonymous"


class PromptDefenseAnalyzeResponse(BaseModel):
    risk_score: float
    triggered_rules: list[dict]
    severity: str
    recommendation: str
    is_safe: bool
    injection_type: str | None = None
    reason: str
    progressive_risk: float = 0.0
    matched_rules: list[str] = Field(default_factory=list)
    matched_patterns: list[str] = Field(default_factory=list)
    layer_scores: Dict[str, float] = Field(default_factory=dict)
    attack_intent: str | None = None
    token_smuggling_detected: bool = False
    dialog_risk_trend: list[float] = Field(default_factory=list)
    new_attack_types: list[str] = Field(default_factory=list)


class OpenClawCheckRequest(BaseModel):
    agent_id: str
    user: str
    action: str
    resource: str
    prompt: str | None = None


class OpenClawCheckResponse(BaseModel):
    allowed: bool
    risk_score: float
    reason: str
