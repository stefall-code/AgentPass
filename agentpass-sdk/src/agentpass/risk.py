from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List
from enum import Enum


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    risk_score: float = Field(..., ge=0.0, le=1.0)
    factors: List[Dict[str, Any]] = []
    recommendations: List[str] = []


class DetectionResult(BaseModel):
    is_detected: bool = False
    risk_score: float = 0.0
    details: Dict[str, Any] = {}


class BaseDetector(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "base_detector"
    version: str = "1.0"

    def detect(self, context: Dict[str, Any]) -> DetectionResult:
        return DetectionResult()


class AnomalyDetector(BaseDetector):
    name: str = "anomaly_detector"

    def detect(self, context: Dict[str, Any]) -> DetectionResult:
        score = 0.0
        details: Dict[str, Any] = {}
        prompt_risk = context.get("prompt_risk_score", 0.0)
        if prompt_risk > 0.7:
            score = 0.6
            details["reason"] = "High prompt risk score"
        elif prompt_risk > 0.4:
            score = 0.3
            details["reason"] = "Medium prompt risk score"
        return DetectionResult(is_detected=score > 0.0, risk_score=score, details=details)


class FraudDetector(BaseDetector):
    name: str = "fraud_detector"

    def detect(self, context: Dict[str, Any]) -> DetectionResult:
        score = 0.0
        details: Dict[str, Any] = {}
        action = context.get("action", "")
        resource = context.get("resource", "")
        if action in ("delete", "admin_access") and resource in ("admin_panel", "system_config"):
            score = 0.8
            details["reason"] = "Suspicious admin-level operation"
        return DetectionResult(is_detected=score > 0.0, risk_score=score, details=details)


class Risk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    detectors: List[Any] = []

    def assess_risk(self, context: Dict[str, Any]) -> RiskAssessment:
        risk_score = 0.0
        factors = []
        recommendations = []

        for detector in self.detectors:
            result = detector.detect(context)
            if result.is_detected:
                risk_score += result.risk_score
                factors.append({
                    "detector": detector.name,
                    "score": result.risk_score,
                    "details": result.details
                })

        risk_score = min(1.0, risk_score)

        if risk_score >= 0.8:
            risk_level = RiskLevel.CRITICAL
            recommendations.append("Block access immediately")
            recommendations.append("Review account activity")
        elif risk_score >= 0.6:
            risk_level = RiskLevel.HIGH
            recommendations.append("Require additional verification")
            recommendations.append("Monitor activity closely")
        elif risk_score >= 0.4:
            risk_level = RiskLevel.MEDIUM
            recommendations.append("Send security alert")
        else:
            risk_level = RiskLevel.LOW

        return RiskAssessment(
            risk_level=risk_level,
            risk_score=risk_score,
            factors=factors,
            recommendations=recommendations
        )
