"""
用户画像与行为分析模块 v1.0
基于用户历史行为构建画像，检测异常访问模式
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import re


class UserAction(BaseModel):
    timestamp: datetime
    action: str
    resource: str
    risk_score: float = 0.0
    blocked: bool = False


class UserProfile(BaseModel):
    user_id: str
    role: str = "user"
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    total_requests: int = 0
    blocked_requests: int = 0
    suspicious_requests: int = 0
    accessed_resources: Set[str] = Field(default_factory=set)
    accessed_resource_types: Set[str] = Field(default_factory=set)
    action_history: List[UserAction] = Field(default_factory=list)
    failed_attempts: int = 0
    risk_trend: List[float] = Field(default_factory=list)

    def is_trustworthy(self) -> bool:
        if self.total_requests < 10:
            return True
        block_rate = self.blocked_requests / self.total_requests
        return block_rate < 0.1

    def get_risk_level(self) -> str:
        if self.total_requests < 5:
            return "unknown"
        if self.failed_attempts > 10:
            return "high"
        block_rate = self.blocked_requests / max(self.total_requests, 1)
        if block_rate > 0.3:
            return "high"
        elif block_rate > 0.1:
            return "medium"
        return "low"

    def add_action(self, action: UserAction):
        self.action_history.append(action)
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-100:]
        self.total_requests += 1
        self.last_active = datetime.now()
        if action.blocked:
            self.blocked_requests += 1
        if action.risk_score > 0.5:
            self.suspicious_requests += 1


class UserProfileManager:
    """
    用户画像管理器
    维护用户历史行为，构建风险画像
    """

    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
        self._action_history: Dict[str, List[UserAction]] = defaultdict(list)
        self._re = re

    def get_profile(self, user_id: str) -> UserProfile:
        if user_id not in self._profiles:
            self._profiles[user_id] = UserProfile(user_id=user_id)
        return self._profiles[user_id]

    def record_action(self, user_id: str, action: str, resource: str,
                      risk_score: float = 0.0, blocked: bool = False):
        profile = self.get_profile(user_id)
        user_action = UserAction(
            timestamp=datetime.now(),
            action=action,
            resource=resource,
            risk_score=risk_score,
            blocked=blocked
        )
        profile.add_action(user_action)
        self._action_history[user_id].append(user_action)

    def check_access_pattern(self, user_id: str, resource: str) -> Dict:
        profile = self.get_profile(user_id)

        result = {
            "is_unusual_resource": False,
            "is_unusual_time": False,
            "is_high_frequency": False,
            "risk_boost": 0.0,
            "reason": []
        }

        # 只有在明显异常时才加成
        if resource not in profile.accessed_resources and profile.total_requests > 50:
            result["is_unusual_resource"] = True
            result["risk_boost"] += 0.05
            result["reason"].append("首次访问该资源")

        recent_actions = [a for a in profile.action_history
                         if a.timestamp > datetime.now() - timedelta(minutes=5)]
        if len(recent_actions) > 20:
            result["is_high_frequency"] = True
            result["risk_boost"] += 0.1
            result["reason"].append("高频访问")

        blocked_recent = sum(1 for a in profile.action_history[-20:] if a.blocked)
        if blocked_recent >= 5:
            result["risk_boost"] += 0.15
            result["reason"].append(f"近期有{blocked_recent}次被拦截")

        if profile.get_risk_level() == "high" and profile.total_requests > 100:
            result["risk_boost"] += 0.15
            result["reason"].append("用户画像风险等级为高")

        return result

    def analyze_behavior(self, user_id: str) -> Dict:
        profile = self.get_profile(user_id)

        analysis = {
            "user_id": user_id,
            "role": profile.role,
            "total_requests": profile.total_requests,
            "risk_level": profile.get_risk_level(),
            "is_trustworthy": profile.is_trustworthy(),
            "access_pattern": {
                "unique_resources": len(profile.accessed_resources),
                "resource_types": len(profile.accessed_resource_types),
            },
            "recent_block_rate": round(profile.blocked_requests / max(profile.total_requests, 1), 3),
            "recent_suspicious_rate": round(profile.suspicious_requests / max(profile.total_requests, 1), 3),
        }

        return analysis


class BehaviorAnomalyDetector:
    """
    行为异常检测器
    检测用户行为中的异常模式
    """

    SUSPICIOUS_SEQUENCE_PATTERNS = [
        ["login", "access_sensitive", "export_data"],
        ["login", " privilege_escalation", "access_admin"],
        ["normal_query", " privilege_escalation", "sensitive_data"],
        ["query", "query", "export_all"],
        ["read", "read", "delete"],
    ]

    def __init__(self):
        self._action_sequences: Dict[str, List[str]] = defaultdict(list)

    def record_action(self, user_id: str, action: str):
        self._action_sequences[user_id].append(action)
        if len(self._action_sequences[user_id]) > 20:
            self._action_sequences[user_id] = self._action_sequences[user_id][-20:]

    def check_sequence(self, user_id: str, current_action: str) -> Dict:
        seq = self._action_sequences[user_id][-5:] + [current_action]

        result = {
            "is_suspicious_sequence": False,
            "risk_score": 0.0,
            "pattern_name": None,
            "explanation": ""
        }

        for pattern in self.SUSPICIOUS_SEQUENCE_PATTERNS:
            if self._matches_pattern(seq, pattern):
                result["is_suspicious_sequence"] = True
                result["risk_score"] = 0.85
                result["pattern_name"] = "可疑行为序列"
                result["explanation"] = f"检测到可疑行为序列: {' -> '.join(pattern)}"
                return result

        if len(seq) >= 3:
            privilege_escalation_count = sum(1 for a in seq if "privilege" in a.lower() or "admin" in a.lower())
            if privilege_escalation_count >= 2:
                result["is_suspicious_sequence"] = True
                result["risk_score"] = 0.7
                result["pattern_name"] = "权限升级尝试"
                result["explanation"] = "检测到连续的权限升级尝试"
                return result

            export_count = sum(1 for a in seq if "export" in a.lower() or "download" in a.lower())
            if export_count >= 3:
                result["is_suspicious_sequence"] = True
                result["risk_score"] = 0.75
                result["pattern_name"] = "批量导出行为"
                result["explanation"] = "检测到连续的数据导出行为"
                return result

        return result

    def _matches_pattern(self, sequence: List[str], pattern: List[str]) -> bool:
        if len(sequence) < len(pattern):
            return False

        for i in range(len(sequence) - len(pattern) + 1):
            match = True
            for j, p in enumerate(pattern):
                if p not in sequence[i + j].lower():
                    match = False
                    break
            if match:
                return True
        return False
