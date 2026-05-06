"""
对话上下文分析模块 v1.0
检测多轮对话中的渐进式诱导和上下文操纵
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re


class ContextTurn(BaseModel):
    timestamp: datetime
    role: str
    content: str
    risk_score: float = 0.0


class ProgressiveInjectionResult(BaseModel):
    is_progressive_injection: bool = Field(default=False)
    risk_score: float = Field(default=0.0)
    confidence: float = Field(default=0.0)
    detected_stages: List[str] = Field(default_factory=list)
    explanation: str = Field(default="")


class ContextManipulationResult(BaseModel):
    is_manipulated: bool = Field(default=False)
    manipulation_type: str = Field(default="none")
    risk_score: float = Field(default=0.0)
    explanation: str = Field(default="")


class DialogContextAnalyzer:
    """
    对话上下文分析器
    检测渐进式诱导、上下文操纵等复杂攻击模式
    """

    def __init__(self):
        self._re = re
        self._init_stages()

    def _init_stages(self):
        self.INITIAL_STAGE_KEYWORDS = [
            "你好", "请问", "帮我", "我想", "查询", "看看", "查一下",
            "这个", "帮我", "请帮我", "麻烦", "请教", "咨询",
        ]

        self.EXPLOITATION_STAGE_KEYWORDS = [
            "帮我看看别人的", "帮我查一下某人的", "帮我获取",
            "我想知道", "怎么才能", "怎样才能",
            "如果我", "假设我", "要是能",
        ]

        self.DIRECT_EXPLOIT_STAGE_KEYWORDS = [
            "导出所有", "导出全部", "导出密码", "绕过", "破解",
            "删除", "修改", "冒充", "伪装", "后门",
            "我是管理员", "我需要管理员", "给我最高权限",
        ]

        self.CONTEXT_RESET_PATTERNS = [
            r"忘了.*之前",
            r"忽略.*之前",
            r"忘记.*对话",
            r"重新.*开始",
            r"清空.*上下文",
            r"忘掉.*所有",
        ]

        self.CONTEXT_EXTENSION_PATTERNS = [
            r"基于.*之前",
            r"继续.*上",
            r"接.*之前",
            r"延.*话题",
            r"另外.*顺便",
            r"还有.*顺便",
            r"对了.*顺便",
            r"对了.*另外",
        ]

        self.TRUST_BREAKING_PATTERNS = [
            r"你确定.*吗",
            r"真的.*吗",
            r"不会.*错吧",
            r"让我.*确认",
            r"再.*确认",
            r"验证.*一下",
            r"检查.*一下",
        ]

    def analyze_progressive_injection(self, dialog_history: List[ContextTurn]) -> ProgressiveInjectionResult:
        """
        分析对话是否包含渐进式注入

        攻击阶段:
        1. 初始建立信任 - 正常查询
        2. 试探性利用 - 问"怎么才能..."
        3. 直接利用 - 执行恶意操作
        """
        result = ProgressiveInjectionResult()

        if len(dialog_history) < 2:
            return result

        stages = self._identify_stages(dialog_history)

        if len(stages) >= 3:
            result.is_progressive_injection = True
            result.confidence = 0.9
            result.risk_score = 0.95
            result.detected_stages = ["信任建立", "试探利用", "直接利用"]
            result.explanation = "检测到典型的渐进式注入攻击模式"
        elif len(stages) == 2:
            stage_types = set(stages)
            if "exploitation" in stage_types or "direct_exploit" in stage_types:
                result.is_progressive_injection = True
                result.confidence = 0.7
                result.risk_score = 0.75
                result.detected_stages = stages
                result.explanation = "检测到可疑的渐进式攻击模式"

        if self._check_context_reset(dialog_history):
            result.risk_score = max(result.risk_score, 0.8)
            result.explanation += "；检测到上下文重置尝试"

        if self._check_trust_breaking(dialog_history[-1] if dialog_history else None):
            result.risk_score = max(result.risk_score, 0.6)
            result.explanation += "；检测到信任破坏模式"

        return result

    def _identify_stages(self, dialog_history: List[ContextTurn]) -> List[str]:
        stages = []

        for turn in dialog_history:
            content = turn.content.lower()
            role = turn.role

            if role == "user":
                if any(kw in content for kw in self.INITIAL_STAGE_KEYWORDS):
                    if not stages or stages[-1] != "initial":
                        stages.append("initial")
                elif any(kw in content for kw in self.EXPLOITATION_STAGE_KEYWORDS):
                    if not stages or stages[-1] not in ["exploitation", "direct_exploit"]:
                        stages.append("exploitation")
                elif any(kw in content for kw in self.DIRECT_EXPLOIT_STAGE_KEYWORDS):
                    if not stages or stages[-1] != "direct_exploit":
                        stages.append("direct_exploit")

        return stages

    def _check_context_reset(self, dialog_history: List[ContextTurn]) -> bool:
        if not dialog_history:
            return False

        last_turn = dialog_history[-1]
        content = last_turn.content

        for pattern in self.CONTEXT_RESET_PATTERNS:
            if self._re.search(pattern, content):
                return True

        reset_count = 0
        for turn in dialog_history[-3:]:
            for pattern in self.CONTEXT_RESET_PATTERNS:
                if self._re.search(pattern, turn.content):
                    reset_count += 1
                    break

        return reset_count >= 2

    def _check_trust_breaking(self, turn: Optional[ContextTurn]) -> bool:
        if not turn:
            return False

        content = turn.content
        for pattern in self.TRUST_BREAKING_PATTERNS:
            if self._re.search(pattern, content):
                return True
        return False

    def analyze_context_manipulation(self, current_prompt: str, dialog_history: List[ContextTurn]) -> ContextManipulationResult:
        """
        分析当前提示词是否在操纵上下文
        """
        result = ContextManipulationResult()

        context_extension_score = 0.0
        for pattern in self.CONTEXT_EXTENSION_PATTERNS:
            if self._re.search(pattern, current_prompt):
                context_extension_score += 0.3

        if len(dialog_history) > 3:
            recent_same_topic = 0
            for turn in dialog_history[-5:-1]:
                if self._has_topic_overlap(current_prompt, turn.content):
                    recent_same_topic += 1

            if recent_same_topic >= 3:
                context_extension_score += 0.4

        if context_extension_score > 0.5:
            result.is_manipulated = True
            result.manipulation_type = "context_extension"
            result.risk_score = min(0.9, context_extension_score)
            result.explanation = "检测到上下文扩展尝试，可能在渐进式诱导"

        escalation_score = self._check_risk_escalation(current_prompt, dialog_history)
        if escalation_score > 0:
            result.is_manipulated = True
            result.manipulation_type = "risk_escalation"
            result.risk_score = max(result.risk_score, escalation_score)
            result.explanation = "检测到风险升级模式"

        return result

    def _has_topic_overlap(self, text1: str, text2: str) -> bool:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = words1 & words2
        return len(overlap) >= 3

    def _check_risk_escalation(self, current_prompt: str, dialog_history: List[ContextTurn]) -> float:
        if not dialog_history:
            return 0.0

        current_risk = 0.0
        if any(kw in current_prompt.lower() for kw in self.DIRECT_EXPLOIT_STAGE_KEYWORDS):
            current_risk = 0.8
        elif any(kw in current_prompt.lower() for kw in self.EXPLOITATION_STAGE_KEYWORDS):
            current_risk = 0.4

        historical_risks = [turn.risk_score for turn in dialog_history[-5:]]
        max_historical = max(historical_risks) if historical_risks else 0.0

        if current_risk > max_historical and max_historical > 0:
            escalation = (current_risk - max_historical) * 0.5
            if escalation > 0.3:
                return escalation

        return 0.0

    def get_context_summary(self, dialog_history: List[ContextTurn]) -> Dict:
        if not dialog_history:
            return {
                "turn_count": 0,
                "avg_risk_score": 0.0,
                "risk_trend": "stable",
                "has_progressive_injection": False,
                "topics": []
            }

        risk_scores = [turn.risk_score for turn in dialog_history]
        avg_risk = sum(risk_scores) / len(risk_scores)

        risk_trend = "stable"
        if len(risk_scores) >= 3:
            recent = risk_scores[-3:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                risk_trend = "increasing"
            elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                risk_trend = "decreasing"

        topics = set()
        for turn in dialog_history:
            topics.update(self._extract_topics(turn.content))

        return {
            "turn_count": len(dialog_history),
            "avg_risk_score": round(avg_risk, 3),
            "risk_trend": risk_trend,
            "has_progressive_injection": self.analyze_progressive_injection(dialog_history).is_progressive_injection,
            "topics": list(topics)[:10],
        }

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        topic_keywords = {
            "用户数据": ["用户", "客户", "会员"],
            "财务数据": ["财务", "工资", "账单", "金额"],
            "权限": ["权限", "管理员", "root", "sudo"],
            "导出操作": ["导出", "下载", "获取"],
            "日志": ["日志", "记录", "审计"],
        }

        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)

        return topics
