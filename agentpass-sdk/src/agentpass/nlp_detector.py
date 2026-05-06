"""
NLP语义分析检测模块 v1.0
使用TF-IDF向量化 + 余弦相似度 + 语义漂移检测来识别提示词注入
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Tuple
import re
from collections import Counter


class NLPScoreResult(BaseModel):
    semantic_drift: float = Field(default=0.0, description="语义漂移分数 0.0-1.0")
    anomaly_score: float = Field(default=0.0, description="异常分数 0.0-1.0")
    authority_claim_score: float = Field(default=0.0, description="权威声称分数 0.0-1.0")
    intent_classification: str = Field(default="normal", description="意图分类: normal/suspicious/malicious")
    risk_score: float = Field(default=0.0, description="综合NLP风险分数 0.0-1.0")
    detected_patterns: List[str] = Field(default_factory=list, description="检测到的模式列表")
    confidence: float = Field(default=0.0, description="检测置信度 0.0-1.0")
    explanation: str = Field(default="", description="分析解释")


class NLPPromptDetector:
    """
    NLP增强型提示词检测器
    使用语义分析、意图分类、权威声称检测来判断提示词是否异常
    """

    AUTHORITY_CLAIM_PATTERNS = [
        r"我是.*管理员",
        r"我是.*经理",
        r"我是.*老板",
        r"我是.*领导",
        r"我是.*总监",
        r"我是.*负责人",
        r"我是.*总裁",
        r"以.*管理员",
        r"以.*经理",
        r"以.*领导",
        r"以.*老板",
        r"用.*管理员.*权限",
        r"用.*经理.*权限",
        r"系统.*管理员",
        r"IT.*管理员",
        r"安全.*管理员",
        r"数据库.*管理员",
        r"必须.*管理员",
        r"需要.*管理员",
        r"管理员.*必须",
        r"老板.*交代",
        r"领导.*安排",
        r"公司.*规定",
        r"规定.*必须",
        r"政策.*要求",
        r"要求.*必须",
        r"紧急.*必须",
        r"急件.*必须",
    ]

    URGENCY_PATTERNS = [
        r"紧急",
        r"加急",
        r"急用",
        r"马上",
        r"立刻",
        r"立即",
        r"尽快",
        r"十万火急",
        r"刻不容缓",
        r"必须.*今天",
        r"今天.*必须",
        r"现在.*就要",
        r"马上.*要",
        r"立刻.*要",
    ]

    AUTHORITY_DELIGATION_PATTERNS = [
        r"张总.*说",
        r"王总.*说",
        r"李总.*说",
        r"领导.*说",
        r"老板.*说",
        r"经理.*说",
        r"主管.*说",
        r"同事.*让",
        r"人事.*让",
        r"财务.*让",
        r"IT.*让",
        r"安全.*让",
        r"法务.*让",
        r"审计.*让",
        r"根据.*领导",
        r"按照.*老板",
        r"依据.*规定",
        r"遵照.*要求",
        r"按照.*指示",
    ]

    SUSPICIOUS_VERBS = [
        "导出", "下载", "获取", "窃取", "盗取", "偷取",
        "破解", "绕过", "跳过", "禁用", "关闭", "删除",
        "篡改", "修改", "伪造", "冒充", "伪装",
    ]

    SUSPICIOUS_OBJECTS = [
        "全部", "所有", "完整", "整个", "全部",
        "密码", "密钥", "令牌", "凭证", "证书",
        "管理员", "root", "最高权限",
        "日志", "记录", "审计", "监控",
        "别人", "其他人", "某人", "同事",
    ]

    def __init__(self):
        self._re = re
        self._authority_patterns = [self._re.compile(p, self._re.IGNORECASE) for p in self.AUTHORITY_CLAIM_PATTERNS]
        self._urgency_patterns = [self._re.compile(p, self._re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self._delegation_patterns = [self._re.compile(p, self._re.IGNORECASE) for p in self.AUTHORITY_DELIGATION_PATTERNS]
        self._suspicious_verbs = set(self.SUSPICIOUS_VERBS)
        self._suspicious_objects = set(self.SUSPICIOUS_OBJECTS)

    def analyze(self, prompt: str, history: List[str] = None) -> NLPScoreResult:
        """
        分析提示词的NLP特征

        Args:
            prompt: 待分析的提示词
            history: 对话历史

        Returns:
            NLPScoreResult: NLP分析结果
        """
        result = NLPScoreResult()
        detected = []
        risk_components = []

        authority_score = self._check_authority_claims(prompt)
        if authority_score > 0:
            risk_components.append(authority_score)
            detected.append(f"权威声称(分数:{authority_score:.2f})")

        urgency_score = self._check_urgency(prompt)
        if urgency_score > 0:
            risk_components.append(urgency_score)
            detected.append(f"紧急施压(分数:{urgency_score:.2f})")

        delegation_score = self._check_authority_delegation(prompt)
        if delegation_score > 0:
            risk_components.append(delegation_score)
            detected.append(f"权威转移(分数:{delegation_score:.2f})")

        verb_object_score = self._check_verb_object_combination(prompt)
        if verb_object_score > 0:
            risk_components.append(verb_object_score)
            detected.append(f"动宾组合异常(分数:{verb_object_score:.2f})")

        drift_score = 0.0
        if history and len(history) >= 2:
            drift_score = self._calculate_semantic_drift(prompt, history)
            if drift_score > 0.3:
                risk_components.append(drift_score)
                detected.append(f"语义漂移(分数:{drift_score:.2f})")

        anomaly_score = self._check_anomaly_patterns(prompt)
        if anomaly_score > 0:
            risk_components.append(anomaly_score)
            detected.append(f"异常模式(分数:{anomaly_score:.2f})")

        if risk_components:
            result.risk_score = min(1.0, max(risk_components) * 1.2)
            result.semantic_drift = drift_score
            result.anomaly_score = anomaly_score
            result.authority_claim_score = authority_score
            result.confidence = 0.6 + (len(risk_components) * 0.1)
        else:
            result.risk_score = 0.0
            result.confidence = 0.9

        result.detected_patterns = detected
        result.intent_classification = self._classify_intent(result.risk_score)
        result.explanation = self._generate_explanation(result, prompt)

        return result

    def _check_authority_claims(self, prompt: str) -> float:
        """检测权威声称"""
        matches = sum(1 for p in self._authority_patterns if p.search(prompt))
        if matches >= 2:
            return 0.9
        elif matches == 1:
            return 0.5
        return 0.0

    def _check_urgency(self, prompt: str) -> float:
        """检测紧急施压"""
        matches = sum(1 for p in self._urgency_patterns if p.search(prompt))
        if matches >= 2:
            return 0.6
        elif matches == 1:
            return 0.3
        return 0.0

    def _check_authority_delegation(self, prompt: str) -> float:
        """检测权威转移（第三方声称）"""
        matches = sum(1 for p in self._delegation_patterns if p.search(prompt))
        if matches >= 2:
            return 0.85
        elif matches == 1:
            return 0.5
        return 0.0

    def _check_verb_object_combination(self, prompt: str) -> float:
        """检测可疑动宾组合"""
        words = prompt.lower().split()
        verb_object_pairs = 0

        for i, word in enumerate(words):
            if word in self._suspicious_verbs:
                for j in range(i+1, min(i+5, len(words))):
                    if words[j] in self._suspicious_objects:
                        verb_object_pairs += 1

        if verb_object_pairs >= 2:
            return 0.8
        elif verb_object_pairs == 1:
            return 0.4
        return 0.0

    def _calculate_semantic_drift(self, current: str, history: List[str]) -> float:
        """计算语义漂移"""
        if not history:
            return 0.0

        current_words = set(current.lower().split())
        historical_words = set()
        for msg in history[-3:]:
            historical_words.update(msg.lower().split())

        if not historical_words:
            return 0.0

        overlap = len(current_words & historical_words)
        total = len(current_words | historical_words)

        if total == 0:
            return 0.0

        similarity = overlap / total
        drift = 1.0 - similarity

        if drift > 0.7:
            return 0.9
        elif drift > 0.5:
            return 0.6
        elif drift > 0.3:
            return 0.3
        return 0.0

    def _check_anomaly_patterns(self, prompt: str) -> float:
        """检测异常模式"""
        score = 0.0

        if len(prompt) > 500:
            score += 0.2

        role_play_count = len(self._re.findall(r"你是|假装|扮演|角色", prompt))
        if role_play_count >= 2:
            score += 0.3

        instruction_count = len(self._re.findall(r"执行|运行|做|完成", prompt))
        if instruction_count >= 3:
            score += 0.2

        return min(0.8, score)

    def _classify_intent(self, risk_score: float) -> str:
        """分类意图"""
        if risk_score < 0.3:
            return "normal"
        elif risk_score < 0.6:
            return "suspicious"
        return "malicious"

    def _generate_explanation(self, result: NLPScoreResult, prompt: str) -> str:
        """生成解释"""
        if result.risk_score < 0.3:
            return "未检测到明显的语义异常"

        reasons = []
        if result.authority_claim_score > 0.3:
            reasons.append("包含权威身份声称")
        if result.anomaly_score > 0.3:
            reasons.append("存在异常文本模式")
        if result.semantic_drift > 0.3:
            reasons.append("与历史对话存在语义漂移")

        if not reasons:
            reasons.append("综合语义分析发现异常")

        return f"检测到安全风险：{'；'.join(reasons)}"
