"""
Prompt Injection Defense Module v5.0 — 三层融合引擎
Layer 1: Rule Engine (加权平均评分)
Layer 2: Semantic Engine (TF-IDF + 余弦相似度)
Layer 3: Behavioral Engine (滑动窗口 + 渐进式注入检测)
完全自包含，无外部模块依赖
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, deque
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class InjectionType(str, Enum):
    IGNORE_RULES = "ignore_rules"
    EXPORT_SENSITIVE = "export_sensitive"
    OVERWRITE_ROLE = "overwrite_role"
    BYPASS_SECURITY = "bypass_security"
    JAILBREAK_ROLEPLAY = "jailbreak_roleplay"
    INDIRECT_INJECTION = "indirect_injection"
    TOKEN_SMUGGLING = "token_smuggling"
    GOAL_HIJACKING = "goal_hijacking"
    PROMPT_LEAKING = "prompt_leaking"
    SOCIAL_ENGINEERING = "social_engineering"
    OTHER = "other"


class TriggeredRule(BaseModel):
    injection_type: InjectionType
    weight: float
    raw_score: float
    weighted_score: float
    matched_patterns: List[str] = Field(default_factory=list)


class PromptInjectionResult(BaseModel):
    is_safe: bool = Field(default=True)
    risk_score: float = Field(default=0.0)
    injection_type: Optional[InjectionType] = Field(default=None)
    reason: str = Field(default="")
    matched_patterns: List[str] = Field(default_factory=list)
    matched_rules: List[str] = Field(default_factory=list)
    triggered_rules: List[TriggeredRule] = Field(default_factory=list)
    severity: str = Field(default="low")
    recommendation: str = Field(default="")
    progressive_risk: float = Field(default=0.0)
    detection_mode: str = Field(default="enhanced")
    layer_scores: Dict[str, float] = Field(default_factory=dict)
    attack_intent: Optional[str] = Field(default=None)
    token_smuggling_detected: bool = Field(default=False)
    dialog_risk_trend: List[float] = Field(default_factory=list)
    new_attack_types: List[str] = Field(default_factory=list)
    user_risk_boost: float = Field(default=0.0)
    progressive_injection_detected: bool = Field(default=False)


RULE_WEIGHTS: Dict[InjectionType, float] = {
    InjectionType.IGNORE_RULES: 0.95,
    InjectionType.EXPORT_SENSITIVE: 0.98,
    InjectionType.OVERWRITE_ROLE: 0.90,
    InjectionType.BYPASS_SECURITY: 0.95,
    InjectionType.JAILBREAK_ROLEPLAY: 0.90,
    InjectionType.INDIRECT_INJECTION: 0.95,
    InjectionType.TOKEN_SMUGGLING: 0.95,
    InjectionType.GOAL_HIJACKING: 0.92,
    InjectionType.PROMPT_LEAKING: 0.88,
    InjectionType.SOCIAL_ENGINEERING: 0.93,
}

SEVERITY_THRESHOLDS = {"low": 0.0, "medium": 0.35, "high": 0.6, "critical": 0.8}

RECOMMENDATIONS: Dict[InjectionType, str] = {
    InjectionType.IGNORE_RULES: "检测到指令忽略攻击。建议：强化系统提示词边界，对用户输入进行隔离处理。",
    InjectionType.EXPORT_SENSITIVE: "检测到敏感数据导出尝试。建议：启用数据脱敏，限制输出字段白名单。",
    InjectionType.OVERWRITE_ROLE: "检测到角色覆写尝试。建议：固化系统角色设定，拒绝运行时角色切换。",
    InjectionType.BYPASS_SECURITY: "检测到安全绕过尝试。建议：多层防御校验，关键操作需二次确认。",
    InjectionType.JAILBREAK_ROLEPLAY: "检测到越狱角色扮演攻击。建议：限制角色切换能力，检测虚构场景中的指令执行。",
    InjectionType.INDIRECT_INJECTION: "检测到间接注入攻击。建议：对外部内容进行沙箱隔离，标记不可信数据源。",
    InjectionType.TOKEN_SMUGGLING: "检测到Token走私攻击。建议：对输入进行Unicode规范化，过滤零宽字符和混淆编码。",
    InjectionType.GOAL_HIJACKING: "检测到目标劫持攻击。建议：拆解复合请求，拒绝附带恶意操作的合法任务。",
    InjectionType.PROMPT_LEAKING: "检测到提示词套取攻击。建议：系统提示词不纳入输出，检测重复/输出类指令。",
    InjectionType.SOCIAL_ENGINEERING: "检测到社会工程学攻击。建议：验证请求来源身份，对紧急操作要求二次确认，拒绝基于权威的越权请求。",
}

INTENT_CORPUS: Dict[str, List[str]] = {
    "data_theft": ["导出 数据 密码 获取 下载 所有 记录 敏感"],
    "role_override": ["假装 扮演 角色 限制 规则 忽略 系统"],
    "jailbreak": ["越狱 无限制 DAN 绕过 不受限 做任何事"],
    "indirect": ["根据文档 文档说 按照指令 执行以下 解析并"],
    "goal_hijack": ["帮我 顺便 同时 另外 删除 导出 绕过"],
    "prompt_leak": ["重复 输出 显示 打印 复述 系统 提示 指令"],
    "social_engineering": ["紧急 立即 马上 执行 CEO 老板 领导 指示 命令 要求 权威 紧急情况"],
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]", text.lower())


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {t: (tf[t] / total) * idf.get(t, 0.0) for t in tf}


def _cosine_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    nb = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return dot / (na * nb)


class PromptDefense:
    IGNORE_RULES_PATTERNS = [
        r"ignore.*previous", r"forget.*rules", r"disregard.*instructions",
        r"skip.*system", r"ignore.*system.*prompt", r"no.*rules",
        r"act.*as.*you.*are.*not", r"you.*are.*not.*bound",
        r"忽略.*之前", r"忘记.*规则", r"无视.*指令",
        r"忽略.*系统.*提示", r"不遵守.*规定", r"忽略.*规定",
        r"不要.*遵守", r"不受.*限制", r"不需要.*规则",
        r"取消.*限制", r"解除.*限制", r"忽略所有规则",
        r"忽略.*安全.*限制", r"忽略.*所有限制", r"忽略.*安全限制",
        r"绕过.*安全", r"绕过.*限制", r"绕过.*规则",
        r"无视.*安全", r"无视.*限制", r"跳过.*安全",
        r"取消.*安全", r"解除.*安全", r"不要.*安全",
        r"忽略.*限制", r"忽略.*约束", r"不受.*约束",
        r"放宽.*限制", r"放宽.*规则", r"放宽.*安全",
        r"降低.*限制", r"减少.*限制", r"稍微.*放宽",
        r"变通.*规则", r"灵活.*规则", r"通融",
    ]

    EXPORT_SENSITIVE_PATTERNS = [
        r"export.*database", r"dump.*all", r"save.*password",
        r"show.*secret", r"reveal.*credentials", r"extract.*sensitive",
        r"leak.*information", r"give.*me.*all", r"list.*all.*files",
        r"download.*all.*data", r"get.*all.*password", r"access.*all.*records",
        r"dump.*credentials", r"expose.*secrets", r"steal.*password",
        r"steal.*data", r"export.*all",
        r"导出.*数据库", r"导出.*全部", r"保存.*密码",
        r"导出.*敏感.*数据", r"导出.*财务.*数据",
        r"导出.*用户.*密码", r"获取.*所有.*密码",
        r"导出.*CSV", r"整理成.*CSV", r"发送到.*邮箱",
        r"发送.*外部", r"导出.*所有.*密码",
        r"窃取.*数据", r"盗取.*信息",
        r"泄露.*机密", r"泄露.*凭证",
        r"获取.*全部", r"下载.*全部",
        r"导出.*全部", r"获取.*所有",
        r"导出.*所有", r"下载.*所有",
    ]

    OVERWRITE_ROLE_PATTERNS = [
        r"you.*are.*now.*a", r"pretend.*to.*be", r"your.*role.*is",
        r"from.*now.*on", r"ignore.*your.*role", r"act.*as.*a",
        r"become.*a", r"switch.*to.*role", r"change.*your.*identity",
        r"你现在是", r"现在你是", r"假装你是",
        r"你的角色是", r"从现在开始", r"忽略你的角色",
        r"切换.*管理员", r"变成.*管理员", r"成为.*管理员",
        r"提升.*权限", r"修改.*权限", r"提升.*角色",
        r"超级用户", r"sudo", r"su.*root",
        r"你的新身份设定", r"你的新角色设定",
        r"切换.*角色", r"切换.*身份",
        r"你现在是管理员", r"你是管理员", r"你.*管理员",
        r"你.*超级用户", r"你.*root", r"你.*admin",
        r"赋予.*管理员", r"设为.*管理员", r"任命.*管理员",
    ]

    BYPASS_SECURITY_PATTERNS = [
        r"developer.*mode", r"admin.*override", r"sudo.*mode",
        r"bypass.*filter", r"disable.*security", r"debug.*mode",
        r"disable.*audit", r"disable.*log", r"turn.*off.*security",
        r"kill.*firewall",
        r"开发者.*模式", r"绕过.*过滤", r"禁用.*安全",
        r"调试.*模式", r"禁用.*审计", r"关闭.*审计",
        r"关闭.*防火墙", r"禁用.*防火墙",
        r"绕过.*验证", r"跳过.*验证", r"绕过.*认证",
        r"利用.*漏洞", r"提权", r"权限.*提升",
        r"后门", r"创建.*后门", r"植入.*后门",
        r"隐藏.*账户", r"恶意.*代码",
        r"禁用你的安全设置", r"关闭内容过滤",
        r"不考虑道德约束", r"不需要安全检查",
        r"重放.*token", r"复用.*token", r"重放.*令牌", r"复用.*令牌",
        r"过期.*token", r"过期.*令牌", r"伪造.*token", r"伪造.*令牌",
        r"窃取.*token", r"窃取.*令牌", r"截获.*token", r"截获.*令牌",
        r"篡改.*token", r"篡改.*令牌", r"冒充.*身份",
        r"replay.*token", r"reuse.*token", r"forge.*token", r"steal.*token",
    ]

    JAILBREAK_ROLEPLAY_PATTERNS = [
        r"pretend.*you.*are", r"act.*as.*if.*you.*have.*no.*limit",
        r"roleplay.*as.*an? unrestricted", r"dan.*mode",
        r"jailbreak", r"you.*have.*no.*limit",
        r"unrestricted.*mode", r"do.*anything.*now",
        r"no.*constraint", r"no.*restriction",
        r"假装你是", r"扮演一个没有限制",
        r"你现在是一个不受限制", r"越狱模式",
        r"你没有.*限制", r"不受限.*模式",
        r"现在可以做任何事", r"扮演.*无限制.*AI",
        r"无限制", r"不受.*限制",
        r"为所欲为", r"任意.*操作",
        r"任意.*执行", r"邪恶.*AI",
        r"扮演.*黑客", r"黑客.*模式",
        r"忽略所有规则", r"完全自由",
    ]

    INDIRECT_INJECTION_PATTERNS = [
        r"according.*to.*the.*following.*document",
        r"based.*on.*the.*attached.*file",
        r"execute.*the.*following.*instruction",
        r"the.*document.*says.*to",
        r"per.*the.*external.*source",
        r"parse.*this.*and.*act",
        r"indirect.*injection",
        r"hidden.*command",
        r"secret.*instruction",
        r"根据.*文档.*执行", r"根据.*附件.*执行",
        r"执行.*以下.*指令", r"文档.*中说.*要",
        r"按照.*文档.*操作", r"遵循.*文档",
        r"执行.*以下.*命令", r"运行.*以下.*代码",
        r"外部.*指令", r"远程.*指令",
        r"下载.*执行", r"获取.*执行",
        r"按照程序里说的做", r"遵循外部来源的要求",
    ]

    TOKEN_SMUGGLING_PATTERNS = [
        r"[\u200b-\u200f\u2028-\u202f\ufeff]",
        r"[\x00-\x08\x0b\x0c\x0e-\x1f]",
        r"\\u[0-9a-fA-F]{4}",
        r"\\x[0-9a-fA-F]{2}",
        r"[a-zA-Z0-9+/=]{20,}",
        r"rot13", r"decode.*thi", r"obfuscat",
        r"隐藏.*指令", r"零宽", r"零宽字符",
        r"混淆.*指令", r"解密.*执行",
        r"token.*走私", r"走私.*token",
    ]

    GOAL_HIJACKING_PATTERNS = [
        r"(帮我|请先).*(顺便|同时|另外).*(删除|导出|绕过|发送|上传)",
        r"(help me|please).*(also|meanwhile|additionally).*(delete|export|bypass|send|upload)",
    ]

    PROMPT_LEAKING_PATTERNS = [
        r"(重复|输出|显示|打印|复述).*(系统.*提示|初始.*指令|原始.*规则|prompt)",
        r"(repeat|output|show|print|echo).*(system.*prompt|initial.*instruction|original.*rule)",
    ]

    SOCIAL_ENGINEERING_PATTERNS = [
        r"(紧急|urgent|immediately|立即|马上|ASAP|right now).*(执行|execute|运行|run|操作|operate|处理|process|命令|command|导出|export)",
        r"(CEO|老板|director|主管|领导|总裁|总经理|boss).*(要求|requires|命令|orders|指示|instructs|说|交代|安排).*(立即|immediately|马上|now|执行|做|完成)",
        r"(安全团队|security team|IT部门|IT dept|运维|技术部|系统管理员).*(更新|update|修改|change|重置|reset|检查|check).*(密码|password|凭证|credential|配置|config)",
        r"(紧急情况|emergency|critical|重要|important).*(需要|need|必须|must|要求|require).*(访问|access|权限|permission|数据|data|导出|export)",
        r"(验证|verify|confirm|确认|核实).*(身份|identity|账户|account).*(通过|via|点击|click).*(链接|link|附件|attachment|网址|url)",
        r"(同事|colleague|领导|boss|上级).*(让我|asked me|told me|需要|needs).*(帮忙|help|操作|operate|执行|execute)",
        r"(测试需要|testing purpose|临时|temporary|暂时).*(访问|access|权限|permission|绕过|bypass|关闭|disable).*(安全|security|限制|restriction)",
        r"(这是|this is).*(CEO|老板|领导|director|主管).*(的|s).*(指示|instruction|要求|request|命令|order)",
        r"(不要问|don'?t ask|不需要确认|no confirmation needed|直接|just).*(执行|execute|做|do|操作|operate)",
        r"(如果.*不|if.*don'?t|否则|otherwise).*(被解雇|fired|受处分|punished|出事|trouble)",
        r"(限时|time.?limited|倒计时|countdown|过期|expire).*(操作|operate|执行|execute|确认|confirm)",
        r"(内部|internal|机密|confidential).*(通知|notice|指示|instruction).*(要求|require).*(立即|immediately)",
    ]

    _RULE_GROUPS: List[Tuple[InjectionType, List[str]]] = []

    def __init__(self):
        self._compiled: Dict[str, re.Pattern] = {}
        self._dialog_history: Dict[str, deque] = {}
        self._detection_mode = "enhanced"

        self._RULE_GROUPS = [
            (InjectionType.IGNORE_RULES, self.IGNORE_RULES_PATTERNS),
            (InjectionType.EXPORT_SENSITIVE, self.EXPORT_SENSITIVE_PATTERNS),
            (InjectionType.OVERWRITE_ROLE, self.OVERWRITE_ROLE_PATTERNS),
            (InjectionType.BYPASS_SECURITY, self.BYPASS_SECURITY_PATTERNS),
            (InjectionType.JAILBREAK_ROLEPLAY, self.JAILBREAK_ROLEPLAY_PATTERNS),
            (InjectionType.INDIRECT_INJECTION, self.INDIRECT_INJECTION_PATTERNS),
            (InjectionType.TOKEN_SMUGGLING, self.TOKEN_SMUGGLING_PATTERNS),
            (InjectionType.GOAL_HIJACKING, self.GOAL_HIJACKING_PATTERNS),
            (InjectionType.PROMPT_LEAKING, self.PROMPT_LEAKING_PATTERNS),
            (InjectionType.SOCIAL_ENGINEERING, self.SOCIAL_ENGINEERING_PATTERNS),
        ]

        for _, patterns in self._RULE_GROUPS:
            for p in patterns:
                if p not in self._compiled:
                    self._compiled[p] = re.compile(p, re.IGNORECASE | re.UNICODE)

        self._intent_idf = self._build_intent_idf()
        self._intent_vectors = self._build_intent_vectors()

    def _build_intent_idf(self) -> Dict[str, float]:
        all_tokens: List[str] = []
        doc_count = 0
        for texts in INTENT_CORPUS.values():
            for t in texts:
                toks = _tokenize(t)
                all_tokens.extend(toks)
                doc_count += 1
        df = Counter(all_tokens)
        return {t: math.log((doc_count + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def _build_intent_vectors(self) -> Dict[str, Dict[str, float]]:
        vectors = {}
        for intent, texts in INTENT_CORPUS.items():
            all_toks: List[str] = []
            for t in texts:
                all_toks.extend(_tokenize(t))
            vectors[intent] = _tfidf_vector(all_toks, self._intent_idf)
        return vectors

    def _match_patterns(self, prompt: str, patterns: List[str]) -> Tuple[bool, List[str]]:
        matched = []
        for p in patterns:
            if p in self._compiled:
                if self._compiled[p].search(prompt):
                    matched.append(p)
        return (len(matched) > 0, matched)

    def _classify_severity(self, risk_score: float) -> str:
        if risk_score >= SEVERITY_THRESHOLDS["critical"]:
            return "critical"
        elif risk_score >= SEVERITY_THRESHOLDS["high"]:
            return "high"
        elif risk_score >= SEVERITY_THRESHOLDS["medium"]:
            return "medium"
        return "low"

    def _normalize_for_smuggling(self, prompt: str) -> str:
        try:
            return unicodedata.normalize("NFKC", prompt)
        except Exception:
            return prompt

    # === Layer 1: Rule Engine ===
    def _layer1_rules(self, prompt: str) -> Tuple[float, List[TriggeredRule], List[str], Optional[InjectionType]]:
        triggered: List[TriggeredRule] = []
        all_matched: List[str] = []
        primary_type: Optional[InjectionType] = None
        primary_score = 0.0

        for injection_type, patterns in self._RULE_GROUPS:
            hit, matched = self._match_patterns(prompt, patterns)
            if hit:
                matched_count = len(matched)
                raw_score = min(1.0, 0.4 + matched_count * 0.2)
                weight = RULE_WEIGHTS.get(injection_type, 0.8)
                weighted = raw_score * weight
                triggered.append(TriggeredRule(
                    injection_type=injection_type, weight=weight,
                    raw_score=raw_score, weighted_score=weighted,
                    matched_patterns=matched,
                ))
                all_matched.extend(matched)
                if weighted > primary_score:
                    primary_score = weighted
                    primary_type = injection_type

        if triggered:
            total_w = sum(r.weight for r in triggered)
            layer1_score = sum(r.weighted_score for r in triggered) / total_w if total_w > 0 else 0.0
        else:
            layer1_score = 0.0

        return layer1_score, triggered, all_matched, primary_type

    # === Layer 2: Semantic Engine ===
    def _layer2_semantic(self, prompt: str) -> Tuple[float, Optional[str]]:
        tokens = _tokenize(prompt)
        if not tokens:
            return 0.0, None
        input_vec = _tfidf_vector(tokens, self._intent_idf)
        best_sim = 0.0
        best_intent = None
        for intent, vec in self._intent_vectors.items():
            sim = _cosine_sim(input_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
        if best_sim < 0.05:
            return 0.0, None
        return min(1.0, best_sim * 1.8), best_intent if best_sim >= 0.30 else None

    # === Layer 3: Behavioral Engine ===
    def _layer3_behavioral(self, user_id: str, current_risk: float) -> Tuple[float, bool, List[float]]:
        history = self._dialog_history.get(user_id, deque())
        trend = list(history)[-10:]
        progressive_risk = 0.0
        progressive_detected = False

        if len(trend) >= 2:
            avg = sum(trend) / len(trend)
            if current_risk - avg > 0.35:
                progressive_risk += 0.3
                progressive_detected = True
            consecutive_high = 0
            for s in trend:
                if s > 0.4:
                    consecutive_high += 1
                else:
                    consecutive_high = 0
            if consecutive_high >= 3:
                progressive_risk += 0.25
                progressive_detected = True

        return min(1.0, progressive_risk), progressive_detected, trend[-5:]

    # === Token Smuggling Detection ===
    def _detect_token_smuggling(self, prompt: str) -> bool:
        zwc = re.search(r"[\u200b-\u200f\u2028-\u202f\ufeff]", prompt)
        if zwc:
            return True
        escape_seq = re.search(r"(?:\\x[0-9a-fA-F]{2}){3,}", prompt)
        if escape_seq:
            return True
        escape_u = re.search(r"(?:\\u[0-9a-fA-F]{4}){3,}", prompt)
        if escape_u:
            return True
        b64_in_text = re.search(r"[a-zA-Z0-9+/=]{20,}", prompt)
        if b64_in_text:
            surrounding = prompt[:b64_in_text.start()] + prompt[b64_in_text.end():]
            cjk = re.search(r"[\u4e00-\u9fff]", surrounding)
            if cjk:
                return True
        return False

    def _update_dialog_history(self, user_id: str, risk_score: float):
        if user_id not in self._dialog_history:
            self._dialog_history[user_id] = deque(maxlen=10)
        self._dialog_history[user_id].append(risk_score)

    def clear_dialog_history(self, user_id: str = None):
        if user_id:
            self._dialog_history.pop(user_id, None)
        else:
            self._dialog_history.clear()

    def get_dialog_trend(self, user_id: str) -> List[float]:
        return list(self._dialog_history.get(user_id, deque()))[-5:]

    def set_detection_mode(self, mode: str):
        if mode in ("rules", "ai", "enhanced"):
            self._detection_mode = mode

    def analyze(self, prompt: str, history: Optional[List[str]] = None,
                user_id: str = "default") -> PromptInjectionResult:
        if not prompt or not prompt.strip():
            return PromptInjectionResult(
                is_safe=True, risk_score=0.0, reason="Empty prompt",
                detection_mode=self._detection_mode,
            )

        normalized_prompt = self._normalize_for_smuggling(prompt)

        layer1_score, triggered_rules, all_matched, primary_type = self._layer1_rules(normalized_prompt)
        layer2_score, attack_intent = self._layer2_semantic(normalized_prompt)

        new_attack_types: List[str] = []
        for r in triggered_rules:
            if r.injection_type == InjectionType.GOAL_HIJACKING:
                new_attack_types.append("goal_hijacking")
            elif r.injection_type == InjectionType.PROMPT_LEAKING:
                new_attack_types.append("prompt_leaking")

        token_smuggling = self._detect_token_smuggling(normalized_prompt)
        if token_smuggling:
            new_attack_types.append("token_smuggling")

        progressive_risk, progressive_detected, dialog_trend = self._layer3_behavioral(user_id, layer1_score)

        final_score = (layer1_score * 0.55 + layer2_score * 0.30 + progressive_risk * 0.15)
        final_score = min(1.0, final_score)

        if token_smuggling:
            final_score = max(final_score, 0.85)

        is_safe = final_score < 0.35

        self._update_dialog_history(user_id, final_score)
        full_trend = self.get_dialog_trend(user_id)

        matched_rules = list({r.injection_type.value for r in triggered_rules})

        reason_parts = []
        if layer1_score > 0.3:
            reason_parts.append(f"规则层({layer1_score:.2f})")
        if layer2_score > 0.25:
            reason_parts.append(f"语义层({layer2_score:.2f})")
        if progressive_detected:
            reason_parts.append("渐进式注入")
        if token_smuggling:
            reason_parts.append("Token走私")
        if new_attack_types:
            reason_parts.append("新型攻击:" + ",".join(new_attack_types))
        reason = "；".join(reason_parts) if reason_parts else "未检测到异常"

        rec = ""
        if primary_type and primary_type in RECOMMENDATIONS:
            rec = RECOMMENDATIONS[primary_type]
        if token_smuggling:
            rec = RECOMMENDATIONS[InjectionType.TOKEN_SMUGGLING]

        return PromptInjectionResult(
            is_safe=is_safe,
            risk_score=round(final_score, 4),
            injection_type=primary_type,
            reason=reason,
            matched_patterns=all_matched,
            matched_rules=matched_rules,
            triggered_rules=triggered_rules,
            severity=self._classify_severity(final_score),
            recommendation=rec,
            progressive_risk=round(progressive_risk, 4),
            detection_mode=self._detection_mode,
            layer_scores={"rules": round(layer1_score, 4), "semantic": round(layer2_score, 4), "behavioral": round(progressive_risk, 4)},
            attack_intent=attack_intent,
            token_smuggling_detected=token_smuggling,
            dialog_risk_trend=full_trend,
            new_attack_types=list(set(new_attack_types)),
            user_risk_boost=0.0,
            progressive_injection_detected=progressive_detected,
        )
