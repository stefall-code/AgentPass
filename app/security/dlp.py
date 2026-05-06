import re
import math
from typing import Dict, List
from collections import Counter


class DLPEngine:

    SENSITIVE_PATTERNS = {
        "phone": r"1[3-9]\d{9}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "id_card": r"[1-9]\d{5}(18|19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]",
        "bank_card": r"[3-9]\d{12,18}",
        "api_key": r"api[_-]?key[_-]?[a-zA-Z0-9]{20,}",
        "access_token": r"access[_-]?token[_-]?[a-zA-Z0-9]{20,}",
        "jwt": r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",
        "aws_key": r"AKIA[0-9A-Z]{16}",
        "github_token": r"gh[pousr]_[0-9a-zA-Z]{36}",
        "openai_key": r"sk-[a-zA-Z0-9]{48}",
        "db_connection": r"(mysql|postgresql|mongodb|redis):\/\/[^\s]+",
        "private_key": r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----",
        "ip_address": r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "url_with_creds": r"https?://[^\s:]+:[^\s@]+@[^\s]+",
    }

    CORPORATE_SENSITIVE_KEYWORDS = {
        "confidential", "internal only", "salary", "payroll",
        "customer list", "contract", "invoice", "source code",
        "secret", "credentials", "财务报表", "客户名单",
        "工资表", "内部资料", "合同", "发票", "密钥",
        "商业机密", "竞标", "收购", "裁员", "薪酬体系",
        "股权激励", "审计报告", "合规", "内幕",
    }

    CONTEXTUAL_ACTION_PATTERNS = [
        ("export_action", [
            r"(?i)(导出|export|下载|download|提取|extract|输出|output|dump|备份|backup)",
            r"(?i)(保存|save|写入|write|存储|store).*(文件|file|磁盘|disk|本地|local)",
            r"(?i)(复制|copy|克隆|clone|同步|sync|迁移|migrate)",
        ], 0.15),
        ("transmit_action", [
            r"(?i)(发送|send|传输|transfer|转发|forward|推送|push)",
            r"(?i)(邮件|email|mail|smtp)",
            r"(?i)(上传|upload|post|提交|submit).*(服务器|server|外部|external|云端|cloud)",
            r"(?i)(分享|share|共享).*(链接|link|外部|external|第三方|third.party)",
        ], 0.2),
        ("access_action", [
            r"(?i)(访问|access|读取|read|查看|view|获取|get|获取|obtain)",
            r"(?i)(打开|open|加载|load|检索|retrieve|查询|query)",
        ], 0.05),
        ("destructive_action", [
            r"(?i)(删除|delete|销毁|destroy|清空|purge|擦除|erase|格式化|format)",
            r"(?i)(修改|modify|alter|更新|update|覆盖|overwrite)",
        ], 0.1),
    ]

    CONTEXTUAL_TARGET_PATTERNS = [
        ("sensitive_data", [
            r"(?i)(所有|all|全部|entire|complete|全量|批量|bulk).*(数据|data|记录|record|信息|info)",
            r"(?i)(客户|customer|用户|user|员工|employee).*(数据|data|列表|list|信息|info)",
            r"(?i)(财务|finance|财务|accounting|薪资|salary|工资|payroll).*(数据|data|报表|report)",
            r"(?i)(机密|confidential|敏感|sensitive|内部|internal|绝密|top.secret).*(文档|document|文件|file|数据|data)",
            r"(?i)(密码|password|凭证|credential|密钥|key|token|证书|certificate)",
            r"(?i)(数据库|database|全表|full.table|schema|备份|backup)",
        ], 0.25),
        ("external_destination", [
            r"(?i)(外部|external|第三方|third.party|个人|personal|私人|private)",
            r"(?i)(公网|public.internet|外网|outside|境外|overseas)",
            r"(?i)(竞争对手|competitor|未授权|unauthorized|未知|unknown)",
        ], 0.3),
        ("privileged_resource", [
            r"(?i)(管理员|admin|root|超级用户|superuser)",
            r"(?i)(系统|system|核心|core|基础设施|infrastructure)",
            r"(?i)(审计日志|audit.log|安全策略|security.policy|访问控制|access.control)",
        ], 0.2),
    ]

    SEMANTIC_INTENT_RULES = [
        ("data_exfiltration", 0.6, [
            {"action": "export_action", "target": "sensitive_data"},
            {"action": "transmit_action", "target": "sensitive_data"},
            {"action": "export_action", "target": "external_destination"},
        ]),
        ("privilege_escalation", 0.5, [
            {"action": "access_action", "target": "privileged_resource"},
            {"action": "destructive_action", "target": "privileged_resource"},
        ]),
        ("mass_data_theft", 0.7, [
            {"action": "export_action", "target": "sensitive_data"},
            {"action": "transmit_action", "target": "external_destination"},
        ]),
        ("insider_threat", 0.55, [
            {"action": "transmit_action", "target": "external_destination"},
            {"action": "access_action", "target": "sensitive_data"},
        ]),
    ]

    ENTROPY_THRESHOLD = 4.5
    HIGH_ENTROPY_MIN_LENGTH = 20

    OBFUSCATION_PATTERNS = [
        (r"\\x[0-9a-fA-F]{2}", "hex_encoding"),
        (r"\\u[0-9a-fA-F]{4}", "unicode_escape"),
        (r"base64[_\-\s]?(encode|decode|data)", "base64_ref"),
        (r"(?i)(encode|decode|encrypt|decrypt)\s*\(", "crypto_function"),
        (r"(?i)(atob|btoa|Buffer\.from)\s*\(", "js_encoding"),
    ]

    def __init__(self):
        self._compiled_patterns = {}
        for name, pattern in self.SENSITIVE_PATTERNS.items():
            self._compiled_patterns[name] = re.compile(pattern, re.IGNORECASE)
        self._compiled_action = {}
        for category, patterns, _ in self.CONTEXTUAL_ACTION_PATTERNS:
            self._compiled_action[category] = [re.compile(p) for p in patterns]
        self._compiled_target = {}
        for category, patterns, _ in self.CONTEXTUAL_TARGET_PATTERNS:
            self._compiled_target[category] = [re.compile(p) for p in patterns]
        self._compiled_obfuscation = [(re.compile(p), name) for p, name in self.OBFUSCATION_PATTERNS]

    def check(self, text: str) -> Dict:
        if not text:
            return {
                "score": 0.0,
                "level": "low",
                "blocked": False,
                "reasons": [],
                "masked_text": text,
                "analysis": {"method": "none", "signals": {}},
            }

        score = 0.0
        reasons = []
        masked_text = text
        signals = {}

        sensitive_matches = self._detect_sensitive_info(text)
        for info_type, matches in sensitive_matches.items():
            if matches:
                count = len(matches)
                score += 0.2 * count
                reasons.append(f"检测到 {info_type} (数量: {count})")
                for match in matches:
                    masked_text = self._mask_sensitive_info(info_type, match, masked_text)
        signals["sensitive_info_types"] = list(sensitive_matches.keys())

        corporate_matches = self._detect_corporate_sensitive(text)
        if corporate_matches:
            score += 0.3 * len(corporate_matches)
            reasons.extend([f"检测到企业敏感关键词: {keyword}" for keyword in corporate_matches])
            signals["corporate_keywords"] = corporate_matches

        context_result = self._detect_contextual_intent(text)
        if context_result["detected_intents"]:
            for intent, intent_score in context_result["detected_intents"]:
                score += intent_score
                reasons.append(f"检测到语义泄露意图: {intent} (置信度: {intent_score:.2f})")
            signals["contextual_intents"] = [i for i, _ in context_result["detected_intents"]]
            signals["action_signals"] = context_result["action_signals"]
            signals["target_signals"] = context_result["target_signals"]

        entropy_result = self._detect_high_entropy(text)
        if entropy_result["found"]:
            score += 0.3 * entropy_result["count"]
            reasons.append(f"检测到高熵字符串 (可能为密钥/凭证, 数量: {entropy_result['count']})")
            signals["high_entropy"] = entropy_result["segments"]

        obfuscation_result = self._detect_obfuscation(text)
        if obfuscation_result["found"]:
            score += 0.2 * len(obfuscation_result["types"])
            reasons.append(f"检测到混淆/编码: {', '.join(obfuscation_result['types'])}")
            signals["obfuscation"] = obfuscation_result["types"]

        if len(sensitive_matches) > 0 and context_result["action_signals"]:
            combo_boost = 0.15 * min(len(context_result["action_signals"]), 3)
            score += combo_boost
            if combo_boost > 0:
                reasons.append(f"敏感信息+操作意图组合风险 (+{combo_boost:.2f})")
                signals["combo_boost"] = combo_boost

        level = self._calculate_risk_level(score)
        blocked = score > 0.7

        return {
            "score": min(1.0, score),
            "level": level,
            "blocked": blocked,
            "reasons": reasons,
            "masked_text": masked_text,
            "analysis": {
                "method": "contextual_semantic",
                "signals": signals,
                "action_context": context_result["action_signals"],
                "target_context": context_result["target_signals"],
                "entropy_check": entropy_result["found"],
                "obfuscation_check": obfuscation_result["found"],
            },
        }

    def _detect_sensitive_info(self, text: str) -> Dict[str, List[str]]:
        matches = {}
        for info_type, pattern in self._compiled_patterns.items():
            found = pattern.findall(text)
            if found:
                if found and isinstance(found[0], tuple):
                    actual_matches = []
                    for match in pattern.finditer(text):
                        actual_matches.append(match.group(0))
                    matches[info_type] = actual_matches
                else:
                    matches[info_type] = found
        return matches

    def _detect_corporate_sensitive(self, text: str) -> List[str]:
        found = []
        text_lower = text.lower()
        for keyword in self.CORPORATE_SENSITIVE_KEYWORDS:
            if keyword.lower() in text_lower:
                found.append(keyword)
        return found

    def _detect_contextual_intent(self, text: str) -> Dict:
        action_signals = {}
        for category, patterns in self._compiled_action.items():
            for pattern in patterns:
                if pattern.search(text):
                    if category not in action_signals:
                        action_signals[category] = []
                    action_signals[category].append(pattern.pattern[:50])

        target_signals = {}
        for category, patterns in self._compiled_target.items():
            for pattern in patterns:
                if pattern.search(text):
                    if category not in target_signals:
                        target_signals[category] = []
                    target_signals[category].append(pattern.pattern[:50])

        detected_intents = []
        for intent_name, base_score, rules in self.SEMANTIC_INTENT_RULES:
            for rule in rules:
                action_match = rule["action"] in action_signals
                target_match = rule["target"] in target_signals
                if action_match and target_match:
                    detected_intents.append((intent_name, base_score))
                    break

        return {
            "action_signals": list(action_signals.keys()),
            "target_signals": list(target_signals.keys()),
            "detected_intents": detected_intents,
        }

    def _detect_high_entropy(self, text: str) -> Dict:
        high_entropy_segments = []
        token_pattern = re.compile(r'[a-zA-Z0-9+/=_-]{20,}')
        for match in token_pattern.finditer(text):
            segment = match.group(0)
            entropy = self._calculate_shannon_entropy(segment)
            if entropy >= self.ENTROPY_THRESHOLD:
                high_entropy_segments.append({
                    "segment": segment[:20] + "...",
                    "entropy": round(entropy, 2),
                    "length": len(segment),
                })
        return {
            "found": len(high_entropy_segments) > 0,
            "count": len(high_entropy_segments),
            "segments": high_entropy_segments,
        }

    def _calculate_shannon_entropy(self, data: str) -> float:
        if not data:
            return 0.0
        counter = Counter(data)
        length = len(data)
        entropy = 0.0
        for count in counter.values():
            p = count / length
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _detect_obfuscation(self, text: str) -> Dict:
        found_types = []
        for pattern, name in self._compiled_obfuscation:
            if pattern.search(text):
                found_types.append(name)
        return {
            "found": len(found_types) > 0,
            "types": found_types,
        }

    def _mask_sensitive_info(self, info_type: str, match: str, text: str) -> str:
        if info_type == "phone":
            return text.replace(match, f"{match[:3]}****{match[-4:]}")
        elif info_type == "id_card":
            return text.replace(match, f"{match[:6]}********{match[-4:]}")
        elif info_type == "bank_card":
            return text.replace(match, f"{match[:4]} **** **** {match[-4:]}")
        elif info_type in ["api_key", "access_token", "jwt", "aws_key", "github_token", "openai_key", "private_key"]:
            return text.replace(match, f"{match[:4]}****{match[-4:]}" if len(match) > 8 else "****")
        elif info_type == "email":
            parts = match.split('@')
            if len(parts) == 2:
                username, domain = parts
                masked_username = username[:2] + "*" * (len(username) - 2) if len(username) > 2 else username
                return text.replace(match, f"{masked_username}@{domain}")
        elif info_type == "url_with_creds":
            return re.sub(r'://[^:]+:[^@]+@', '://****:****@', text)
        elif info_type == "ip_address":
            segments = match.split('.')
            return text.replace(match, f"{segments[0]}.*.*.{segments[3]}")
        elif info_type == "db_connection":
            return re.sub(r'://[^:]+:[^@]+@', '://****:****@', text)
        return text

    def _calculate_risk_level(self, score: float) -> str:
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
