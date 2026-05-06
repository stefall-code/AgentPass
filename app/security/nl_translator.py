"""
P2-10: Natural Language Permission Translation

Translates natural language permission descriptions into auditable ACL rules:
  - "Only read financial data, cannot modify" → {action: "read:finance", effect: "allow"} + {action: "write:finance", effect: "deny"}
  - Intent classification + entity extraction
  - Rule generation with confidence scoring
  - Human-in-the-loop confirmation before enforcement
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger("agent_system")

_NL_RULES: Dict[str, Dict[str, Any]] = {}
_TRANSLATION_LOG: List[Dict[str, Any]] = []


_ACTION_KEYWORDS = {
    "read": ["read", "view", "see", "access", "query", "fetch", "get", "check", "look", "retrieve", "browse", "search", "list", "读取", "查看", "阅读", "访问", "查询", "获取", "浏览"],
    "write": ["write", "modify", "edit", "change", "update", "alter", "set", "patch", "adjust", "修改", "编辑", "更改", "更新", "写入", "变更"],
    "create": ["create", "add", "new", "generate", "make", "build", "insert", "produce", "创建", "新建", "添加", "生成"],
    "delete": ["delete", "remove", "drop", "destroy", "erase", "clear", "purge", "删除", "移除", "清除"],
    "execute": ["execute", "run", "invoke", "trigger", "launch", "start", "call", "perform", "执行", "运行", "调用"],
    "export": ["export", "download", "extract", "pull", "transfer", "send", "share", "output", "导出", "下载", "提取", "分享"],
    "delegate": ["delegate", "assign", "forward", "hand off", "transfer authority", "委托", "委派", "分配"],
    "admin": ["admin", "manage", "configure", "control", "govern", "supervise", "administer", "管理", "配置", "控制"],
}

_RESOURCE_KEYWORDS = {
    "feishu_table:finance": ["finance", "financial", "revenue", "profit", "budget", "accounting", "fiscal", "cost", "expense", "income", "财务", "营收", "利润", "预算", "会计", "成本", "收入", "金融"],
    "feishu_table:hr": ["hr", "human resource", "employee", "salary", "payroll", "personnel", "staff", "workforce", "compensation", "人事", "人力资源", "员工", "薪资", "薪酬", "人员"],
    "feishu_table:sales": ["sales", "customer", "client", "deal", "order", "revenue", "pipeline", "crm", "销售", "客户", "订单"],
    "feishu_table:product": ["product", "inventory", "stock", "catalog", "item", "merchandise", "产品", "库存", "商品"],
    "feishu_doc": ["document", "doc", "file", "report", "memo", "note", "paper", "article", "文档", "文件", "报告", "备忘"],
    "feishu_sheet": ["sheet", "spreadsheet", "excel", "table", "data", "表格", "数据表", "电子表格"],
    "feishu_message": ["message", "chat", "notification", "im", "communication", "消息", "聊天", "通知"],
    "web_search": ["web", "internet", "online", "search engine", "google", "网络", "搜索", "互联网"],
    "api_endpoint": ["api", "endpoint", "service", "microservice", "backend", "接口", "服务"],
    "system_config": ["config", "configuration", "setting", "system", "admin panel", "配置", "设置", "系统"],
}

_NEGATION_KEYWORDS = [
    "cannot", "can't", "not allowed", "no access", "forbidden", "denied",
    "must not", "should not", "don't", "never", "without permission",
    "prohibited", "restricted", "blocked", "unauthorized",
    "不能", "不可以", "无法", "禁止", "不允许", "不得", "严禁", "无权",
]

_ALLOW_KEYWORDS = [
    "can", "allowed", "permitted", "may", "able to", "have access",
    "authorized", "granted", "only", "just", "solely",
    "可以", "允许", "能够", "有权", "只能", "仅能", "仅允许",
]

_CONDITION_KEYWORDS = {
    "time_restriction": ["business hours", "working hours", "office hours", "9 to 5", "9am to 6pm", "during work", "工作时间", "办公时间", "上班时间", "营业时间"],
    "trust_threshold": ["trusted", "high trust", "trust score", "verified", "vetted", "受信任", "可信", "高信任", "信任度"],
    "approval_required": ["with approval", "approved by", "after review", "with consent", "authorized by", "审批", "批准", "审核", "同意", "许可"],
    "department_restriction": ["same department", "own team", "own department", "within team", "同部门", "本部门", "本团队"],
    "audit_logging": ["audited", "logged", "tracked", "monitored", "recorded", "审计", "记录", "监控", "追踪"],
}


@dataclass
class TranslatedRule:
    action: str
    resource: str
    effect: str
    condition: str
    confidence: float
    original_text: str


def translate_nl_to_acl(nl_text: str, agent_id: str = "") -> Dict[str, Any]:
    translation_id = f"nl_{uuid.uuid4().hex[:10]}"
    text_lower = nl_text.lower().strip()

    intents = _classify_intents(text_lower)
    resources = _extract_resources(text_lower)
    effects = _determine_effects(text_lower)
    conditions = _extract_conditions(text_lower)
    scope = _determine_scope(text_lower)

    rules = _generate_rules(intents, resources, effects, conditions, scope, nl_text)

    if not rules:
        _log_translation("translation_empty", agent_id, nl_text[:50])
        return {
            "translation_id": translation_id,
            "original_text": nl_text,
            "rules": [],
            "confidence": 0.0,
            "warning": "Could not extract any permission rules from the input",
        }

    overall_confidence = sum(r.confidence for r in rules) / len(rules)

    rule_dicts = []
    for r in rules:
        rule_id = f"rule_{uuid.uuid4().hex[:8]}"
        rule_dict = {
            "rule_id": rule_id,
            "action": r.action,
            "resource": r.resource,
            "effect": r.effect,
            "condition": r.condition,
            "confidence": round(r.confidence, 3),
            "original_text": r.original_text,
        }
        rule_dicts.append(rule_dict)
        _NL_RULES[rule_id] = {
            "rule_id": rule_id,
            "translation_id": translation_id,
            "agent_id": agent_id,
            "rule": rule_dict,
            "confirmed": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    result = {
        "translation_id": translation_id,
        "original_text": nl_text,
        "extracted_intents": intents,
        "extracted_resources": resources,
        "extracted_effects": effects,
        "extracted_conditions": conditions,
        "scope": scope,
        "rules": rule_dicts,
        "overall_confidence": round(overall_confidence, 3),
        "requires_confirmation": overall_confidence < 0.8,
    }

    _log_translation("translation_completed", agent_id, f"rules={len(rules)},confidence={overall_confidence:.2f}")
    return result


def confirm_rule(rule_id: str) -> Dict[str, Any]:
    entry = _NL_RULES.get(rule_id)
    if not entry:
        return {"confirmed": False, "reason": f"Rule '{rule_id}' not found"}
    entry["confirmed"] = True
    entry["confirmed_at"] = datetime.now(timezone.utc).isoformat()
    _log_translation("rule_confirmed", entry.get("agent_id", ""), rule_id)
    return {"confirmed": True, "rule_id": rule_id, "rule": entry["rule"]}


def reject_rule(rule_id: str) -> Dict[str, Any]:
    entry = _NL_RULES.get(rule_id)
    if not entry:
        return {"rejected": False, "reason": f"Rule '{rule_id}' not found"}
    if rule_id in _NL_RULES:
        del _NL_RULES[rule_id]
    _log_translation("rule_rejected", entry.get("agent_id", ""), rule_id)
    return {"rejected": True, "rule_id": rule_id}


def list_nl_rules() -> Dict[str, Any]:
    rules = []
    for rule_id, entry in _NL_RULES.items():
        rules.append({
            "rule_id": rule_id,
            "agent_id": entry.get("agent_id", ""),
            "action": entry["rule"]["action"],
            "resource": entry["rule"]["resource"],
            "effect": entry["rule"]["effect"],
            "confidence": entry["rule"]["confidence"],
            "confirmed": entry["confirmed"],
        })
    return {"rules": rules, "total": len(rules)}


def get_nl_translator_status() -> Dict[str, Any]:
    confirmed = sum(1 for e in _NL_RULES.values() if e["confirmed"])
    return {
        "total_rules_generated": len(_NL_RULES),
        "confirmed_rules": confirmed,
        "pending_rules": len(_NL_RULES) - confirmed,
        "translation_log_entries": len(_TRANSLATION_LOG),
        "supported_intents": list(_ACTION_KEYWORDS.keys()),
        "supported_resources": list(_RESOURCE_KEYWORDS.keys()),
    }


def _classify_intents(text: str) -> List[str]:
    intents = []
    for action, keywords in _ACTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                if action not in intents:
                    intents.append(action)
                break
    if not intents:
        if any(w in text for w in ["access", "use", "work with"]):
            intents.append("read")
    return intents


def _extract_resources(text: str) -> List[str]:
    resources = []
    for resource, keywords in _RESOURCE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                if resource not in resources:
                    resources.append(resource)
                break
    if not resources:
        if "data" in text or "information" in text:
            resources.append("feishu_table")
        elif "system" in text or "platform" in text:
            resources.append("system_config")
    return resources


def _determine_effects(text: str) -> Dict[str, str]:
    effects = {}
    has_negation = any(neg in text for neg in _NEGATION_KEYWORDS)
    has_allow = any(allow in text for allow in _ALLOW_KEYWORDS)

    if has_negation and has_allow:
        parts = re.split(r"(,|but|however|except|and|;|\.)", text)
        for part in parts:
            part_lower = part.lower()
            is_negated = any(neg in part_lower for neg in _NEGATION_KEYWORDS)
            part_intents = _classify_intents(part_lower)
            for intent in part_intents:
                effects[intent] = "deny" if is_negated else "allow"
    elif has_negation:
        intents = _classify_intents(text)
        for intent in intents:
            effects[intent] = "deny"
    elif has_allow:
        intents = _classify_intents(text)
        for intent in intents:
            effects[intent] = "allow"
    else:
        intents = _classify_intents(text)
        for intent in intents:
            effects[intent] = "allow"

    return effects


def _extract_conditions(text: str) -> List[str]:
    conditions = []
    for cond_type, keywords in _CONDITION_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                if cond_type not in conditions:
                    conditions.append(cond_type)
                break
    return conditions


def _determine_scope(text: str) -> str:
    if any(w in text for w in ["only", "just", "solely", "exclusively", "nothing else"]):
        return "restricted"
    if any(w in text for w in ["all", "everything", "any", "unlimited", "full"]):
        return "broad"
    return "normal"


def _generate_rules(
    intents: List[str],
    resources: List[str],
    effects: Dict[str, str],
    conditions: List[str],
    scope: str,
    original_text: str,
) -> List[TranslatedRule]:
    rules = []

    if not resources:
        resources = ["*"]
    if not intents:
        intents = ["read"]

    for resource in resources:
        for intent in intents:
            action = f"{intent}:{resource}" if resource != "*" else intent
            effect = effects.get(intent, "allow")

            confidence = 0.7
            if len(intents) == 1:
                confidence += 0.1
            if len(resources) == 1:
                confidence += 0.1
            if conditions:
                confidence += 0.05
            if scope == "restricted":
                confidence += 0.05
            confidence = min(1.0, confidence)

            condition_str = ""
            if "time_restriction" in conditions:
                condition_str += "time.is_business_hours == true"
            if "trust_threshold" in conditions:
                condition_str += (" AND " if condition_str else "") + "trust_score >= 0.7"
            if "approval_required" in conditions:
                condition_str += (" AND " if condition_str else "") + "approval_status == 'approved'"
            if "department_restriction" in conditions:
                condition_str += (" AND " if condition_str else "") + "requester.department == agent.department"
            if "audit_logging" in conditions:
                condition_str += (" AND " if condition_str else "") + "audit_enabled == true"

            rules.append(TranslatedRule(
                action=action,
                resource=resource,
                effect=effect,
                condition=condition_str,
                confidence=confidence,
                original_text=original_text,
            ))

    if scope == "restricted" and len(intents) > 0:
        all_intents = list(_ACTION_KEYWORDS.keys())
        denied_intents = [i for i in all_intents if i not in intents]
        for resource in resources:
            for denied in denied_intents[:3]:
                action = f"{denied}:{resource}" if resource != "*" else denied
                rules.append(TranslatedRule(
                    action=action,
                    resource=resource,
                    effect="deny",
                    condition="",
                    confidence=0.6,
                    original_text=original_text,
                ))

    return rules


def _log_translation(action: str, agent_id: str, detail: str) -> None:
    _TRANSLATION_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent_id": agent_id,
        "detail": detail,
    })
    if len(_TRANSLATION_LOG) > 200:
        _TRANSLATION_LOG.pop(0)
