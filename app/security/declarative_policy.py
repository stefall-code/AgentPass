"""
P2-8: Declarative Policy Configuration (YAML/JSON) — Cedar-style Policy Engine

Replaces scattered Python policy definitions with declarative YAML/JSON policies:
  - Policies defined in structured format (name, target, rules, conditions, effects)
  - Condition expression evaluator (trust_score, role, time, department, etc.)
  - Policy priority and conflict resolution (deny-override by default)
  - Hot-reload without code changes
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("agent_system")

_POLICY_STORE: Dict[str, Dict[str, Any]] = {}
_POLICY_EVAL_LOG: List[Dict[str, Any]] = []


@dataclass
class PolicyRule:
    action: str
    condition: str
    effect: str  # "allow" or "deny"
    priority: int = 50
    reason: str = ""


@dataclass
class Policy:
    name: str
    target: str
    rules: List[PolicyRule]
    description: str = ""
    enabled: bool = True
    version: str = "1.0"
    created_at: str = ""
    priority: int = 50


def load_policy_from_dict(policy_data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        name = policy_data.get("name", "")
        if not name:
            return {"loaded": False, "reason": "Policy name is required"}

        target = policy_data.get("target", "*")
        description = policy_data.get("description", "")
        enabled = policy_data.get("enabled", True)
        version = policy_data.get("version", "1.0")
        policy_priority = policy_data.get("priority", 50)

        rules = []
        for r in policy_data.get("rules", []):
            rule = PolicyRule(
                action=r.get("action", "*"),
                condition=r.get("condition", "true"),
                effect=r.get("effect", "deny"),
                priority=r.get("priority", 50),
                reason=r.get("reason", ""),
            )
            if rule.effect not in ("allow", "deny"):
                return {"loaded": False, "reason": f"Invalid effect '{rule.effect}' in rule for action '{rule.action}'"}
            rules.append(rule)

        if not rules:
            return {"loaded": False, "reason": "Policy must have at least one rule"}

        policy = Policy(
            name=name,
            target=target,
            rules=rules,
            description=description,
            enabled=enabled,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            priority=policy_priority,
        )

        _POLICY_STORE[name] = {
            "policy": policy,
            "raw": policy_data,
        }

        _log_eval("policy_loaded", name, f"target={target},rules={len(rules)}")
        return {
            "loaded": True,
            "name": name,
            "target": target,
            "rules_count": len(rules),
            "version": version,
        }

    except Exception as e:
        return {"loaded": False, "reason": f"Parse error: {str(e)[:100]}"}


def load_policy_from_json(json_str: str) -> Dict[str, Any]:
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {"loaded": False, "reason": f"JSON parse error: {str(e)}"}
    return load_policy_from_dict(data)


def load_policy_from_yaml_like(yaml_str: str) -> Dict[str, Any]:
    data = _simple_yaml_parse(yaml_str)
    if isinstance(data, dict):
        return load_policy_from_dict(data)
    return {"loaded": False, "reason": "Failed to parse YAML-like input"}


def evaluate_policy(
    agent_id: str,
    action: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if context is None:
        context = {}

    ctx = _build_eval_context(agent_id, context)

    matched_rules = []
    for policy_name, store_entry in _POLICY_STORE.items():
        policy = store_entry["policy"]
        if not policy.enabled:
            continue

        if policy.target != "*" and policy.target != agent_id:
            continue

        for rule in policy.rules:
            if not _action_matches(rule.action, action):
                continue

            cond_result = _evaluate_condition(rule.condition, ctx)
            if cond_result:
                matched_rules.append({
                    "policy": policy_name,
                    "rule_action": rule.action,
                    "effect": rule.effect,
                    "priority": rule.priority,
                    "reason": rule.reason,
                    "condition": rule.condition,
                })

    if not matched_rules:
        from app.permission import check_permission
        if check_permission(context.get("role", "operator"), action):
            result = {
                "decision": "allow",
                "reason": "RBAC fallback: role-based permission",
                "matched_rules": [{"policy": "rbac_fallback", "effect": "allow", "reason": "role has permission"}],
                "agent_id": agent_id,
                "action": action,
            }
        else:
            result = {
                "decision": "deny",
                "reason": "No matching policy rule found — default deny",
                "matched_rules": [],
                "agent_id": agent_id,
                "action": action,
            }
    else:
        matched_rules.sort(key=lambda r: r["priority"], reverse=True)
        highest = matched_rules[0]

        deny_rules = [r for r in matched_rules if r["effect"] == "deny"]
        allow_rules = [r for r in matched_rules if r["effect"] == "allow"]

        if deny_rules:
            top_deny = deny_rules[0]
            decision = "deny"
            reason = top_deny.get("reason") or f"Denied by policy '{top_deny['policy']}' (priority {top_deny['priority']})"
            highest = top_deny
        else:
            decision = "allow"
            reason = allow_rules[0].get("reason") or f"Allowed by policy '{allow_rules[0]['policy']}'"

        result = {
            "decision": decision,
            "reason": reason,
            "matched_rules": matched_rules,
            "agent_id": agent_id,
            "action": action,
        }

    _log_eval(
        "policy_evaluated",
        agent_id,
        f"action={action},decision={result['decision']},rules={len(matched_rules)}",
    )
    return result


def list_policies() -> Dict[str, Any]:
    policies = []
    for name, store_entry in _POLICY_STORE.items():
        p = store_entry["policy"]
        policies.append({
            "name": p.name,
            "target": p.target,
            "rules_count": len(p.rules),
            "enabled": p.enabled,
            "version": p.version,
            "priority": p.priority,
        })
    return {"policies": policies, "total": len(policies)}


def get_policy(name: str) -> Dict[str, Any]:
    entry = _POLICY_STORE.get(name)
    if not entry:
        return {"found": False}
    p = entry["policy"]
    return {
        "found": True,
        "name": p.name,
        "target": p.target,
        "description": p.description,
        "enabled": p.enabled,
        "version": p.version,
        "priority": p.priority,
        "rules": [
            {"action": r.action, "condition": r.condition, "effect": r.effect, "priority": r.priority, "reason": r.reason}
            for r in p.rules
        ],
        "raw": entry["raw"],
    }


def delete_policy(name: str) -> Dict[str, Any]:
    if name not in _POLICY_STORE:
        return {"deleted": False, "reason": f"Policy '{name}' not found"}
    del _POLICY_STORE[name]
    _log_eval("policy_deleted", name, "")
    return {"deleted": True, "name": name}


def toggle_policy(name: str, enabled: bool) -> Dict[str, Any]:
    entry = _POLICY_STORE.get(name)
    if not entry:
        return {"toggled": False, "reason": f"Policy '{name}' not found"}
    entry["policy"].enabled = enabled
    _log_eval("policy_toggled", name, f"enabled={enabled}")
    return {"toggled": True, "name": name, "enabled": enabled}


def get_policy_engine_status() -> Dict[str, Any]:
    return {
        "loaded_policies": len(_POLICY_STORE),
        "eval_log_entries": len(_POLICY_EVAL_LOG),
        "policy_names": list(_POLICY_STORE.keys()),
    }


def _build_eval_context(agent_id: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    ctx = {
        "agent_id": agent_id,
        "role": extra.get("role", "basic"),
        "trust_score": extra.get("trust_score", 0.5),
        "department": extra.get("department", ""),
        "time.hour": now.hour,
        "time.day": now.strftime("%A"),
        "time.is_business_hours": 8 <= now.hour < 22,
        "chain_length": extra.get("chain_length", 0),
        "risk_score": extra.get("risk_score", 0.0),
        "resource": extra.get("resource", ""),
    }
    ctx.update(extra)
    return ctx


def _action_matches(rule_action: str, request_action: str) -> bool:
    if rule_action == "*":
        return True
    if rule_action == request_action:
        return True
    rule_parts = rule_action.split(":")
    req_parts = request_action.split(":")
    for i in range(min(len(rule_parts), len(req_parts))):
        if rule_parts[i] == "*":
            continue
        if rule_parts[i] != req_parts[i]:
            return False
    return True


def _evaluate_condition(condition: str, ctx: Dict[str, Any]) -> bool:
    if not condition or condition.strip().lower() == "true":
        return True
    if condition.strip().lower() == "false":
        return False

    try:
        expr = condition
        for key, val in ctx.items():
            placeholder = key.replace(".", "_")
            if isinstance(val, str):
                expr = expr.replace(key, f"'{val}'")
            elif isinstance(val, bool):
                expr = expr.replace(key, str(val))
            else:
                expr = expr.replace(key, str(val))

        expr = expr.replace(" AND ", " and ")
        expr = expr.replace(" OR ", " or ")
        expr = expr.replace(" NOT ", " not ")

        expr = re.sub(r'(\d+)\.\.(\d+)', lambda m: f'list(range({m.group(1)},{int(m.group(2))+1}))', expr)
        expr = re.sub(r'IN\s+', 'in ', expr, flags=re.IGNORECASE)

        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.'\"()[] ,><=!&|+-*/%inandrnotfls")
        if not all(c in allowed_chars for c in expr if c not in '\t\n'):
            return False

        result = eval(expr, {"__builtins__": {}}, {})
        return bool(result)
    except Exception:
        return False


def _simple_yaml_parse(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    result: Dict[str, Any] = {}
    current_policy: Dict[str, Any] = {}
    current_rules: List[Dict[str, Any]] = []
    in_rules = False
    current_rule: Dict[str, Any] = {}

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("- name:"):
            if current_policy and current_rules:
                current_policy["rules"] = current_rules
                result = current_policy
            current_policy = {"name": stripped.split(":", 1)[1].strip().strip('"').strip("'")}
            current_rules = []
            in_rules = False
            continue

        if stripped.startswith("target:"):
            current_policy["target"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if stripped.startswith("description:"):
            current_policy["description"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if stripped.startswith("version:"):
            current_policy["version"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            continue
        if stripped.startswith("enabled:"):
            val = stripped.split(":", 1)[1].strip().lower()
            current_policy["enabled"] = val == "true"
            continue
        if stripped.startswith("priority:"):
            try:
                current_policy["priority"] = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue

        if stripped == "rules:" or stripped.startswith("rules:"):
            in_rules = True
            continue

        if in_rules:
            if stripped.startswith("- action:"):
                if current_rule:
                    current_rules.append(current_rule)
                current_rule = {"action": stripped.split(":", 1)[1].strip().strip('"').strip("'")}
                continue
            if stripped.startswith("action:") and current_rule:
                current_rule["action"] = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                continue
            if stripped.startswith("condition:"):
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if current_rule:
                    current_rule["condition"] = val
                continue
            if stripped.startswith("effect:"):
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if current_rule:
                    current_rule["effect"] = val
                continue
            if stripped.startswith("reason:"):
                val = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                if current_rule:
                    current_rule["reason"] = val
                continue

    if current_rule:
        current_rules.append(current_rule)
    if current_policy:
        current_policy["rules"] = current_rules
        result = current_policy

    return result if result else None


def _log_eval(action: str, policy_name: str, detail: str) -> None:
    _POLICY_EVAL_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "policy_name": policy_name,
        "detail": detail,
    })
    if len(_POLICY_EVAL_LOG) > 200:
        _POLICY_EVAL_LOG.pop(0)
