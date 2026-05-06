from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import yaml
import fnmatch
import ipaddress


class Priority(str, Enum):
    DENY_OVERRIDE = "deny_override"
    ALLOW_OVERRIDE = "allow_override"


class PolicyRule(BaseModel):
    resource: str
    action: str
    effect: str = Field(..., pattern="^(allow|deny)$")
    priority: int = Field(default=0, description="Higher priority rules are evaluated first")
    conditions: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    @field_validator('effect')
    @classmethod
    def validate_effect(cls, v):
        if v not in ["allow", "deny"]:
            raise ValueError("effect must be 'allow' or 'deny'")
        return v


class Policy(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    rules: List[PolicyRule] = Field(default_factory=list)
    version: str = "1.0"
    priority_strategy: Priority = Priority.DENY_OVERRIDE

    def __init__(self, **data):
        super().__init__(**data)
        self._sort_rules()

    def _sort_rules(self):
        self.rules = sorted(self.rules, key=lambda r: -r.priority)

    def evaluate(self, resource: str, action: str, context: Dict[str, Any] = None) -> bool:
        result = self._evaluate_internal(resource, action, context)
        return result["allowed"]

    def explain(self, resource: str, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        return self._evaluate_internal(resource, action, context)

    def _evaluate_internal(self, resource: str, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        matched_rules = []
        final_decision = False

        for rule in self.rules:
            if self._match_resource(rule.resource, resource) and self._match_action(rule.action, action):
                condition_result = self._evaluate_conditions(rule.conditions, context)
                if condition_result["passed"]:
                    matched_rules.append({
                        "rule": rule,
                        "conditions_met": True,
                        "condition_details": condition_result
                    })
                elif rule.conditions:
                    matched_rules.append({
                        "rule": rule,
                        "conditions_met": False,
                        "condition_details": condition_result
                    })

        if not matched_rules:
            return {
                "allowed": False,
                "reason": "No matching rules found",
                "matched_rules": [],
                "final_decision_by": "default_deny"
            }

        matched_and_applicable = [r for r in matched_rules if r["conditions_met"]]

        if not matched_and_applicable:
            return {
                "allowed": False,
                "reason": "All matching rules failed conditions",
                "matched_rules": [
                    {
                        "resource": r["rule"].resource,
                        "action": r["rule"].action,
                        "effect": r["rule"].effect,
                        "priority": r["rule"].priority,
                        "conditions_met": r["conditions_met"],
                        "description": r["rule"].description
                    }
                    for r in matched_rules
                ],
                "final_decision_by": "conditions_not_met",
                "priority_strategy": self.priority_strategy.value
            }

        top_rule = matched_and_applicable[0]

        if self.priority_strategy == Priority.DENY_OVERRIDE:
            final_decision = (top_rule["rule"].effect == "allow")

        elif self.priority_strategy == Priority.ALLOW_OVERRIDE:
            final_decision = (top_rule["rule"].effect == "allow")

        return {
            "allowed": final_decision,
            "reason": f"Rule '{top_rule['rule'].resource}:{top_rule['rule'].action}' ({top_rule['rule'].effect}) matched at priority {top_rule['rule'].priority}",
            "matched_rules": [
                {
                    "resource": r["rule"].resource,
                    "action": r["rule"].action,
                    "effect": r["rule"].effect,
                    "priority": r["rule"].priority,
                    "conditions_met": r["conditions_met"],
                    "description": r["rule"].description
                }
                for r in matched_rules
            ],
            "final_decision_by": f"rule: {top_rule['rule'].id}" if hasattr(top_rule['rule'], 'id') else f"priority: {top_rule['rule'].priority}",
            "priority_strategy": self.priority_strategy.value
        }

    def _match_resource(self, rule_resource: str, request_resource: str) -> bool:
        if rule_resource == "*":
            return True
        if rule_resource == request_resource:
            return True
        if "*" in rule_resource:
            return fnmatch.fnmatch(request_resource, rule_resource)
        if rule_resource.endswith("/**"):
            prefix = rule_resource[:-3]
            return request_resource.startswith(prefix)
        if "/*" in rule_resource:
            parts = rule_resource.split("/*")
            if len(parts) == 2:
                prefix, suffix = parts
                if suffix:
                    return request_resource.startswith(prefix) and (request_resource[len(prefix):] or "").count("/") == 0
                return request_resource.startswith(prefix)
        return False

    def _match_action(self, rule_action: str, request_action: str) -> bool:
        if rule_action == "*":
            return True
        if rule_action == request_action:
            return True
        if "*" in rule_action:
            return fnmatch.fnmatch(request_action, rule_action)
        return False

    def _evaluate_conditions(self, conditions: Optional[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        if not conditions:
            return {"passed": True, "details": []}

        details = []
        passed = True

        for key, condition in conditions.items():
            if key == "time":
                time_result = self._evaluate_time_condition(condition, context)
                details.append({"type": "time", "result": time_result})
                if not time_result["passed"]:
                    passed = False

            elif key == "ip":
                ip_result = self._evaluate_ip_condition(condition, context)
                details.append({"type": "ip", "result": ip_result})
                if not ip_result["passed"]:
                    passed = False

            elif key == "resource_tag":
                tag_result = self._evaluate_resource_tag_condition(condition, context)
                details.append({"type": "resource_tag", "result": tag_result})
                if not tag_result["passed"]:
                    passed = False

            elif key == "role":
                role_result = self._evaluate_role_condition(condition, context)
                details.append({"type": "role", "result": role_result})
                if not role_result["passed"]:
                    passed = False

            elif key == "custom":
                custom_result = self._evaluate_custom_condition(condition, context)
                details.append({"type": "custom", "result": custom_result})
                if not custom_result["passed"]:
                    passed = False

            else:
                if key not in context:
                    passed = False
                    details.append({
                        "type": "attribute",
                        "condition": condition,
                        "passed": False,
                        "reason": f"Context key '{key}' not found"
                    })
                elif isinstance(condition, dict):
                    operator = condition.get("operator", "eq")
                    expected = condition.get("value")
                    actual = context.get(key)

                    if not self._evaluate_operator(operator, actual, expected):
                        passed = False
                        details.append({
                            "type": "attribute",
                            "key": key,
                            "operator": operator,
                            "expected": expected,
                            "actual": actual,
                            "passed": False
                        })
                    else:
                        details.append({
                            "type": "attribute",
                            "key": key,
                            "operator": operator,
                            "expected": expected,
                            "actual": actual,
                            "passed": True
                        })
                else:
                    if context.get(key) != condition:
                        passed = False
                        details.append({
                            "type": "attribute",
                            "key": key,
                            "expected": condition,
                            "actual": context.get(key),
                            "passed": False
                        })
                    else:
                        details.append({
                            "type": "attribute",
                            "key": key,
                            "expected": condition,
                            "actual": context.get(key),
                            "passed": True
                        })

        return {"passed": passed, "details": details}

    def _evaluate_time_condition(self, condition: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now()
        current_hour = now.hour
        current_day = now.strftime("%A").lower()

        if isinstance(condition, str):
            condition = {"hours": condition}

        if "hours" in condition:
            hours_str = condition["hours"]
            if hours_str == "workhours":
                if 9 <= current_hour < 18:
                    return {"passed": True, "type": "workhours"}
                return {"passed": False, "type": "workhours", "current_hour": current_hour}
            elif "-" in hours_str:
                start, end = hours_str.split("-")
                start_hour = int(start.strip())
                end_hour = int(end.strip())
                if start_hour <= current_hour < end_hour:
                    return {"passed": True, "type": "range", "start": start_hour, "end": end_hour, "current_hour": current_hour}
                return {"passed": False, "type": "range", "start": start_hour, "end": end_hour, "current_hour": current_hour}

        if "start" in condition and "end" in condition:
            start_hour = int(condition["start"])
            end_hour = int(condition["end"])
            if start_hour <= current_hour < end_hour:
                return {"passed": True, "type": "range", "start": start_hour, "end": end_hour, "current_hour": current_hour}
            return {"passed": False, "type": "range", "start": start_hour, "end": end_hour, "current_hour": current_hour}

        if "days" in condition:
            allowed_days = [d.lower() for d in condition["days"]]
            if current_day in allowed_days:
                return {"passed": True, "type": "days", "allowed": allowed_days, "current_day": current_day}
            return {"passed": False, "type": "days", "allowed": allowed_days, "current_day": current_day}

        return {"passed": True, "type": "unconditional"}

    def _evaluate_ip_condition(self, condition: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        request_ip_str = context.get("ip_address")
        if not request_ip_str:
            return {"passed": False, "reason": "No IP address in context"}

        try:
            request_ip = ipaddress.ip_address(request_ip_str)
        except ValueError:
            return {"passed": False, "reason": f"Invalid IP address: {request_ip_str}"}

        if isinstance(condition, str):
            condition = {"allow": condition}

        if "allow" in condition:
            allowed_str = condition["allow"]
            if allowed_str == "private":
                if request_ip.is_private:
                    return {"passed": True, "type": "private", "ip": str(request_ip)}
                return {"passed": False, "type": "private", "ip": str(request_ip)}
            elif allowed_str == "loopback":
                if request_ip.is_loopback:
                    return {"passed": True, "type": "loopback", "ip": str(request_ip)}
                return {"passed": False, "type": "loopback", "ip": str(request_ip)}
            elif "/" in allowed_str:
                network = ipaddress.ip_network(allowed_str, strict=False)
                if request_ip in network:
                    return {"passed": True, "type": "network", "network": allowed_str, "ip": str(request_ip)}
                return {"passed": False, "type": "network", "network": allowed_str, "ip": str(request_ip)}
            else:
                try:
                    allowed_ip = ipaddress.ip_address(allowed_str)
                    if request_ip == allowed_ip:
                        return {"passed": True, "type": "exact", "ip": str(request_ip)}
                    return {"passed": False, "type": "exact", "expected": allowed_str, "ip": str(request_ip)}
                except ValueError:
                    return {"passed": False, "reason": f"Invalid allowed IP: {allowed_str}"}

        if "deny" in condition:
            denied_str = condition["deny"]
            if "/" in denied_str:
                network = ipaddress.ip_network(denied_str, strict=False)
                if request_ip in network:
                    return {"passed": False, "type": "denied_network", "network": denied_str, "ip": str(request_ip)}
            else:
                try:
                    denied_ip = ipaddress.ip_address(denied_str)
                    if request_ip == denied_ip:
                        return {"passed": False, "type": "denied_exact", "expected": denied_str, "ip": str(request_ip)}
                except ValueError:
                    pass

        return {"passed": True, "type": "unconditional", "ip": str(request_ip)}

    def _evaluate_resource_tag_condition(self, condition: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        resource_tags = context.get("resource_tags", {})
        if isinstance(condition, str):
            condition = {"require": [condition]}

        if "require" in condition:
            required_tags = condition["require"]
            missing_tags = [tag for tag in required_tags if tag not in resource_tags]
            if not missing_tags:
                return {"passed": True, "type": "require", "required": required_tags, "found": list(resource_tags.keys())}
            return {"passed": False, "type": "require", "required": required_tags, "missing": missing_tags}

        if "exclude" in condition:
            excluded_tags = condition["exclude"]
            found_excluded = [tag for tag in excluded_tags if tag in resource_tags]
            if not found_excluded:
                return {"passed": True, "type": "exclude", "excluded": excluded_tags, "found": []}
            return {"passed": False, "type": "exclude", "excluded": excluded_tags, "found": found_excluded}

        return {"passed": True, "type": "unconditional"}

    def _evaluate_role_condition(self, condition: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        user_role = context.get("role", "")
        if isinstance(condition, str):
            condition = {"require": condition}

        if "require" in condition:
            required_role = condition["require"]
            if required_role == "*" or user_role == required_role:
                return {"passed": True, "type": "require", "required": required_role, "actual": user_role}
            if isinstance(required_role, list):
                if user_role in required_role:
                    return {"passed": True, "type": "require", "required": required_role, "actual": user_role}
                return {"passed": False, "type": "require", "required": required_role, "actual": user_role}
            return {"passed": False, "type": "require", "required": required_role, "actual": user_role}

        if "deny" in condition:
            denied_role = condition["deny"]
            if user_role == denied_role:
                return {"passed": False, "type": "deny", "denied": denied_role, "actual": user_role}

        return {"passed": True, "type": "unconditional", "actual": user_role}

    def _evaluate_custom_condition(self, condition: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        if callable(condition):
            try:
                result = condition(context)
                return {"passed": result, "type": "callable"}
            except Exception as e:
                return {"passed": False, "type": "callable", "error": str(e)}
        return {"passed": True, "type": "unconditional"}

    def _evaluate_operator(self, operator: str, actual: Any, expected: Any) -> bool:
        if operator == "eq" or operator == "==":
            return actual == expected
        elif operator == "ne" or operator == "!=":
            return actual != expected
        elif operator == "gt" or operator == ">":
            return actual > expected
        elif operator == "ge" or operator == ">=":
            return actual >= expected
        elif operator == "lt" or operator == "<":
            return actual < expected
        elif operator == "le" or operator == "<=":
            return actual <= expected
        elif operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected
        elif operator == "contains":
            return expected in actual
        elif operator == "startswith":
            return str(actual).startswith(str(expected))
        elif operator == "endswith":
            return str(actual).endswith(str(expected))
        elif operator == "regex":
            import re
            return bool(re.match(str(expected), str(actual)))
        return False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Policy":
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: str = None) -> str:
        data = self.model_dump(mode='json')
        yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
        if yaml_path:
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
        return yaml_str
