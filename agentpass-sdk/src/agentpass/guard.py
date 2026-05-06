from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from typing import Dict, Any, Optional, List
from .auth import Auth
from .policy import Policy, PolicyRule
from .risk import Risk, RiskLevel, AnomalyDetector, FraudDetector
from .audit import Audit, AuditEvent
from .prompt_defense import PromptDefense


class Guard(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    secret: str
    auth: Auth = Field(default=None)
    policies: Dict[str, Policy] = Field(default_factory=dict)
    risk: Risk = Field(default=None)
    audit: Audit = Field(default=None)
    prompt_defense: PromptDefense = Field(default=None)
    _is_initialized: bool = PrivateAttr(default=False)

    def __init__(self, **data):
        super().__init__(**data)
        if not self._is_initialized:
            self._setup_default_components()

    def _setup_default_components(self):
        self.auth = Auth(secret_key=self.secret)
        self.audit = Audit(storage_backend=None)
        detectors = [
            AnomalyDetector(name="anomaly_detector", version="1.0"),
            FraudDetector(name="fraud_detector", version="1.0")
        ]
        self.risk = Risk(detectors=detectors)
        self.prompt_defense = PromptDefense()
        self.policies = {
            "default": Policy(
                id="default",
                name="Default Policy",
                rules=[
                    PolicyRule(resource="*", action="read_doc", effect="allow", priority=50),
                    PolicyRule(resource="*", action="write_doc", effect="allow", priority=50),
                    PolicyRule(resource="admin_panel", action="*", effect="deny", priority=100),
                    PolicyRule(resource="*", action="*", effect="deny", priority=0),
                ]
            )
        }
        self._is_initialized = True

    def issue_token(self, agent_id: str, role: str = "user", **extra_claims) -> str:
        payload = {"sub": agent_id, "role": role, **extra_claims}
        token = self.auth.create_access_token(payload)
        self._log_audit_event(
            event_type="token_issued",
            user_id=agent_id,
            action="issue_token",
            status="success",
            details={"role": role}
        )
        return token

    def check(self, token: str, action: str, resource: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        payload = self.auth.verify_token(token)
        if not payload:
            self._log_audit_event(
                event_type="access_check",
                action=action,
                resource=resource,
                status="deny",
                details={"reason": "invalid_token"}
            )
            return {"allowed": False, "reason": "Invalid or expired token", "risk_level": "unknown"}

        agent_id = payload.get("sub", "unknown")
        role = payload.get("role", "user")
        context = context or {}

        prompt_injection_result = None
        if "prompt" in context:
            prompt_injection_result = self.prompt_defense.analyze(context["prompt"])
            if not prompt_injection_result.is_safe:
                self._log_audit_event(
                    event_type="prompt_injection_detected",
                    user_id=agent_id,
                    resource=resource,
                    action=action,
                    status="block",
                    details={
                        "risk_score": prompt_injection_result.risk_score,
                        "reason": prompt_injection_result.reason
                    }
                )
                return {
                    "allowed": False,
                    "reason": prompt_injection_result.reason,
                    "risk_level": "critical" if prompt_injection_result.risk_score > 0.7 else "high",
                    "risk_score": prompt_injection_result.risk_score,
                    "agent_id": agent_id,
                    "role": role,
                    "prompt_injection": prompt_injection_result.model_dump()
                }

        risk_context = {"user_id": agent_id, "resource": resource, "action": action, "role": role}
        risk_assessment = self.risk.assess_risk(risk_context)
        authorized = self._check_authorization(role, resource, action)

        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            decision, allowed, reason = "block", False, f"High risk detected: {risk_assessment.risk_level.value}"
        elif not authorized:
            decision, allowed, reason = "deny", False, f"Role '{role}' is not authorized for {action} on {resource}"
        else:
            decision, allowed, reason = "allow", True, "Access granted"

        self._log_audit_event(
            event_type="access_check",
            user_id=agent_id,
            resource=resource,
            action=action,
            status=decision,
            details={"role": role, "risk_level": risk_assessment.risk_level.value, "risk_score": risk_assessment.risk_score}
        )

        return {
            "allowed": allowed,
            "reason": reason,
            "risk_level": risk_assessment.risk_level.value,
            "risk_score": risk_assessment.risk_score,
            "agent_id": agent_id,
            "role": role
        }

    def batch_check(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量权限检查，单次调用处理多个请求

        Args:
            requests: 列表，每项包含 token, action, resource, context(可选)

        Returns:
            与输入等长的结果列表
        """
        results = []
        for req in requests:
            token = req.get("token", "")
            action = req.get("action", "")
            resource = req.get("resource", "")
            context = req.get("context")
            try:
                result = self.check(token, action, resource, context)
            except Exception as e:
                result = {"allowed": False, "reason": f"Batch check error: {str(e)}", "risk_level": "unknown"}
            results.append(result)
        return results

    def explain(self, agent_id: str, action: str, resource: str) -> Dict[str, Any]:
        """
        返回该Agent在该资源上的权限说明，不实际执行检查

        Args:
            agent_id: Agent标识
            action: 操作类型
            resource: 资源标识

        Returns:
            权限说明字典，包含匹配的规则、角色权限等
        """
        matched_rules = []
        for policy_id, policy in self.policies.items():
            for rule in policy.rules:
                if self._rule_matches(rule, resource, action):
                    matched_rules.append({
                        "policy_id": policy_id,
                        "policy_name": policy.name,
                        "resource": rule.resource,
                        "action": rule.action,
                        "effect": rule.effect,
                        "priority": rule.priority,
                    })

        matched_rules.sort(key=lambda r: r["priority"], reverse=True)

        final_effect = "deny"
        if matched_rules:
            final_effect = matched_rules[0]["effect"]

        return {
            "agent_id": agent_id,
            "action": action,
            "resource": resource,
            "final_effect": final_effect,
            "matched_rules": matched_rules,
            "explanation": self._build_explanation(agent_id, action, resource, matched_rules, final_effect),
        }

    def _rule_matches(self, rule: PolicyRule, resource: str, action: str) -> bool:
        resource_match = rule.resource == "*" or rule.resource == resource
        action_match = rule.action == "*" or rule.action == action
        return resource_match and action_match

    def _build_explanation(self, agent_id: str, action: str, resource: str,
                           rules: List[Dict], final_effect: str) -> str:
        if not rules:
            return f"Agent '{agent_id}' 对资源 '{resource}' 执行 '{action}'：无匹配规则，默认拒绝。"
        top = rules[0]
        if final_effect == "allow":
            return (f"Agent '{agent_id}' 可以对 '{resource}' 执行 '{action}'。"
                    f"由策略 '{top['policy_name']}' 中的规则允许（优先级 {top['priority']}）。")
        return (f"Agent '{agent_id}' 被禁止对 '{resource}' 执行 '{action}'。"
                f"由策略 '{top['policy_name']}' 中的规则拒绝（优先级 {top['priority']}）。")

    def _check_authorization(self, role: str, resource: str, action: str) -> bool:
        for policy in self.policies.values():
            if policy.evaluate(resource, action, {"role": role}):
                return True
        return False

    def add_policy(self, policy: Policy) -> None:
        self.policies[policy.id] = policy

    def authenticate(self, token: str) -> Optional[Dict[str, Any]]:
        return self.auth.verify_token(token)

    def authorize(self, user_id: str, resource: str, action: str) -> bool:
        return self._check_authorization(user_id, resource, action)

    def assess_and_protect(self, user_id: str, resource: str, action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        context.update({"user_id": user_id, "resource": resource, "action": action})
        risk_assessment = self.risk.assess_risk(context)
        authorized = self._check_authorization(user_id, resource, action)

        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            decision, reason = "block", f"High risk detected: {risk_assessment.risk_level.value}"
        elif not authorized:
            decision, reason = "deny", "Access not authorized"
        else:
            decision, reason = "allow", "Access granted"

        self._log_audit_event(
            event_type="access_control",
            user_id=user_id,
            resource=resource,
            action=action,
            status=decision,
            details={
                "risk_level": risk_assessment.risk_level.value,
                "risk_score": risk_assessment.risk_score,
                "recommendations": risk_assessment.recommendations
            }
        )

        return {
            "decision": decision,
            "reason": reason,
            "risk_assessment": risk_assessment.model_dump(),
            "authorized": authorized
        }

    def _log_audit_event(self, **kwargs):
        event = AuditEvent(**kwargs)
        self.audit.log_event(event)

    def clear_dialog_history(self, user_id: str = None):
        self.prompt_defense.clear_dialog_history(user_id)

    def analyze_prompt(self, prompt: str, history: List[str] = None, user_id: str = "default") -> Dict[str, Any]:
        result = self.prompt_defense.analyze(prompt, history=history, user_id=user_id)
        return result.model_dump()

    def check_with_context(self, agent_id: str, action: str, resource: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        role = context.get("role", "user")
        user = context.get("user", agent_id)

        prompt_risk_score = context.get("prompt_risk_score", 0.0)
        if prompt_risk_score > 0.0:
            risk_assessment = self.risk.assess_risk({
                "user_id": agent_id, "resource": resource,
                "action": action, "role": role, "prompt_risk_score": prompt_risk_score
            })
        else:
            risk_assessment = self.risk.assess_risk({
                "user_id": agent_id, "resource": resource, "action": action, "role": role
            })

        authorized = self._check_authorization(role, resource, action)

        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            decision, allowed, reason = "block", False, f"High risk detected: {risk_assessment.risk_level.value}"
        elif not authorized:
            decision, allowed, reason = "deny", False, f"Role '{role}' is not authorized for {action} on {resource}"
        else:
            decision, allowed, reason = "allow", True, "Access granted"

        self._log_audit_event(
            event_type="openclaw_check",
            user_id=agent_id,
            resource=resource,
            action=action,
            status=decision,
            details={"user": user, "role": role, "risk_level": risk_assessment.risk_level.value,
                     "risk_score": risk_assessment.risk_score, "source": "openclaw"}
        )

        return {
            "allowed": allowed,
            "reason": reason,
            "risk_level": risk_assessment.risk_level.value,
            "risk_score": risk_assessment.risk_score,
            "agent_id": agent_id,
            "role": role,
            "user": user
        }
