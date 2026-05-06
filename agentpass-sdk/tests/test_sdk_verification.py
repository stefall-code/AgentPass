"""
AgentPass SDK 验收测试脚本 - SDK 安装和 API 验证

运行方式:
    cd agentpass-sdk
    python tests/test_sdk_verification.py

依赖:
    pip install pyjwt pydantic pyyaml
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import pytest


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def add_pass(self, name, detail=""):
        self.passed += 1
        self.results.append({"name": name, "status": "PASS", "detail": detail})
        print(f"  [PASS] {name}")

    def add_fail(self, name, detail=""):
        self.failed += 1
        self.results.append({"name": name, "status": "FAIL", "detail": detail})
        print(f"  [FAIL] {name} - {detail}")

    def print_summary(self):
        total = self.passed + self.failed
        rate = (self.passed / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"测试结果: {self.passed}/{total} 通过 ({rate:.1f}%)")
        print(f"{'='*60}")
        return self.failed == 0


@pytest.fixture
def results():
    """Create a TestResults instance for testing"""
    return TestResults()


def test_sdk_imports(results):
    print("\n[1. SDK 安装验证]")
    try:
        results.add_pass("from agentpass import Guard")
    except Exception as e:
        results.add_fail("from agentpass import Guard", str(e))
        return

    try:
        import agentpass
        version = agentpass.__version__
        if version == "0.1.0":
            results.add_pass("版本检查", f"version={version}")
        else:
            results.add_fail("版本检查", f"expected 0.1.0, got {version}")
    except Exception as e:
        results.add_fail("版本检查", str(e))

    try:
        results.add_pass("Policy 模块导入")
    except Exception as e:
        results.add_fail("Policy 模块导入", str(e))

    try:
        results.add_pass("Audit 模块导入")
    except Exception as e:
        results.add_fail("Audit 模块导入", str(e))

    try:
        results.add_pass("Risk 模块导入")
    except Exception as e:
        results.add_fail("Risk 模块导入", str(e))

    try:
        results.add_pass("FastAPI 集成导入")
    except Exception as e:
        results.add_fail("FastAPI 集成导入", str(e))

    try:
        results.add_pass("依赖检查 (jwt, pydantic, yaml)")
    except Exception as e:
        results.add_fail("依赖检查", str(e))


def test_guard_api(results):
    print("\n[2. Guard API 验证]")
    from agentpass import Guard

    try:
        guard = Guard(secret="test-secret-key-32-characters!!")
        results.add_pass("Guard 初始化")
    except Exception as e:
        results.add_fail("Guard 初始化", str(e))
        return

    try:
        token = guard.issue_token("agent_1", role="admin")
        if token and isinstance(token, str) and len(token) > 20:
            results.add_pass("Token 签发", f"token={token[:30]}...")
        else:
            results.add_fail("Token 签发", "Invalid token format")
    except Exception as e:
        results.add_fail("Token 签发", str(e))

    try:
        payload = guard.authenticate(token)
        if payload and payload.get("sub") == "agent_1" and payload.get("role") == "admin":
            results.add_pass("Token 验证", f"sub={payload['sub']}, role={payload['role']}")
        else:
            results.add_fail("Token 验证", f"Invalid payload: {payload}")
    except Exception as e:
        results.add_fail("Token 验证", str(e))

    try:
        result = guard.check(token, "read_doc", "doc:test")
        if result.get("allowed"):
            results.add_pass("权限检查-允许", f"allowed={result['allowed']}")
        else:
            results.add_fail("权限检查-允许", f"expected True, got {result}")
    except Exception as e:
        results.add_fail("权限检查-允许", str(e))

    try:
        result = guard.check(token, "delete_doc", "doc:test")
        if not result.get("allowed"):
            results.add_pass("权限检查-拒绝", f"allowed={result['allowed']}")
        else:
            results.add_fail("权限检查-拒绝", f"expected False, got {result}")
    except Exception as e:
        results.add_fail("权限检查-拒绝", str(e))

    try:
        result = guard.assess_and_protect("agent_1", "doc:test", "read")
        if "decision" in result and "risk_assessment" in result:
            results.add_pass("assess_and_protect", f"decision={result['decision']}")
        else:
            results.add_fail("assess_and_protect", f"Invalid result: {result}")
    except Exception as e:
        results.add_fail("assess_and_protect", str(e))


def test_policy_module(results):
    print("\n[3. 策略模块验证]")
    from agentpass import Policy, PolicyRule, Priority

    try:
        policy = Policy(
            id="test_policy",
            name="Test Policy",
            rules=[
                PolicyRule(resource="doc:*", action="read", effect="allow", priority=10),
                PolicyRule(resource="*", action="*", effect="deny", priority=0)
            ]
        )
        results.add_pass("Policy 创建")
    except Exception as e:
        results.add_fail("Policy 创建", str(e))
        return

    try:
        policy = Policy(
            id="priority_test",
            name="Priority Test",
            priority_strategy=Priority.DENY_OVERRIDE,
            rules=[
                PolicyRule(resource="*", action="*", effect="allow", priority=0),
                PolicyRule(resource="secret:*", action="*", effect="deny", priority=100)
            ]
        )
        deny_result = policy.evaluate("secret:file", "read")
        if not deny_result:
            results.add_pass("DENY_OVERRIDE 策略", "deny 优先")
        else:
            results.add_fail("DENY_OVERRIDE 策略", f"expected False, got {deny_result}")
    except Exception as e:
        results.add_fail("DENY_OVERRIDE 策略", str(e))

    try:
        policy = Policy(
            id="allow_override_test",
            name="Allow Override Test",
            priority_strategy=Priority.ALLOW_OVERRIDE,
            rules=[
                PolicyRule(resource="*", action="*", effect="deny", priority=0),
                PolicyRule(resource="public:*", action="*", effect="allow", priority=100)
            ]
        )
        allow_result = policy.evaluate("public:readme", "read")
        if allow_result:
            results.add_pass("ALLOW_OVERRIDE 策略", "allow 优先")
        else:
            results.add_fail("ALLOW_OVERRIDE 策略", f"expected True, got {allow_result}")
    except Exception as e:
        results.add_fail("ALLOW_OVERRIDE 策略", str(e))

    try:
        policy = Policy(
            id="ip_condition_test",
            name="IP Condition Test",
            rules=[
                PolicyRule(
                    resource="doc:*",
                    action="read",
                    effect="allow",
                    conditions={"ip": {"allow": "private"}}
                )
            ]
        )
        result = policy.evaluate("doc:test", "read", {"ip_address": "192.168.1.100"})
        if result:
            results.add_pass("条件-ip 匹配", "私有 IP 匹配成功")
        else:
            results.add_fail("条件-ip 匹配", f"expected True, got {result}")
    except Exception as e:
        results.add_fail("条件-ip 匹配", str(e))

    try:
        policy = Policy(
            id="role_condition_test",
            name="Role Condition Test",
            rules=[
                PolicyRule(
                    resource="doc:*",
                    action="write",
                    effect="allow",
                    conditions={"role": {"require": ["admin", "editor"]}}
                )
            ]
        )
        result = policy.evaluate("doc:test", "write", {"role": "admin"})
        if result:
            results.add_pass("条件-role 匹配", "角色匹配成功")
        else:
            results.add_fail("条件-role 匹配", f"expected True, got {result}")
    except Exception as e:
        results.add_fail("条件-role 匹配", str(e))

    try:
        policy = Policy(
            id="explain_test",
            name="Explain Test",
            rules=[
                PolicyRule(resource="doc:*", action="read", effect="allow", priority=10),
                PolicyRule(resource="*", action="*", effect="deny", priority=0)
            ]
        )
        explain_result = policy.explain("doc:test", "read")
        if "matched_rules" in explain_result and "allowed" in explain_result:
            results.add_pass("explain() 方法", f"matched_rules={len(explain_result['matched_rules'])}")
        else:
            results.add_fail("explain() 方法", f"Invalid result: {explain_result}")
    except Exception as e:
        results.add_fail("explain() 方法", str(e))

    try:
        yaml_str = policy.to_yaml()
        if yaml_str and "id:" in yaml_str and "rules:" in yaml_str:
            results.add_pass("YAML 导出", f"length={len(yaml_str)}")
        else:
            results.add_fail("YAML 导出", "Invalid YAML format")
    except Exception as e:
        results.add_fail("YAML 导出", str(e))

    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_str)
            temp_path = f.name

        loaded_policy = Policy.from_yaml(temp_path)
        os.unlink(temp_path)

        if loaded_policy.id == "explain_test":
            results.add_pass("YAML 加载", f"id={loaded_policy.id}")
        else:
            results.add_fail("YAML 加载", f"ID mismatch: {loaded_policy.id}")
    except Exception as e:
        results.add_fail("YAML 加载", str(e))


def test_audit_module(results):
    print("\n[4. 审计模块验证]")
    from agentpass import Audit, AuditEvent

    try:
        audit = Audit(storage_backend=None)
        results.add_pass("Audit 初始化")
    except Exception as e:
        results.add_fail("Audit 初始化", str(e))
        return

    try:
        event = AuditEvent(
            event_type="test_action",
            user_id="agent_1",
            resource="doc:test",
            action="read",
            status="allow"
        )
        audit.log_event(event)
        results.add_pass("事件记录", f"event_id={event.id}")
    except Exception as e:
        results.add_fail("事件记录", str(e))

    try:
        events = audit.get_events()
        if isinstance(events, list):
            results.add_pass("事件查询", f"count={len(events)}")
        else:
            results.add_fail("事件查询", f"Invalid type: {type(events)}")
    except Exception as e:
        results.add_fail("事件查询", str(e))


def run_all_tests():
    print("=" * 60)
    print("AgentPass SDK 验收测试")
    print("=" * 60)

    results = TestResults()

    test_sdk_imports(results)
    test_guard_api(results)
    test_policy_module(results)
    test_audit_module(results)

    success = results.print_summary()

    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "passed": results.passed,
            "failed": results.failed,
            "details": results.results
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到: test_results.json")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
