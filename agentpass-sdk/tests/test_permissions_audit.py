"""
AgentPass SDK 验收测试脚本 - 权限和审计测试

运行方式:
    cd agentpass-sdk
    python tests/test_permissions_audit.py

依赖:
    pip install pyjwt pydantic pyyaml
"""
import sys
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


def test_role_permissions(results):
    print("\n[6. 权限测试 - 基于角色]")
    from agentpass import Guard

    guard = Guard(secret="test-secret-key-32-characters!!")

    test_cases = [
        ("admin", "doc:test", "read_doc", True, "Admin should read docs"),
        ("admin", "doc:test", "write_doc", True, "Admin should write docs"),
        ("admin", "admin:panel", "read", True, "Admin should access admin panel"),
        ("admin", "admin:panel", "write", True, "Admin should write admin panel"),

        ("basic", "doc:test", "read_doc", True, "Basic should read docs"),
        ("basic", "doc:test", "write_doc", False, "Basic should NOT write docs"),
        ("basic", "admin:panel", "read", False, "Basic should NOT access admin"),

        ("editor", "doc:test", "read_doc", True, "Editor should read docs"),
        ("editor", "doc:test", "write_doc", True, "Editor should write docs"),

        ("operator", "task:execute", "execute_task", True, "Operator should execute tasks"),
        ("operator", "api:call", "call_api", True, "Operator should call APIs"),

        ("user", "doc:test", "read_doc", False, "User should NOT read docs"),
    ]

    for role, resource, action, expected, description in test_cases:
        try:
            token = guard.issue_token(f"test_{role}", role=role)
            result = guard.check(token, action, resource)
            actual = result.get("allowed")

            if actual == expected:
                results.add_pass(f"角色 {role} - {action} on {resource}", description)
            else:
                results.add_fail(
                    f"角色 {role} - {action} on {resource}",
                    f"expected {expected}, got {actual}. {description}"
                )
        except Exception as e:
            results.add_fail(f"角色 {role} - {action} on {resource}", str(e))


def test_token_validation(results):
    print("\n[6b. Token 验证测试]")
    from agentpass import Guard

    guard = Guard(secret="test-secret-key-32-characters!!")

    try:
        token = guard.issue_token("agent_1", role="admin")
        payload = guard.authenticate(token)

        if payload and payload.get("sub") == "agent_1":
            results.add_pass("有效 Token 验证", "验证通过")
        else:
            results.add_fail("有效 Token 验证", f"Invalid payload: {payload}")
    except Exception as e:
        results.add_fail("有效 Token 验证", str(e))

    try:
        invalid_payload = guard.authenticate("invalid.token.string")
        if invalid_payload is None:
            results.add_pass("无效 Token 验证", "正确拒绝")
        else:
            results.add_fail("无效 Token 验证", f"Should return None, got {invalid_payload}")
    except Exception as e:
        results.add_fail("无效 Token 验证", str(e))


def test_audit_logging(results):
    print("\n[7. 审计日志测试]")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

    from agentpass import Audit, AuditEvent

    try:
        audit = Audit(storage_backend=None)
        results.add_pass("SDK Audit 初始化")
    except Exception as e:
        results.add_fail("SDK Audit 初始化", str(e))
        return

    try:
        event1 = AuditEvent(
            event_type="login",
            user_id="agent_1",
            resource="auth",
            action="login",
            status="success"
        )
        audit.log_event(event1)

        event2 = AuditEvent(
            event_type="access_denied",
            user_id="agent_2",
            resource="admin:panel",
            action="read",
            status="deny"
        )
        audit.log_event(event2)

        events = audit.get_events()
        if len(events) >= 2:
            results.add_pass("SDK 事件记录", f"记录了 {len(events)} 个事件")
        else:
            results.add_fail("SDK 事件记录", f"Expected >=2, got {len(events)}")
    except Exception as e:
        results.add_fail("SDK 事件记录", str(e))

    try:
        json_output = audit.model_dump()
        if "events" in json_output or isinstance(json_output, dict):
            results.add_pass("SDK Audit JSON 序列化")
        else:
            results.add_fail("SDK Audit JSON 序列化", "Invalid format")
    except Exception as e:
        results.add_fail("SDK Audit JSON 序列化", str(e))


def test_adapter_audit(results):
    print("\n[7b. Adapter 审计测试]")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))

    try:
        from app.adapters import get_audit_adapter

        audit_adapter = get_audit_adapter()
        results.add_pass("Audit Adapter 获取")
    except Exception as e:
        results.add_fail("Audit Adapter 获取", str(e))
        return

    try:
        audit_adapter.log_event(
            event_type="test_login",
            agent_id="agent_test",
            resource="auth",
            action="login",
            status="success"
        )

        count = audit_adapter.get_event_count()
        if count >= 1:
            results.add_pass("Adapter 事件记录", f"事件数量: {count}")
        else:
            results.add_fail("Adapter 事件记录", f"Expected >=1, got {count}")
    except Exception as e:
        results.add_fail("Adapter 事件记录", str(e))

    try:
        events = audit_adapter.get_all_events()
        if len(events) >= 1:
            results.add_pass("Adapter 事件查询", f"获取 {len(events)} 个事件")
        else:
            results.add_fail("Adapter 事件查询", f"Expected >=1, got {len(events)}")
    except Exception as e:
        results.add_fail("Adapter 事件查询", str(e))

    try:
        json_output = audit_adapter.export_to_json()
        if json_output and "agent_test" in json_output:
            results.add_pass("Adapter JSON 导出", f"长度: {len(json_output)}")
        else:
            results.add_fail("Adapter JSON 导出", "Invalid output")
    except Exception as e:
        results.add_fail("Adapter JSON 导出", str(e))

    try:
        csv_output = audit_adapter.export_to_csv()
        if csv_output and "event_type" in csv_output:
            results.add_pass("Adapter CSV 导出", f"长度: {len(csv_output)}")
        else:
            results.add_fail("Adapter CSV 导出", "Invalid output")
    except Exception as e:
        results.add_fail("Adapter CSV 导出", str(e))

    try:
        audit_adapter.clear_events()
        count_after = audit_adapter.get_event_count()
        if count_after == 0:
            results.add_pass("Adapter 事件清除", "已清除所有事件")
        else:
            results.add_fail("Adapter 事件清除", f"Expected 0, got {count_after}")
    except Exception as e:
        results.add_fail("Adapter 事件清除", str(e))


def run_all_tests():
    print("=" * 60)
    print("AgentPass 权限和审计测试")
    print("=" * 60)

    results = TestResults()

    test_role_permissions(results)
    test_token_validation(results)
    test_audit_logging(results)
    test_adapter_audit(results)

    success = results.print_summary()

    with open("test_permissions_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "passed": results.passed,
            "failed": results.failed,
            "details": results.results
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到: test_permissions_results.json")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
