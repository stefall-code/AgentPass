"""
AgentPass SDK 验收测试脚本 - API 验证

运行方式:
    cd agentpass-sdk
    python tests/test_api_verification.py

依赖:
    pip install requests
    需要先启动: python examples/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import time
import requests
import pytest


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
        self.base_url = "http://127.0.0.1:8000"

    def add_pass(self, name, detail=""):
        self.passed += 1
        self.results.append({"name": name, "status": "PASS", "detail": detail})
        print(f"  [PASS] {name}")

    def add_fail(self, name, detail=""):
        self.failed += 1
        self.results.append({"name": name, "status": "FAIL", "detail": detail})
        print(f"  [FAIL] {name} - {detail}")

    def add_skip(self, name, reason=""):
        self.results.append({"name": name, "status": "SKIP", "detail": reason})
        print(f"  [SKIP] {name} - {reason}")

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


def wait_for_server(url, timeout=10):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def test_demo_app_endpoints(results):
    print("\n[5. Demo App 回归测试]")

    base_url = results.base_url
    if not wait_for_server(base_url):
        results.add_skip("Demo App 启动检测", "服务未运行，请先启动 python examples/app.py")
        return

    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "healthy":
            results.add_pass("健康检查", "返回 healthy")
        else:
            results.add_fail("健康检查", f"status={r.status_code}, body={r.text}")
    except Exception as e:
        results.add_fail("健康检查", str(e))

    try:
        r = requests.post(
            f"{base_url}/login",
            params={"username": "admin", "password": "admin123"},
            timeout=5
        )
        if r.status_code == 200:
            admin_token = r.json().get("access_token")
            if admin_token:
                results.add_pass("Admin 登录", "获得 token")
                results.admin_token = admin_token
            else:
                results.add_fail("Admin 登录", "未返回 token")
        else:
            results.add_fail("Admin 登录", f"status={r.status_code}")
    except Exception as e:
        results.add_fail("Admin 登录", str(e))

    try:
        r = requests.post(
            f"{base_url}/login",
            params={"username": "user", "password": "user123"},
            timeout=5
        )
        if r.status_code == 200:
            user_token = r.json().get("access_token")
            if user_token:
                results.add_pass("User 登录", "获得 token")
                results.user_token = user_token
            else:
                results.add_fail("User 登录", "未返回 token")
        else:
            results.add_fail("User 登录", f"status={r.status_code}")
    except Exception as e:
        results.add_fail("User 登录", str(e))

    try:
        if hasattr(results, 'admin_token'):
            headers = {"Authorization": f"Bearer {results.admin_token}"}
            r = requests.get(f"{base_url}/profile", headers=headers, timeout=5)
            if r.status_code == 200:
                profile = r.json()
                results.add_pass("获取 Profile", f"user_id={profile.get('user_id')}")
            else:
                results.add_fail("获取 Profile", f"status={r.status_code}")
        else:
            results.add_skip("获取 Profile", "需要 admin token")
    except Exception as e:
        results.add_fail("获取 Profile", str(e))

    try:
        if hasattr(results, 'user_token'):
            headers = {"Authorization": f"Bearer {results.user_token}"}
            r = requests.get(f"{base_url}/documents", headers=headers, timeout=5)
            if r.status_code == 200:
                docs = r.json()
                if "documents" in docs:
                    results.add_pass("列表文档", f"返回 {len(docs['documents'])} 个文档")
                else:
                    results.add_fail("列表文档", f"格式错误: {docs}")
            else:
                results.add_fail("列表文档", f"status={r.status_code}")
        else:
            results.add_skip("列表文档", "需要 user token")
    except Exception as e:
        results.add_fail("列表文档", str(e))

    try:
        if hasattr(results, 'user_token'):
            headers = {"Authorization": f"Bearer {results.user_token}"}
            r = requests.get(f"{base_url}/documents/1", headers=headers, timeout=5)
            if r.status_code == 200:
                doc = r.json()
                results.add_pass("获取文档", f"id={doc.get('id')}")
            else:
                results.add_fail("获取文档", f"status={r.status_code}")
        else:
            results.add_skip("获取文档", "需要 user token")
    except Exception as e:
        results.add_fail("获取文档", str(e))

    try:
        if hasattr(results, 'admin_token'):
            headers = {"Authorization": f"Bearer {results.admin_token}"}
            r = requests.delete(f"{base_url}/documents/1", headers=headers, timeout=5)
            if r.status_code == 200:
                results.add_pass("删除文档-Admin", "成功删除")
            else:
                results.add_fail("删除文档-Admin", f"status={r.status_code}")
        else:
            results.add_skip("删除文档-Admin", "需要 admin token")
    except Exception as e:
        results.add_fail("删除文档-Admin", str(e))


def test_permission_check(results):
    print("\n[6. 权限测试]")

    base_url = results.base_url
    if not hasattr(results, 'admin_token') or not hasattr(results, 'user_token'):
        results.add_skip("权限测试", "需要先完成登录测试")
        return

    admin_headers = {"Authorization": f"Bearer {results.admin_token}"}
    user_headers = {"Authorization": f"Bearer {results.user_token}"}

    try:
        r = requests.get(f"{base_url}/documents", headers=admin_headers, timeout=5)
        if r.status_code == 200:
            results.add_pass("Admin 读文档", "allowed")
        else:
            results.add_fail("Admin 读文档", f"status={r.status_code}")
    except Exception as e:
        results.add_fail("Admin 读文档", str(e))

    try:
        r = requests.get(f"{base_url}/documents", headers=user_headers, timeout=5)
        if r.status_code == 200:
            results.add_pass("User 读文档", "allowed")
        else:
            results.add_fail("User 读文档", f"status={r.status_code}")
    except Exception as e:
        results.add_fail("User 读文档", str(e))

    try:
        if hasattr(results, 'user_token'):
            r = requests.delete(f"{base_url}/documents/1", headers=user_headers, timeout=5)
            if r.status_code == 403:
                results.add_pass("User 删除文档", "正确拒绝 403")
            else:
                results.add_fail("User 删除文档", f"expected 403, got {r.status_code}")
        else:
            results.add_skip("User 删除文档", "需要 user token")
    except Exception as e:
        results.add_fail("User 删除文档", str(e))

    try:
        r = requests.get(f"{base_url}/documents", timeout=5)
        if r.status_code == 401:
            results.add_pass("缺失 Token", "正确返回 401")
        else:
            results.add_fail("缺失 Token", f"expected 401, got {r.status_code}")
    except Exception as e:
        results.add_fail("缺失 Token", str(e))

    try:
        r = requests.get(
            f"{base_url}/documents",
            headers={"Authorization": "Bearer invalid.token.here"},
            timeout=5
        )
        if r.status_code == 401:
            results.add_pass("无效 Token", "正确返回 401")
        else:
            results.add_fail("无效 Token", f"expected 401, got {r.status_code}")
    except Exception as e:
        results.add_fail("无效 Token", str(e))


def run_all_tests():
    print("=" * 60)
    print("AgentPass API 验收测试")
    print("=" * 60)
    print("注意: 请先启动 Demo App: python examples/app.py")

    results = TestResults()

    test_demo_app_endpoints(results)
    test_permission_check(results)

    success = results.print_summary()

    with open("test_api_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "passed": results.passed,
            "failed": results.failed,
            "details": results.results
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存到: test_api_results.json")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
