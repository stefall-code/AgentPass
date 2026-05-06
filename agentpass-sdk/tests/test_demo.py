import sys
sys.path.insert(0, 'src')

from agentpass import Guard


def test_issue_token_basic():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_1", role="admin")
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 0


def test_issue_token_with_extra_claims():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_2", role="user", department="engineering", level=5)
    assert token is not None
    payload = guard.auth.verify_token(token)
    assert payload["sub"] == "agent_2"
    assert payload["role"] == "user"
    assert payload["department"] == "engineering"
    assert payload["level"] == 5


def test_verify_valid_token():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_1", role="admin")
    payload = guard.auth.verify_token(token)
    assert payload is not None
    assert payload["sub"] == "agent_1"
    assert payload["role"] == "admin"


def test_verify_invalid_token():
    guard = Guard(secret="test-secret-key-1234567890")
    payload = guard.auth.verify_token("invalid.token.here")
    assert payload is None


def test_allowed_action():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_1", role="admin")
    result = guard.check(token=token, action="read_doc", resource="internal_doc")
    assert result["allowed"] is True
    assert result["reason"] == "Access granted"


def test_denied_action():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_1", role="admin")
    result = guard.check(token=token, action="delete_doc", resource="internal_doc")
    assert result["allowed"] is False
    assert "not authorized" in result["reason"]


def test_risk_level():
    guard = Guard(secret="test-secret-key-1234567890")
    token = guard.issue_token("agent_1", role="admin")
    result = guard.check(token=token, action="read_doc", resource="internal_doc")
    assert result["risk_level"] == "low"
    assert result["risk_score"] == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
