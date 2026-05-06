import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ENVIRONMENT"] = "test"


class TestConfigSecurity:
    def test_validate_security_detects_default_jwt_secret(self):
        from app.config import settings
        warnings = settings.validate_security()
        assert "JWT_SECRET" in warnings

    def test_validate_security_returns_list(self):
        from app.config import settings
        result = settings.validate_security()
        assert isinstance(result, list)


class TestIdentityApiKeyHash:
    def test_hash_api_key_returns_string(self):
        from app.identity import _hash_api_key
        result = _hash_api_key("test-key-123")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_verify_api_key_sha256_fallback(self):
        from app.identity import _hash_api_key, _verify_api_key
        hashed = _hash_api_key("test-key-123")
        try:
            import bcrypt
            if hashed.startswith("$2b$") or hashed.startswith("$2a$"):
                assert _verify_api_key("test-key-123", hashed)
                assert not _verify_api_key("wrong-key", hashed)
            else:
                assert _verify_api_key("test-key-123", hashed)
                assert not _verify_api_key("wrong-key", hashed)
        except ImportError:
            assert _verify_api_key("test-key-123", hashed)
            assert not _verify_api_key("wrong-key", hashed)

    def test_hash_is_deterministic_for_sha256(self):
        from app.identity import _hash_api_key
        try:
            import bcrypt
        except ImportError:
            h1 = _hash_api_key("same-key")
            h2 = _hash_api_key("same-key")
            assert h1 == h2


class TestMiddlewareRequestID:
    def test_request_id_pattern_accepts_valid(self):
        from app.middleware import _REQUEST_ID_PATTERN
        assert _REQUEST_ID_PATTERN.match("abc123")
        assert _REQUEST_ID_PATTERN.match("req-123-456")
        assert _REQUEST_ID_PATTERN.match("uuid_abc_def")

    def test_request_id_pattern_rejects_invalid(self):
        from app.middleware import _REQUEST_ID_PATTERN
        assert not _REQUEST_ID_PATTERN.match("")
        assert not _REQUEST_ID_PATTERN.match("a" * 65)
        assert not _REQUEST_ID_PATTERN.match("<script>alert(1)</script>")
        assert not _REQUEST_ID_PATTERN.match("id with spaces")
        assert not _REQUEST_ID_PATTERN.match("id\nwith\nnewlines")


class TestEd25519Auth:
    def test_generate_keypair(self):
        from app.security.ed25519_auth import generate_keypair
        result = generate_keypair("test-agent-keypair")
        assert "private_key_b64" in result
        assert "public_key_b64" in result
        assert "fingerprint" in result
        assert result["agent_id"] == "test-agent-keypair"

    def test_register_and_challenge(self):
        from app.security.ed25519_auth import (
            generate_keypair, issue_challenge, sign_challenge_locally,
            verify_challenge_response,
        )
        kp = generate_keypair("test-agent-challenge")
        challenge = issue_challenge("test-agent-challenge")
        assert challenge["challenge"] is not None
        assert challenge["challenge_id"] is not None

        signature = sign_challenge_locally(kp["private_key_b64"], challenge["challenge"])
        result = verify_challenge_response(
            challenge["challenge_id"], signature, "test-agent-challenge"
        )
        assert result["verified"] is True
        assert "session_token" in result

    def test_session_token_is_secure(self):
        from app.security.ed25519_auth import (
            generate_keypair, issue_challenge, sign_challenge_locally,
            verify_challenge_response,
        )
        kp = generate_keypair("test-agent-session")
        challenge = issue_challenge("test-agent-session")
        signature = sign_challenge_locally(kp["private_key_b64"], challenge["challenge"])
        result = verify_challenge_response(
            challenge["challenge_id"], signature, "test-agent-session"
        )
        token = result["session_token"]
        assert len(token) >= 32
        assert all(c.isalnum() or c in "-_=" for c in token)


class TestOwaspMemoryIntegrity:
    def test_write_and_verify(self):
        from app.security.owasp_shield import write_memory, verify_memory_integrity
        write_memory("agent-a", "key1", "value1", "private")
        write_memory("agent-a", "key2", "value2", "shared")
        result = verify_memory_integrity()
        assert result["total_entries"] >= 2
        assert result["chain_valid"] is True
        assert result["tampered"] == 0

    def test_tamper_detection(self):
        from app.security.owasp_shield import write_memory, verify_memory_integrity, _MEMORY_STORE
        write_memory("agent-b", "key1", "original", "private")
        mem_key = "agent-b:key1"
        if mem_key in _MEMORY_STORE:
            _MEMORY_STORE[mem_key]["value"] = "tampered"
            result = verify_memory_integrity()
            assert result["tampered"] >= 1


class TestAuditLogEvent:
    def test_log_event_does_not_crash(self):
        from app.audit import log_event
        log_event(
            action="test:action",
            resource="test:resource",
            decision="allow",
            reason="unit test",
            agent_id="test-agent",
            context={"test": True},
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
