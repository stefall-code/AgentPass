"""
P2-7: Ed25519 Challenge-Response Authentication

Replaces HMAC API Key comparison with Ed25519 challenge-response:
  - Agent generates Ed25519 keypair at registration
  - Broker sends random challenge at each auth
  - Agent signs challenge with private key
  - Broker verifies signature with public key
  - Private key never leaves the Agent
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError

logger = logging.getLogger("agent_system")

_AGENT_KEYS: Dict[str, Dict[str, Any]] = {}
_PERSIST_LOADED = False
_PENDING_CHALLENGES: Dict[str, Dict[str, Any]] = {}
_AUTH_SESSIONS: Dict[str, Dict[str, Any]] = {}
_CHALLENGE_LOG: List[Dict[str, Any]] = []

CHALLENGE_EXPIRY_SECONDS = 300
MAX_CHALLENGE_ATTEMPTS = 3


def _ensure_persist_loaded():
    global _PERSIST_LOADED
    if _PERSIST_LOADED:
        return
    _PERSIST_LOADED = True
    try:
        from app.db import SessionLocal
        from app.models import Ed25519KeyRow
        with SessionLocal() as db:
            rows = db.query(Ed25519KeyRow).all()
            for row in rows:
                _AGENT_KEYS[row.agent_id] = {
                    "agent_id": row.agent_id,
                    "public_key_b64": row.public_key_b64,
                    "public_key_fingerprint": row.public_key_fingerprint,
                    "registered_at": row.registered_at,
                    "auth_count": row.auth_count,
                    "last_auth": row.last_auth,
                }
            if rows:
                logger.info("Ed25519: loaded %d keys from database", len(rows))
    except Exception as e:
        logger.warning("Ed25519: failed to load keys from database: %s", e)


def _persist_key(agent_id: str, key_data: Dict[str, Any]):
    try:
        from app.db import SessionLocal
        from app.models import Ed25519KeyRow
        with SessionLocal() as db:
            existing = db.query(Ed25519KeyRow).filter_by(agent_id=agent_id).first()
            if existing:
                existing.public_key_b64 = key_data["public_key_b64"]
                existing.public_key_fingerprint = key_data["public_key_fingerprint"]
                existing.registered_at = key_data["registered_at"]
            else:
                db.add(Ed25519KeyRow(
                    agent_id=agent_id,
                    public_key_b64=key_data["public_key_b64"],
                    public_key_fingerprint=key_data["public_key_fingerprint"],
                    registered_at=key_data["registered_at"],
                    auth_count=0,
                    last_auth=None,
                ))
            db.commit()
    except Exception as e:
        logger.warning("Ed25519: failed to persist key for %s: %s", agent_id, e)


def _persist_auth_update(agent_id: str, auth_count: int, last_auth: str):
    try:
        from app.db import SessionLocal
        from app.models import Ed25519KeyRow
        with SessionLocal() as db:
            row = db.query(Ed25519KeyRow).filter_by(agent_id=agent_id).first()
            if row:
                row.auth_count = auth_count
                row.last_auth = last_auth
                db.commit()
    except Exception as e:
        logger.warning("Ed25519: failed to persist auth update for %s: %s", agent_id, e)


def generate_keypair(agent_id: str) -> Dict[str, Any]:
    _ensure_persist_loaded()
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key

    private_key_b64 = base64.b64encode(bytes(signing_key)).decode("ascii")
    public_key_b64 = base64.b64encode(bytes(verify_key)).decode("ascii")
    key_fingerprint = hashlib.sha256(bytes(verify_key)).hexdigest()[:16]

    _AGENT_KEYS[agent_id] = {
        "agent_id": agent_id,
        "public_key_b64": public_key_b64,
        "public_key_fingerprint": key_fingerprint,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "auth_count": 0,
        "last_auth": None,
    }

    _persist_key(agent_id, _AGENT_KEYS[agent_id])
    _log_challenge("keypair_generated", agent_id, f"fingerprint={key_fingerprint}")

    return {
        "agent_id": agent_id,
        "private_key_b64": private_key_b64,
        "public_key_b64": public_key_b64,
        "fingerprint": key_fingerprint,
        "warning": "PRIVATE KEY MUST NEVER BE SENT TO THE SERVER — keep it locally!",
    }


def register_public_key(agent_id: str, public_key_b64: str) -> Dict[str, Any]:
    _ensure_persist_loaded()
    try:
        pub_bytes = base64.b64decode(public_key_b64)
        verify_key = VerifyKey(pub_bytes)
        fingerprint = hashlib.sha256(pub_bytes).hexdigest()[:16]
    except Exception as e:
        _log_challenge("register_failed", agent_id, f"invalid_key:{str(e)[:50]}")
        return {"registered": False, "reason": f"Invalid public key: {str(e)[:80]}"}

    _AGENT_KEYS[agent_id] = {
        "agent_id": agent_id,
        "public_key_b64": public_key_b64,
        "public_key_fingerprint": fingerprint,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "auth_count": 0,
        "last_auth": None,
    }

    _persist_key(agent_id, _AGENT_KEYS[agent_id])
    _log_challenge("public_key_registered", agent_id, f"fingerprint={fingerprint}")
    return {"registered": True, "agent_id": agent_id, "fingerprint": fingerprint}


def issue_challenge(agent_id: str) -> Dict[str, Any]:
    _ensure_persist_loaded()
    if agent_id not in _AGENT_KEYS:
        return {"challenge": None, "reason": f"Agent '{agent_id}' has no registered Ed25519 public key"}

    challenge_bytes = os.urandom(32)
    challenge_b64 = base64.b64encode(challenge_bytes).decode("ascii")
    challenge_id = f"ch_{uuid.uuid4().hex[:12]}"
    issued_at = time.time()

    _PENDING_CHALLENGES[challenge_id] = {
        "challenge_id": challenge_id,
        "agent_id": agent_id,
        "challenge_b64": challenge_b64,
        "issued_at": issued_at,
        "attempts": 0,
        "verified": False,
    }

    _log_challenge("challenge_issued", agent_id, f"challenge_id={challenge_id}")

    return {
        "challenge_id": challenge_id,
        "challenge": challenge_b64,
        "expires_in": CHALLENGE_EXPIRY_SECONDS,
        "algorithm": "Ed25519",
    }


def verify_challenge_response(
    challenge_id: str,
    signature_b64: str,
    agent_id: str,
) -> Dict[str, Any]:
    challenge = _PENDING_CHALLENGES.get(challenge_id)
    if not challenge:
        _log_challenge("verify_failed", agent_id, "challenge_not_found")
        return {"verified": False, "reason": "Challenge not found or expired"}

    if challenge["agent_id"] != agent_id:
        _log_challenge("verify_failed", agent_id, "agent_mismatch")
        return {"verified": False, "reason": "Challenge was issued to a different agent"}

    if challenge["verified"]:
        _log_challenge("verify_failed", agent_id, "challenge_already_used")
        return {"verified": False, "reason": "Challenge already used — request a new one"}

    if time.time() - challenge["issued_at"] > CHALLENGE_EXPIRY_SECONDS:
        del _PENDING_CHALLENGES[challenge_id]
        _log_challenge("verify_failed", agent_id, "challenge_expired")
        return {"verified": False, "reason": "Challenge expired — request a new one"}

    challenge["attempts"] += 1
    if challenge["attempts"] > MAX_CHALLENGE_ATTEMPTS:
        del _PENDING_CHALLENGES[challenge_id]
        _log_challenge("verify_failed", agent_id, "max_attempts_exceeded")
        return {"verified": False, "reason": f"Max {MAX_CHALLENGE_ATTEMPTS} attempts exceeded"}

    agent_key = _AGENT_KEYS.get(agent_id)
    if not agent_key:
        return {"verified": False, "reason": "Agent public key not found"}

    try:
        pub_bytes = base64.b64decode(agent_key["public_key_b64"])
        verify_key = VerifyKey(pub_bytes)

        sig_bytes = base64.b64decode(signature_b64)
        challenge_bytes = base64.b64decode(challenge["challenge_b64"])

        verify_key.verify(challenge_bytes, sig_bytes)

        challenge["verified"] = True
        agent_key["auth_count"] += 1
        agent_key["last_auth"] = datetime.now(timezone.utc).isoformat()

        _persist_auth_update(agent_id, agent_key["auth_count"], agent_key["last_auth"])

        session_token = secrets.token_urlsafe(32)

        _AUTH_SESSIONS[session_token] = {
            "session_token": session_token,
            "agent_id": agent_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "challenge_id": challenge_id,
            "fingerprint": agent_key["public_key_fingerprint"],
        }

        del _PENDING_CHALLENGES[challenge_id]

        _log_challenge("verify_success", agent_id, f"session={session_token[:12]}")
        return {
            "verified": True,
            "agent_id": agent_id,
            "session_token": session_token,
            "fingerprint": agent_key["public_key_fingerprint"],
        }

    except (BadSignatureError, Exception) as e:
        _log_challenge("verify_failed", agent_id, f"bad_signature:{str(e)[:50]}")
        return {"verified": False, "reason": f"Signature verification failed: {str(e)[:80]}"}


def sign_challenge_locally(private_key_b64: str, challenge_b64: str) -> str:
    priv_bytes = base64.b64decode(private_key_b64)
    signing_key = SigningKey(priv_bytes)
    challenge_bytes = base64.b64decode(challenge_b64)
    signed = signing_key.sign(challenge_bytes)
    return base64.b64encode(signed.signature).decode("ascii")


def get_ed25519_status() -> Dict[str, Any]:
    _ensure_persist_loaded()
    expired = sum(
        1 for c in _PENDING_CHALLENGES.values()
        if time.time() - c["issued_at"] > CHALLENGE_EXPIRY_SECONDS
    )
    return {
        "registered_agents": len(_AGENT_KEYS),
        "pending_challenges": len(_PENDING_CHALLENGES),
        "expired_challenges": expired,
        "active_sessions": len(_AUTH_SESSIONS),
        "auth_log_entries": len(_CHALLENGE_LOG),
    }


def get_agent_auth_info(agent_id: str) -> Dict[str, Any]:
    _ensure_persist_loaded()
    info = _AGENT_KEYS.get(agent_id)
    if not info:
        return {"found": False}
    return {
        "found": True,
        "agent_id": agent_id,
        "fingerprint": info["public_key_fingerprint"],
        "auth_count": info["auth_count"],
        "last_auth": info["last_auth"],
        "registered_at": info["registered_at"],
    }


def _log_challenge(action: str, agent_id: str, detail: str) -> None:
    _CHALLENGE_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent_id": agent_id,
        "detail": detail,
    })
    if len(_CHALLENGE_LOG) > 200:
        _CHALLENGE_LOG.pop(0)
