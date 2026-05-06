from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from typing import Dict, Optional, Any, List

from sqlalchemy import select, delete

from app.config import settings
from app import database, permission
from app.db import SessionLocal
from app.models import AgentRow, IssuedTokenRow


def _hash_api_key(api_key: str) -> str:
    try:
        import bcrypt
        return bcrypt.hashpw(api_key.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")
    except ImportError:
        raw = f"{settings.JWT_SECRET}:{api_key}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()


def _verify_api_key(api_key: str, stored_hash: str) -> bool:
    try:
        import bcrypt
        if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
            return bcrypt.checkpw(api_key.encode("utf-8"), stored_hash.encode("utf-8"))
    except ImportError:
        pass
    raw = f"{settings.JWT_SECRET}:{api_key}".encode("utf-8")
    return hmac.compare_digest(hashlib.sha256(raw).hexdigest(), stored_hash)


def _decode_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw)


def _hydrate_agent(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not row:
        return None
    row["metadata"] = _decode_json(row.pop("metadata_json", "{}"))
    row["attributes"] = _decode_json(row.pop("attributes_json", "{}"))
    return row


def _row_to_agent_dict(row: AgentRow) -> Dict[str, Any]:
    d = {
        "agent_id": row.agent_id,
        "name": row.name,
        "role": row.role,
        "api_key_hash": row.api_key_hash,
        "status": row.status,
        "status_reason": row.status_reason,
        "metadata_json": row.metadata_json,
        "attributes_json": row.attributes_json,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "last_login_at": row.last_login_at,
    }
    return _hydrate_agent(d)


def generate_agent_id() -> str:
    return f"agent_{secrets.token_hex(4)}"


def generate_api_key() -> str:
    return f"sk_agent_{secrets.token_urlsafe(24)}"


def generate_execution_id(prefix: str = "exec") -> str:
    return f"{prefix}_{secrets.token_hex(6)}"


def create_agent(
    name: str,
    role: str = "basic",
    metadata: Optional[Dict[str, Any]] = None,
    attributes: Optional[Dict[str, Any]] = None,
    allow_admin: bool = False,
    preset_agent_id: Optional[str] = None,
    preset_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    permission.validate_role(role, allow_admin=allow_admin)
    agent_id = preset_agent_id or generate_agent_id()
    api_key = preset_api_key or generate_api_key()
    now = database.utc_now()
    metadata = metadata or {}
    attributes = attributes or {}

    with SessionLocal() as db:
        try:
            row = AgentRow(
                agent_id=agent_id,
                name=name,
                role=role,
                api_key_hash=_hash_api_key(api_key),
                status="active",
                status_reason=None,
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                attributes_json=json.dumps(attributes, ensure_ascii=False),
                created_at=now,
                updated_at=now,
            )
            db.add(row)
            db.commit()
        except Exception as exc:
            db.rollback()
            if preset_agent_id:
                raise
            raise ValueError("Agent creation failed. Try again.") from exc

    agent = get_agent(agent_id)
    if not agent:
        raise RuntimeError("Agent creation succeeded but lookup failed.")
    agent["api_key"] = api_key
    return agent


def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    with SessionLocal() as db:
        row = db.get(AgentRow, agent_id)
    if not row:
        return None
    return _row_to_agent_dict(row)


def authenticate_agent(agent_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    agent = get_agent(agent_id)
    if not agent:
        return None
    if not _verify_api_key(api_key, agent["api_key_hash"]):
        return None
    return agent


def record_login(agent_id: str) -> None:
    now = database.utc_now()
    with SessionLocal() as db:
        row = db.get(AgentRow, agent_id)
        if row:
            row.last_login_at = now
            row.updated_at = now
            db.commit()


def list_agents() -> List[Dict[str, Any]]:
    with SessionLocal() as db:
        rows = db.execute(
            select(
                AgentRow.agent_id, AgentRow.name, AgentRow.role,
                AgentRow.status, AgentRow.status_reason,
                AgentRow.created_at, AgentRow.last_login_at,
            ).order_by(AgentRow.created_at)
        ).all()
    return [
        {
            "agent_id": r[0], "name": r[1], "role": r[2],
            "status": r[3], "status_reason": r[4],
            "created_at": r[5], "last_login_at": r[6],
        }
        for r in rows
    ]


def update_status(agent_id: str, status: str, reason: str) -> Dict[str, Any]:
    permission.validate_status(status)
    now = database.utc_now()
    with SessionLocal() as db:
        row = db.get(AgentRow, agent_id)
        if not row:
            raise ValueError("Agent not found.")
        row.status = status
        row.status_reason = reason
        row.updated_at = now
        db.commit()

    return {
        "agent_id": agent_id,
        "status": status,
        "status_reason": reason,
        "updated_at": now,
    }


def update_agent(
    agent_id: str,
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent = get_agent(agent_id)
    if not agent:
        raise ValueError("Agent not found.")

    now = database.utc_now()
    with SessionLocal() as db:
        row = db.get(AgentRow, agent_id)
        if not row:
            raise ValueError("Agent not found.")

        if name is not None:
            row.name = name
        if attributes is not None:
            merged = {**agent.get("attributes", {}), **attributes}
            row.attributes_json = json.dumps(merged, ensure_ascii=False)
        row.updated_at = now
        db.commit()

    updated = get_agent(agent_id)
    return {
        "agent_id": updated["agent_id"],
        "name": updated["name"],
        "attributes": updated["attributes"],
        "updated_at": now,
    }


def delete_agent(agent_id: str) -> bool:
    agent = get_agent(agent_id)
    if not agent:
        raise ValueError("Agent not found.")

    with SessionLocal() as db:
        db.execute(delete(IssuedTokenRow).where(IssuedTokenRow.agent_id == agent_id))
        row = db.get(AgentRow, agent_id)
        if row:
            db.delete(row)
        db.commit()
    return True


def ensure_demo_agents() -> None:
    for item in settings.DEMO_AGENTS:
        if get_agent(item["agent_id"]):
            continue
        create_agent(
            name=item["name"],
            role=item["role"],
            attributes=item.get("attributes", {}),
            allow_admin=True,
            preset_agent_id=item["agent_id"],
            preset_api_key=item["api_key"],
        )


def sync_demo_agents(reset_state: bool = False) -> None:
    now = database.utc_now()
    max_retries = 3
    retry_count = 0
    valid_ids = {item["agent_id"] for item in settings.DEMO_AGENTS}

    while retry_count < max_retries:
        try:
            with SessionLocal() as db:
                for item in settings.DEMO_AGENTS:
                    existing = db.get(AgentRow, item["agent_id"])
                    attrs_json = json.dumps(item.get("attributes", {}), ensure_ascii=False)
                    if existing:
                        existing.name = item["name"]
                        existing.role = item["role"]
                        existing.api_key_hash = _hash_api_key(item["api_key"])
                        existing.attributes_json = attrs_json
                        existing.updated_at = now
                        if reset_state:
                            existing.status = "active"
                            existing.status_reason = None
                            existing.last_login_at = None
                    else:
                        db.add(AgentRow(
                            agent_id=item["agent_id"],
                            name=item["name"],
                            role=item["role"],
                            api_key_hash=_hash_api_key(item["api_key"]),
                            status="active",
                            status_reason=None,
                            metadata_json="{}",
                            attributes_json=attrs_json,
                            created_at=now,
                            updated_at=now,
                            last_login_at=None,
                        ))
                db.query(AgentRow).filter(~AgentRow.agent_id.in_(valid_ids)).delete(synchronize_session=False)
                db.commit()
            return
        except Exception:
            retry_count += 1
            if retry_count >= max_retries:
                import traceback
                traceback.print_exc()
                return
            import time
            time.sleep(0.5)
