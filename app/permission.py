from __future__ import annotations


ROLE_PERMISSIONS = {
    "basic": ["read_doc"],
    "editor": ["read_doc", "write_doc"],
    "operator": ["read_doc", "write_doc", "execute_task", "call_api", "delegate_task"],
    "admin": [
        "read_doc",
        "write_doc",
        "execute_task",
        "call_api",
        "delegate_task",
        "view_audit",
        "manage_agents",
    ],
}

SELF_REGISTER_ROLES = {"basic", "editor", "operator"}
VALID_AGENT_STATUSES = {"active", "suspended", "disabled"}
VALID_SENSITIVITIES = {"public", "internal", "confidential"}


def check_permission(role: str, action: str) -> bool:
    if action in ROLE_PERMISSIONS.get(role, []):
        return True
    action_parts = action.split(":")
    if len(action_parts) >= 2:
        underscore_action = "_".join(action_parts[:2])
        if underscore_action in ROLE_PERMISSIONS.get(role, []):
            return True
    for perm in ROLE_PERMISSIONS.get(role, []):
        if action.startswith(perm.replace("_", ":")):
            return True
    return False


def list_permissions(role: str) -> list[str]:
    return ROLE_PERMISSIONS.get(role, []).copy()


def validate_role(role: str, allow_admin: bool = False) -> None:
    if role not in ROLE_PERMISSIONS:
        raise ValueError(f"Unknown role: {role}")
    if role == "admin" and not allow_admin:
        raise ValueError("Admin role cannot be requested from the public register endpoint.")


def validate_status(status: str) -> None:
    if status not in VALID_AGENT_STATUSES:
        raise ValueError(f"Unsupported agent status: {status}")


def validate_sensitivity(sensitivity: str) -> None:
    if sensitivity not in VALID_SENSITIVITIES:
        raise ValueError(f"Unsupported document sensitivity: {sensitivity}")
