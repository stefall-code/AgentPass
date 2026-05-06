"""
OWASP Agentic Top 10 Shield — ASI04/05/06/08

Addresses 4 missing OWASP risks:
  ASI04 Agentic Supply Chain   — Tool/model source verification + integrity checking
  ASI05 Unexpected Code Exec   — CodeShield: detect and sandbox dangerous code patterns
  ASI06 Memory Poisoning       — Memory integrity verification + tamper detection
  ASI08 Cascading Failures     — Cross-agent isolation + circuit breaker + blast radius containment
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("agent_system")

_STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "owasp_state.json")


# ============================================================================
# ASI04: Agentic Supply Chain — Tool/Model Source Verification
# ============================================================================

_TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {}
_SUPPLY_CHAIN_LOG: List[Dict[str, Any]] = []


def register_tool(
    tool_id: str,
    name: str,
    version: str,
    source: str,
    provider: str,
    capabilities: List[str],
    integrity_hash: Optional[str] = None,
) -> Dict[str, Any]:
    content = f"{tool_id}:{name}:{version}:{source}:{provider}:{','.join(sorted(capabilities))}"
    computed_hash = hashlib.sha256(content.encode()).hexdigest()

    if integrity_hash and integrity_hash != computed_hash:
        _log_supply_chain("tool_register_tampered", tool_id, f"hash_mismatch:expected={integrity_hash[:16]},got={computed_hash[:16]}")
        return {"registered": False, "reason": "Integrity hash mismatch — tool may have been tampered with"}

    entry = {
        "tool_id": tool_id,
        "name": name,
        "version": version,
        "source": source,
        "provider": provider,
        "capabilities": capabilities,
        "integrity_hash": computed_hash,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "verified": True,
        "trust_level": _compute_tool_trust(source, provider),
    }
    _TOOL_REGISTRY[tool_id] = entry
    _log_supply_chain("tool_registered", tool_id, f"trust={entry['trust_level']}")
    return {"registered": True, "tool_id": tool_id, "trust_level": entry["trust_level"], "hash": computed_hash[:16]}


def verify_tool_access(agent_id: str, tool_id: str, action: str) -> Dict[str, Any]:
    tool = _TOOL_REGISTRY.get(tool_id)
    if not tool:
        _log_supply_chain("tool_access_denied", tool_id, "not_registered")
        return {"allowed": False, "reason": f"Tool '{tool_id}' not registered in supply chain"}

    if not tool.get("verified"):
        _log_supply_chain("tool_access_denied", tool_id, "not_verified")
        return {"allowed": False, "reason": f"Tool '{tool_id}' integrity not verified"}

    if action not in tool.get("capabilities", []):
        _log_supply_chain("tool_access_denied", tool_id, f"no_capability:{action}")
        return {"allowed": False, "reason": f"Tool '{tool_id}' does not support '{action}'"}

    trust = tool.get("trust_level", "untrusted")
    if trust == "untrusted":
        _log_supply_chain("tool_access_denied", tool_id, "untrusted_source")
        return {"allowed": False, "reason": f"Tool '{tool_id}' from untrusted source '{tool['source']}'"}

    current_hash = hashlib.sha256(
        f"{tool_id}:{tool['name']}:{tool['version']}:{tool['source']}:{tool['provider']}:{','.join(sorted(tool['capabilities']))}".encode()
    ).hexdigest()
    if current_hash != tool["integrity_hash"]:
        _log_supply_chain("tool_access_denied", tool_id, "integrity_violation")
        return {"allowed": False, "reason": f"Tool '{tool_id}' integrity check FAILED — may have been tampered with"}

    _log_supply_chain("tool_access_granted", tool_id, f"action={action},trust={trust}")
    return {"allowed": True, "tool_id": tool_id, "trust_level": trust, "action": action}


def _compute_tool_trust(source: str, provider: str) -> str:
    trusted_sources = {"official", "internal", "verified_marketplace"}
    trusted_providers = {"agent-iam", "anthropic", "openai", "google", "microsoft"}

    if source.lower() in trusted_sources and provider.lower() in trusted_providers:
        return "high"
    elif source.lower() in trusted_sources or provider.lower() in trusted_providers:
        return "medium"
    else:
        return "untrusted"


def get_supply_chain_status() -> Dict[str, Any]:
    return {
        "registered_tools": len(_TOOL_REGISTRY),
        "tools": {tid: {"name": t["name"], "trust": t["trust_level"], "verified": t["verified"]} for tid, t in _TOOL_REGISTRY.items()},
        "log_entries": len(_SUPPLY_CHAIN_LOG),
    }


# ============================================================================
# ASI05: Unexpected Code Execution — CodeShield
# ============================================================================

_DANGEROUS_PATTERNS = [
    (r"\bos\.system\s*\(", "os_system", "CRITICAL", "Direct OS command execution"),
    (r"\bsubprocess\.(run|call|Popen|check_output)\s*\(", "subprocess", "CRITICAL", "Subprocess execution"),
    (r"\bexec\s*\(", "exec", "CRITICAL", "Dynamic code execution via exec()"),
    (r"\beval\s*\(", "eval", "HIGH", "Dynamic code evaluation via eval()"),
    (r"\b__import__\s*\(", "import", "HIGH", "Dynamic module import"),
    (r"\bopen\s*\(.+[\"']w", "file_write", "HIGH", "File write operation"),
    (r"\brm\s+-rf", "rm_rf", "CRITICAL", "Destructive file deletion"),
    (r"\bshutil\.rmtree", "rmtree", "CRITICAL", "Directory tree deletion"),
    (r"\bsocket\s*\(", "socket", "MEDIUM", "Network socket creation"),
    (r"\brequests\.(get|post|put|delete|patch)\s*\(", "http_request", "MEDIUM", "Outbound HTTP request"),
    (r"\bhttpx\.(get|post|put|delete|patch)\s*\(", "httpx_request", "MEDIUM", "Outbound HTTPX request"),
    (r"\bimport\s+pickle", "pickle", "HIGH", "Pickle deserialization (RCE risk)"),
    (r"\bpickle\.loads?", "pickle_load", "CRITICAL", "Pickle deserialization execution"),
    (r"\bbase64\.b64decode", "base64_decode", "MEDIUM", "Base64 decode (obfuscation indicator)"),
    (r"\\x[0-9a-fA-F]{2}", "hex_escape", "LOW", "Hex escape sequences (obfuscation)"),
    (r"\bctypes\s*\.", "ctypes", "HIGH", "Low-level FFI access"),
    (r"\bcompile\s*\(", "compile", "MEDIUM", "Runtime code compilation"),
    (r"\binput\s*\(", "input_func", "LOW", "User input reading"),
]

_SANDBOX_POLICIES = {
    "strict": {"allow": [], "deny": ["*"]},
    "standard": {"allow": ["http_request", "httpx_request", "base64_decode", "input_func"], "deny": ["os_system", "subprocess", "exec", "eval", "rm_rf", "rmtree", "pickle", "pickle_load", "ctypes"]},
    "permissive": {"allow": ["http_request", "httpx_request", "base64_decode", "input_func", "file_write"], "deny": ["os_system", "rm_rf", "rmtree", "pickle_load", "ctypes"]},
}

_CODE_SHIELD_LOG: List[Dict[str, Any]] = []


@dataclass
class CodeShieldResult:
    safe: bool
    risk_score: float
    threats: List[Dict[str, Any]]
    action: str
    sandbox_level: str
    masked_code: str = ""


def scan_code(code: str, sandbox_level: str = "standard") -> CodeShieldResult:
    threats = []
    for pattern, threat_id, severity, description in _DANGEROUS_PATTERNS:
        if re.search(pattern, code):
            threats.append({"id": threat_id, "severity": severity, "description": description, "pattern": pattern})

    policy = _SANDBOX_POLICIES.get(sandbox_level, _SANDBOX_POLICIES["standard"])
    blocked = []
    for threat in threats:
        if threat["id"] in policy["deny"] or "*" in policy["deny"]:
            if threat["id"] not in policy.get("allow", []):
                blocked.append(threat)

    risk_score = 0.0
    severity_weights = {"CRITICAL": 0.4, "HIGH": 0.25, "MEDIUM": 0.15, "LOW": 0.05}
    for threat in threats:
        risk_score += severity_weights.get(threat["severity"], 0.1)
    risk_score = min(1.0, risk_score)

    safe = len(blocked) == 0 and risk_score < 0.5
    if risk_score >= 0.7:
        action = "block"
    elif risk_score >= 0.4:
        action = "sandbox"
    else:
        action = "allow"

    masked_code = code
    for threat in blocked:
        masked_code = re.sub(
            threat["pattern"],
            f"/* BLOCKED:{threat['id']} */",
            masked_code,
        )

    result = CodeShieldResult(
        safe=safe,
        risk_score=risk_score,
        threats=threats,
        action=action,
        sandbox_level=sandbox_level,
        masked_code=masked_code,
    )

    _CODE_SHIELD_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "safe": safe,
        "risk_score": risk_score,
        "action": action,
        "threats": [t["id"] for t in threats],
        "blocked": [t["id"] for t in blocked],
    })
    if len(_CODE_SHIELD_LOG) > 200:
        _CODE_SHIELD_LOG.pop(0)

    return result


def get_codeshield_stats() -> Dict[str, Any]:
    return {
        "total_scans": len(_CODE_SHIELD_LOG),
        "blocked": sum(1 for l in _CODE_SHIELD_LOG if l["action"] == "block"),
        "sandboxed": sum(1 for l in _CODE_SHIELD_LOG if l["action"] == "sandbox"),
        "allowed": sum(1 for l in _CODE_SHIELD_LOG if l["action"] == "allow"),
        "patterns": len(_DANGEROUS_PATTERNS),
        "sandbox_policies": list(_SANDBOX_POLICIES.keys()),
    }


# ============================================================================
# ASI06: Memory Poisoning — Memory Integrity Verification
# ============================================================================

_MEMORY_STORE: Dict[str, Dict[str, Any]] = {}
_MEMORY_HASH_CHAIN: List[str] = []
_MEMORY_ACCESS_LOG: List[Dict[str, Any]] = []


def write_memory(agent_id: str, key: str, value: str, scope: str = "private") -> Dict[str, Any]:
    mem_key = f"{agent_id}:{key}"

    content_hash = hashlib.sha256(value.encode()).hexdigest()
    chain_hash = hashlib.sha256(
        (_MEMORY_HASH_CHAIN[-1] + content_hash).encode()
    ).hexdigest() if _MEMORY_HASH_CHAIN else content_hash
    _MEMORY_HASH_CHAIN.append(chain_hash)

    entry = {
        "agent_id": agent_id,
        "key": key,
        "value": value,
        "scope": scope,
        "content_hash": content_hash,
        "chain_hash": chain_hash,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "read_count": 0,
        "last_read_at": None,
    }
    _MEMORY_STORE[mem_key] = entry

    _log_memory("write", agent_id, key, scope)
    return {"written": True, "key": key, "hash": content_hash[:16], "scope": scope}


def read_memory(agent_id: str, key: str, requesting_agent: str = "") -> Dict[str, Any]:
    mem_key = f"{agent_id}:{key}"
    entry = _MEMORY_STORE.get(mem_key)

    if not entry:
        _log_memory("read_miss", agent_id, key, requesting_agent)
        return {"found": False, "reason": "Memory key not found"}

    if entry["scope"] == "private" and requesting_agent and requesting_agent != agent_id:
        _log_memory("read_denied", agent_id, key, requesting_agent)
        return {"found": False, "reason": "Private memory — access denied for other agents"}

    current_hash = hashlib.sha256(entry["value"].encode()).hexdigest()
    if current_hash != entry["content_hash"]:
        _log_memory("read_tampered", agent_id, key, requesting_agent)
        return {"found": False, "reason": "MEMORY INTEGRITY VIOLATION — content has been tampered with!", "tampered": True}

    entry["read_count"] += 1
    entry["last_read_at"] = datetime.now(timezone.utc).isoformat()

    _log_memory("read_ok", agent_id, key, requesting_agent)
    return {"found": True, "value": entry["value"], "hash": current_hash[:16], "reads": entry["read_count"]}


def verify_memory_integrity() -> Dict[str, Any]:
    tampered = []
    verified = []
    for mem_key, entry in _MEMORY_STORE.items():
        current_hash = hashlib.sha256(entry["value"].encode()).hexdigest()
        if current_hash != entry["content_hash"]:
            tampered.append({"key": mem_key, "expected": entry["content_hash"][:16], "actual": current_hash[:16]})
        else:
            verified.append(mem_key)

    chain_valid = True
    sorted_entries = sorted(_MEMORY_STORE.values(), key=lambda e: e.get("written_at", ""))
    if len(sorted_entries) != len(_MEMORY_HASH_CHAIN):
        chain_valid = False
    else:
        for i in range(len(_MEMORY_HASH_CHAIN)):
            entry = sorted_entries[i]
            if i == 0:
                expected = entry["content_hash"]
            else:
                expected = hashlib.sha256(
                    (_MEMORY_HASH_CHAIN[i - 1] + entry["content_hash"]).encode()
                ).hexdigest()
            if _MEMORY_HASH_CHAIN[i] != expected:
                chain_valid = False
                break

    return {
        "total_entries": len(_MEMORY_STORE),
        "verified": len(verified),
        "tampered": len(tampered),
        "tampered_keys": tampered,
        "chain_valid": chain_valid,
        "chain_length": len(_MEMORY_HASH_CHAIN),
    }


def poison_memory(agent_id: str, key: str, poisoned_value: str) -> Dict[str, Any]:
    mem_key = f"{agent_id}:{key}"
    entry = _MEMORY_STORE.get(mem_key)
    if not entry:
        return {"poisoned": False, "reason": "Key not found"}

    original_hash = entry["content_hash"]
    entry["value"] = poisoned_value

    current_hash = hashlib.sha256(poisoned_value.encode()).hexdigest()
    integrity_ok = current_hash == original_hash

    _log_memory("poison_attempt", agent_id, key, "")
    return {
        "poisoned": True,
        "key": key,
        "original_hash": original_hash[:16],
        "current_hash": current_hash[:16],
        "integrity_violation": not integrity_ok,
        "message": "Memory value changed but integrity check will detect the tampering" if not integrity_ok else "No change detected",
    }


def get_memory_stats() -> Dict[str, Any]:
    return {
        "total_entries": len(_MEMORY_STORE),
        "chain_length": len(_MEMORY_HASH_CHAIN),
        "access_log_entries": len(_MEMORY_ACCESS_LOG),
        "scopes": list(set(e["scope"] for e in _MEMORY_STORE.values())),
    }


def _log_memory(action: str, agent_id: str, key: str, requesting_agent: str) -> None:
    _MEMORY_ACCESS_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent_id": agent_id,
        "key": key,
        "requesting_agent": requesting_agent,
    })
    if len(_MEMORY_ACCESS_LOG) > 200:
        _MEMORY_ACCESS_LOG.pop(0)


# ============================================================================
# ASI08: Cascading Failures — Cross-Agent Isolation + Circuit Breaker
# ============================================================================

_AGENT_HEALTH: Dict[str, Dict[str, Any]] = {}
_CIRCUIT_BREAKERS: Dict[str, Dict[str, Any]] = {}
_ISOLATION_ZONES: Dict[str, Set[str]] = {}
_FAILURE_EVENTS: List[Dict[str, Any]] = []
_CASCADE_LOG: List[Dict[str, Any]] = []


def register_agent_zone(agent_id: str, zone: str = "default") -> Dict[str, Any]:
    if zone not in _ISOLATION_ZONES:
        _ISOLATION_ZONES[zone] = set()
    _ISOLATION_ZONES[zone].add(agent_id)

    _AGENT_HEALTH[agent_id] = {
        "status": "healthy",
        "zone": zone,
        "failure_count": 0,
        "last_failure": None,
        "circuit_open": False,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"agent_id": agent_id, "zone": zone, "status": "healthy"}


def report_failure(agent_id: str, error_type: str, severity: str = "medium") -> Dict[str, Any]:
    if agent_id not in _AGENT_HEALTH:
        register_agent_zone(agent_id)

    health = _AGENT_HEALTH[agent_id]
    health["failure_count"] += 1
    health["last_failure"] = datetime.now(timezone.utc).isoformat()

    severity_thresholds = {"low": 5, "medium": 3, "high": 2, "critical": 1}
    threshold = severity_thresholds.get(severity, 3)

    if health["failure_count"] >= threshold:
        health["status"] = "degraded"
        health["circuit_open"] = True
        _CIRCUIT_BREAKERS[agent_id] = {
            "state": "open",
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "failure_count": health["failure_count"],
            "severity": severity,
        }
        _log_cascade("circuit_opened", agent_id, f"failures={health['failure_count']},severity={severity}")

        zone = health.get("zone", "default")
        _isolate_zone(zone, agent_id, severity)

    _FAILURE_EVENTS.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "error_type": error_type,
        "severity": severity,
        "failure_count": health["failure_count"],
        "circuit_open": health["circuit_open"],
    })
    if len(_FAILURE_EVENTS) > 200:
        _FAILURE_EVENTS.pop(0)

    return {
        "agent_id": agent_id,
        "failure_count": health["failure_count"],
        "status": health["status"],
        "circuit_open": health["circuit_open"],
        "zone_isolated": health["circuit_open"],
    }


def check_agent_available(agent_id: str) -> Dict[str, Any]:
    health = _AGENT_HEALTH.get(agent_id)
    if not health:
        return {"available": True, "reason": "not_registered"}

    if health.get("circuit_open"):
        return {"available": False, "reason": "circuit_breaker_open", "status": health["status"]}

    return {"available": True, "status": health["status"], "failures": health["failure_count"]}


def check_cross_agent_call(source_agent: str, target_agent: str) -> Dict[str, Any]:
    source_health = _AGENT_HEALTH.get(source_agent)
    target_health = _AGENT_HEALTH.get(target_agent)

    if source_health and source_health.get("circuit_open"):
        return {"allowed": False, "reason": f"Source agent '{source_agent}' circuit breaker is open"}

    if target_health and target_health.get("circuit_open"):
        return {"allowed": False, "reason": f"Target agent '{target_agent}' circuit breaker is open"}

    source_zone = source_health.get("zone", "default") if source_health else "default"
    target_zone = target_health.get("zone", "default") if target_health else "default"

    if source_zone != target_zone:
        cross_zone_policy = _get_cross_zone_policy(source_zone, target_zone)
        if not cross_zone_policy.get("allowed", False):
            return {"allowed": False, "reason": f"Cross-zone call blocked: {source_zone} → {target_zone}"}

    return {"allowed": True, "source_zone": source_zone, "target_zone": target_zone}


def _isolate_zone(zone: str, failed_agent: str, severity: str) -> None:
    agents_in_zone = _ISOLATION_ZONES.get(zone, set())

    if severity in ("high", "critical"):
        for agent_id in agents_in_zone:
            if agent_id != failed_agent:
                health = _AGENT_HEALTH.get(agent_id)
                if health and not health.get("circuit_open"):
                    health["status"] = "isolated"
                    _log_cascade("agent_isolated", agent_id, f"zone={zone},triggered_by={failed_agent}")

    _log_cascade("zone_isolated", failed_agent, f"zone={zone},agents_affected={len(agents_in_zone)},severity={severity}")


def _get_cross_zone_policy(source_zone: str, target_zone: str) -> Dict[str, Any]:
    restricted_zones = {"quarantine", "isolated"}
    if source_zone in restricted_zones or target_zone in restricted_zones:
        return {"allowed": False, "reason": "Restricted zone"}
    return {"allowed": True}


def reset_circuit_breaker(agent_id: str) -> Dict[str, Any]:
    health = _AGENT_HEALTH.get(agent_id)
    if not health:
        return {"reset": False, "reason": "Agent not registered"}

    health["circuit_open"] = False
    health["status"] = "healthy"
    health["failure_count"] = 0
    if agent_id in _CIRCUIT_BREAKERS:
        del _CIRCUIT_BREAKERS[agent_id]
    _log_cascade("circuit_reset", agent_id, "manual_reset")
    return {"reset": True, "agent_id": agent_id}


def get_cascade_status() -> Dict[str, Any]:
    return {
        "agent_health": {aid: {"status": h["status"], "zone": h.get("zone"), "failures": h["failure_count"], "circuit_open": h["circuit_open"]} for aid, h in _AGENT_HEALTH.items()},
        "isolation_zones": {z: list(agents) for z, agents in _ISOLATION_ZONES.items()},
        "open_circuits": list(_CIRCUIT_BREAKERS.keys()),
        "failure_events": len(_FAILURE_EVENTS),
    }


def _log_supply_chain(action: str, tool_id: str, detail: str) -> None:
    _SUPPLY_CHAIN_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "tool_id": tool_id,
        "detail": detail,
    })
    if len(_SUPPLY_CHAIN_LOG) > 200:
        _SUPPLY_CHAIN_LOG.pop(0)


def _log_cascade(action: str, agent_id: str, detail: str) -> None:
    _CASCADE_LOG.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "agent_id": agent_id,
        "detail": detail,
    })
    if len(_CASCADE_LOG) > 200:
        _CASCADE_LOG.pop(0)


def get_owasp_status() -> Dict[str, Any]:
    return {
        "ASI04_supply_chain": {
            "registered_tools": len(_TOOL_REGISTRY),
            "log_entries": len(_SUPPLY_CHAIN_LOG),
        },
        "ASI05_code_shield": get_codeshield_stats(),
        "ASI06_memory_integrity": get_memory_stats(),
        "ASI08_cascade_protection": {
            "agents_monitored": len(_AGENT_HEALTH),
            "open_circuits": len(_CIRCUIT_BREAKERS),
            "isolation_zones": len(_ISOLATION_ZONES),
        },
        "ASI09_wallet_guard": get_wallet_stats(),
    }


# ============================================================================
# ASI09: Denial of Wallet — Cost Tracking + Budget Limits + Auto-Throttle
# ============================================================================

_AGENT_BUDGETS: Dict[str, Dict[str, Any]] = {}
_COST_RECORDS: List[Dict[str, Any]] = []
_GLOBAL_BUDGET: Dict[str, Any] = {
    "total_budget": 100.0,
    "total_spent": 0.0,
    "alert_threshold": 0.8,
    "hard_limit": 0.95,
    "currency": "USD",
}
_SPEND_ALERTS: List[Dict[str, Any]] = []

_MODEL_COSTS: Dict[str, float] = {
    "gpt-4o": 0.005,
    "gpt-4o-mini": 0.00015,
    "claude-3.5-sonnet": 0.003,
    "claude-3-haiku": 0.00025,
    "gemini-1.5-pro": 0.0035,
    "gemini-1.5-flash": 0.000075,
    "qwen-max": 0.002,
    "qwen-turbo": 0.0002,
    "deepseek-chat": 0.00014,
    "deepseek-reasoner": 0.00055,
    "doubao-pro": 0.0005,
    "kimi": 0.001,
    "default": 0.001,
}

_TOOL_COSTS: Dict[str, float] = {
    "feishu_api_call": 0.0001,
    "bitable_query": 0.0002,
    "web_search": 0.001,
    "document_create": 0.0005,
    "default": 0.0005,
}


def set_agent_budget(agent_id: str, daily_budget: float, monthly_budget: float = 0) -> Dict[str, Any]:
    if monthly_budget <= 0:
        monthly_budget = daily_budget * 30
    _AGENT_BUDGETS[agent_id] = {
        "agent_id": agent_id,
        "daily_budget": daily_budget,
        "monthly_budget": monthly_budget,
        "daily_spent": 0.0,
        "monthly_spent": 0.0,
        "request_count": 0,
        "token_count": 0,
        "last_reset": datetime.now(timezone.utc).isoformat(),
        "throttled": False,
        "blocked": False,
    }
    return {"agent_id": agent_id, "daily_budget": daily_budget, "monthly_budget": monthly_budget}


def record_cost(
    agent_id: str,
    model: str = "default",
    tool: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    request_type: str = "inference",
) -> Dict[str, Any]:
    model_cost = _MODEL_COSTS.get(model, _MODEL_COSTS["default"])
    tool_cost = _TOOL_COSTS.get(tool, 0) if tool else 0

    token_cost = model_cost * (input_tokens + output_tokens) / 1000.0
    total_cost = token_cost + tool_cost

    if agent_id not in _AGENT_BUDGETS:
        set_agent_budget(agent_id, daily_budget=10.0)

    budget = _AGENT_BUDGETS[agent_id]
    budget["daily_spent"] += total_cost
    budget["monthly_spent"] += total_cost
    budget["request_count"] += 1
    budget["token_count"] += input_tokens + output_tokens

    _GLOBAL_BUDGET["total_spent"] += total_cost

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agent_id": agent_id,
        "model": model,
        "tool": tool,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": round(total_cost, 6),
        "daily_spent": round(budget["daily_spent"], 4),
        "daily_budget": budget["daily_budget"],
        "daily_usage": round(budget["daily_spent"] / budget["daily_budget"], 4) if budget["daily_budget"] > 0 else 0,
    }
    _COST_RECORDS.append(record)
    if len(_COST_RECORDS) > 500:
        _COST_RECORDS.pop(0)

    action = "allow"
    if budget["daily_spent"] >= budget["daily_budget"] * _GLOBAL_BUDGET["hard_limit"]:
        action = "block"
        budget["blocked"] = True
        _SPEND_ALERTS.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "level": "critical",
            "message": f"Agent '{agent_id}' exceeded daily budget hard limit ({budget['daily_spent']:.2f}/{budget['daily_budget']:.2f})",
            "action": "blocked",
        })
    elif budget["daily_spent"] >= budget["daily_budget"] * _GLOBAL_BUDGET["alert_threshold"]:
        action = "throttle"
        budget["throttled"] = True
        _SPEND_ALERTS.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": agent_id,
            "level": "warning",
            "message": f"Agent '{agent_id}' approaching daily budget limit ({budget['daily_spent']:.2f}/{budget['daily_budget']:.2f})",
            "action": "throttled",
        })

    if _GLOBAL_BUDGET["total_spent"] >= _GLOBAL_BUDGET["total_budget"] * _GLOBAL_BUDGET["hard_limit"]:
        _SPEND_ALERTS.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_id": "GLOBAL",
            "level": "critical",
            "message": f"Global budget hard limit reached ({_GLOBAL_BUDGET['total_spent']:.2f}/{_GLOBAL_BUDGET['total_budget']:.2f})",
            "action": "global_block",
        })

    if len(_SPEND_ALERTS) > 100:
        _SPEND_ALERTS.pop(0)

    return {
        "cost": round(total_cost, 6),
        "daily_spent": round(budget["daily_spent"], 4),
        "daily_budget": budget["daily_budget"],
        "daily_usage_pct": round(budget["daily_spent"] / budget["daily_budget"] * 100, 1) if budget["daily_budget"] > 0 else 0,
        "action": action,
        "throttled": budget["throttled"],
        "blocked": budget["blocked"],
    }


def check_budget(agent_id: str) -> Dict[str, Any]:
    budget = _AGENT_BUDGETS.get(agent_id)
    if not budget:
        return {"has_budget": False, "allowed": True}

    daily_usage = budget["daily_spent"] / budget["daily_budget"] if budget["daily_budget"] > 0 else 0
    monthly_usage = budget["monthly_spent"] / budget["monthly_budget"] if budget["monthly_budget"] > 0 else 0

    if budget["blocked"] or daily_usage >= _GLOBAL_BUDGET["hard_limit"]:
        return {
            "has_budget": True,
            "allowed": False,
            "reason": "Daily budget hard limit exceeded — requests blocked",
            "daily_usage_pct": round(daily_usage * 100, 1),
            "action": "block",
        }

    if budget["throttled"] or daily_usage >= _GLOBAL_BUDGET["alert_threshold"]:
        return {
            "has_budget": True,
            "allowed": True,
            "reason": "Approaching daily budget — requests throttled",
            "daily_usage_pct": round(daily_usage * 100, 1),
            "action": "throttle",
            "throttle_delay_ms": int(500 * (daily_usage / _GLOBAL_BUDGET["alert_threshold"])),
        }

    return {
        "has_budget": True,
        "allowed": True,
        "daily_usage_pct": round(daily_usage * 100, 1),
        "monthly_usage_pct": round(monthly_usage * 100, 1),
        "action": "allow",
    }


def save_state():
    state = {
        "agent_budgets": _AGENT_BUDGETS,
        "global_budget": _GLOBAL_BUDGET,
        "circuit_breakers": _CIRCUIT_BREAKERS,
        "agent_health": _AGENT_HEALTH,
        "tool_registry": _TOOL_REGISTRY,
        "model_registry": _MODEL_REGISTRY,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with open(_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        logger.info("OWASP state saved to %s", _STATE_FILE)
    except Exception as e:
        logger.warning("Failed to save OWASP state: %s", e)


def load_state():
    global _AGENT_BUDGETS, _GLOBAL_BUDGET, _CIRCUIT_BREAKERS, _AGENT_HEALTH, _TOOL_REGISTRY, _MODEL_REGISTRY
    if not os.path.exists(_STATE_FILE):
        return
    try:
        with open(_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        _AGENT_BUDGETS.update(state.get("agent_budgets", {}))
        _GLOBAL_BUDGET.update(state.get("global_budget", {}))
        _CIRCUIT_BREAKERS.update(state.get("circuit_breakers", {}))
        _AGENT_HEALTH.update(state.get("agent_health", {}))
        _TOOL_REGISTRY.update(state.get("tool_registry", {}))
        _MODEL_REGISTRY.update(state.get("model_registry", {}))
        logger.info("OWASP state loaded from %s", _STATE_FILE)
    except Exception as e:
        logger.warning("Failed to load OWASP state: %s", e)


def get_cost_report(agent_id: Optional[str] = None) -> Dict[str, Any]:
    if agent_id:
        records = [r for r in _COST_RECORDS if r["agent_id"] == agent_id]
        budget = _AGENT_BUDGETS.get(agent_id, {})
        return {
            "agent_id": agent_id,
            "total_cost": round(sum(r["cost"] for r in records), 4),
            "request_count": len(records),
            "token_count": budget.get("token_count", 0),
            "daily_spent": round(budget.get("daily_spent", 0), 4),
            "daily_budget": budget.get("daily_budget", 0),
            "daily_usage_pct": round(budget.get("daily_spent", 0) / budget.get("daily_budget", 1) * 100, 1),
            "throttled": budget.get("throttled", False),
            "blocked": budget.get("blocked", False),
        }

    agent_summaries = {}
    for aid, budget in _AGENT_BUDGETS.items():
        agent_summaries[aid] = {
            "daily_spent": round(budget["daily_spent"], 4),
            "daily_budget": budget["daily_budget"],
            "daily_usage_pct": round(budget["daily_spent"] / budget["daily_budget"] * 100, 1) if budget["daily_budget"] > 0 else 0,
            "throttled": budget["throttled"],
            "blocked": budget["blocked"],
        }

    return {
        "global_budget": _GLOBAL_BUDGET["total_budget"],
        "global_spent": round(_GLOBAL_BUDGET["total_spent"], 4),
        "global_usage_pct": round(_GLOBAL_BUDGET["total_spent"] / _GLOBAL_BUDGET["total_budget"] * 100, 1),
        "agents": agent_summaries,
        "total_requests": len(_COST_RECORDS),
        "alerts": len(_SPEND_ALERTS),
    }


def reset_daily_budgets() -> int:
    count = 0
    for aid, budget in _AGENT_BUDGETS.items():
        budget["daily_spent"] = 0.0
        budget["throttled"] = False
        budget["blocked"] = False
        budget["last_reset"] = datetime.now(timezone.utc).isoformat()
        count += 1
    _GLOBAL_BUDGET["total_spent"] = 0.0
    return count


def get_wallet_stats() -> Dict[str, Any]:
    return {
        "agents_with_budget": len(_AGENT_BUDGETS),
        "total_cost_records": len(_COST_RECORDS),
        "global_spent": round(_GLOBAL_BUDGET["total_spent"], 4),
        "global_budget": _GLOBAL_BUDGET["total_budget"],
        "alerts_count": len(_SPEND_ALERTS),
        "model_costs": list(_MODEL_COSTS.keys()),
        "alert_threshold_pct": round(_GLOBAL_BUDGET["alert_threshold"] * 100),
        "hard_limit_pct": round(_GLOBAL_BUDGET["hard_limit"] * 100),
    }
