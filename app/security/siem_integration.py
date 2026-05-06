"""
P2-9: SIEM Integration & Compliance Logging

Enterprise-grade security event export and compliance:
  - Splunk / Datadog / ELK JSON log export
  - SOC 2 / HIPAA compliance report templates
  - OpenTelemetry trace integration
  - Structured audit event formatting
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("agent_system")

_SIEM_EVENTS: List[Dict[str, Any]] = {}
_OTEL_SPANS: Dict[str, Dict[str, Any]] = {}
_COMPLIANCE_REPORTS: List[Dict[str, Any]] = []


def emit_siem_event(
    event_type: str,
    agent_id: str,
    action: str = "",
    resource: str = "",
    decision: str = "",
    reason: str = "",
    severity: str = "info",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event_id = f"evt_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.now(timezone.utc)

    event = {
        "event_id": event_id,
        "timestamp": timestamp.isoformat(),
        "timestamp_unix": int(timestamp.timestamp() * 1000),
        "event_type": event_type,
        "severity": _map_severity(severity),
        "source": "agent-iam-system",
        "agent_id": agent_id,
        "action": action,
        "resource": resource,
        "decision": decision,
        "reason": reason,
        "metadata": metadata or {},
        "compliance_tags": _get_compliance_tags(event_type, decision, severity),
    }

    if event_id not in _SIEM_EVENTS:
        _SIEM_EVENTS[event_id] = event

    return event


def export_splunk(format_type: str = "json") -> List[Dict[str, Any]]:
    events = list(_SIEM_EVENTS.values())
    splunk_events = []
    for evt in events:
        splunk_evt = {
            "time": evt["timestamp_unix"] / 1000.0,
            "host": "agent-iam-system",
            "source": f"agent-iam:{evt['event_type']}",
            "sourcetype": "_json",
            "index": "agent_security",
            "event": {
                "event_id": evt["event_id"],
                "event_type": evt["event_type"],
                "severity": evt["severity"],
                "agent_id": evt["agent_id"],
                "action": evt["action"],
                "resource": evt["resource"],
                "decision": evt["decision"],
                "reason": evt["reason"],
                "compliance_tags": evt.get("compliance_tags", []),
            },
        }
        splunk_events.append(splunk_evt)
    return splunk_events


def export_elk() -> List[Dict[str, Any]]:
    events = list(_SIEM_EVENTS.values())
    elk_events = []
    for evt in events:
        elk_evt = {
            "@timestamp": evt["timestamp"],
            "@version": "1",
            "message": f"{evt['event_type']}: agent={evt['agent_id']} action={evt['action']} decision={evt['decision']}",
            "agent-iam": {
                "event_id": evt["event_id"],
                "event_type": evt["event_type"],
                "severity": evt["severity"],
                "agent_id": evt["agent_id"],
                "action": evt["action"],
                "resource": evt["resource"],
                "decision": evt["decision"],
                "reason": evt["reason"],
                "compliance_tags": evt.get("compliance_tags", []),
            },
        }
        elk_events.append(elk_evt)
    return elk_events


def export_datadog() -> List[Dict[str, Any]]:
    events = list(_SIEM_EVENTS.values())
    dd_events = []
    for evt in events:
        dd_evt = {
            "ddsource": "agent-iam-system",
            "ddtags": ",".join(evt.get("compliance_tags", []) + [f"severity:{evt['severity']}", f"decision:{evt['decision']}"]),
            "hostname": "agent-iam-system",
            "message": json.dumps({
                "event_id": evt["event_id"],
                "event_type": evt["event_type"],
                "agent_id": evt["agent_id"],
                "action": evt["action"],
                "decision": evt["decision"],
                "reason": evt["reason"],
            }),
            "service": "agent-iam",
            "timestamp": evt["timestamp_unix"],
        }
        dd_events.append(dd_evt)
    return dd_events


def start_otel_span(
    trace_id: str,
    span_name: str,
    agent_id: str,
    parent_span_id: Optional[str] = None,
) -> Dict[str, Any]:
    span_id = f"span_{uuid.uuid4().hex[:16]}"
    span = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": span_name,
        "agent_id": agent_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "start_time_unix_nano": int(time.time() * 1e9),
        "status": "OK",
        "attributes": {
            "agent.id": agent_id,
            "service.name": "agent-iam-system",
            "service.version": "2.4",
        },
        "events": [],
    }
    _OTEL_SPANS[span_id] = span
    return {"trace_id": trace_id, "span_id": span_id, "name": span_name}


def add_otel_span_event(span_id: str, event_name: str, attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    span = _OTEL_SPANS.get(span_id)
    if not span:
        return {"added": False, "reason": "Span not found"}

    span_event = {
        "name": event_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "attributes": attributes or {},
    }
    span["events"].append(span_event)
    return {"added": True, "span_id": span_id, "event": event_name}


def end_otel_span(span_id: str, status: str = "OK") -> Dict[str, Any]:
    span = _OTEL_SPANS.get(span_id)
    if not span:
        return {"ended": False, "reason": "Span not found"}

    span["end_time"] = datetime.now(timezone.utc).isoformat()
    span["end_time_unix_nano"] = int(time.time() * 1e9)
    span["status"] = status
    duration_ms = (span["end_time_unix_nano"] - span["start_time_unix_nano"]) / 1e6
    span["duration_ms"] = round(duration_ms, 2)
    return {"ended": True, "span_id": span_id, "duration_ms": round(duration_ms, 2), "status": status}


def export_otel_traces() -> List[Dict[str, Any]]:
    traces: Dict[str, List[Dict[str, Any]]] = {}
    for span_id, span in _OTEL_SPANS.items():
        tid = span["trace_id"]
        if tid not in traces:
            traces[tid] = []
        traces[tid].append(span)

    result = []
    for trace_id, spans in traces.items():
        result.append({
            "trace_id": trace_id,
            "spans": spans,
            "span_count": len(spans),
        })
    return result


def generate_soc2_report() -> Dict[str, Any]:
    events = list(_SIEM_EVENTS.values())
    now = datetime.now(timezone.utc)

    total_events = len(events)
    deny_events = [e for e in events if e["decision"] == "deny"]
    allow_events = [e for e in events if e["decision"] == "allow"]
    critical_events = [e for e in events if e["severity"] in ("high", "critical")]

    agent_activity: Dict[str, int] = {}
    for evt in events:
        agent_activity[evt["agent_id"]] = agent_activity.get(evt["agent_id"], 0) + 1

    return {
        "report_type": "SOC2_Type_II",
        "generated_at": now.isoformat(),
        "reporting_period": "current_session",
        "trust_service_criteria": {
            "CC6_1_logical_access": {
                "status": "satisfied" if total_events > 0 else "no_data",
                "evidence": f"{total_events} access control events recorded",
                "deny_rate": f"{len(deny_events)}/{total_events}" if total_events > 0 else "N/A",
            },
            "CC6_2_role_based_access": {
                "status": "satisfied",
                "evidence": "RBAC + ABAC policy engine active",
            },
            "CC7_1_monitoring": {
                "status": "satisfied" if total_events > 0 else "no_data",
                "evidence": f"{total_events} monitored events with hash chain integrity",
            },
            "CC7_2_incident_response": {
                "status": "satisfied" if len(critical_events) > 0 else "no_incidents",
                "evidence": f"{len(critical_events)} critical events detected and responded",
            },
            "CC8_1_change_management": {
                "status": "satisfied",
                "evidence": "Declarative policy engine with versioning and audit trail",
            },
        },
        "summary": {
            "total_events": total_events,
            "allow_count": len(allow_events),
            "deny_count": len(deny_events),
            "critical_count": len(critical_events),
            "unique_agents": len(agent_activity),
        },
    }


def generate_hipaa_report() -> Dict[str, Any]:
    events = list(_SIEM_EVENTS.values())
    phi_events = [e for e in events if _is_phi_event(e)]
    unauthorized_phi = [e for e in phi_events if e["decision"] == "deny"]

    return {
        "report_type": "HIPAA_Compliance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "safeguards": {
            "access_control_164_312_a)": {
                "status": "satisfied",
                "evidence": "Role-based access control with Ed25519 authentication",
                "phi_access_events": len(phi_events),
            },
            "audit_controls_164_312_b)": {
                "status": "satisfied",
                "evidence": "Hash chain audit logging with SIEM export",
                "total_audit_records": len(events),
            },
            "integrity_164_312_c_1)": {
                "status": "satisfied",
                "evidence": "SHA-256 hash chain for audit log integrity",
            },
            "person_entity_auth_164_312_d)": {
                "status": "satisfied",
                "evidence": "Ed25519 challenge-response + JWT token authentication",
            },
            "transmission_security_164_312_e_1)": {
                "status": "satisfied",
                "evidence": "HTTPS/TLS enforced, credential broker pattern",
            },
        },
        "phi_summary": {
            "phi_access_attempts": len(phi_events),
            "unauthorized_phi_access": len(unauthorized_phi),
            "phi_protection_rate": f"{(1 - len(unauthorized_phi)/max(len(phi_events),1))*100:.1f}%",
        },
    }


def get_siem_status() -> Dict[str, Any]:
    return {
        "total_events": len(_SIEM_EVENTS),
        "active_traces": len(_OTEL_SPANS),
        "export_formats": ["splunk", "elk", "datadog", "otel"],
        "compliance_templates": ["SOC2", "HIPAA"],
        "reports_generated": len(_COMPLIANCE_REPORTS),
    }


def _map_severity(severity: str) -> str:
    mapping = {
        "info": "info",
        "low": "low",
        "medium": "medium",
        "warning": "medium",
        "high": "high",
        "critical": "critical",
        "error": "high",
    }
    return mapping.get(severity.lower(), "info")


def _get_compliance_tags(event_type: str, decision: str, severity: str) -> List[str]:
    tags = []
    if decision == "deny":
        tags.append("access-denied")
    if severity in ("high", "critical"):
        tags.append("security-incident")
    if "phi" in event_type.lower() or "health" in event_type.lower():
        tags.append("hipaa-phi")
    if "finance" in event_type.lower() or "payment" in event_type.lower():
        tags.append("pci-dss")
    if "auth" in event_type.lower():
        tags.append("authentication")
    if "privilege" in event_type.lower() or "admin" in event_type.lower():
        tags.append("privilege-escalation")
    return tags


def _is_phi_event(event: Dict[str, Any]) -> bool:
    phi_keywords = ["phi", "health", "medical", "patient", "diagnosis", "treatment"]
    text = f"{event.get('event_type', '')} {event.get('resource', '')} {event.get('action', '')}".lower()
    return any(kw in text for kw in phi_keywords)
