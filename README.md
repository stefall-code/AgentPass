# AgentPass v2.5 — AI Agent 安全治理平台

> JWT authentication, RBAC/ABAC policy engine, prompt injection defense, real-time audit, risk auto-lock, cross-platform governance, explainable IAM, Feishu security gateway

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The server starts at `http://127.0.0.1:8000` and auto-opens in your browser.
Close the browser tab and the server exits automatically within 3 seconds.

## v2.5 Highlights

### 🛡️ Feishu Security Gateway

All Feishu API requests MUST pass through the local IAM system:

```
Feishu CLI → IAMTransport → IAM Check → Decision → Feishu API
```

- **IAMTransport** — `httpx.AsyncBaseTransport` interceptor, zero bypass
- **mapRequestToAction** — HTTP path → IAM action mapping
- **callIAMCheck** — Issue token + check permission via `/api/delegate/check`
- **Fail-closed** — IAM unreachable = all requests blocked
- **Header injection** — `X-Agent-ID` / `X-Trust-Score` / `X-Risk-Score`
- **Gateway Console** — `/gateway` page with stats, check, demos, audit log

### 🧠 Explainable IAM

Every authorization decision has a human-readable explanation:

- 8-step decision trace (token validation → auto-revoke → revocation → replay → chain → capability → dynamic policy → trust)
- Risk analysis, trust analysis, final reason, suggestion
- Explain buttons on ALL pages (7 pages covered)

### 🌐 Cross-Platform Unified Governance

- **Platform Adapter** — Feishu / Web / API request normalization
- **Orchestrator** — Platform metadata injection into tokens
- **Dynamic Policy** — Platform risk weights + enterprise data access rules
- **Governance Console** — `/governance` with traffic visualization + real-time event stream + agent control

## Features

- **JWT Authentication** — Access/refresh tokens with JWKS verification
- **RBAC + ABAC Policy Engine** — Priority-based rule evaluation with time constraints
- **Delegation Chain** — Visual graph with depth limit enforcement + auto-revoke
- **Prompt Injection Defense** — 9 attack types + 3-layer fusion engine
- **OpenClaw Integration** — Cross-agent data leak detection
- **Semantic Role Drift Detection** — TF-IDF + cosine distance analysis
- **Zero-Trust Context Isolation** — AES-256 encrypted session contexts
- **Real-Time Audit** — WebSocket push with hash chain integrity
- **Risk Auto-Lock** — Automatic agent suspension on repeated denials
- **Token Constraints** — IP binding + usage limits + expiration
- **Governance Center** — Approval queue, risk dashboard, permission diff, threat map
- **Permission Suggestions** — Least-privilege analysis with unused resource detection
- **Access Heatmap** — Daily access pattern visualization
- **Heartbeat Shutdown** — Server auto-exits when browser disconnects

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Security Command Center | Main dashboard with Feishu interaction |
| `/feishu` | Feishu Security Console | Feishu message interaction + attack demos |
| `/gateway` | IAM Gateway | Feishu API security gateway console |
| `/governance` | Unified Governance | Cross-platform governance console |
| `/audit` | Audit Center | 8-column audit table + execution + Explain |
| `/chain` | Chain Viewer | SVG delegation chain + attack replay |
| `/trust` | Trust Dashboard | Trust score ranking + degrade/revoke demos |
| `/risk` | Threat Radar | Risk analysis + decision distribution chart |
| `/v22` | v2.2 Legacy | Original unified governance page |

## Architecture

```
main.py                    # Entry point + heartbeat shutdown + browser auto-open
app/
  config.py                # Settings (Pydantic BaseSettings)
  database.py              # SQLAlchemy + SQLite
  identity.py              # Agent CRUD + risk lock
  auth.py                  # JWT issue / verify / refresh
  permission.py            # RBAC/ABAC policy engine
  audit.py                 # Hash-chain audit log
  ws.py                    # WebSocket real-time push
  middleware.py             # Rate limit / error handler / timing / request ID
  schemas.py               # Pydantic request/response models
  platform/                # Cross-platform adapter
    __init__.py            # PlatformRequest + normalize_request
    adapter.py             # Platform risk weights
  feishu/                  # Feishu integration
    __init__.py            # Module exports
    client.py              # FeishuClient with IAM Gateway
    router.py              # Feishu webhook + API routes
    iam_gateway.py         # IAMTransport + mapRequestToAction + callIAMCheck + logAudit
  orchestrator/            # Task orchestration
    orchestrator.py        # Platform-aware token issuance + trust scoring
  policy/                  # Dynamic policy engine
    dynamic_policy.py      # Platform risk + enterprise data rules
  delegation/              # Delegation engine
    engine.py              # Token issuance + check + auto-revoke
  explainer/               # Explainable IAM
    __init__.py            # explain_decision() — 8-step decision trace
  connectors/              # Platform connectors
  cost/                    # Cost tracking
  routers/                 # FastAPI route modules
    admin.py               # Dashboard, audit, role matrix, demo reset
    agents.py              # Agent CRUD, access, delegate
    auth.py                # Login, register, introspect, revoke
    approval.py            # Approval queue + Feishu card push
    insights.py            # Risk dashboard, delegation graph, reputation
    drift.py               # Semantic drift detection
    context.py             # Zero-trust context isolation
    resources.py           # Resource listing
    websocket.py           # WebSocket endpoints with heartbeat
    platforms.py           # Cross-platform management API
    delegation.py          # Delegation IAM + trust + audit
    governance.py          # Unified governance API
    explain.py             # Explainable IAM API
    gateway.py             # Feishu Security Gateway API
  services/                # Background tasks + reputation + daily stats
agentpass-sdk/             # Prompt defense SDK
frontend/                  # Vanilla JS + CSS (Apple Design)
  iam-explain.js           # Explainable IAM modal component
  feishu-section.js        # Feishu interaction (index.html)
  feishu.js                # Feishu security console
  governance.js            # Governance console
  audit.js                 # Audit center
  chain.js                 # Chain viewer
  trust.js                 # Trust dashboard
  risk.js                  # Threat radar
  gateway.html             # Gateway console page
```

## API Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/login` | POST | Agent login |
| `/api/register` | POST | Agent registration |
| `/api/auth/introspect` | GET | Token introspection |
| `/api/auth/revoke` | POST | Token revocation |
| `/api/agents/access` | POST | Resource access check |
| `/api/agents/delegate` | POST | Delegate permission |
| `/api/admin/dashboard` | GET | Governance dashboard |
| `/api/admin/approvals` | GET | Approval queue |
| `/api/delegate/issue-root` | POST | Issue root delegation token |
| `/api/delegate/check` | POST | Check delegation permission |
| `/api/delegate/trust` | GET | Get all agent trust scores |
| `/api/delegate/trust/reset` | POST | Reset trust scores |
| `/api/delegate/audit/logs` | GET | Delegation audit logs |
| `/api/gateway/stats` | GET | Gateway statistics |
| `/api/gateway/audit` | GET | Gateway audit log |
| `/api/gateway/check` | POST | Manual IAM check |
| `/api/gateway/map-action` | POST | Path → Action mapping test |
| `/api/gateway/demo/escalation` | POST | Escalation attack demo |
| `/api/gateway/demo/bypass-attempt` | POST | Bypass attempt demo |
| `/api/explain/result` | POST | Explain IAM decision |
| `/api/governance/overview` | GET | Cross-platform overview |
| `/api/governance/revoke-agent` | POST | Revoke agent via governance |
| `/api/feishu/webhook` | POST | Feishu event callback |
| `/api/feishu/test` | POST | Feishu message test |
| `/api/insights/risk-dashboard` | GET | Risk gauges per agent |
| `/api/insights/delegation-graph` | GET | Delegation chain graph |
| `/api/insights/permission-diff` | GET | Permission comparison |
| `/api/prompt-defense/analyze` | POST | Prompt injection analysis |
| `/api/drift/analyze` | POST | Semantic drift analysis |
| `/api/context/seal` | POST | Context isolation seal |

## Demo Agents

| Agent ID | Capabilities |
|----------|-------------|
| doc_agent | read:doc, write:doc:public, delegate:data_agent |
| data_agent | read:feishu_table, read:feishu_table:finance, read:feishu_table:hr |
| external_agent | write:doc:public, read:web |

## License

MIT
