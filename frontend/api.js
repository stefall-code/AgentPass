const BASE = '';

export async function request(method, path, options = {}) {
  const { headers = {}, body, params } = options;
  let url = BASE + path;
  if (params) {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== '')).toString();
    if (qs) url += '?' + qs;
  }
  const opts = { method, headers: { 'Content-Type': 'application/json', ...headers } };
  if (body) opts.body = JSON.stringify(body);
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    let msg = text;
    try { msg = JSON.parse(text).detail || text; } catch {}
    throw new Error(msg || resp.statusText);
  }
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) return resp.json();
  return resp.text();
}

export function authHeaders(token) {
  return token ? { Authorization: 'Bearer ' + token } : {};
}

export const api = {
  getOverview: () => request('GET', '/api/overview'),
  getHealth: () => request('GET', '/api/health'),
  getRoleMatrix: () => request('GET', '/api/admin/role-matrix'),

  login: (agentId, apiKey, opts = {}) => request('POST', '/api/login', {
    body: { agent_id: agentId, api_key: apiKey, bound_ip: opts.bound_ip, usage_limit: opts.usage_limit, expires_in_minutes: opts.expires_minutes },
  }),

  register: (name, role, attrs = {}) => request('POST', '/api/register', {
    body: { name, role, attributes: attrs },
  }),

  getMe: (headers) => request('GET', '/api/me', { headers }),
  introspectToken: (headers) => request('GET', '/api/auth/introspect', { headers }),
  revokeToken: (headers) => request('POST', '/api/auth/revoke', { headers }),
  refreshToken: (headers, refreshToken) => request('POST', '/api/auth/refresh', { headers, body: { refresh_token: refreshToken } }),

  listAgents: (headers) => request('GET', '/api/agents', { headers }),
  getAgent: (headers, agentId) => request('GET', '/api/agents/' + agentId, { headers }),
  updateAgentStatus: (headers, agentId, status, reason) => request('POST', '/api/admin/agents/' + agentId + '/status', { headers, body: { status, reason } }),

  accessResource: (headers, action, resource) => request('POST', '/api/agents/access', { headers, body: { action, resource } }),
  executeTask: (headers, taskName) => request('POST', '/api/tasks/execute', { headers, body: { task_name: taskName } }),
  callIntegration: (headers, integrationName) => request('POST', '/api/integrations/call', { headers, body: { service_name: integrationName } }),
  delegate: (headers, fromAgent, toAgent, taskName) => request('POST', '/api/agents/delegate', { headers, body: { target_agent_id: toAgent, task_name: taskName } }),

  policyTrace: (headers, params) => request('POST', '/api/insights/policy-trace', { headers, body: { agent_id: params.agent_id, action: params.action, resource: params.resource, resource_sensitivity: (params.resource_meta && params.resource_meta.sensitivity) || 'public' } }),

  getDashboard: (headers) => request('GET', '/api/admin/dashboard', { headers }),
  getAuditLogs: (headers, params = {}) => request('GET', '/api/admin/audit/logs', { headers, params }),
  exportAudit: (headers, format) => request('GET', '/api/admin/audit/export', { headers, params: { format } }),
  verifyIntegrity: (headers) => request('GET', '/api/admin/audit/verify-integrity', { headers }),

  delegationGraph: (headers) => request('GET', '/api/insights/delegation-graph', { headers }),
  riskDashboard: (headers) => request('GET', '/api/insights/risk-dashboard', { headers }),
  permissionSuggestions: (headers) => request('GET', '/api/insights/permission-suggestions', { headers }),
  accessHeatmap: (headers) => request('GET', '/api/insights/access-heatmap', { headers }),
  permissionDiff: (headers, agentA, agentB) => request('GET', '/api/insights/permission-diff', { headers, params: { agent_a: agentA, agent_b: agentB } }),

  resetDemo: (headers) => request('POST', '/api/admin/demo/reset', { headers }),
};
