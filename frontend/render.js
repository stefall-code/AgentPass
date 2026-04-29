// 渲染模块
import { escapeHtml } from '/static/utils.js';
import state, { getOrderedSessionIds, getTestSummary } from '/static/state.js';

// 元素对象
const el = {
  healthPill: document.getElementById("healthPill"),
  healthCaption: document.getElementById("healthCaption"),
  featureCount: document.getElementById("featureCount"),
  demoAgentCount: document.getElementById("demoAgentCount"),
  docCount: document.getElementById("docCount"),
  activeTokenCount: document.getElementById("activeTokenCount"),
  denyCount: document.getElementById("denyCount"),
  suspendedCount: document.getElementById("suspendedCount"),
  featureChips: document.getElementById("featureChips"),
  policyNotes: document.getElementById("policyNotes"),
  demoAgentCards: document.getElementById("demoAgentCards"),
  demoDocumentCards: document.getElementById("demoDocumentCards"),
  sessionVault: document.getElementById("sessionVault"),
  vaultCount: document.getElementById("vaultCount"),
  activeSessionMini: document.getElementById("activeSessionMini"),
  miniSessionDot: document.getElementById("miniSessionDot"),
  profileCard: document.getElementById("profileCard"),
  tokenInspector: document.getElementById("tokenInspector"),
  tokenTextarea: document.getElementById("tokenTextarea"),
  tokenStatusPill: document.getElementById("tokenStatusPill"),
  roleMatrix: document.getElementById("roleMatrix"),
  scenarioTimeline: document.getElementById("scenarioTimeline"),
  docContentInput: document.getElementById("docContentInput"),
  apiMethodSelect: document.getElementById("apiMethodSelect"),
  apiPathInput: document.getElementById("apiPathInput"),
  apiBodyInput: document.getElementById("apiBodyInput"),
  apiUseAuthSelect: document.getElementById("apiUseAuthSelect"),
  apiPresetSelect: document.getElementById("apiPresetSelect"),
  apiResponseOutput: document.getElementById("apiResponseOutput"),
  dashboardSummary: document.getElementById("dashboardSummary"),
  allowBar: document.getElementById("allowBar"),
  denyBar: document.getElementById("denyBar"),
  allowLabel: document.getElementById("allowLabel"),
  denyLabel: document.getElementById("denyLabel"),
  roleBars: document.getElementById("roleBars"),
  sensitivityBars: document.getElementById("sensitivityBars"),
  topActions: document.getElementById("topActions"),
  recentDenials: document.getElementById("recentDenials"),
  agentsTable: document.getElementById("agentsTable"),
  auditTable: document.getElementById("auditTable"),
  consoleOutput: document.getElementById("consoleOutput"),
  toastStack: document.getElementById("toastStack"),
  registerResult: document.getElementById("registerResult"),
};

/**
 * 设置健康状态
 * @param {string} status 状态
 */
export function setHealth(status) {
  if (status === "ok") {
    el.healthPill.className = "pill success";
    el.healthPill.textContent = "Online";
    el.healthCaption.textContent = "FastAPI 服务正常，SQLite 数据库已连接。";
    return;
  }
  el.healthPill.className = "pill danger";
  el.healthPill.textContent = "Offline";
  el.healthCaption.textContent = "后端服务不可达，请先启动 uvicorn。";
}

/**
 * 构建概览卡片
 * @param {string} title 标题
 * @param {string} subtitle 副标题
 * @param {string[]} pills 标签
 * @returns {string} HTML字符串
 */
function buildOverviewCard(title, subtitle, pills = []) {
  return `
    <article class="overview-card">
      <strong>${escapeHtml(title)}</strong>
      <small>${escapeHtml(subtitle)}</small>
      ${pills.length ? `<div class="chips" style="margin-top:10px;">${pills.join("")}</div>` : ""}
    </article>
  `;
}

/**
 * 渲染概览
 */
export function renderOverview() {
  const overview = state.overview;
  if (!overview) {
    return;
  }

  el.featureCount.textContent = overview.features.length;
  el.demoAgentCount.textContent = overview.demo_agents.length;
  el.docCount.textContent = overview.demo_documents.length;

  el.featureChips.innerHTML = overview.features.map((feature) => `<span class="chip">${escapeHtml(feature)}</span>`).join("");

  const notes = state.dashboard?.policy_notes || [
    "Confidential resources require admin role.",
    "Agent allowlist is checked against the target resource.",
    "Repeated denied requests lead to automatic suspension.",
    "JWT can be bound to IP and limited by usage count.",
  ];
  el.policyNotes.innerHTML = notes
    .map((note) => `<div class="note-item"><strong>Policy</strong><small>${escapeHtml(note)}</small></div>`)
    .join("");

  el.demoAgentCards.innerHTML = overview.demo_agents
    .map((agent) =>
      buildOverviewCard(
        agent.name || agent.agent_id,
        `${agent.agent_id} · role: ${agent.role}`,
        [`<span class="pill primary">${escapeHtml(agent.role)}</span>`],
      ),
    )
    .join("");

  el.demoDocumentCards.innerHTML = overview.demo_documents
    .map((doc) =>
      buildOverviewCard(
        doc.doc_id,
        `sensitivity: ${doc.sensitivity}`,
        [`<span class="pill neutral">${escapeHtml(doc.sensitivity)}</span>`],
      ),
    )
    .join("");
}

/**
 * 渲染会话库
 * @param {object} DEMO_PRESETS 演示预设
 */
export function renderSessionVault(DEMO_PRESETS) {
  const ids = getOrderedSessionIds(DEMO_PRESETS);
  el.vaultCount.textContent = String(ids.filter((id) => state.sessions[id]).length);
  el.vaultCount.className = "pill neutral";

  el.sessionVault.innerHTML = ids
    .map((agentId) => {
      const existing = state.sessions[agentId];
      const preset = DEMO_PRESETS[agentId] || {};
      const session = existing || { ...preset, agentId };
      const isActive = state.activeSessionId === agentId;
      const tokenMeta = session.tokenMeta;
      const statusClass = tokenMeta?.active ? "success" : session.token ? "warning" : "neutral";
      const statusLabel = tokenMeta?.active
        ? "token active"
        : session.token
          ? "token cached"
          : session.source === "demo"
            ? "demo ready"
            : "not logged in";

      return `
        <article class="session-card ${isActive ? "active" : ""}">
          <div class="metric-top">
            <div>
              <strong>${escapeHtml(session.label || agentId)}</strong>
              <div class="helper-text">${escapeHtml(agentId)}</div>
            </div>
            <span class="pill ${statusClass}">${escapeHtml(statusLabel)}</span>
          </div>
          <div class="session-meta">
            <span class="pill neutral">${escapeHtml(session.profile?.role || session.roleHint || "-")}</span>
            <span class="pill neutral">${escapeHtml(session.profile?.status || "unknown")}</span>
            ${
              tokenMeta?.usage_remaining !== undefined
                ? `<span class="pill neutral">remain ${tokenMeta.usage_remaining}</span>`
                : ""
            }
          </div>
          <div class="session-actions">
            <button class="ghost-button small" data-action="use-session" data-agent-id="${escapeHtml(agentId)}" type="button">切换</button>
            <button class="ghost-button small" data-action="login-session" data-agent-id="${escapeHtml(agentId)}" type="button">登录</button>
            <button class="ghost-button small" data-action="refresh-session" data-agent-id="${escapeHtml(agentId)}" type="button">刷新</button>
            <button class="ghost-button small" data-action="revoke-session" data-agent-id="${escapeHtml(agentId)}" type="button">撤销</button>
          </div>
        </article>
      `;
    })
    .join("");
}

/**
 * 渲染角色矩阵
 */
export function renderRoleMatrix() {
  const permissions = state.dashboard?.role_permissions || {
    basic: ["read_doc"],
    editor: ["read_doc", "write_doc"],
    operator: ["read_doc", "execute_task", "call_api", "delegate_task"],
    admin: [
      "read_doc",
      "write_doc",
      "execute_task",
      "call_api",
      "delegate_task",
      "view_audit",
      "manage_agents",
    ],
  };
  const actions = Array.from(new Set(Object.values(permissions).flat()));
  el.roleMatrix.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Role</th>
          ${actions.map((action) => `<th>${escapeHtml(action)}</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${Object.entries(permissions)
          .map(
            ([role, allowed]) => `
              <tr>
                <td><strong>${escapeHtml(role)}</strong></td>
                ${actions
                  .map(
                    (action) =>
                      `<td>${allowed.includes(action) ? '<span class="pill success">yes</span>' : '<span class="pill neutral">-</span>'}</td>`,
                  )
                  .join("")}
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

/**
 * 渲染活跃会话
 */
export function renderActiveSession() {
  const session = state.sessions[state.activeSessionId];
  if (!session) {
    el.activeSessionMini.className = "empty-state";
    el.activeSessionMini.textContent = "当前没有激活会话。";
    el.miniSessionDot.className = "status-dot";
    el.profileCard.className = "profile-card empty-state";
    el.profileCard.textContent = "请选择一个 Agent 会话并完成登录。";
    el.tokenInspector.className = "empty-state";
    el.tokenInspector.textContent = "登录后将展示 token 元数据。";
    el.tokenTextarea.value = "";
    el.tokenStatusPill.className = "pill neutral";
    el.tokenStatusPill.textContent = "No Token";
    return;
  }

  el.activeSessionMini.className = "";
  el.activeSessionMini.innerHTML = `
    <strong>${escapeHtml(session.label || session.agentId)}</strong>
    <div class="helper-text">${escapeHtml(session.agentId)}</div>
    <div class="chips" style="margin-top:10px;">
      <span class="pill primary">${escapeHtml(session.profile?.role || session.roleHint || "-")}</span>
      <span class="pill ${session.token ? "success" : "neutral"}">${session.token ? "authenticated" : "anonymous"}</span>
    </div>
  `;
  el.miniSessionDot.className = `status-dot ${session.token ? "active" : ""}`;

  if (!session.profile) {
    el.profileCard.className = "profile-card empty-state";
    el.profileCard.textContent = "当前会话还没有 profile 信息，点击刷新当前会话或先登录。";
  } else {
    const allowedResources = session.profile.attributes?.allowed_resources || [];
    el.profileCard.className = "profile-card";
    el.profileCard.innerHTML = `
      <div class="profile-grid">
        <div class="profile-item"><span>Agent ID</span><strong>${escapeHtml(session.profile.agent_id)}</strong></div>
        <div class="profile-item"><span>Name</span><strong>${escapeHtml(session.profile.name)}</strong></div>
        <div class="profile-item"><span>Role</span><strong>${escapeHtml(session.profile.role)}</strong></div>
        <div class="profile-item"><span>Status</span><strong>${escapeHtml(session.profile.status)}</strong></div>
      </div>
      <div class="chips" style="margin-top:14px;">
        ${session.profile.permissions.map((item) => `<span class="chip">${escapeHtml(item)}</span>`).join("")}
      </div>
      <div class="profile-item" style="margin-top:14px;">
        <span>Allowed Resources</span>
        <strong>${allowedResources.length ? escapeHtml(allowedResources.join(", ")) : "-"}</strong>
      </div>
    `;
  }

  el.tokenTextarea.value = session.token || "";
  if (!session.tokenMeta) {
    el.tokenInspector.className = "empty-state";
    el.tokenInspector.textContent = session.token
      ? "当前有 token，但尚未加载元数据。点击检查当前 Token即可查看。"
      : "当前没有 token。";
    el.tokenStatusPill.className = session.token ? "pill warning" : "pill neutral";
    el.tokenStatusPill.textContent = session.token ? "Cached Token" : "No Token";
    return;
  }

  const meta = session.tokenMeta;
  el.tokenStatusPill.className = meta.active ? "pill success" : "pill warning";
  el.tokenStatusPill.textContent = meta.active ? "Token Active" : "Token Inactive";
  el.tokenInspector.className = "";
  el.tokenInspector.innerHTML = `
    <div class="profile-grid">
      <div class="profile-item"><span>JTI</span><strong>${escapeHtml(meta.jti)}</strong></div>
      <div class="profile-item"><span>Bound IP</span><strong>${escapeHtml(meta.bound_ip || "none")}</strong></div>
      <div class="profile-item"><span>Issued At</span><strong>${escapeHtml(meta.issued_at)}</strong></div>
      <div class="profile-item"><span>Expires At</span><strong>${escapeHtml(meta.expires_at)}</strong></div>
      <div class="profile-item"><span>Usage Limit</span><strong>${escapeHtml(String(meta.usage_limit))}</strong></div>
      <div class="profile-item"><span>Usage Remaining</span><strong>${escapeHtml(String(meta.usage_remaining))}</strong></div>
    </div>
  `;
}

/**
 * 渲染条形图
 * @param {HTMLElement} container 容器元素
 * @param {object} map 数据映射
 */
export function renderBars(container, map) {
  const entries = Object.entries(map || {});
  if (!entries.length) {
    container.className = "bar-list empty-state";
    container.textContent = "暂无数据。";
    return;
  }

  const max = Math.max(...entries.map(([, count]) => count), 1);
  container.className = "bar-list";
  container.innerHTML = entries
    .map(
      ([name, count]) => `
        <div class="bar-row">
          <div class="bar-row-top">
            <span>${escapeHtml(name)}</span>
            <span>${escapeHtml(String(count))}</span>
          </div>
          <div class="bar-row-track">
            <div class="bar-row-fill" style="width:${(count / max) * 100}%"></div>
          </div>
        </div>
      `,
    )
    .join("");
}

/**
 * 渲染仪表盘
 */
export function renderDashboard() {
  if (!state.dashboard) {
    el.dashboardSummary.className = "dashboard-summary empty-state";
    el.dashboardSummary.textContent = "需要一个 admin 会话才能加载完整治理统计。";
    el.activeTokenCount.textContent = "0";
    el.denyCount.textContent = "0";
    el.suspendedCount.textContent = "0";
    el.allowBar.style.width = "0%";
    el.denyBar.style.width = "0%";
    el.allowLabel.textContent = "allow: 0";
    el.denyLabel.textContent = "deny: 0";
    el.roleBars.className = "bar-list empty-state";
    el.roleBars.textContent = "等待 admin 数据。";
    el.sensitivityBars.className = "bar-list empty-state";
    el.sensitivityBars.textContent = "等待 admin 数据。";
    el.topActions.className = "chips empty-state";
    el.topActions.textContent = "等待 admin 数据。";
    el.recentDenials.className = "notes-list empty-state";
    el.recentDenials.textContent = "等待 admin 数据。";
    return;
  }

  const snapshot = state.dashboard.snapshot;
  const audit = state.dashboard.audit;
  const allowTotal = audit.allow || 0;
  const denyTotal = audit.deny || 0;
  const total = Math.max(allowTotal + denyTotal, 1);

  el.activeTokenCount.textContent = String(snapshot.tokens.active || 0);
  el.denyCount.textContent = String(denyTotal);
  el.suspendedCount.textContent = String(snapshot.agents.by_status?.suspended || 0);

  el.dashboardSummary.className = "dashboard-summary";
  el.dashboardSummary.innerHTML = `
    <div class="dashboard-card-grid">
      <div class="quick-card">
        <span>Agents</span>
        <strong>${snapshot.agents.total}</strong>
      </div>
      <div class="quick-card">
        <span>Audit Events</span>
        <strong>${audit.total}</strong>
      </div>
      <div class="quick-card">
        <span>Active Tokens</span>
        <strong>${snapshot.tokens.active}</strong>
      </div>
      <div class="quick-card">
        <span>Documents</span>
        <strong>${snapshot.documents.total}</strong>
      </div>
    </div>
  `;

  el.allowBar.style.width = `${(allowTotal / total) * 100}%`;
  el.denyBar.style.width = `${(denyTotal / total) * 100}%`;
  el.allowLabel.textContent = `allow: ${allowTotal}`;
  el.denyLabel.textContent = `deny: ${denyTotal}`;

  renderBars(el.roleBars, snapshot.agents.by_role);
  renderBars(el.sensitivityBars, snapshot.documents.by_sensitivity);

  el.topActions.className = "chips";
  el.topActions.innerHTML = audit.top_actions.length
    ? audit.top_actions.map((item) => `<span class="chip">${escapeHtml(item.action)} · ${item.count}</span>`).join("")
    : '<span class="empty-state">暂无操作数据。</span>';

  el.recentDenials.className = "notes-list";
  el.recentDenials.innerHTML = audit.recent_denials.length
    ? audit.recent_denials
        .map(
          (item) => `
            <div class="note-item">
              <strong>${escapeHtml(item.agent_id || "-")} · ${escapeHtml(item.action)}</strong>
              <small>${escapeHtml(item.reason)}</small>
            </div>
          `,
        )
        .join("")
    : '<div class="empty-state">最近没有拒绝事件。</div>';
}

/**
 * 渲染Agent表格
 * @param {array} items Agent列表
 */
export function renderAgentsTable(items) {
  if (!items?.length) {
    el.agentsTable.className = "table-shell empty-state";
    el.agentsTable.textContent = "暂无 Agent 数据。";
    return;
  }
  el.agentsTable.className = "table-shell";
  el.agentsTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>agent_id</th>
          <th>role</th>
          <th>status</th>
          <th>status_reason</th>
          <th>actions</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map(
            (item) => `
              <tr>
                <td>
                  <strong>${escapeHtml(item.agent_id)}</strong>
                  <div class="table-subtle">${escapeHtml(item.name)}</div>
                </td>
                <td>${escapeHtml(item.role)}</td>
                <td>${escapeHtml(item.status)}</td>
                <td>${escapeHtml(item.status_reason || "-")}</td>
                <td>
                  <div class="action-group">
                    <button class="action-button" data-agent-id="${escapeHtml(item.agent_id)}" data-next-status="active" data-tone="success" type="button">active</button>
                    <button class="action-button" data-agent-id="${escapeHtml(item.agent_id)}" data-next-status="suspended" data-tone="danger" type="button">suspend</button>
                    <button class="action-button" data-agent-id="${escapeHtml(item.agent_id)}" data-next-status="disabled" data-tone="danger" type="button">disable</button>
                  </div>
                </td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

/**
 * 渲染审计表格
 * @param {array} items 审计日志列表
 */
export function renderAuditTable(items) {
  if (!items?.length) {
    el.auditTable.className = "table-shell empty-state";
    el.auditTable.textContent = "暂无审计日志。";
    return;
  }
  el.auditTable.className = "table-shell";
  el.auditTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>time</th>
          <th>agent</th>
          <th>action</th>
          <th>resource</th>
          <th>decision</th>
          <th>reason</th>
        </tr>
      </thead>
      <tbody>
        ${items
          .map(
            (item) => `
              <tr>
                <td>${escapeHtml(item.created_at)}</td>
                <td>${escapeHtml(item.agent_id || "-")}</td>
                <td>${escapeHtml(item.action)}</td>
                <td>${escapeHtml(item.resource)}</td>
                <td class="${item.decision === "allow" ? "decision-allow" : "decision-deny"}">${escapeHtml(item.decision)}</td>
                <td>${escapeHtml(item.reason)}</td>
              </tr>
            `,
          )
          .join("")}
      </tbody>
    </table>
  `;
}

/**
 * 渲染时间线
 */
export function renderTimeline() {
  if (!state.scenarioTimeline.length) {
    el.scenarioTimeline.className = "timeline empty-state";
    el.scenarioTimeline.textContent = "还没有运行任何场景。";
    return;
  }
  el.scenarioTimeline.className = "timeline";
  el.scenarioTimeline.innerHTML = state.scenarioTimeline
    .map(
      (item) => `
        <article class="timeline-item">
          <div class="timeline-item-header">
            <strong>${escapeHtml(item.title)}</strong>
            <span class="pill ${item.tone === "error" ? "danger" : item.tone === "warn" ? "warning" : "success"}">${escapeHtml(item.time)}</span>
          </div>
          <small>${escapeHtml(item.detail)}</small>
        </article>
      `,
    )
    .join("");
}

/**
 * 记录控制台日志
 * @param {string} title 标题
 * @param {*} payload 内容
 * @param {string} tone 语气
 */
export function logConsole(title, payload, tone = "info") {
  const prefix = tone === "error" ? "[ERROR]" : tone === "warn" ? "[WARN]" : "[INFO]";
  const text = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  const time = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  el.consoleOutput.textContent = `${prefix} ${time} ${title}\n${text}\n\n${el.consoleOutput.textContent}`;
}

/**
 * 显示提示信息
 * @param {string} title 标题
 * @param {string} detail 详情
 * @param {string} tone 语气
 */
export function pushToast(title, detail, tone = "success") {
  const node = document.createElement("div");
  node.className = `toast ${tone}`;
  node.innerHTML = `<strong>${escapeHtml(title)}</strong><small>${escapeHtml(detail)}</small>`;
  el.toastStack.prepend(node);
  window.setTimeout(() => {
    node.remove();
  }, 3600);
}

/**
 * 渲染测试结果摘要
 */
export function renderTestSummary() {
  const container = document.getElementById("testSummaryContent");
  if (!container) return;

  const summary = getTestSummary();
  if (summary.total === 0) {
    container.innerHTML = '<div class="empty-state" style="padding:8px;">尚未运行测试</div>';
    return;
  }

  const passRate = summary.total > 0 ? Math.round((summary.passed / summary.total) * 100) : 0;
  const rateClass = passRate === 100 ? "success" : passRate >= 70 ? "warning" : "danger";

  container.innerHTML = `
    <div style="display:flex;gap:12px;align-items:center;justify-content:center;padding:8px 0;">
      <div style="text-align:center;">
        <span class="pill ${rateClass}" style="font-size:16px;padding:4px 10px;">${passRate}%</span>
        <div class="helper-text" style="margin-top:2px;">通过率</div>
      </div>
      <div style="text-align:center;">
        <strong style="color:var(--success);">${summary.passed}</strong>
        <div class="helper-text">通过</div>
      </div>
      <div style="text-align:center;">
        <strong style="color:var(--danger);">${summary.failed}</strong>
        <div class="helper-text">失败</div>
      </div>
    </div>
    <div style="margin-top:6px;">
      ${Object.entries(summary.scenarios)
        .map(([id, result]) => {
          const ok = result.failed === 0;
          return `<div style="display:flex;justify-content:space-between;padding:2px 0;font-size:12px;">
            <span>${escapeHtml(id)}</span>
            <span class="pill ${ok ? "success" : "danger"}" style="font-size:10px;">${ok ? "PASS" : `${result.failed} FAIL`}</span>
          </div>`;
        })
        .join("")}
    </div>
  `;
}

/**
 * 渲染实时审计事件（来自WebSocket）
 * @param {object} event 审计事件
 */
export function renderLiveAuditEvent(event) {
  const container = document.getElementById("liveAuditFeed");
  if (!container) return;

  const decisionClass = event.decision === "allow" ? "success" : "danger";
  const line = document.createElement("div");
  line.className = "note-item";
  line.style.cssText = "padding:4px 0;font-size:11px;";
  line.innerHTML = `
    <strong><span class="pill ${decisionClass}" style="font-size:9px;padding:1px 4px;">${escapeHtml(event.decision)}</span> ${escapeHtml(event.agent_id || "?")} → ${escapeHtml(event.action)} ${escapeHtml(event.resource)}</strong>
    <small>${escapeHtml(event.reason || "")}</small>
  `;
  container.insertBefore(line, container.firstChild);
  // 限制显示条数，防止内存增长
  while (container.children.length > 30) {
    container.removeChild(container.lastChild);
  }
}

// 导出元素对象
export { el };
