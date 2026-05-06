const DELEGATE_BASE = '/api/delegate';
const AUDIT_BASE = '/api/delegate/audit';
const WS_BASE = location.protocol === 'https:' ? 'wss:' : 'ws:';

let _auditWs = null;
let _wsConnected = false;
let _wsReconnectAttempts = 0;
const _WS_MAX_DELAY = 30000;
let _logEntries = [];
let _trustScores = {};
let _detailCache = {};

function fetchJSON(url, options = {}) {
    return fetch(url, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } }).then(r => r.json());
}

function connectAuditWS() {
    const wsUrl = `${WS_BASE}//${location.host}/ws/audit`;
    try {
        _auditWs = new WebSocket(wsUrl);
        _auditWs.onopen = () => {
            _wsConnected = true;
            _wsReconnectAttempts = 0;
            updateWSStatus(true);
            pushLiveEvent('allow', 'system', 'WebSocket connected');
        };
        _auditWs.onmessage = (e) => {
            try { handleAuditMessage(JSON.parse(e.data)); } catch {}
        };
        _auditWs.onclose = () => {
            _wsConnected = false;
            updateWSStatus(false);
            _wsReconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(2, _wsReconnectAttempts - 1) + Math.random() * 1000, _WS_MAX_DELAY);
            setTimeout(connectAuditWS, delay);
        };
        _auditWs.onerror = () => { _auditWs.close(); };
    } catch {
        setTimeout(connectAuditWS, 5000);
    }
}

function updateWSStatus(connected) {
    const el = document.getElementById('wsStatus');
    if (!el) return;
    el.innerHTML = `<span class="ac-live-dot ${connected ? 'on' : 'off'}"></span>${connected ? '已连接' : '未连接'}`;
}

function handleAuditMessage(msg) {
    const decision = msg.decision || 'allow';
    const isAutoRevoke = msg.context && msg.context.auto_revoked;
    let type = decision === 'allow' ? 'allow' : 'deny';
    if (isAutoRevoke) type = 'revoke';
    pushLiveEvent(type, msg.agent_id || '?', msg.action || '?');

    const entry = buildLogEntry(msg);
    _logEntries.unshift(entry);
    appendRowToTable(entry, true);
    refreshStats();
}

function pushLiveEvent(type, agent, action) {
    const stream = document.getElementById('liveStream');
    if (!stream) return;
    const now = new Date().toLocaleTimeString();
    const el = document.createElement('div');
    el.className = 'ac-ev-item';
    el.innerHTML = `<span class="ac-ev-dot ${type}"></span><span class="ac-ev-time">${now}</span><span class="ac-ev-agent">${agent}</span><span class="ac-ev-action">${action}</span>`;
    const placeholder = stream.querySelector('.ac-empty');
    if (placeholder) placeholder.remove();
    stream.prepend(el);
    while (stream.children.length > 80) stream.removeChild(stream.lastChild);
}

function buildLogEntry(log) {
    const ctx = log.context || {};
    const delegationChain = ctx.delegation_chain || ctx.chain || log.chain || null;
    const delegationSteps = ctx.delegation_steps || null;
    const delegationChainDisplay = ctx.delegation_chain_display || (delegationChain ? delegationChain.join(' → ') : null);
    return {
        id: log.id || Date.now(),
        timestamp: log.timestamp || log.created_at || new Date().toISOString(),
        agent_id: log.agent_id || '-',
        action: log.action || '-',
        resource: log.resource || '-',
        decision: log.decision || 'allow',
        reason: log.reason || '-',
        risk_score: ctx.risk_score ?? log.risk_score ?? null,
        trust_score_before: ctx.trust_score_before ?? null,
        trust_score_after: ctx.trust_score_after ?? null,
        auto_revoked: !!(ctx.auto_revoked || log.auto_revoked),
        token_id: log.token_id || ctx.token || null,
        chain: delegationChain,
        chain_detail: ctx.chain_detail || null,
        delegation_chain: delegationChain,
        delegation_steps: delegationSteps,
        delegation_chain_display: delegationChainDisplay,
        revoked: ctx.revoked ?? null,
        platform: ctx.platform || 'web',
        entry_point: ctx.entry_point || 'frontend',
        blocked_at: ctx.blocked_at || log.blocked_at || '',
        prompt_risk_score: ctx.prompt_risk_score ?? log.prompt_risk_score ?? null,
        attack_types: ctx.attack_types || log.attack_types || [],
        attack_intent: ctx.attack_intent || log.attack_intent || '',
        severity: ctx.severity || log.severity || '',
        context: ctx,
    };
}

async function executeAction() {
    const agentId = document.getElementById('execAgent').value;
    const action = document.getElementById('execAction').value.trim();
    const user = document.getElementById('execUser').value.trim();
    const btn = document.getElementById('execBtn');
    const resultPanel = document.getElementById('execResult');

    if (!action) { alert('请输入 Action'); return; }

    btn.disabled = true;
    btn.textContent = '⏳ 执行中...';
    resultPanel.style.display = 'none';

    try {
        const trustBefore = await fetchJSON(`${DELEGATE_BASE}/trust`);
        const scoreBefore = (trustBefore.agents || {})[agentId]?.trust_score;

        const issueResp = await fetchJSON(`${DELEGATE_BASE}/issue-root`, {
            method: 'POST',
            body: JSON.stringify({ agent_id: agentId, delegated_user: user }),
        });

        if (!issueResp.token) {
            showExecResult(false, 'Token 签发失败', null, scoreBefore, null, []);
            return;
        }

        const token = issueResp.token;
        const jti = issueResp.jti || '';

        const steps = [];
        steps.push({ name: 'Token 签发', pass: true, detail: `jti=${jti.substring(0, 12)}...` });

        if (scoreBefore !== undefined) {
            steps.push({ name: '信任评分', pass: scoreBefore >= 0.3, detail: `score=${scoreBefore.toFixed(2)}` });
        }

        const checkResp = await fetchJSON(`${DELEGATE_BASE}/check`, {
            method: 'POST',
            body: JSON.stringify({ token, action }),
        });

        const allowed = checkResp.allowed;
        steps.push({ name: 'Capability 匹配', pass: allowed, detail: checkResp.reason || '' });
        steps.push({ name: '最终决策', pass: allowed, detail: allowed ? 'ALLOW' : 'DENY' });

        const trustAfter = await fetchJSON(`${DELEGATE_BASE}/trust`);
        const scoreAfter = (trustAfter.agents || {})[agentId]?.trust_score;
        _trustScores = trustAfter.agents || {};

        const riskScore = checkResp.risk_score ?? (allowed ? 0.1 : 0.9);

        const entry = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            agent_id: agentId,
            action: action,
            resource: '',
            decision: allowed ? 'allow' : 'deny',
            reason: checkResp.reason || '',
            risk_score: riskScore,
            trust_score_before: scoreBefore,
            trust_score_after: scoreAfter,
            auto_revoked: !!checkResp.auto_revoked,
            token_id: token,
            chain: issueResp.chain || checkResp.chain || null,
            chain_detail: null,
            revoked: null,
            context: { jti, capabilities: issueResp.capabilities, delegated_user: user },
        };

        _logEntries.unshift(entry);
        appendRowToTable(entry, true);
        refreshStats();

        showExecResult(allowed, checkResp.reason || '', riskScore, scoreBefore, scoreAfter, steps);

    } catch (e) {
        showExecResult(false, e.message, null, null, null, []);
    } finally {
        btn.disabled = false;
        btn.textContent = '⚡ 执行操作';
    }
}

function showExecResult(allowed, reason, riskScore, trustBefore, trustAfter, steps) {
    const panel = document.getElementById('execResult');
    const cls = allowed ? 'allowed' : 'denied';
    const icon = allowed ? '🟢' : '🔴';
    const label = allowed ? 'ALLOW' : 'DENY';

    const pipelineHtml = steps.length ? `<div class="ac-trace-pipeline" style="margin-top:10px">
        ${steps.map(s => `<span class="ac-trace-step ${s.pass ? 'pass' : 'fail'}">${s.pass ? '✓' : '✗'} ${s.name}</span>`).join('')}
    </div>` : '';

    const trustDeltaHtml = (trustBefore !== null && trustAfter !== null && trustBefore !== trustAfter)
        ? `<div style="margin-top:8px"><span class="ac-trust-delta ${trustAfter < trustBefore ? 'down' : 'up'}">Trust: ${trustBefore.toFixed(2)} → ${trustAfter.toFixed(2)} (${trustAfter < trustBefore ? '' : '+'}${(trustAfter - trustBefore).toFixed(2)})</span></div>`
        : '';

    panel.className = `ac-result-panel ${cls}`;
    panel.style.display = 'block';

    var explainBtnData = {
        agent_id: document.getElementById('execAgent')?.value || '',
        action: document.getElementById('execAction')?.value || '',
        decision: allowed ? 'allow' : 'deny',
        reason: reason || '',
        trust_score: trustAfter || trustBefore,
        risk_score: riskScore || 0,
        chain_detail: [],
        blocked_at: allowed ? '' : 'policy_check',
        auto_revoked: false,
        prompt_risk_score: null,
        attack_types: [],
        attack_intent: '',
        severity: '',
    };

    panel.innerHTML = `
        <div style="font-size:1.3rem;font-weight:700;margin-bottom:6px">${icon} ${label}</div>
        <div style="font-size:0.82rem;color:rgba(255,255,255,0.6)">${reason}</div>
        ${riskScore !== null ? `<div style="margin-top:6px;font-size:0.82rem">Risk: <span class="${riskScore > 0.7 ? 'ac-risk-high' : (riskScore > 0.3 ? 'ac-risk-med' : 'ac-risk-low')}">${riskScore.toFixed(2)}</span></div>` : ''}
        ${trustDeltaHtml}
        ${pipelineHtml}
        <div style="margin-top:10px">${IAM_EXPLAIN.makeBtn('🧠 Explain This Decision', explainBtnData, 'iam-explain-btn-lg')}</div>
    `;
}

async function loadAuditLogs() {
    const agent = document.getElementById('filterAgent').value.trim();
    const decision = document.getElementById('filterResult').value;
    const action = document.getElementById('filterAction').value.trim();
    const limit = parseInt(document.getElementById('filterLimit').value) || 100;

    const params = new URLSearchParams();
    params.set('limit', limit);
    if (agent) params.set('agent_id', agent);
    if (decision) params.set('decision', decision);
    if (action) params.set('action', action);

    try {
        const resp = await fetch(`${AUDIT_BASE}/logs?${params}`);
        const data = await resp.json();
        const logs = data.logs || data || [];
        _logEntries = logs.map(buildLogEntry);
        renderTable();
        refreshStats();
    } catch (e) {
        document.getElementById('auditTableBody').innerHTML = `<tr><td colspan="9" style="color:#f87171;text-align:center;padding:20px">加载失败: ${e.message}</td></tr>`;
    }
}

function renderTable() {
    const tbody = document.getElementById('auditTableBody');
    if (!_logEntries.length) {
        tbody.innerHTML = '<tr><td colspan="9" class="ac-empty">无审计记录</td></tr>';
        return;
    }
    tbody.innerHTML = _logEntries.map((entry, i) => renderRow(entry, i, false)).join('');
}

function appendRowToTable(entry, isNew) {
    const tbody = document.getElementById('auditTableBody');
    const placeholder = tbody.querySelector('.ac-empty');
    if (placeholder) placeholder.remove();

    const tr = document.createElement('tr');
    tr.innerHTML = renderRowCells(entry, 0);
    if (isNew) tr.className = 'new-row';

    const detailTr = document.createElement('tr');
    detailTr.className = 'ac-detail-row';
    detailTr.id = `detail-${entry.id}`;
    detailTr.innerHTML = `<td colspan="9"><div class="ac-detail-content">${renderDetailContent(entry)}</div></td>`;

    if (tbody.firstChild) {
        tbody.insertBefore(detailTr, tbody.firstChild);
        tbody.insertBefore(tr, detailTr);
    } else {
        tbody.appendChild(tr);
        tbody.appendChild(detailTr);
    }

    while (tbody.children.length > 200) {
        tbody.removeChild(tbody.lastChild);
        if (tbody.lastChild) tbody.removeChild(tbody.lastChild);
    }
}

function renderRow(entry, index, isNew) {
    return `<tr class="${isNew ? 'new-row' : ''}">${renderRowCells(entry, index)}</tr>
            <tr class="ac-detail-row" id="detail-${entry.id}"><td colspan="9"><div class="ac-detail-content">${renderDetailContent(entry)}</div></td></tr>`;
}

function renderRowCells(entry, index) {
    const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : '-';
    const isAutoRevoke = entry.auto_revoked;
    const decision = entry.decision || 'allow';

    let badgeClass = decision === 'allow' ? 'ac-badge-allow' : 'ac-badge-deny';
    let badgeText = decision.toUpperCase();
    if (isAutoRevoke) { badgeClass = 'ac-badge-revoke'; badgeText = 'AUTO-REVOKE'; }

    const risk = entry.risk_score;
    let riskHtml = '<span style="color:rgba(255,255,255,0.3)">—</span>';
    if (risk !== null && risk !== undefined) {
        const riskCls = risk > 0.7 ? 'ac-risk-high' : (risk > 0.3 ? 'ac-risk-med' : 'ac-risk-low');
        riskHtml = `<span class="${riskCls}">${risk.toFixed(2)}</span>`;
    }

    let trustHtml = '<span style="color:rgba(255,255,255,0.3)">—</span>';
    const tb = entry.trust_score_before;
    const ta = entry.trust_score_after;
    if (tb !== null && ta !== null && tb !== undefined && ta !== undefined) {
        const delta = ta - tb;
        const cls = delta < 0 ? 'down' : (delta > 0 ? 'up' : '');
        const arrow = delta < 0 ? '▼' : (delta > 0 ? '▲' : '—');
        trustHtml = `<span class="ac-trust-delta ${cls}">${tb.toFixed(2)} → ${ta.toFixed(2)} ${arrow} ${delta >= 0 ? '+' : ''}${delta.toFixed(2)}</span>`;
    } else if (ta !== null && ta !== undefined) {
        trustHtml = `<span style="color:rgba(255,255,255,0.5)">${ta.toFixed(2)}</span>`;
    }

    const reasonShort = (entry.reason || '-').length > 40 ? entry.reason.substring(0, 38) + '...' : (entry.reason || '-');

    let platformTag = '';
    if (entry.platform === 'feishu') {
        platformTag = '<span style="display:inline-block;background:rgba(59,130,246,0.2);color:#60a5fa;font-size:0.65rem;padding:1px 6px;border-radius:8px;margin-left:4px;vertical-align:middle">💬 Feishu</span>';
    }

    let chainHtml = '<span style="color:rgba(255,255,255,0.3)">—</span>';
    const dChain = entry.delegation_chain || entry.chain;
    if (dChain && dChain.length > 0) {
        const chainItems = dChain.map((c, i) => {
            let icon = '';
            let color = '#c4b5fd';
            if (c.startsWith('user:')) {
                icon = '👤';
                color = '#60a5fa';
            } else if (c === 'doc_agent') {
                icon = '📄';
                color = '#fbbf24';
            } else if (c === 'data_agent') {
                icon = '📊';
                color = '#34d399';
            } else if (c === 'external_agent') {
                icon = '🌐';
                color = '#f87171';
            }
            return `<span style="color:${color};font-weight:600;white-space:nowrap">${icon} ${c}</span>`;
        });
        chainHtml = chainItems.join('<span style="color:rgba(255,255,255,0.3);margin:0 2px">→</span>');
    }

    var explainBtnData = {
        agent_id: entry.agent_id || '',
        action: entry.action || '',
        decision: entry.decision === 'allow' ? 'allow' : 'deny',
        reason: entry.reason || '',
        trust_score: entry.trust_score_after || entry.trust_score_before,
        risk_score: entry.risk_score || 0,
        chain_detail: dChain || [],
        blocked_at: entry.blocked_at || '',
        auto_revoked: !!entry.auto_revoked,
        prompt_risk_score: entry.prompt_risk_score || null,
        attack_types: entry.attack_types || [],
        attack_intent: entry.attack_intent || '',
        severity: entry.severity || '',
    };

    return `
        <td style="white-space:nowrap;color:rgba(255,255,255,0.45);font-size:0.72rem">${ts}</td>
        <td style="color:#c4b5fd;font-weight:500">${entry.agent_id}${platformTag}</td>
        <td style="max-width:260px;font-size:0.72rem">${chainHtml}</td>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${entry.action}">${entry.action}</td>
        <td><span class="ac-badge ${badgeClass}">${badgeText}</span></td>
        <td style="max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:rgba(255,255,255,0.5);font-size:0.72rem" title="${entry.reason || ''}">${reasonShort}</td>
        <td>${riskHtml}</td>
        <td>${trustHtml}</td>
        <td style="white-space:nowrap">${IAM_EXPLAIN.makeBtn('🧠', explainBtnData, 'iam-explain-btn-sm')} <button class="ac-expand-btn" onclick="toggleDetail('${entry.id}')">展开</button></td>
    `;
}

function toggleDetail(id) {
    const row = document.getElementById(`detail-${id}`);
    if (!row) return;
    row.classList.toggle('open');
    const btn = row.previousElementSibling?.querySelector('.ac-expand-btn');
    if (btn) btn.textContent = row.classList.contains('open') ? '收起' : '展开';
}

function renderDetailContent(entry) {
    const ctx = entry.context || {};
    const sections = [];

    if (entry.platform && entry.platform !== 'web') {
        const platformLabel = entry.platform === 'feishu' ? '💬 Feishu' : entry.platform;
        const entryPointLabel = entry.entry_point || '-';
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Platform</div><div class="ac-detail-value">${platformLabel} (entry: ${entryPointLabel})</div></div>`);
    }

    if (entry.token_id) {
        const tokenDisplay = entry.token_id.length > 60 ? entry.token_id.substring(0, 57) + '...' : entry.token_id;
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Token</div><div class="ac-detail-value">${tokenDisplay}</div></div>`);
    }

    if (ctx.jti) {
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">JTI</div><div class="ac-detail-value">${ctx.jti}</div></div>`);
    }

    const dChain = entry.delegation_chain || entry.chain;
    if (dChain && dChain.length) {
        const chainSteps = dChain.map((c, i) => {
            let icon = '';
            let color = '#c4b5fd';
            let bg = 'rgba(196,181,253,0.08)';
            let borderColor = 'rgba(196,181,253,0.3)';
            if (c.startsWith('user:')) {
                icon = '👤';
                color = '#60a5fa';
                bg = 'rgba(96,165,250,0.08)';
                borderColor = 'rgba(96,165,250,0.3)';
            } else if (c === 'doc_agent') {
                icon = '📄';
                color = '#fbbf24';
                bg = 'rgba(251,191,36,0.08)';
                borderColor = 'rgba(251,191,36,0.3)';
            } else if (c === 'data_agent') {
                icon = '📊';
                color = '#34d399';
                bg = 'rgba(52,211,153,0.08)';
                borderColor = 'rgba(52,211,153,0.3)';
            } else if (c === 'external_agent') {
                icon = '🌐';
                color = '#f87171';
                bg = 'rgba(248,113,113,0.08)';
                borderColor = 'rgba(248,113,113,0.3)';
            }
            return `<span class="ac-trace-step pass" style="border-color:${borderColor};background:${bg};color:${color}">${icon} ${c}</span>`;
        });
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">🔗 Delegation Chain（委派链）</div><div class="ac-trace-pipeline">${chainSteps.join('<span style="color:rgba(255,255,255,0.3);margin:0 2px;font-weight:700">→</span>')}</div></div>`);
    }

    const delegationSteps = entry.delegation_steps || ctx.delegation_steps;
    if (delegationSteps && delegationSteps.length > 0) {
        let stepsHtml = '<div style="margin-top:8px">';
        stepsHtml += '<table style="width:100%;border-collapse:collapse;font-size:0.72rem">';
        stepsHtml += '<tr style="color:rgba(255,255,255,0.4);font-size:0.65rem;text-transform:uppercase"><th style="padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08)">From</th><th style="padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08)">To</th><th style="padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08)">Action</th><th style="padding:4px 8px;text-align:left;border-bottom:1px solid rgba(255,255,255,0.08)">Type</th></tr>';
        for (const step of delegationSteps) {
            const typeLabel = step.type === 'delegate' ? '<span style="color:#fbbf24">🔄 委派</span>' : '<span style="color:#34d399">⚡ 执行</span>';
            stepsHtml += `<tr><td style="padding:4px 8px;color:#60a5fa">${step.from}</td><td style="padding:4px 8px;color:#c4b5fd">${step.to}</td><td style="padding:4px 8px;color:rgba(255,255,255,0.6)">${step.action}</td><td style="padding:4px 8px">${typeLabel}</td></tr>`;
        }
        stepsHtml += '</table></div>';
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">📋 Delegation Steps（委派步骤）</div>${stepsHtml}</div>`);
    }

    if (ctx.capabilities && ctx.capabilities.length) {
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Capabilities</div><div class="ac-detail-value">${ctx.capabilities.join(', ')}</div></div>`);
    }

    const revoked = entry.revoked;
    if (revoked !== null && revoked !== undefined) {
        const revokeStatus = revoked ? '<span style="color:#ef4444;font-weight:700">🔥 REVOKED</span>' : '<span style="color:#34d399">✓ Active</span>';
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Revoke Status</div><div>${revokeStatus}</div></div>`);
    }

    if (entry.auto_revoked) {
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Auto-Revoke</div><div style="color:#ef4444;font-weight:700">🔥 AUTO-REVOKED — Trust score dropped below threshold, all tokens invalidated</div></div>`);
    }

    const traceSteps = [];
    traceSteps.push({ name: 'Token Valid', pass: !revoked && !entry.auto_revoked });
    traceSteps.push({ name: 'Capability Match', pass: entry.decision === 'allow' });
    traceSteps.push({ name: 'Trust Check', pass: (entry.trust_score_after ?? 1) >= 0.3 });
    traceSteps.push({ name: 'Final Decision', pass: entry.decision === 'allow' });
    sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Policy Trace</div><div class="ac-trace-pipeline">${traceSteps.map(s => `<span class="ac-trace-step ${s.pass ? 'pass' : 'fail'}">${s.pass ? '✓' : '✗'} ${s.name}</span>`).join('')}</div></div>`);

    if (ctx.delegated_user) {
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Delegated User</div><div class="ac-detail-value">${ctx.delegated_user}</div></div>`);
    }

    if (ctx._chain_hash) {
        sections.push(`<div class="ac-detail-section"><div class="ac-detail-label">Chain Hash</div><div class="ac-detail-value">${ctx._chain_hash}</div></div>`);
    }

    var explainData = {
        agent_id: entry.agent_id || '',
        action: entry.action || '',
        decision: entry.decision === 'allow' ? 'allow' : 'deny',
        reason: entry.reason || '',
        trust_score: entry.trust_score_after || entry.trust_score_before,
        risk_score: entry.risk_score || 0,
        chain_detail: dChain || [],
        blocked_at: entry.blocked_at || '',
        auto_revoked: entry.auto_revoked || false,
        prompt_risk_score: entry.prompt_risk_score || null,
        attack_types: entry.attack_types || [],
        attack_intent: entry.attack_intent || '',
        severity: entry.severity || '',
    };
    sections.push(`<div class="ac-detail-section" style="margin-top:8px">${IAM_EXPLAIN.makeBtn('🧠 Explain Decision', explainData, 'iam-explain-btn-lg')}</div>`);

    return sections.join('');
}

function refreshStats() {
    const total = _logEntries.length;
    const allow = _logEntries.filter(l => l.decision === 'allow').length;
    const deny = _logEntries.filter(l => l.decision === 'deny').length;
    const revoke = _logEntries.filter(l => l.auto_revoked).length;
    const highRisk = _logEntries.filter(l => l.risk_score !== null && l.risk_score > 0.7).length;

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statAllow').textContent = allow;
    document.getElementById('statDeny').textContent = deny;
    document.getElementById('statRevoke').textContent = revoke;
    document.getElementById('statHighRisk').textContent = highRisk;
}

async function verifyIntegrity() {
    const card = document.getElementById('integrityCard');
    const result = document.getElementById('integrityResult');
    card.style.display = 'block';
    result.innerHTML = '<div style="color:rgba(255,255,255,0.4)">验证中...</div>';
    try {
        const data = await fetchJSON(`${DELEGATE_BASE}/audit/integrity`);
        const valid = data.valid !== false;
        document.getElementById('statIntegrity').textContent = valid ? '✓' : '✗';
        document.getElementById('statIntegrity').style.color = valid ? '#34d399' : '#ef4444';
        let html = `<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
            <span class="${valid ? 'ac-integrity-ok' : 'ac-integrity-fail'}">${valid ? '✓' : '✗'}</span>
            <span style="font-weight:600;color:${valid ? '#34d399' : '#ef4444'}">${valid ? '哈希链完整' : '哈希链异常'}</span>
        </div>`;
        if (data.total_entries || data.total_logs) html += `<div style="font-size:0.82rem;color:rgba(255,255,255,0.5)">总条目: ${data.total_entries || data.total_logs} | 验证通过: ${data.verified_count || data.verified || data.total_entries || data.total_logs}</div>`;
        if (data.broken_links && data.broken_links.length) {
            html += `<div style="margin-top:6px;color:#f87171;font-size:0.82rem">断裂位置: ${data.broken_links.map(b => `#${b.index || b}`).join(', ')}</div>`;
        }
        result.innerHTML = html;
    } catch (e) {
        result.innerHTML = `<div style="color:#f87171">验证失败: ${e.message}</div>`;
    }
}

async function exportAudit() {
    try {
        const resp = await fetch(`${AUDIT_BASE}/export?format=json`);
        if (!resp.ok) {
            const errText = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${errText.substring(0, 200)}`);
        }
        const blob = await resp.blob();
        if (blob.size === 0) {
            throw new Error('导出数据为空');
        }
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audit_export_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
        alert('导出失败: ' + e.message);
    }
}

connectAuditWS();
loadAuditLogs();
