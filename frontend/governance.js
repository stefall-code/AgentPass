var GOV = (function() {
    var BASE = '/api/governance';
    var refreshTimer = null;
    var currentFilter = null;

    function fetchJSON(url, opts) {
        opts = opts || {};
        var headers = {};
        if (opts.body) {
            headers['Content-Type'] = 'application/json';
        }
        if (opts.headers) {
            Object.assign(headers, opts.headers);
        }
        return fetch(url, Object.assign({}, opts, { headers: headers }))
            .then(function(r) {
                if (!r.ok) {
                    return r.text().then(function(t) {
                        throw new Error('HTTP ' + r.status + ': ' + t.substring(0, 200));
                    });
                }
                return r.json();
            });
    }

    function getPlatformTag(platform) {
        var cls = { feishu: 'tag-feishu', web: 'tag-web', api: 'tag-api' }[platform] || 'tag-web';
        var icon = { feishu: '💬', web: '🌐', api: '⚡' }[platform] || '🌐';
        return '<span class="platform-tag ' + cls + '">' + icon + ' ' + platform.toUpperCase() + '</span>';
    }

    function getStatusBadge(status) {
        if (status === 'success') return '<span style="color:#34d399;font-weight:700">✅ ALLOW</span>';
        if (status === 'auto_revoked') return '<span style="color:#ef4444;font-weight:700">🔥 REVOKED</span>';
        if (status === 'denied') return '<span style="color:#ef4444;font-weight:700">❌ DENY</span>';
        if (status === 'error') return '<span style="color:#fbbf24;font-weight:700">⚠️ ERROR</span>';
        return '<span style="color:rgba(255,255,255,0.4)">' + (status || '?') + '</span>';
    }

    function getTrustColor(score) {
        if (score == null) return 'rgba(255,255,255,0.3)';
        if (score >= 0.7) return '#34d399';
        if (score >= 0.5) return '#fbbf24';
        return '#ef4444';
    }

    function getTrustStatus(score) {
        if (score == null) return { cls: 'status-safe', text: '—' };
        if (score >= 0.7) return { cls: 'status-safe', text: '🟢 SAFE' };
        if (score >= 0.5) return { cls: 'status-warn', text: '🟡 WARN' };
        if (score >= 0.3) return { cls: 'status-danger', text: '🔴 DANGER' };
        return { cls: 'status-revoked', text: '🔥 REVOKED' };
    }

    function formatTime(ts) {
        if (!ts) return '';
        var d = new Date(ts * 1000);
        return d.toLocaleTimeString();
    }

    function showGlobalAlert(msg, type) {
        console.log('[GOV] showGlobalAlert:', msg, type);
        var debugEl = document.getElementById('govDebug');
        if (debugEl) debugEl.innerHTML += '<div>[ALERT] ' + msg + '</div>';
        var existing = document.querySelector('.gov-alert');
        if (existing) existing.remove();
        var el = document.createElement('div');
        el.className = 'gov-alert';
        if (type === 'success') {
            el.style.background = 'linear-gradient(135deg,rgba(52,211,153,0.92),rgba(16,185,129,0.92))';
        } else if (type === 'error') {
            el.style.background = 'linear-gradient(135deg,rgba(239,68,68,0.92),rgba(220,38,38,0.92))';
        } else {
            el.style.background = 'linear-gradient(135deg,rgba(139,92,246,0.92),rgba(167,139,250,0.92))';
        }
        el.textContent = msg;
        document.body.appendChild(el);
        setTimeout(function() {
            el.style.transition = 'opacity 0.5s';
            el.style.opacity = '0';
            setTimeout(function() { if (el.parentNode) el.remove(); }, 500);
        }, 5000);
    }

    function updateOverview(data) {
        var summary = data.summary || {};
        var e;
        e = document.getElementById('statTotal'); if (e) e.textContent = summary.total_events || 0;
        e = document.getElementById('statSuccess'); if (e) e.textContent = summary.total_success || 0;
        e = document.getElementById('statDenied'); if (e) e.textContent = summary.total_denied || 0;
        e = document.getElementById('statDenyRate'); if (e) e.textContent = Math.round((summary.deny_rate || 0) * 100) + '%';
        e = document.getElementById('statRevoked'); if (e) e.textContent = (summary.auto_revoked_agents || []).length;

        var ps = data.platform_stats || {};
        ['feishu', 'web', 'api'].forEach(function(p) {
            var s = ps[p] || {};
            var cap = p.charAt(0).toUpperCase() + p.slice(1);
            e = document.getElementById('ps' + cap + 'Total'); if (e) e.textContent = s.total || 0;
            e = document.getElementById('ps' + cap + 'Deny'); if (e) e.textContent = s.denied || 0;
            e = document.getElementById('ps' + cap + 'Risk'); if (e) e.textContent = (s.avg_risk || 0).toFixed(2);
        });

        var agents = data.top_risky_agents || [];
        var list = document.getElementById('agentList');
        if (list) {
            var html = '';
            agents.forEach(function(a) {
                var st = a.auto_revoked ? { cls: 'status-revoked', text: '🔥 REVOKED' } : getTrustStatus(a.trust_score);
                var pct = Math.max(0, Math.min(100, (a.trust_score || 0) * 100));
                html += '<div class="agent-row">';
                html += '<div><div class="agent-name">' + a.agent_id + '</div>';
                html += '<div class="trust-bar"><div class="trust-fill" style="width:' + pct + '%;background:' + getTrustColor(a.trust_score) + '"></div></div></div>';
                html += '<div style="text-align:right"><div class="agent-trust" style="color:' + getTrustColor(a.trust_score) + '">' + (a.trust_score != null ? a.trust_score.toFixed(2) : '—') + '</div>';
                html += '<span class="agent-status ' + st.cls + '">' + st.text + '</span></div>';
                html += '</div>';
            });
            list.innerHTML = html;
        }
    }

    function updateEvents(events) {
        var stream = document.getElementById('eventStream');
        if (!stream) return;

        var html = '';
        (events || []).slice().reverse().forEach(function(ev) {
            var platform = ev.platform || 'web';
            var explainData = {
                agent_id: ev.agent_id || ev.agent || '',
                action: ev.action || '',
                decision: ev.result === 'success' ? 'allow' : 'deny',
                reason: '',
                trust_score: ev.trust_score,
                risk_score: ev.platform_risk || 0,
                chain_detail: ev.chain || [],
                blocked_at: ev.blocked_at || '',
                auto_revoked: ev.auto_revoked || false,
                prompt_risk_score: ev.prompt_risk_score || null,
                attack_types: ev.attack_types || [],
                attack_intent: ev.attack_intent || '',
                severity: ev.severity || '',
            };
            html += '<div class="event-item">';
            html += '<div class="event-header">';
            html += getPlatformTag(platform) + ' ' + getStatusBadge(ev.result);
            html += '<span class="event-time">' + formatTime(ev.timestamp) + '</span>';
            html += '</div>';
            html += '<div class="event-detail">' + (ev.agent_id || ev.agent || '') + ' → ' + (ev.action || '') + (ev.trust_score != null && ev.trust_score !== undefined ? ' | trust: ' + Number(ev.trust_score).toFixed(2) : '') + '</div>';
            html += '<div style="margin-top:6px">' + IAM_EXPLAIN.makeBtn('🧠 Explain', explainData) + '</div>';
            html += '</div>';
        });
        stream.innerHTML = html || '<div style="color:rgba(255,255,255,0.2);text-align:center;padding:16px;font-size:0.82rem">暂无事件</div>';
    }

    async function refresh() {
        try {
            var data = await fetchJSON(BASE + '/overview');
            updateOverview(data);
            var eventsData = await fetchJSON(BASE + '/events?limit=30');
            updateEvents(eventsData.events);
        } catch (e) {
            console.error('Governance refresh error:', e);
        }
    }

    async function runDemo() {
        console.log('[GOV] runDemo called');
        showGlobalAlert('🎬 正在执行跨平台演示...', 'info');
        try {
            var result = await fetchJSON(BASE + '/demo/cross-platform', { method: 'POST' });
            console.log('[GOV] runDemo result:', result);
            var steps = result.steps || [];
            var summary = result.summary || {};

            var lines = ['🎬 跨平台演示完成'];
            steps.forEach(function(s) {
                var icon = s.status === 'success' ? '✅' : (s.status === 'denied' ? '❌' : (s.status === 'revoked' ? '🔥' : '⚠️'));
                lines.push(icon + ' Step' + s.step + ' ' + s.platform + ': ' + s.status);
            });

            var alertType = (summary.feishu_after_revoke === 'auto_revoked') ? 'error' : 'success';
            showGlobalAlert(lines.join('  |  '), alertType);
            refresh();
        } catch (e) {
            console.error('[GOV] runDemo error:', e);
            showGlobalAlert('❌ 演示执行失败: ' + e.message, 'error');
        }
    }

    async function demoPlatform(platform) {
        console.log('[GOV] demoPlatform called with:', platform);
        var messages = {
            feishu: '帮我生成财务报告',
            web: '帮我查一下财务数据',
            api: '读取薪资数据',
        };
        var platformName = { feishu: '飞书', web: 'Web', api: 'API' }[platform] || platform;
        showGlobalAlert('⏳ 正在执行 ' + platformName + ' 演示...', 'info');
        try {
            console.log('[GOV] demoPlatform started:', platform);
            var resetResult = await fetchJSON(BASE + '/reset-all', { method: 'POST' });
            console.log('[GOV] reset-all result:', resetResult);

            var result = await fetchJSON('/api/feishu/test', {
                method: 'POST',
                body: JSON.stringify({
                    user_id: platform + '_demo_user',
                    message: messages[platform] || '帮我查数据',
                    platform: platform
                }),
            });
            console.log('[GOV] feishu/test result:', result);

            var status = result.status || 'unknown';
            var trust = result.trust_score != null ? result.trust_score.toFixed(2) : '—';
            var msg, alertType;

            if (status === 'success') {
                msg = '✅ ' + platformName + ' 平台请求成功 — trust: ' + trust;
                alertType = 'success';
            } else if (status === 'denied') {
                msg = '❌ ' + platformName + ' 平台请求被拒绝 — trust: ' + trust;
                alertType = 'error';
            } else if (status === 'auto_revoked') {
                msg = '🔥 ' + platformName + ' 平台 Agent 已被封禁 — trust: ' + trust;
                alertType = 'error';
            } else {
                msg = '⚠️ ' + platformName + ' 平台返回: ' + status + ' — trust: ' + trust;
                alertType = 'info';
            }

            console.log('[GOV] showing alert:', msg, alertType);
            showGlobalAlert(msg, alertType);
            refresh();
        } catch (e) {
            console.error('[GOV] demoPlatform error:', e);
            showGlobalAlert('❌ ' + platformName + ' 演示失败: ' + e.message, 'error');
        }
    }

    async function demoEscalation() {
        showGlobalAlert('🔒 正在执行越权拦截演示...', 'info');
        try {
            var result = await fetchJSON('/api/delegate/demo/escalation-attack', { method: 'POST' });
            var steps = result.steps || [];
            var blocked = steps.find(function(s) { return s.success === false; });
            if (blocked) {
                showGlobalAlert('🔒 越权拦截成功! external_agent → data_agent 被拒绝: ' + (blocked.reason || '').substring(0, 80), 'error');
            } else {
                showGlobalAlert('⚠️ 越权拦截演示完成', 'info');
            }
            refresh();
        } catch (e) {
            showGlobalAlert('❌ 越权拦截演示失败: ' + e.message, 'error');
        }
    }

    async function runCollaboration() {
        var reportEl = document.getElementById('collabReport');
        if (reportEl) {
            reportEl.style.display = 'block';
            reportEl.textContent = '⏳ 正在执行三Agent协作（真实后端调用）...\n\n';
            reportEl.textContent += '👤 用户发起请求: "综合分析AI技术趋势"\n';
            reportEl.textContent += '   → 触发协作任务类型（关键词"综合"匹配）\n\n';
        }
        showGlobalAlert('🔗 正在执行三Agent协作演示...', 'info');
        try {
            var resetResult = await fetchJSON(BASE + '/reset-all', { method: 'POST' });
            var result = await fetchJSON('/api/feishu/test', {
                method: 'POST',
                body: JSON.stringify({ user_id: 'gov_collab_user', message: '综合分析AI技术趋势' }),
            });
            var status = result.status || 'unknown';
            var trust = result.trust_score != null ? result.trust_score.toFixed(2) : '—';
            var capability = result.capability || 'N/A';
            var chain = result.chain || [];
            var platformRisk = result.platform_risk;
            var content = result.content || '';

            if (reportEl) {
                var lines = [];
                lines.push('━━━ 协作执行结果 ━━━');
                lines.push('status: ' + status);
                lines.push('capability: ' + capability);
                lines.push('trust_score: ' + trust);
                if (chain.length > 0) lines.push('delegation_chain: ' + chain.join(' → '));
                if (platformRisk != null) lines.push('platform_risk: ' + platformRisk.toFixed(2));
                lines.push('');
                lines.push('━━━ 协作报告 ━━━');
                lines.push(content || '（无内容）');
                reportEl.textContent = lines.join('\n');
            }

            if (status === 'success' || status === 'degraded') {
                var msg = '✅ 三Agent协作成功! ';
                msg += 'trust: ' + trust + ' | ';
                msg += 'capability: ' + capability + ' | ';
                msg += 'chain: ' + chain.join(' → ');
                showGlobalAlert(msg, 'success');
            } else {
                showGlobalAlert('⚠️ 协作返回: ' + status + ' trust: ' + trust, 'info');
            }
            refresh();
        } catch (e) {
            if (reportEl) {
                reportEl.textContent = '❌ 三Agent协作演示失败: ' + e.message;
            }
            showGlobalAlert('❌ 三Agent协作演示失败: ' + e.message, 'error');
        }
    }

    async function revokeAgent(agentId) {
        try {
            var result = await fetchJSON(BASE + '/revoke-agent', {
                method: 'POST',
                body: JSON.stringify({ agent_id: agentId, reason: 'Manual revocation from governance console' }),
            });
            showGlobalAlert('🔥 Agent ' + agentId + ' 已被撤销 — trust: ' + (result.trust_before || 0).toFixed(2) + ' → 0.00', 'error');
            refresh();
        } catch (e) {
            console.error('Revoke error:', e);
            showGlobalAlert('❌ 撤销失败: ' + e.message, 'error');
        }
    }

    async function resetAgent(agentId) {
        try {
            var result = await fetchJSON(BASE + '/reset-agent?agent_id=' + agentId, { method: 'POST' });
            showGlobalAlert('🔄 Agent ' + agentId + ' 已重置 — trust: ' + (result.trust_score || 0).toFixed(2), 'success');
            refresh();
        } catch (e) {
            console.error('Reset error:', e);
            showGlobalAlert('❌ 重置失败: ' + e.message, 'error');
        }
    }

    function filterPlatform(platform) {
        currentFilter = currentFilter === platform ? null : platform;
        refresh();
    }

    function startAutoRefresh() {
        if (refreshTimer) clearInterval(refreshTimer);
        refreshTimer = setInterval(refresh, 3000);
    }

    refresh();
    startAutoRefresh();

    return {
        refresh: refresh,
        runDemo: runDemo,
        runCollaboration: runCollaboration,
        demoEscalation: demoEscalation,
        demoPlatform: demoPlatform,
        revokeAgent: revokeAgent,
        resetAgent: resetAgent,
        filterPlatform: filterPlatform,
    };
})();
