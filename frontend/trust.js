﻿var TE = (function() {
    var BASE = '/api/delegate';

    var canvas = document.getElementById('fieldCanvas');
    var ctx = canvas ? canvas.getContext('2d') : null;
    var canvasOk = !!(canvas && ctx);
    var W = 800, H = 600, dpr = 1;

    var _trustData = null;
    var _autoRevokeData = null;
    var _prevScores = {};
    var _trustHistory = {};
    var _selectedAgent = null;
    var _recentActions = {};
    var particles = [];

    var AGENT_DEFS = {
        'doc_agent': { icon: '📄', color: '#0071e3', rgb: '0,113,227', caps: ['read:doc', 'write:doc:public'], pos: { x: 0.25, y: 0.45 } },
        'data_agent': { icon: '📊', color: '#10b981', rgb: '16,185,129', caps: ['read:feishu_table'], pos: { x: 0.5, y: 0.55 } },
        'external_agent': { icon: '🌐', color: '#f59e0b', rgb: '245,158,11', caps: ['write:doc:public'], pos: { x: 0.75, y: 0.45 } },
    };

    function fetchJSON(url, opts) {
        opts = opts || {};
        return fetch(url, Object.assign({}, opts, { headers: Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}) })).then(function(r) { return r.json(); });
    }

    function getLevel(score, isRevoked) {
        if (isRevoked || score < 0.3) return 'revoked';
        if (score < 0.5) return 'danger';
        if (score < 0.7) return 'warning';
        return 'safe';
    }

    function getLevelColor(level) {
        switch (level) {
            case 'safe': return '#34d399';
            case 'warning': return '#fbbf24';
            case 'danger': return '#f87171';
            case 'revoked': return '#ef4444';
            default: return '#636366';
        }
    }

    function getLevelLabel(level) {
        switch (level) {
            case 'safe': return '🟢 STABLE';
            case 'warning': return '🟡 FLUCTUATING';
            case 'danger': return '🔴 UNSTABLE';
            case 'revoked': return '🔥 COLLAPSED';
            default: return '—';
        }
    }

    if (canvasOk) {
        function resize() {
            dpr = window.devicePixelRatio || 1;
            var rect = canvas.parentElement.getBoundingClientRect();
            W = rect.width || 800;
            H = rect.height || 600;
            canvas.width = Math.round(W * dpr);
            canvas.height = Math.round(H * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        function drawGrid() {
            ctx.strokeStyle = 'rgba(0,113,227,0.05)';
            ctx.lineWidth = 0.5;
            var step = 50;
            for (var x = 0; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
            for (var y = 0; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
        }

        function drawConnections() {
            var agents = _trustData ? _trustData.agents : {};
            var keys = Object.keys(AGENT_DEFS);
            for (var i = 0; i < keys.length; i++) {
                for (var j = i + 1; j < keys.length; j++) {
                    var a = AGENT_DEFS[keys[i]];
                    var b = AGENT_DEFS[keys[j]];
                    var ax = a.pos.x * W, ay = a.pos.y * H;
                    var bx = b.pos.x * W, by = b.pos.y * H;
                    var scoreA = agents[keys[i]] ? agents[keys[i]].trust_score : 0.5;
                    var scoreB = agents[keys[j]] ? agents[keys[j]].trust_score : 0.5;
                    var avg = (scoreA + scoreB) / 2;
                    var alpha = Math.max(0.02, avg * 0.08);
                    ctx.beginPath();
                    ctx.moveTo(ax, ay);
                    ctx.lineTo(bx, by);
                    ctx.strokeStyle = 'rgba(0,113,227,' + alpha.toFixed(3) + ')';
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        function spawnEnergyParticle(agentId) {
            var def = AGENT_DEFS[agentId];
            if (!def) return;
            var agents = _trustData ? _trustData.agents : {};
            var score = agents[agentId] ? agents[agentId].trust_score : 0.5;
            var isRevoked = _autoRevokeData && _autoRevokeData.auto_revoked_agents && (agentId in _autoRevokeData.auto_revoked_agents);
            var level = getLevel(score, isRevoked);
            var color = getLevelColor(level);
            var cx = def.pos.x * W;
            var cy = def.pos.y * H;
            var angle = Math.random() * Math.PI * 2;
            var speed = 0.3 + Math.random() * 0.5;
            particles.push({
                x: cx, y: cy,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                life: 1, color: color, rgb: color.replace('#', '').match(/.{2}/g).map(function(h) { return parseInt(h, 16); }).join(','),
                size: 1.5 + Math.random() * 2,
                maxLife: 0.6 + Math.random() * 0.4,
            });
        }

        function updateParticles() {
            for (var i = particles.length - 1; i >= 0; i--) {
                var p = particles[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= 0.01;
                if (p.life <= 0) particles.splice(i, 1);
            }
        }

        function drawParticles() {
            particles.forEach(function(p) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + p.rgb + ',' + (p.life * 0.3).toFixed(2) + ')';
                ctx.fill();
            });
        }

        function loop() {
            ctx.clearRect(0, 0, W, H);
            if (W < 10 || H < 10) { requestAnimationFrame(loop); return; }
            drawGrid();
            drawConnections();
            updateParticles();
            drawParticles();
            requestAnimationFrame(loop);
        }

        window.addEventListener('resize', resize);
        resize();
        loop();

        setInterval(function() {
            var keys = Object.keys(AGENT_DEFS);
            var agentId = keys[Math.floor(Math.random() * keys.length)];
            spawnEnergyParticle(agentId);
        }, 300);
    }

    function renderEnergyNodes() {
        var container = document.getElementById('energyNodes');
        if (!container) return;
        var agents = _trustData ? _trustData.agents : {};
        var autoRevokedAgents = (_autoRevokeData && _autoRevokeData.auto_revoked_agents) || {};

        var html = '';
        Object.keys(AGENT_DEFS).forEach(function(id) {
            var def = AGENT_DEFS[id];
            var info = agents[id];
            var score = info ? info.trust_score : 0;
            var isRevoked = id in autoRevokedAgents;
            var level = getLevel(score, isRevoked);
            var color = getLevelColor(level);
            var label = getLevelLabel(level);
            var prevScore = _prevScores[id];
            var delta = prevScore !== undefined ? score - prevScore : 0;
            var deltaHtml = '';
            if (delta !== 0 && prevScore !== undefined) {
                var arrow = delta < 0 ? '▼' : '▲';
                var dColor = delta < 0 ? '#ef4444' : '#34d399';
                deltaHtml = '<div style="font-size:0.6rem;color:' + dColor + ';font-weight:700">' + arrow + ' ' + (delta >= 0 ? '+' : '') + delta.toFixed(2) + '</div>';
            }

            var left = def.pos.x * 100;
            var top = def.pos.y * 100;

            html += '<div class="te-energy-node' + (_selectedAgent === id ? ' selected' : '') + '" style="left:' + left + '%;top:' + top + '%;transform:translate(-50%,-50%)" onclick="TE.selectAgent(\'' + id + '\')">';
            html += '<div class="te-node-core ' + level + '">';
            html += '<div class="te-node-ring ' + level + '"></div>';
            html += '<div class="te-node-ring2 ' + level + '"></div>';
            html += def.icon;
            html += '</div>';
            html += '<div class="te-node-label" style="color:' + def.color + '">' + id.replace('_', ' ') + '</div>';
            html += '<div class="te-node-score" style="color:' + color + '">' + score.toFixed(2) + '</div>';
            html += deltaHtml;
            html += '<div class="te-node-status" style="color:' + color + '">' + label + '</div>';
            html += '</div>';
        });
        container.innerHTML = html;
    }

    function updateStats() {
        var agents = _trustData ? _trustData.agents : {};
        var autoRevokedAgents = (_autoRevokeData && _autoRevokeData.auto_revoked_agents) || {};
        var safe = 0, warning = 0, danger = 0, revoked = 0;
        Object.keys(agents).forEach(function(id) {
            var level = getLevel(agents[id].trust_score, id in autoRevokedAgents);
            if (level === 'safe') safe++;
            else if (level === 'warning') warning++;
            else if (level === 'danger') danger++;
            else revoked++;
        });
        var e;
        e = document.getElementById('statAgents'); if (e) e.textContent = Object.keys(agents).length;
        e = document.getElementById('statSafe'); if (e) e.textContent = safe;
        e = document.getElementById('statWarning'); if (e) e.textContent = warning;
        e = document.getElementById('statDanger'); if (e) e.textContent = danger;
        e = document.getElementById('statRevoked'); if (e) e.textContent = revoked;
        e = document.getElementById('trustThreshold'); if (e) e.textContent = (_trustData && _trustData.threshold != null) ? _trustData.threshold : '0.5';
        e = document.getElementById('autoRevokeThreshold'); if (e) e.textContent = (_autoRevokeData && _autoRevokeData.threshold != null) ? _autoRevokeData.threshold : '0.3';
    }

    function selectAgent(agentId) {
        _selectedAgent = agentId;
        renderEnergyNodes();
        renderDetail(agentId);
    }

    function renderDetail(agentId) {
        var content = document.getElementById('detailContent');
        var historySection = document.getElementById('historySection');
        var actionsSection = document.getElementById('actionsSection');
        var riskSection = document.getElementById('riskSection');
        if (!content) return;

        var agents = _trustData ? _trustData.agents : {};
        var autoRevokedAgents = (_autoRevokeData && _autoRevokeData.auto_revoked_agents) || {};
        var info = agents[agentId];
        var def = AGENT_DEFS[agentId];
        if (!info || !def) {
            content.innerHTML = '<div class="te-detail-empty">Agent 数据不可用</div>';
            if (historySection) historySection.style.display = 'none';
            if (actionsSection) actionsSection.style.display = 'none';
            if (riskSection) riskSection.className = 'te-risk-reasons';
            return;
        }

        var score = info.trust_score;
        var isRevoked = agentId in autoRevokedAgents;
        var level = getLevel(score, isRevoked);
        var color = getLevelColor(level);
        var badge = '<span class="te-badge te-badge-' + level + '">' + getLevelLabel(level) + '</span>';

        var html = '';
        html += '<div class="te-detail-row"><span class="te-detail-key">Agent</span><span class="te-detail-val" style="color:' + def.color + '">' + def.icon + ' ' + agentId + '</span></div>';
        html += '<div class="te-detail-row"><span class="te-detail-key">Trust Score</span><span class="te-detail-val" style="color:' + color + ';font-size:1.1rem;font-weight:800">' + score.toFixed(2) + '</span></div>';
        html += '<div class="te-detail-row"><span class="te-detail-key">Status</span><span class="te-detail-val">' + badge + '</span></div>';
        html += '<div class="te-detail-row"><span class="te-detail-key">Capabilities</span><span class="te-detail-val" style="font-size:0.7rem">' + def.caps.join(', ') + '</span></div>';
        if (isRevoked) {
            var revokeInfo = autoRevokedAgents[agentId];
            html += '<div class="te-detail-row"><span class="te-detail-key">Revoke Reason</span><span class="te-detail-val" style="color:#ef4444;font-size:0.7rem">' + (revokeInfo && revokeInfo.reason ? revokeInfo.reason : 'Trust below threshold') + '</span></div>';
        }

        var explainData = {
            agent_id: agentId,
            action: def.caps[0] || '',
            decision: (level === 'revoked' || level === 'danger') ? 'deny' : 'allow',
            reason: isRevoked ? 'Trust score collapsed below auto-revoke threshold' : (level === 'warning' ? 'Trust score declining' : 'Normal access'),
            trust_score: score,
            risk_score: 1 - score,
            chain_detail: [],
            blocked_at: (level === 'revoked' || level === 'danger') ? 'trust_check' : '',
            auto_revoked: isRevoked,
            prompt_risk_score: null,
            attack_types: [],
            attack_intent: '',
            severity: '',
        };
        html += '<div style="margin-top:12px">' + IAM_EXPLAIN.makeBtn('🧠 Explain Decision', explainData, 'iam-explain-btn-lg') + '</div>';

        content.innerHTML = html;

        if (historySection) {
            historySection.style.display = 'block';
            drawHistory(agentId);
        }

        if (actionsSection) {
            actionsSection.style.display = 'block';
            renderRecentActions(agentId);
        }

        if (riskSection) {
            if (level === 'danger' || level === 'revoked' || level === 'warning') {
                riskSection.className = 'te-risk-reasons show';
                var reasons = [];
                if (isRevoked) {
                    reasons.push('Trust score collapsed below auto-revoke threshold');
                    reasons.push('All tokens for this agent have been revoked');
                }
                if (level === 'warning') reasons.push('Trust score declining — monitor closely');
                if (level === 'danger') reasons.push('Trust score critically low — auto-revoke imminent');
                if (score < 0.5) reasons.push('Multiple denied requests detected');
                document.getElementById('riskReasons').innerHTML = reasons.map(function(r) { return '<div class="te-risk-item">' + r + '</div>'; }).join('');
            } else {
                riskSection.className = 'te-risk-reasons';
            }
        }
    }

    function drawHistory(agentId) {
        var hCanvas = document.getElementById('historyCanvas');
        if (!hCanvas) return;
        var hCtx = hCanvas.getContext('2d');
        var rect = hCanvas.getBoundingClientRect();
        var w = rect.width || 300;
        var h = rect.height || 80;
        var d = window.devicePixelRatio || 1;
        hCanvas.width = Math.round(w * d);
        hCanvas.height = Math.round(h * d);
        hCtx.setTransform(d, 0, 0, d, 0, 0);

        var history = _trustHistory[agentId] || [];
        if (!_trustHistory[agentId]) {
            var agents = _trustData ? _trustData.agents : {};
            var score = agents[agentId] ? agents[agentId].trust_score : 0.5;
            history = [score, score, score, score, score];
        }

        hCtx.clearRect(0, 0, w, h);

        hCtx.strokeStyle = 'rgba(255,255,255,0.04)';
        hCtx.lineWidth = 0.5;
        for (var i = 0; i <= 4; i++) {
            var y = (h / 4) * i;
            hCtx.beginPath(); hCtx.moveTo(0, y); hCtx.lineTo(w, y); hCtx.stroke();
        }

        var level = getLevel(history[history.length - 1], false);
        var lineColor = getLevelColor(level);

        hCtx.beginPath();
        var step = w / Math.max(history.length - 1, 1);
        history.forEach(function(val, idx) {
            var x = idx * step;
            var y = h - (val * h);
            if (idx === 0) hCtx.moveTo(x, y);
            else hCtx.lineTo(x, y);
        });
        hCtx.strokeStyle = lineColor;
        hCtx.lineWidth = 2;
        hCtx.stroke();

        hCtx.beginPath();
        history.forEach(function(val, idx) {
            var x = idx * step;
            var y = h - (val * h);
            hCtx.lineTo(x, y);
        });
        hCtx.lineTo(w, h);
        hCtx.lineTo(0, h);
        hCtx.closePath();
        var grad = hCtx.createLinearGradient(0, 0, 0, h);
        grad.addColorStop(0, lineColor.replace(')', ',0.15)').replace('rgb', 'rgba'));
        grad.addColorStop(1, 'rgba(0,0,0,0)');
        hCtx.fillStyle = grad;
        hCtx.fill();

        var lastVal = history[history.length - 1];
        var lastX = (history.length - 1) * step;
        var lastY = h - (lastVal * h);
        hCtx.beginPath();
        hCtx.arc(lastX, lastY, 4, 0, Math.PI * 2);
        hCtx.fillStyle = lineColor;
        hCtx.fill();
    }

    function renderRecentActions(agentId) {
        var list = document.getElementById('actionsList');
        if (!list) return;
        var actions = _recentActions[agentId] || [];
        if (actions.length === 0) {
            list.innerHTML = '<div style="color:rgba(255,255,255,0.2);font-size:0.7rem">暂无行为记录</div>';
            return;
        }
        list.innerHTML = actions.map(function(a) {
            return '<div class="te-action-item"><span class="te-action-dot ' + a.type + '"></span><span class="te-action-text">' + a.action + '</span></div>';
        }).join('');
    }

    async function loadData() {
        try {
            var results = await Promise.all([
                fetchJSON(BASE + '/trust'),
                fetchJSON(BASE + '/auto-revoke/list'),
            ]);
            var trust = results[0];
            var autoRevoke = results[1];

            var newScores = {};
            if (trust.agents) {
                Object.keys(trust.agents).forEach(function(id) {
                    var newScore = trust.agents[id].trust_score;
                    var prevScore = _prevScores[id];
                    if (prevScore !== undefined && Math.abs(newScore - prevScore) > 0.001) {
                        if (!_trustHistory[id]) _trustHistory[id] = [prevScore];
                        _trustHistory[id].push(newScore);
                        if (_trustHistory[id].length > 20) _trustHistory[id].shift();
                    }
                    newScores[id] = newScore;
                });
            }
            _prevScores = newScores;
            _trustData = trust;
            _autoRevokeData = autoRevoke;

            updateStats();
            renderEnergyNodes();
            if (_selectedAgent) renderDetail(_selectedAgent);
        } catch (e) {
            console.error('加载信任数据失败:', e);
        }
    }

    async function resetTrust() {
        try {
            await fetchJSON(BASE + '/trust/reset', { method: 'POST' });
            _prevScores = {};
            _trustHistory = {};
            _recentActions = {};
            await loadData();
        } catch (e) {
            console.error('重置信任分失败:', e);
        }
    }

    async function demoDegrade() {
        var logEl = document.getElementById('demoLog');
        var insightEl = document.getElementById('demoInsight');
        if (logEl) logEl.innerHTML = '';
        if (insightEl) insightEl.innerHTML = '';
        try {
            var data = await fetchJSON(BASE + '/demo/trust-degrade', { method: 'POST' });
            await loadData();
            var steps = data.steps || [];
            var html = '<div class="te-demo-log">';
            html += '<div style="margin-bottom:8px;font-weight:600;color:#fbbf24;font-size:0.82rem">📉 信任降级: ' + (data.flow || '') + '</div>';
            steps.forEach(function(s) {
                var badge = '';
                if (s.status === 'BLOCKED_BY_TRUST') badge = '<span class="badge deny">🚫 BLOCKED</span>';
                else if (s.allowed === false) badge = '<span class="badge deny">DENY</span>';
                else if (s.allowed === true) badge = '<span class="badge allow">ALLOW</span>';
                else badge = '<span class="badge" style="background:rgba(167,139,250,0.15);color:#c4b5fd">INFO</span>';
                var trustInfo = s.trust_score_before !== undefined ? s.trust_score_before + ' → ' + s.trust_score_after : '';
                html += '<div class="te-demo-step"><span class="num">#' + (s.step || '') + '</span><span class="action">' + (s.action || '') + '</span>' + badge + (trustInfo ? '<span style="color:rgba(255,255,255,0.3);font-size:0.6rem">' + trustInfo + '</span>' : '') + '</div>';
            });
            html += '</div>';
            if (logEl) logEl.innerHTML = html;
            if (insightEl) insightEl.innerHTML = '<div class="te-insight-box yellow" style="margin-top:12px"><div class="te-insight-title yellow">💡 信任降级机制</div><div style="color:rgba(255,255,255,0.65)">' + (data.key_insight || '每次被拒绝的请求都会降低信任评分，连续拒绝将触发自动撤销') + '</div></div>';
        } catch (e) {
            if (logEl) logEl.innerHTML = '<div style="color:#f87171;padding:12px">演示失败: ' + e.message + '</div>';
        }
    }

    async function demoAutoRevoke() {
        var logEl = document.getElementById('demoLog');
        var insightEl = document.getElementById('demoInsight');
        if (logEl) logEl.innerHTML = '';
        if (insightEl) insightEl.innerHTML = '';
        try {
            var data = await fetchJSON(BASE + '/demo/auto-revoke', { method: 'POST' });
            await loadData();
            var steps = data.steps || [];
            var html = '<div class="te-demo-log">';
            html += '<div style="margin-bottom:8px;font-weight:600;color:#ef4444;font-size:0.82rem">🔥 自动撤销: ' + (data.flow || '') + '</div>';
            steps.forEach(function(s) {
                var badge = '';
                if (s.status === 'AUTO_REVOKED' || s.auto_revoked) badge = '<span class="badge revoke">🔥 REVOKED</span>';
                else if (s.allowed === false) badge = '<span class="badge deny">DENY</span>';
                else if (s.allowed === true) badge = '<span class="badge allow">ALLOW</span>';
                else badge = '<span class="badge" style="background:rgba(167,139,250,0.15);color:#c4b5fd">INFO</span>';
                var trustInfo = s.trust_score_before !== undefined ? s.trust_score_before + ' → ' + s.trust_score_after : '';
                html += '<div class="te-demo-step"><span class="num">#' + (s.step || '') + '</span><span class="action">' + (s.action || '') + '</span>' + badge + (trustInfo ? '<span style="color:rgba(255,255,255,0.3);font-size:0.6rem">' + trustInfo + '</span>' : '') + '</div>';
            });
            html += '</div>';
            if (logEl) logEl.innerHTML = html;
            var color = data.auto_revoked_triggered ? 'red' : 'yellow';
            if (insightEl) insightEl.innerHTML = '<div class="te-insight-box ' + color + '" style="margin-top:12px"><div class="te-insight-title ' + color + '">' + (data.auto_revoked_triggered ? '🔥 安全闭环' : '⚠️ 信任降级') + '</div><div style="color:rgba(255,255,255,0.65)">' + (data.key_insight || '信任评分正在下降') + '</div></div>';
        } catch (e) {
            if (logEl) logEl.innerHTML = '<div style="color:#f87171;padding:12px">演示失败: ' + e.message + '</div>';
        }
    }

    loadData();

    setInterval(loadData, 5000);

    return {
        refresh: loadData,
        resetTrust: resetTrust,
        selectAgent: selectAgent,
        demoDegrade: demoDegrade,
        demoAutoRevoke: demoAutoRevoke,
    };
})();
