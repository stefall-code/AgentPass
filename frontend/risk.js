var RR = (function() {
    var BASE = '/api/delegate';
    var AUDIT_BASE = '/api/delegate/audit';
    var WS_BASE = location.protocol === 'https:' ? 'wss:' : 'ws:';

    var canvas = document.getElementById('radarCanvas');
    var ctx = canvas ? canvas.getContext('2d') : null;
    var canvasOk = !!(canvas && ctx);
    var W = 800, H = 620, dpr = 1;
    var radarAngle = 0;
    var signals = [];
    var burstEffects = [];
    var pulseRings = [];

    var stats = { total: 0, denied: 0, highRisk: 0, autoRevoke: 0, revokedAgents: 0 };
    var attackCounts = { replay: 0, escalation: 0, autoRevoke: 0, capDeny: 0 };
    var _ws = null;

    function fetchJSON(url, opts) {
        opts = opts || {};
        return fetch(url, Object.assign({}, opts, { headers: Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}) })).then(function(r) { return r.json(); });
    }

    if (canvasOk) {
        function resize() {
            dpr = window.devicePixelRatio || 1;
            var rect = canvas.parentElement.getBoundingClientRect();
            W = rect.width || 800;
            H = rect.height || 620;
            canvas.width = Math.round(W * dpr);
            canvas.height = Math.round(H * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        function getRadarCenter() {
            return { x: W * 0.5, y: H * 0.52 };
        }

        function getRadarRadius() {
            return Math.min(W, H) * 0.38;
        }

        function drawRadarBase() {
            var c = getRadarCenter();
            var r = getRadarRadius();

            for (var i = 1; i <= 4; i++) {
                var ringR = r * (i / 4);
                ctx.beginPath();
                ctx.arc(c.x, c.y, ringR, 0, Math.PI * 2);
                ctx.strokeStyle = 'rgba(0,113,227,' + (0.12 - i * 0.02).toFixed(2) + ')';
                ctx.lineWidth = 0.5;
                ctx.stroke();
            }

            ctx.beginPath();
            ctx.moveTo(c.x - r, c.y);
            ctx.lineTo(c.x + r, c.y);
            ctx.strokeStyle = 'rgba(0,113,227,0.06)';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(c.x, c.y - r);
            ctx.lineTo(c.x, c.y + r);
            ctx.strokeStyle = 'rgba(0,113,227,0.06)';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(c.x - r * 0.707, c.y - r * 0.707);
            ctx.lineTo(c.x + r * 0.707, c.y + r * 0.707);
            ctx.strokeStyle = 'rgba(0,113,227,0.04)';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(c.x + r * 0.707, c.y - r * 0.707);
            ctx.lineTo(c.x - r * 0.707, c.y + r * 0.707);
            ctx.strokeStyle = 'rgba(0,113,227,0.04)';
            ctx.lineWidth = 0.5;
            ctx.stroke();

            var labels = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
            for (var i = 0; i < 4; i++) {
                var ringR = r * ((i + 1) / 4);
                ctx.font = '8px sans-serif';
                ctx.fillStyle = 'rgba(255,255,255,0.08)';
                ctx.textAlign = 'left';
                ctx.fillText(labels[i], c.x + 4, c.y - ringR + 10);
            }
        }

        function drawSweep() {
            var c = getRadarCenter();
            var r = getRadarRadius();
            radarAngle += 0.015;

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(c.x, c.y);
            ctx.arc(c.x, c.y, r, radarAngle - 0.5, radarAngle, false);
            ctx.closePath();
            var grad = ctx.createRadialGradient(c.x, c.y, 0, c.x, c.y, r);
            grad.addColorStop(0, 'rgba(0,113,227,0.12)');
            grad.addColorStop(1, 'rgba(0,113,227,0.02)');
            ctx.fillStyle = grad;
            ctx.fill();
            ctx.restore();

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(c.x, c.y);
            ctx.lineTo(c.x + Math.cos(radarAngle) * r, c.y + Math.sin(radarAngle) * r);
            ctx.strokeStyle = 'rgba(0,113,227,0.4)';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.restore();
        }

        function addSignal(risk, type, agentId) {
            var c = getRadarCenter();
            var r = getRadarRadius();
            var dist = r * (1 - risk) * 0.9 + r * 0.1;
            var angle = Math.random() * Math.PI * 2;
            var x = c.x + Math.cos(angle) * dist;
            var y = c.y + Math.sin(angle) * dist;

            var color, size, glowSize;
            switch (type) {
                case 'revoke':
                    color = '#ef4444'; size = 6; glowSize = 20;
                    break;
                case 'deny':
                    color = '#f87171'; size = 4; glowSize = 12;
                    break;
                case 'high_risk':
                    color = '#fbbf24'; size = 4; glowSize = 10;
                    break;
                default:
                    color = '#34d399'; size = 3; glowSize = 6;
            }

            signals.push({
                x: x, y: y, color: color,
                rgb: color.replace('#', '').match(/.{2}/g).map(function(h) { return parseInt(h, 16); }).join(','),
                size: size, glowSize: glowSize,
                life: 1, maxLife: type === 'revoke' ? 1 : 0.7,
                type: type, agentId: agentId || '',
            });

            if (type === 'revoke') {
                for (var i = 0; i < 16; i++) {
                    var bAngle = (Math.PI * 2 / 16) * i + Math.random() * 0.3;
                    var speed = 2 + Math.random() * 3;
                    burstEffects.push({
                        x: x, y: y,
                        vx: Math.cos(bAngle) * speed, vy: Math.sin(bAngle) * speed,
                        life: 1, rgb: '239,68,68', size: 2 + Math.random() * 2,
                    });
                }
                pulseRings.push({ x: x, y: y, radius: 5, maxRadius: 60, life: 1, color: '#ef4444' });
            } else if (type === 'deny') {
                pulseRings.push({ x: x, y: y, radius: 3, maxRadius: 30, life: 1, color: '#f87171' });
            }
        }

        function updateSignals() {
            for (var i = signals.length - 1; i >= 0; i--) {
                signals[i].life -= 0.003;
                if (signals[i].life <= 0) signals.splice(i, 1);
            }
            for (var i = burstEffects.length - 1; i >= 0; i--) {
                var b = burstEffects[i];
                b.x += b.vx; b.y += b.vy;
                b.vx *= 0.95; b.vy *= 0.95;
                b.life -= 0.03;
                if (b.life <= 0) burstEffects.splice(i, 1);
            }
            for (var i = pulseRings.length - 1; i >= 0; i--) {
                var p = pulseRings[i];
                p.radius += (p.maxRadius - p.radius) * 0.08;
                p.life -= 0.02;
                if (p.life <= 0) pulseRings.splice(i, 1);
            }
        }

        function drawSignals() {
            signals.forEach(function(s) {
                var alpha = Math.min(1, s.life / s.maxLife);
                ctx.save();
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.glowSize * alpha, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + s.rgb + ',' + (alpha * 0.15).toFixed(2) + ')';
                ctx.fill();
                ctx.restore();

                ctx.save();
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
                ctx.fillStyle = s.color;
                ctx.shadowColor = s.color;
                ctx.shadowBlur = 10 * alpha;
                ctx.fill();
                ctx.restore();
            });

            burstEffects.forEach(function(b) {
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.size * b.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + b.rgb + ',' + (b.life * 0.8).toFixed(2) + ')';
                ctx.fill();
            });

            pulseRings.forEach(function(p) {
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
                ctx.strokeStyle = p.color;
                ctx.lineWidth = 1.5 * p.life;
                ctx.globalAlpha = p.life * 0.5;
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
        }

        function drawCenterDot() {
            var c = getRadarCenter();
            ctx.save();
            ctx.beginPath();
            ctx.arc(c.x, c.y, 3, 0, Math.PI * 2);
            ctx.fillStyle = '#0071e3';
            ctx.shadowColor = '#0071e3';
            ctx.shadowBlur = 8;
            ctx.fill();
            ctx.restore();
        }

        function loop() {
            ctx.clearRect(0, 0, W, H);
            if (W < 10 || H < 10) { requestAnimationFrame(loop); return; }
            drawRadarBase();
            drawSweep();
            updateSignals();
            drawSignals();
            drawCenterDot();
            requestAnimationFrame(loop);
        }

        window.addEventListener('resize', resize);
        resize();
        loop();
    }

    function connectWS() {
        var wsUrl = WS_BASE + '//' + location.host + '/ws/audit';
        try {
            _ws = new WebSocket(wsUrl);
            _ws.onmessage = function(e) {
                try { handleWSMessage(JSON.parse(e.data)); } catch (err) {}
            };
            _ws.onclose = function() { setTimeout(connectWS, 3000); };
            _ws.onerror = function() { if (_ws) _ws.close(); };
        } catch (err) { setTimeout(connectWS, 5000); }
    }

    function handleWSMessage(msg) {
        var decision = msg.decision || 'allow';
        var isAutoRevoke = msg.context && msg.context.auto_revoked;
        var riskScore = (msg.context && msg.context.risk_score) || 0;
        var agentId = msg.agent_id || 'unknown';
        var action = msg.action || '';

        var type = 'allow';
        if (isAutoRevoke) type = 'revoke';
        else if (decision === 'deny') type = 'deny';
        else if (riskScore > 0.7) type = 'high_risk';

        addSignal(riskScore || (type === 'allow' ? 0.1 : 0.6), type, agentId);
        pushAttackLog(type, agentId, action);

        stats.total++;
        if (decision === 'deny') stats.denied++;
        if (riskScore > 0.7) stats.highRisk++;
        if (isAutoRevoke) stats.autoRevoke++;
        updateStats();
    }

    function pushAttackLog(type, agent, action) {
        var log = document.getElementById('attackLog');
        if (!log) return;
        var now = new Date().toLocaleTimeString();
        var placeholder = log.querySelector('div[style]');
        if (placeholder && placeholder.textContent.indexOf('等待') >= 0) placeholder.remove();
        var explainData = {
            agent_id: agent,
            action: action || '',
            decision: (type === 'deny' || type === 'revoke') ? 'deny' : 'allow',
            reason: type === 'revoke' ? 'Auto-revoked: trust collapsed' : (type === 'deny' ? 'Request denied by policy' : 'Allowed'),
            trust_score: null,
            risk_score: type === 'revoke' ? 0.95 : (type === 'deny' ? 0.7 : 0.1),
            chain_detail: [],
            blocked_at: type === 'deny' ? 'policy_check' : (type === 'revoke' ? 'auto_revoke' : ''),
            auto_revoked: type === 'revoke',
            prompt_risk_score: null,
            attack_types: [],
            attack_intent: '',
            severity: '',
        };
        var el = document.createElement('div');
        el.className = 'rr-attack-item';
        el.innerHTML = '<span class="rr-attack-dot ' + type + '"></span><span class="rr-attack-time">' + now + '</span><span class="rr-attack-agent">' + agent + '</span><span class="rr-attack-action">' + action + '</span>' + IAM_EXPLAIN.makeBtn('🧠', explainData, 'iam-explain-btn-sm');
        log.prepend(el);
        while (log.children.length > 60) log.removeChild(log.lastChild);
    }

    function updateStats() {
        var e;
        e = document.getElementById('statTotal'); if (e) e.textContent = stats.total;
        e = document.getElementById('statDenied'); if (e) e.textContent = stats.denied;
        var denyRate = stats.total > 0 ? ((stats.denied / stats.total) * 100).toFixed(1) + '%' : '0%';
        e = document.getElementById('statDenyRate'); if (e) e.textContent = denyRate;
        e = document.getElementById('statHighRisk'); if (e) e.textContent = stats.highRisk;
        e = document.getElementById('statAutoRevoke'); if (e) e.textContent = stats.autoRevoke;
        e = document.getElementById('statRevokedAgents'); if (e) e.textContent = stats.revokedAgents;

        var maxCount = Math.max(attackCounts.replay, attackCounts.escalation, attackCounts.autoRevoke, attackCounts.capDeny, 1);
        e = document.getElementById('typeReplay'); if (e) e.textContent = attackCounts.replay;
        e = document.getElementById('typeEscalation'); if (e) e.textContent = attackCounts.escalation;
        e = document.getElementById('typeAutoRevoke'); if (e) e.textContent = attackCounts.autoRevoke;
        e = document.getElementById('typeCapDeny'); if (e) e.textContent = attackCounts.capDeny;
        e = document.getElementById('barReplay'); if (e) e.style.width = (attackCounts.replay / maxCount * 100) + '%';
        e = document.getElementById('barEscalation'); if (e) e.style.width = (attackCounts.escalation / maxCount * 100) + '%';
        e = document.getElementById('barAutoRevoke'); if (e) e.style.width = (attackCounts.autoRevoke / maxCount * 100) + '%';
        e = document.getElementById('barCapDeny'); if (e) e.style.width = (attackCounts.capDeny / maxCount * 100) + '%';
    }

    function renderTopThreats(agents, autoRevokedAgents) {
        var el = document.getElementById('topThreats');
        if (!el) return;
        var CORE_AGENTS = ['doc_agent', 'data_agent', 'external_agent'];
        var entries = Object.entries(agents).filter(function(pair) {
            return CORE_AGENTS.indexOf(pair[0]) >= 0;
        });
        var demoEntries = Object.entries(agents).filter(function(pair) {
            return CORE_AGENTS.indexOf(pair[0]) < 0;
        });
        if (!entries.length && !demoEntries.length) {
            el.innerHTML = '<div style="color:rgba(255,255,255,0.25);text-align:center;padding:16px;font-size:0.82rem">无 Agent 数据</div>';
            return;
        }
        var sorted = entries.sort(function(a, b) { return a[1].trust_score - b[1].trust_score; });
        var html = '';
        sorted.forEach(function(pair) {
            var id = pair[0], info = pair[1];
            var risk = 1 - info.trust_score;
            var isRevoked = id in autoRevokedAgents;
            var level, levelClass, badge;
            if (isRevoked || risk > 0.7) { level = 'CRITICAL'; levelClass = 'critical'; badge = '<span class="rr-badge rr-badge-critical">🔥 CRITICAL</span>'; }
            else if (risk > 0.5) { level = 'HIGH'; levelClass = 'high'; badge = '<span class="rr-badge rr-badge-high">HIGH</span>'; }
            else if (risk > 0.3) { level = 'MEDIUM'; levelClass = 'medium'; badge = '<span class="rr-badge rr-badge-medium">MEDIUM</span>'; }
            else { level = 'LOW'; levelClass = 'low'; badge = '<span class="rr-badge rr-badge-low">LOW</span>'; }

            var riskColor = risk > 0.7 ? '#ef4444' : (risk > 0.5 ? '#f87171' : (risk > 0.3 ? '#fbbf24' : '#34d399'));

            html += '<div class="rr-threat-item ' + levelClass + '">';
            html += '<div class="rr-threat-agent"><span>' + id + '</span>' + badge + '</div>';
            html += '<div class="rr-threat-detail">risk: <span class="rr-threat-risk" style="color:' + riskColor + '">' + risk.toFixed(2) + '</span> · trust: ' + info.trust_score.toFixed(2) + (isRevoked ? ' · 🔥 REVOKED' : '') + '</div>';
            var explainData = {
                agent_id: id,
                action: 'risk_assessment',
                decision: (levelClass === 'critical' || levelClass === 'high') ? 'deny' : 'allow',
                reason: isRevoked ? 'Agent auto-revoked due to trust collapse' : ('Risk level: ' + level + ', trust: ' + info.trust_score.toFixed(2)),
                trust_score: info.trust_score,
                risk_score: risk,
                chain_detail: [],
                blocked_at: isRevoked ? 'auto_revoke' : '',
                auto_revoked: isRevoked,
                prompt_risk_score: null,
                attack_types: [],
                attack_intent: '',
                severity: '',
            };
            html += '<div style="margin-top:4px">' + IAM_EXPLAIN.makeBtn('🧠 Explain', explainData) + '</div>';
            html += '</div>';
        });
        if (demoEntries.length > 0) {
            html += '<div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.06)">';
            html += '<div style="font-size:0.65rem;color:rgba(255,255,255,0.3);margin-bottom:6px">演示 Agent (' + demoEntries.length + ')</div>';
            var demoSorted = demoEntries.sort(function(a, b) { return a[1].trust_score - b[1].trust_score; });
            demoSorted.slice(0, 5).forEach(function(pair) {
                var id = pair[0], info = pair[1];
                var risk = 1 - info.trust_score;
                var isRevoked = id in autoRevokedAgents;
                var riskColor = risk > 0.7 ? '#ef4444' : (risk > 0.5 ? '#f87171' : (risk > 0.3 ? '#fbbf24' : '#34d399'));
                html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:0.7rem">';
                html += '<span style="color:rgba(255,255,255,0.5)">' + id.replace('demo_', '') + '</span>';
                html += '<span style="color:' + riskColor + ';font-weight:600">risk ' + risk.toFixed(2) + '</span>';
                html += '</div>';
            });
            if (demoEntries.length > 5) {
                html += '<div style="font-size:0.6rem;color:rgba(255,255,255,0.2);text-align:center;padding:4px">... +' + (demoEntries.length - 5) + ' more</div>';
            }
            html += '</div>';
        }
        el.innerHTML = html;
    }

    async function loadData() {
        try {
            var results = await Promise.all([
                fetchJSON(AUDIT_BASE + '/logs?limit=500'),
                fetchJSON(BASE + '/trust'),
                fetchJSON(BASE + '/auto-revoke/list'),
                fetchJSON(BASE + '/revoked/list'),
            ]);
            var logs = results[0];
            var trust = results[1];
            var autoRevoke = results[2];
            var revoked = results[3];

            var entries = Array.isArray(logs) ? logs : (logs.logs || []);
            var agents = trust.agents || {};
            var autoRevokedAgents = (autoRevoke && autoRevoke.auto_revoked_agents) || {};
            var revokedTokens = (revoked && revoked.revoked_tokens) || [];

            stats.total = entries.length;
            stats.denied = entries.filter(function(l) { return l.decision === 'deny'; }).length;
            stats.highRisk = entries.filter(function(l) { var c = l.context || {}; return c.risk_score > 0.7; }).length;
            stats.autoRevoke = entries.filter(function(l) { var c = l.context || {}; return c.auto_revoked || l.auto_revoked; }).length;
            stats.revokedAgents = Object.keys(autoRevokedAgents).length;

            attackCounts.replay = entries.filter(function(l) { var c = l.context || {}; return c.attack_type === 'replay'; }).length;
            attackCounts.escalation = entries.filter(function(l) { var c = l.context || {}; return c.attack_type === 'escalation'; }).length;
            attackCounts.autoRevoke = stats.autoRevoke;
            attackCounts.capDeny = entries.filter(function(l) { return l.decision === 'deny' && !(l.context && l.context.auto_revoked); }).length;

            updateStats();
            renderTopThreats(agents, autoRevokedAgents);

            var recentDeny = entries.filter(function(l) { return l.decision === 'deny' || (l.context && l.context.auto_revoked); }).slice(0, 20);
            recentDeny.reverse();
            recentDeny.forEach(function(l) {
                var type = (l.context && l.context.auto_revoked) ? 'revoke' : 'deny';
                var riskScore = (l.context && l.context.risk_score) || 0.5;
                addSignal(riskScore, type, l.agent_id);
            });

        } catch (e) {
            console.error('加载风险数据失败:', e);
        }
    }

    async function simulateAttack() {
        try {
            var data = await fetchJSON(BASE + '/demo/auto-revoke', { method: 'POST' });
            var steps = data.steps || [];
            steps.forEach(function(s, idx) {
                setTimeout(function() {
                    var risk = 0.5 + (idx / steps.length) * 0.5;
                    var type = 'allow';
                    if (s.status === 'AUTO_REVOKED' || s.auto_revoked) {
                        type = 'revoke';
                        attackCounts.autoRevoke++;
                    } else if (s.allowed === false) {
                        type = 'deny';
                        attackCounts.capDeny++;
                        stats.denied++;
                    } else if (s.allowed === true) {
                        type = 'allow';
                    }
                    stats.total++;
                    addSignal(risk, type, 'external_agent');
                    pushAttackLog(type, 'external_agent', s.action || '');
                    updateStats();
                }, idx * 500);
            });
            setTimeout(function() { loadData(); }, steps.length * 500 + 1000);
        } catch (e) {
            pushAttackLog('deny', 'system', '模拟失败: ' + e.message);
        }
    }

    async function clearRevoked() {
        try {
            await Promise.all([
                fetchJSON(BASE + '/auto-revoke/clear', { method: 'POST' }),
                fetchJSON(BASE + '/revoked/clear', { method: 'POST' }),
            ]);
            await loadData();
        } catch (e) {
            console.error('清除撤销记录失败:', e);
        }
    }

    loadData();
    connectWS();

    setInterval(loadData, 8000);

    return {
        refresh: loadData,
        simulateAttack: simulateAttack,
        clearRevoked: clearRevoked,
    };
})();
