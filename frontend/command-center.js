﻿﻿﻿(function() {
    var BASE = '/api/delegate';
    var WS_BASE = location.protocol === 'https:' ? 'wss:' : 'ws:';

    var canvas = document.getElementById('networkCanvas');
    var ctx = canvas ? canvas.getContext('2d') : null;
    var canvasOk = !!(canvas && ctx);

    var W = 800, H = 600, dpr = 1;
    var particles = [];
    var burstEffects = [];
    var ws = null;
    var stats = { total: 0, allow: 0, deny: 0, revoke: 0, attacks: 0, revokedAgents: 0, activeAgents: 0 };
    var trustScores = {};
    var autoRevoked = {};

    var AGENT_DEFS = {
        'user': { label: 'USER', color: '#c4b5fd', rgb: '196,181,253', caps: ['issue token'], icon: '👤' },
        'doc_agent': { label: 'doc_agent', color: '#0071e3', rgb: '0,113,227', caps: ['read:doc', 'write:doc:public'], icon: '📄' },
        'data_agent': { label: 'data_agent', color: '#10b981', rgb: '16,185,129', caps: ['read:feishu_table'], icon: '📊' },
        'external_agent': { label: 'ext_agent', color: '#f59e0b', rgb: '245,158,11', caps: ['write:doc:public'], icon: '🌐' },
    };

    var EDGE_DEFS = [
        { from: 'user', to: 'doc_agent' },
        { from: 'user', to: 'data_agent' },
        { from: 'user', to: 'external_agent' },
        { from: 'doc_agent', to: 'data_agent' },
        { from: 'external_agent', to: 'data_agent' },
    ];

    function hexToRgb(hex) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return r + ',' + g + ',' + b;
    }

    function renderAgentGrid() {
        var grid = document.getElementById('ccAgentGrid');
        if (!grid) return;
        var html = '';
        var ids = Object.keys(AGENT_DEFS);
        ids.forEach(function(id, idx) {
            var def = AGENT_DEFS[id];
            if (id === 'user') return;
            var trust = trustScores[id];
            var isRevoked = id in autoRevoked;
            var trustColor = isRevoked ? '#ef4444' : (trust == null ? '#636366' : (trust < 0.3 ? '#ef4444' : (trust < 0.7 ? '#fbbf24' : '#34d399')));
            var statusText = isRevoked ? '🔥 REVOKED' : (trust == null ? '⏳ PENDING' : (trust < 0.3 ? '🔴 DANGER' : (trust < 0.7 ? '🟡 WARNING' : '🟢 SAFE')));
            var trustPct = trust != null ? Math.round(trust * 100) : 0;
            var borderColor = isRevoked ? '#ef4444' : def.color;
            var glowStyle = isRevoked ? 'box-shadow:0 0 20px rgba(239,68,68,0.2);border-color:rgba(239,68,68,0.3)' : 'box-shadow:0 0 20px rgba(' + def.rgb + ',0.1);border-color:rgba(' + hexToRgb(def.color) + ',0.15)';
            var floatDelay = (idx * 0.5) + 's';
            var circumference = 2 * Math.PI * 28;
            var dashOffset = circumference - (trustPct / 100) * circumference;

            html += '<div class="cc-card" style="--card-accent:' + borderColor + ';' + glowStyle + '">';
            html += '<span class="icon" style="animation-delay:' + floatDelay + '">' + def.icon + '</span>';
            html += '<div class="name" style="color:' + def.color + '">' + def.label + '</div>';
            html += '<div class="caps">' + def.caps.join(' · ') + '</div>';
            html += '<div style="display:flex;align-items:center;gap:12px;justify-content:center">';
            html += '<svg width="64" height="64" viewBox="0 0 64 64" style="transform:rotate(-90deg)">';
            html += '<circle cx="32" cy="32" r="28" fill="none" stroke="rgba(255,255,255,0.04)" stroke-width="4"/>';
            html += '<circle cx="32" cy="32" r="28" fill="none" stroke="' + trustColor + '" stroke-width="4" stroke-linecap="round" stroke-dasharray="' + circumference.toFixed(1) + '" stroke-dashoffset="' + dashOffset.toFixed(1) + '" style="transition:stroke-dashoffset 0.8s ease"/>';
            html += '</svg>';
            html += '<div style="text-align:left">';
            html += '<div class="cc-trust-val" style="color:' + trustColor + '">' + (trust != null ? trust.toFixed(2) : '—') + '</div>';
            html += '<div class="cc-status" style="color:' + trustColor + '">' + statusText + '</div>';
            html += '</div></div>';
            html += '<div class="cc-trust-bar" style="margin-top:12px"><div class="cc-trust-fill" style="width:' + trustPct + '%;background:' + trustColor + '"></div></div>';
            html += '</div>';
        });
        grid.innerHTML = html;
    }

    function updateAgentPanel() {
        var el = document.getElementById('ccAgentStatus');
        if (!el) return;
        var html = '';
        var ids = Object.keys(AGENT_DEFS);
        ids.forEach(function(id) {
            var def = AGENT_DEFS[id];
            if (id === 'user') return;
            var trust = trustScores[id];
            var isRevoked = id in autoRevoked;
            var trustColor = isRevoked ? '#ef4444' : (trust == null ? '#636366' : (trust < 0.3 ? '#ef4444' : (trust < 0.7 ? '#fbbf24' : '#34d399')));
            var statusText = isRevoked ? 'REVOKED' : (trust == null ? '—' : (trust < 0.3 ? 'DANGER' : (trust < 0.7 ? 'WARNING' : 'SAFE')));
            var trustPct = trust != null ? Math.round(trust * 100) : 0;
            var revokedStyle = isRevoked ? 'text-decoration:line-through;opacity:0.4' : '';

            html += '<div class="cc-agent-row">';
            html += '<div style="display:flex;align-items:center;gap:8px">';
            html += '<span style="font-size:1rem">' + def.icon + '</span>';
            html += '<div>';
            html += '<div style="font-size:0.72rem;font-weight:700;color:' + def.color + ';' + revokedStyle + '">' + def.label + '</div>';
            html += '<div style="font-size:0.55rem;color:rgba(255,255,255,0.25)">' + def.caps[0] + '</div>';
            html += '</div></div>';
            html += '<div style="text-align:right">';
            html += '<div style="font-size:0.8rem;font-weight:800;color:' + trustColor + '">' + (trust != null ? trust.toFixed(2) : '—') + '</div>';
            html += '<div style="font-size:0.5rem;color:' + trustColor + ';font-weight:600;letter-spacing:0.04em">' + statusText + '</div>';
            html += '</div>';
            html += '</div>';
        });
        el.innerHTML = html;

        var healthRing = document.getElementById('ccHealthRing');
        var healthPct = document.getElementById('ccHealthPct');
        if (healthRing && healthPct) {
            var total = stats.activeAgents || 1;
            var revoked = stats.revokedAgents || 0;
            var health = Math.max(0, Math.round(((total - revoked) / total) * 100));
            var circumference = 2 * Math.PI * 20;
            var offset = circumference - (health / 100) * circumference;
            healthRing.setAttribute('stroke-dashoffset', offset.toFixed(1));
            var hColor = health > 70 ? '#34d399' : (health > 40 ? '#fbbf24' : '#ef4444');
            healthRing.setAttribute('stroke', hColor);
            healthPct.textContent = health + '%';
            healthPct.style.color = hColor;
        }
    }

    function updateUI() {
        var el = function(id) { return document.getElementById(id); };
        var e;
        e = el('ccActiveAgents'); if (e) e.textContent = Math.max(0, (stats.activeAgents || 0) - (stats.revokedAgents || 0));
        e = el('ccAttacks'); if (e) e.textContent = stats.attacks;
        e = el('ccRevoked'); if (e) e.textContent = stats.revokedAgents;
        e = el('ccTotal'); if (e) e.textContent = stats.total;
        e = el('ccAllow'); if (e) e.textContent = stats.allow;
        e = el('ccDeny'); if (e) e.textContent = stats.deny;
        e = el('ccRevokeCount'); if (e) e.textContent = stats.revoke;

        var threatFill = el('ccThreatFill');
        var threatLabel = el('ccThreatLabel');
        if (threatFill && threatLabel) {
            var total = stats.total || 1;
            var denyRate = (stats.deny + stats.revoke) / total;
            var pct = Math.min(100, Math.round(denyRate * 100));
            var tColor, tText;
            if (pct < 15) { tColor = '#34d399'; tText = 'LOW'; }
            else if (pct < 40) { tColor = '#fbbf24'; tText = 'MEDIUM'; }
            else if (pct < 70) { tColor = '#f97316'; tText = 'HIGH'; }
            else { tColor = '#ef4444'; tText = 'CRITICAL'; }
            threatFill.style.width = Math.max(5, pct) + '%';
            threatFill.style.background = tColor;
            threatLabel.textContent = tText;
            threatLabel.style.color = tColor;
        }
    }

    function pushEvent(type, agent, action) {
        var stream = document.getElementById('ccEventStream');
        if (!stream) return;
        var now = new Date().toLocaleTimeString();
        var colors = { allow: '#34d399', deny: '#f87171', revoke: '#ef4444', info: '#c4b5fd' };
        var bgColors = { allow: 'rgba(52,211,153,0.08)', deny: 'rgba(248,113,113,0.08)', revoke: 'rgba(239,68,68,0.1)', info: 'rgba(196,181,253,0.06)' };
        var el = document.createElement('div');
        el.className = 'cc-event';
        el.innerHTML = '<span style="color:rgba(255,255,255,0.15)">' + now + '</span> <span style="color:' + (colors[type] || colors.info) + ';font-weight:700;letter-spacing:0.04em">' + type.toUpperCase() + '</span> <span style="color:#c4b5fd;font-weight:600">' + agent + '</span> <span style="color:rgba(255,255,255,0.3)">' + action + '</span>';
        stream.prepend(el);
        while (stream.children.length > 50) stream.removeChild(stream.lastChild);
    }

    if (canvasOk) {
        function resize() {
            dpr = window.devicePixelRatio || 1;
            W = canvas.offsetWidth || canvas.clientWidth || window.innerWidth;
            H = canvas.offsetHeight || canvas.clientHeight || window.innerHeight;
            if (W < 10) W = window.innerWidth;
            if (H < 10) H = window.innerHeight;
            canvas.width = Math.round(W * dpr);
            canvas.height = Math.round(H * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        function drawGrid() {
            ctx.strokeStyle = 'rgba(0,113,227,0.1)';
            ctx.lineWidth = 0.5;
            var step = 50;
            for (var x = 0; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
            for (var y = 0; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
        }

        function drawEdges() {
            var nodePositions = {
                'user': { x: W * 0.5, y: H * 0.15 },
                'doc_agent': { x: W * 0.22, y: H * 0.48 },
                'data_agent': { x: W * 0.5, y: H * 0.58 },
                'external_agent': { x: W * 0.78, y: H * 0.48 },
            };
            EDGE_DEFS.forEach(function(e) {
                var from = nodePositions[e.from];
                var to = nodePositions[e.to];
                if (!from || !to) return;
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = 'rgba(0,113,227,0.12)';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
        }

        function spawnParticle(fromId, toId, type) {
            var nodePositions = {
                'user': { x: W * 0.5, y: H * 0.15 },
                'doc_agent': { x: W * 0.22, y: H * 0.48 },
                'data_agent': { x: W * 0.5, y: H * 0.58 },
                'external_agent': { x: W * 0.78, y: H * 0.48 },
            };
            var from = nodePositions[fromId];
            var to = nodePositions[toId];
            if (!from || !to) return;
            var color, speed, size;
            switch (type) {
                case 'allow': color = '#34d399'; speed = 3; size = 4; break;
                case 'deny': color = '#f87171'; speed = 2; size = 5; break;
                case 'revoke': color = '#ef4444'; speed = 2; size = 6; break;
                default: color = '#0071e3'; speed = 2.5; size = 3;
            }
            var dx = to.x - from.x, dy = to.y - from.y;
            var dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
            particles.push({
                x: from.x, y: from.y,
                vx: (dx / dist) * speed, vy: (dy / dist) * speed,
                targetX: to.x, targetY: to.y,
                color: color, rgb: hexToRgb(color), size: size, type: type,
                life: 1, trail: [],
            });
        }

        function spawnBurst(x, y, color) {
            var rgb = hexToRgb(color);
            for (var i = 0; i < 16; i++) {
                var angle = (Math.PI * 2 / 16) * i + Math.random() * 0.3;
                var speed = 2 + Math.random() * 3;
                burstEffects.push({
                    x: x, y: y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
                    life: 1, rgb: rgb, size: 2 + Math.random() * 2,
                });
            }
        }

        function update() {
            for (var i = particles.length - 1; i >= 0; i--) {
                var p = particles[i];
                p.trail.push({ x: p.x, y: p.y, life: 1 });
                if (p.trail.length > 10) p.trail.shift();
                p.trail.forEach(function(t) { t.life -= 0.1; });
                p.x += p.vx;
                p.y += p.vy;
                var dx = p.targetX - p.x, dy = p.targetY - p.y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 15) {
                    if (p.type === 'deny' || p.type === 'replay') {
                        spawnBurst(p.x, p.y, p.type === 'deny' ? '#f87171' : '#fbbf24');
                        p.life = 0;
                    } else {
                        p.life -= 0.05;
                    }
                }
                p.life -= 0.01;
                if (p.life <= 0) particles.splice(i, 1);
            }
            for (var i = burstEffects.length - 1; i >= 0; i--) {
                var b = burstEffects[i];
                b.x += b.vx; b.y += b.vy;
                b.vx *= 0.95; b.vy *= 0.95;
                b.life -= 0.04;
                if (b.life <= 0) burstEffects.splice(i, 1);
            }
        }

        function draw() {
            ctx.clearRect(0, 0, W, H);
            if (W < 10 || H < 10) return;
            drawGrid();
            drawEdges();
            particles.forEach(function(p) {
                p.trail.forEach(function(t) {
                    if (t.life <= 0) return;
                    ctx.beginPath();
                    ctx.arc(t.x, t.y, p.size * t.life * 0.5, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(' + p.rgb + ',' + Math.max(0, t.life * 0.4).toFixed(2) + ')';
                    ctx.fill();
                });
                ctx.save();
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.shadowColor = p.color;
                ctx.shadowBlur = 12;
                ctx.fill();
                ctx.restore();
            });
            burstEffects.forEach(function(b) {
                ctx.beginPath();
                ctx.arc(b.x, b.y, b.size * b.life, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(' + b.rgb + ',' + Math.max(0, b.life * 0.8).toFixed(2) + ')';
                ctx.fill();
            });
        }

        function loop() {
            update();
            draw();
            requestAnimationFrame(loop);
        }

        window.addEventListener('resize', resize);
        resize();
        loop();
    }

    function fetchJSON(url, opts) {
        opts = opts || {};
        return fetch(url, Object.assign({}, opts, { headers: Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}) })).then(function(r) { return r.json(); });
    }

    function loadData() {
        Promise.all([
            fetchJSON(BASE + '/trust'),
            fetchJSON(BASE + '/auto-revoke/list'),
        ]).then(function(results) {
            var trust = results[0];
            var autoRevokeData = results[1];
            var agentsMap = trust.agents || {};
            trustScores = {};
            Object.keys(agentsMap).forEach(function(id) {
                trustScores[id] = agentsMap[id].trust_score;
            });
            autoRevoked = (autoRevokeData && autoRevokeData.auto_revoked_agents) || {};
            stats.activeAgents = Object.keys(agentsMap).length;
            stats.revokedAgents = Object.keys(autoRevoked).length;
            updateUI();
            updateAgentPanel();
            renderAgentGrid();
        }).catch(function() {});
    }

    function connectWS() {
        var wsUrl = WS_BASE + '//' + location.host + '/ws/audit';
        try {
            ws = new WebSocket(wsUrl);
            ws.onmessage = function(e) {
                try { handleWSMessage(JSON.parse(e.data)); } catch (err) {}
            };
            ws.onclose = function() { setTimeout(connectWS, 3000); };
            ws.onerror = function() { if (ws) ws.close(); };
        } catch (err) { setTimeout(connectWS, 5000); }
    }

    function handleWSMessage(msg) {
        var decision = msg.decision || 'allow';
        var isAutoRevoke = msg.context && msg.context.auto_revoked;
        var agentId = msg.agent_id || 'unknown';
        var action = msg.action || '';

        stats.total++;
        if (decision === 'allow') stats.allow++;
        else { stats.deny++; stats.attacks++; }
        if (isAutoRevoke) { stats.revoke++; stats.revokedAgents++; }

        var type = decision === 'allow' ? 'allow' : 'deny';
        if (isAutoRevoke) type = 'revoke';

        if (canvasOk) {
            spawnParticle('user', agentId || 'data_agent', type);
            if (isAutoRevoke) {
                var nodePositions = {
                    'doc_agent': { x: W * 0.22, y: H * 0.48 },
                    'data_agent': { x: W * 0.5, y: H * 0.58 },
                    'external_agent': { x: W * 0.78, y: H * 0.48 },
                };
                var pos = nodePositions[agentId];
                if (pos) spawnBurst(pos.x, pos.y, '#ef4444');
            }
        }

        pushEvent(type, agentId, action);
        updateUI();
        loadData();
    }

    function simulateAttack() {
        pushEvent('info', 'system', '🔥 Attack simulation started');
        fetchJSON(BASE + '/demo/auto-revoke', { method: 'POST' }).then(function(data) {
            loadData();
            var steps = data.steps || [];
            steps.forEach(function(step, idx) {
                setTimeout(function() {
                    if (step.status === 'AUTO_REVOKED' || step.auto_revoked) {
                        if (canvasOk) spawnParticle('user', 'external_agent', 'revoke');
                        pushEvent('revoke', 'external_agent', step.action || 'AUTO-REVOKED');
                        stats.revoke++;
                    } else if (step.allowed === true) {
                        if (canvasOk) spawnParticle('user', 'external_agent', 'allow');
                        pushEvent('allow', 'external_agent', step.action || '');
                        stats.allow++;
                    } else if (step.allowed === false) {
                        if (canvasOk) spawnParticle('user', 'external_agent', 'deny');
                        pushEvent('deny', 'external_agent', step.action || '');
                        stats.deny++;
                        stats.attacks++;
                    }
                    stats.total++;
                    updateUI();
                    renderAgentGrid();
                    updateAgentPanel();
                }, idx * 600);
            });
        }).catch(function(e) {
            pushEvent('deny', 'system', 'Simulation failed');
        });
    }

    function resetAll() {
        Promise.all([
            fetchJSON(BASE + '/trust/reset', { method: 'POST' }),
            fetchJSON(BASE + '/auto-revoke/clear', { method: 'POST' }),
            fetchJSON(BASE + '/used-tokens/clear', { method: 'POST' }),
        ]).then(function() {
            stats = { total: 0, allow: 0, deny: 0, revoke: 0, attacks: 0, revokedAgents: 0, activeAgents: 0 };
            particles = [];
            burstEffects = [];
            loadData();
            updateUI();
            renderAgentGrid();
            updateAgentPanel();
        }).catch(function() {});
    }

    var simBtn = document.getElementById('ccSimulateBtn');
    if (simBtn) simBtn.addEventListener('click', simulateAttack);
    var resetBtn = document.getElementById('ccResetBtn');
    if (resetBtn) resetBtn.addEventListener('click', resetAll);

    loadData();
    connectWS();
    renderAgentGrid();

    setInterval(function() {
        var agentIds = ['doc_agent', 'data_agent', 'external_agent'];
        var from = agentIds[Math.floor(Math.random() * agentIds.length)];
        if (!(from in autoRevoked) && canvasOk) {
            spawnParticle('user', from, 'info');
        }
    }, 5000);
})();
