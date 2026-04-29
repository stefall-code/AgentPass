﻿﻿var CP = (function() {
    var BASE = '/api/delegate';
    var WS_BASE = location.protocol === 'https:' ? 'wss:' : 'ws:';

    var canvas = document.getElementById('playerCanvas');
    var ctx = canvas ? canvas.getContext('2d') : null;
    var canvasOk = !!(canvas && ctx);
    var W = 800, H = 520, dpr = 1;

    var steps = [];
    var currentStep = -1;
    var isPlaying = false;
    var playTimer = null;
    var playSpeed = 1200;
    var selectedScene = null;
    var particles = [];
    var burstEffects = [];
    var nodeStates = {};
    var counts = { allow: 0, deny: 0, revoke: 0 };
    var _trustData = null;
    var _autoRevokeData = null;
    var _auditWs = null;

    var NODE_POS = {
        'user': { x: 0.5, y: 0.12 },
        'prompt_defense': { x: 0.5, y: 0.28 },
        'doc_agent': { x: 0.22, y: 0.48 },
        'data_agent': { x: 0.5, y: 0.62 },
        'external_agent': { x: 0.78, y: 0.48 },
        'policy_engine': { x: 0.82, y: 0.82 },
    };

    var NODE_COLORS = {
        'user': { color: '#c4b5fd', rgb: '196,181,253', icon: '👤' },
        'prompt_defense': { color: '#a78bfa', rgb: '167,139,250', icon: '🛡' },
        'doc_agent': { color: '#0071e3', rgb: '0,113,227', icon: '📄' },
        'data_agent': { color: '#10b981', rgb: '16,185,129', icon: '📊' },
        'external_agent': { color: '#f59e0b', rgb: '245,158,11', icon: '🌐' },
        'policy_engine': { color: '#af52de', rgb: '175,82,222', icon: '🛡' },
    };

    function hexToRgb(hex) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return r + ',' + g + ',' + b;
    }

    function fetchJSON(url, opts) {
        opts = opts || {};
        return fetch(url, Object.assign({}, opts, { headers: Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}) })).then(function(r) { return r.json(); });
    }

    function connectWS() {
        var wsUrl = WS_BASE + '//' + location.host + '/ws/audit';
        try {
            _auditWs = new WebSocket(wsUrl);
            _auditWs.onopen = function() { updateWSStatus(true); };
            _auditWs.onmessage = function(e) {
                try { handleWSMessage(JSON.parse(e.data)); } catch (err) {}
            };
            _auditWs.onclose = function() { updateWSStatus(false); setTimeout(connectWS, 3000); };
            _auditWs.onerror = function() { if (_auditWs) _auditWs.close(); };
        } catch (err) { setTimeout(connectWS, 5000); }
    }

    function updateWSStatus(connected) {
        var dot = document.getElementById('wsDot');
        var label = document.getElementById('wsLabel');
        if (dot) dot.className = 'cp-live-dot ' + (connected ? 'on' : 'off');
        if (label) label.textContent = connected ? '已连接' : '未连接';
    }

    function handleWSMessage(msg) {
        var decision = msg.decision || 'allow';
        var action = msg.action || '';
        var isAutoRevoke = msg.context && msg.context.auto_revoked;
        var isPromptBlocked = action === 'prompt_injection_blocked';
        var type = isPromptBlocked ? 'deny' : (isAutoRevoke ? 'revoke' : (decision === 'allow' ? 'allow' : 'deny'));
        var agent = msg.agent_id || '?';
        var displayAction = isPromptBlocked ? '🛡️ prompt_injection_blocked' : action;
        pushLiveEvent(type, agent, displayAction);
    }

    function pushLiveEvent(type, agent, action) {
        var stream = document.getElementById('liveStream');
        if (!stream) return;
        var now = new Date().toLocaleTimeString();
        var el = document.createElement('div');
        el.className = 'cp-ev-item';
        el.innerHTML = '<span class="cp-ev-dot ' + type + '"></span><span class="cp-ev-time">' + now + '</span><span class="cp-ev-agent">' + agent + '</span><span class="cp-ev-action">' + action + '</span>';
        var placeholder = stream.querySelector('.cp-empty');
        if (placeholder) placeholder.remove();
        stream.prepend(el);
        while (stream.children.length > 60) stream.removeChild(stream.lastChild);
    }

    async function loadTrustData() {
        try {
            var results = await Promise.all([
                fetchJSON(BASE + '/trust'),
                fetchJSON(BASE + '/auto-revoke/list'),
            ]);
            _trustData = results[0];
            _autoRevokeData = results[1];
        } catch (e) {}
    }

    function getTrustScore(agentId) {
        if (!_trustData || !_trustData.agents) return null;
        return _trustData.agents[agentId] ? _trustData.agents[agentId].trust_score : null;
    }

    if (canvasOk) {
        function resize() {
            dpr = window.devicePixelRatio || 1;
            var rect = canvas.parentElement.getBoundingClientRect();
            W = rect.width || 800;
            H = rect.height || 520;
            if (W < 10) W = 800;
            if (H < 10) H = 520;
            canvas.width = Math.round(W * dpr);
            canvas.height = Math.round(H * dpr);
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        function getPos(id) {
            var p = NODE_POS[id];
            if (!p) return { x: W * 0.5, y: H * 0.5 };
            return { x: p.x * W, y: p.y * H };
        }

        function drawGrid() {
            ctx.strokeStyle = 'rgba(0,113,227,0.06)';
            ctx.lineWidth = 0.5;
            var step = 50;
            for (var x = 0; x < W; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
            for (var y = 0; y < H; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
        }

        function drawEdges() {
            var edgeDefs = [
                ['user', 'prompt_defense'],
                ['prompt_defense', 'doc_agent'], ['prompt_defense', 'data_agent'], ['prompt_defense', 'external_agent'],
                ['doc_agent', 'data_agent'], ['external_agent', 'data_agent'],
            ];
            edgeDefs.forEach(function(e) {
                var from = getPos(e[0]);
                var to = getPos(e[1]);
                var fromState = nodeStates[e[0]];
                var toState = nodeStates[e[1]];
                var active = (fromState && fromState.active) || (toState && toState.active);
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = active ? 'rgba(0,113,227,0.2)' : 'rgba(255,255,255,0.04)';
                ctx.lineWidth = active ? 1.5 : 0.5;
                ctx.stroke();
            });
        }

        function drawNodes() {
            Object.keys(NODE_POS).forEach(function(id) {
                var pos = getPos(id);
                var def = NODE_COLORS[id] || { color: '#636366', rgb: '99,99,102', icon: '?' };
                var state = nodeStates[id] || {};
                var isActive = state.active;
                var isRevoked = state.revoked;
                var isDenied = state.denied;
                var radius = isActive ? 28 : 22;

                if (isRevoked) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.arc(pos.x, pos.y, radius + 8, 0, Math.PI * 2);
                    ctx.fillStyle = 'rgba(239,68,68,0.08)';
                    ctx.fill();
                    ctx.restore();
                }

                ctx.save();
                ctx.beginPath();
                ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
                var alpha = isActive ? 0.15 : 0.04;
                ctx.fillStyle = isRevoked ? 'rgba(239,68,68,' + alpha + ')' : (isDenied ? 'rgba(248,113,113,' + alpha + ')' : 'rgba(' + def.rgb + ',' + alpha + ')');
                ctx.fill();
                ctx.strokeStyle = isRevoked ? 'rgba(239,68,68,0.6)' : (isDenied ? 'rgba(248,113,113,0.5)' : (isActive ? def.color : 'rgba(255,255,255,0.08)'));
                ctx.lineWidth = isActive ? 2 : 1;
                if (isActive) {
                    ctx.shadowColor = def.color;
                    ctx.shadowBlur = 20;
                }
                ctx.stroke();
                ctx.restore();

                ctx.save();
                ctx.font = (isActive ? '20px' : '16px') + ' sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(def.icon, pos.x, pos.y);
                ctx.restore();

                ctx.save();
                ctx.font = (isActive ? 'bold 11px' : '10px') + ' sans-serif';
                ctx.textAlign = 'center';
                ctx.fillStyle = isActive ? def.color : 'rgba(255,255,255,0.3)';
                ctx.fillText(id.replace('_', ' '), pos.x, pos.y + radius + 14);
                ctx.restore();

                if (isActive && state.trust != null) {
                    ctx.save();
                    ctx.font = '9px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillStyle = state.trust < 0.3 ? '#ef4444' : (state.trust < 0.7 ? '#fbbf24' : '#34d399');
                    ctx.fillText('trust: ' + state.trust.toFixed(2), pos.x, pos.y + radius + 26);
                    ctx.restore();
                }

                if (isRevoked) {
                    ctx.save();
                    ctx.font = 'bold 10px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.fillStyle = '#ef4444';
                    ctx.fillText('🔥 REVOKED', pos.x, pos.y + radius + 38);
                    ctx.restore();
                }
            });
        }

        function spawnParticle(fromId, toId, type) {
            var from = getPos(fromId);
            var to = getPos(toId);
            var color, speed, size;
            switch (type) {
                case 'allow': color = '#34d399'; speed = 3; size = 5; break;
                case 'deny': color = '#f87171'; speed = 2; size = 6; break;
                case 'revoke': color = '#ef4444'; speed = 2; size = 7; break;
                default: color = '#0071e3'; speed = 2.5; size = 4;
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
            for (var i = 0; i < 20; i++) {
                var angle = (Math.PI * 2 / 20) * i + Math.random() * 0.3;
                var speed = 2 + Math.random() * 4;
                burstEffects.push({
                    x: x, y: y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
                    life: 1, rgb: rgb, size: 2 + Math.random() * 3,
                });
            }
        }

        function updateParticles() {
            for (var i = particles.length - 1; i >= 0; i--) {
                var p = particles[i];
                p.trail.push({ x: p.x, y: p.y, life: 1 });
                if (p.trail.length > 12) p.trail.shift();
                p.trail.forEach(function(t) { t.life -= 0.08; });
                p.x += p.vx;
                p.y += p.vy;
                var dx = p.targetX - p.x, dy = p.targetY - p.y;
                var dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 20) {
                    if (p.type === 'deny' || p.type === 'replay') {
                        spawnBurst(p.x, p.y, p.type === 'deny' ? '#f87171' : '#fbbf24');
                        p.life = 0;
                    } else if (p.type === 'revoke') {
                        spawnBurst(p.x, p.y, '#ef4444');
                        p.life = 0;
                    } else {
                        p.life -= 0.05;
                    }
                }
                p.life -= 0.008;
                if (p.life <= 0) particles.splice(i, 1);
            }
            for (var i = burstEffects.length - 1; i >= 0; i--) {
                var b = burstEffects[i];
                b.x += b.vx; b.y += b.vy;
                b.vx *= 0.94; b.vy *= 0.94;
                b.life -= 0.03;
                if (b.life <= 0) burstEffects.splice(i, 1);
            }
        }

        function drawParticles() {
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
                ctx.shadowBlur = 15;
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
            ctx.clearRect(0, 0, W, H);
            if (W < 10 || H < 10) { requestAnimationFrame(loop); return; }
            drawGrid();
            drawEdges();
            drawNodes();
            updateParticles();
            drawParticles();
            requestAnimationFrame(loop);
        }

        window.addEventListener('resize', resize);
        resize();
        loop();
    }

    function resetNodeStates() {
        nodeStates = {};
        Object.keys(NODE_POS).forEach(function(id) {
            nodeStates[id] = { active: false, revoked: false, denied: false, trust: null };
        });
    }

    function updateCounts() {
        var e;
        e = document.getElementById('pAllow'); if (e) e.textContent = counts.allow;
        e = document.getElementById('pDeny'); if (e) e.textContent = counts.deny;
        e = document.getElementById('pRevoke'); if (e) e.textContent = counts.revoke;
    }

    function updateProgress() {
        var total = steps.length;
        var pct = total > 0 ? ((currentStep + 1) / total) * 100 : 0;
        var fill = document.getElementById('progressFill');
        if (fill) fill.style.width = pct + '%';
        var label = document.getElementById('stepLabel');
        if (label) label.textContent = (currentStep + 1) + ' / ' + total;
    }

    function updateTimeline() {
        var tl = document.getElementById('timeline');
        if (!tl) return;
        var html = '';
        steps.forEach(function(s, i) {
            var cls = '';
            if (i < currentStep) {
                cls = 'done';
                if (s.type === 'deny') cls += ' deny-step';
                if (s.type === 'revoke') cls += ' revoke-step';
            } else if (i === currentStep) {
                cls = 'current';
            }
            html += '<div class="cp-tl-step ' + cls + '" onclick="CP.goToStep(' + i + ')">' + (i + 1) + '. ' + s.label + '</div>';
        });
        tl.innerHTML = html;
    }

    function updateDetail(stepIdx) {
        var content = document.getElementById('detailContent');
        var whyBlocked = document.getElementById('whyBlocked');
        var whyReasons = document.getElementById('whyReasons');
        if (!content || stepIdx < 0 || stepIdx >= steps.length) {
            if (content) content.innerHTML = '<div style="color:rgba(255,255,255,0.3)">播放后点击步骤查看详情</div>';
            if (whyBlocked) whyBlocked.className = 'cp-why-blocked';
            return;
        }

        var s = steps[stepIdx];
        var typeBadge = s.type === 'allow' ? '<span class="cp-badge cp-badge-allow">ALLOW</span>' :
                        (s.type === 'deny' ? '<span class="cp-badge cp-badge-deny">DENY</span>' :
                        (s.type === 'revoke' ? '<span class="cp-badge cp-badge-revoke">🔥 REVOKED</span>' :
                        '<span class="cp-badge" style="background:rgba(167,139,250,0.15);color:#c4b5fd">INFO</span>'));

        var html = '';
        html += '<div class="cp-detail-row"><span class="cp-detail-key">Step</span><span class="cp-detail-val">' + (stepIdx + 1) + ' / ' + steps.length + '</span></div>';
        html += '<div class="cp-detail-row"><span class="cp-detail-key">From</span><span class="cp-detail-val" style="color:' + (NODE_COLORS[s.from] || {}).color + '">' + s.from + '</span></div>';
        html += '<div class="cp-detail-row"><span class="cp-detail-key">To</span><span class="cp-detail-val" style="color:' + (NODE_COLORS[s.to] || {}).color + '">' + s.to + '</span></div>';
        html += '<div class="cp-detail-row"><span class="cp-detail-key">Action</span><span class="cp-detail-val">' + (s.action || '—') + '</span></div>';
        html += '<div class="cp-detail-row"><span class="cp-detail-key">Decision</span><span class="cp-detail-val">' + typeBadge + '</span></div>';
        if (s.trust != null) {
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Trust</span><span class="cp-detail-val" style="color:' + (s.trust < 0.3 ? '#ef4444' : (s.trust < 0.7 ? '#fbbf24' : '#34d399')) + '">' + s.trust.toFixed(2) + '</span></div>';
        }
        if (s.reason) {
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Reason</span><span class="cp-detail-val">' + s.reason + '</span></div>';
        }
        if (s.prompt_risk_score != null) {
            var riskColor = s.prompt_risk_score > 0.7 ? '#ef4444' : (s.prompt_risk_score > 0.35 ? '#fbbf24' : '#34d399');
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Prompt Risk</span><span class="cp-detail-val" style="color:' + riskColor + '">' + s.prompt_risk_score.toFixed(2) + '</span></div>';
        }
        if (s.attack_types && s.attack_types.length > 0) {
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Attack Types</span><span class="cp-detail-val" style="color:#f87171">' + s.attack_types.join(', ') + '</span></div>';
        }
        if (s.attack_intent) {
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Attack Intent</span><span class="cp-detail-val" style="color:#fbbf24">' + s.attack_intent + '</span></div>';
        }
        if (s.severity) {
            var sevColor = s.severity === 'critical' ? '#ef4444' : (s.severity === 'high' ? '#f87171' : '#fbbf24');
            html += '<div class="cp-detail-row"><span class="cp-detail-key">Severity</span><span class="cp-detail-val" style="color:' + sevColor + '">' + s.severity + '</span></div>';
        }
        var explainData = {
            agent_id: s.to || '',
            action: s.action || '',
            decision: s.type === 'allow' ? 'allow' : 'deny',
            reason: s.reason || '',
            trust_score: s.trust,
            risk_score: 0,
            chain_detail: [],
            blocked_at: s.type === 'deny' ? (s.action === 'prompt_injection_blocked' ? 'prompt_defense' : 'check') : '',
            auto_revoked: s.type === 'revoke',
            prompt_risk_score: s.prompt_risk_score,
            attack_types: s.attack_types || [],
            attack_intent: s.attack_intent || '',
            severity: s.severity || '',
        };
        html += '<div style="margin-top:10px">' + IAM_EXPLAIN.makeBtn('🧠 Explain Decision', explainData, 'iam-explain-btn-lg') + '</div>';
        content.innerHTML = html;

        if ((s.type === 'deny' || s.type === 'revoke') && whyBlocked && whyReasons) {
            whyBlocked.className = 'cp-why-blocked show';
            var reasons = [];
            if (s.reason) reasons.push(s.reason);
            if (s.type === 'deny') {
                if (s.reason && s.reason.indexOf('capability') >= 0) reasons.push('Capability scope mismatch');
                if (s.reason && s.reason.indexOf('trust') >= 0) reasons.push('Trust score below threshold');
                if (reasons.length === 0) { reasons.push('Capability scope mismatch'); reasons.push('Trust score below threshold'); }
            }
            if (s.type === 'revoke') {
                reasons.push('Trust score dropped below auto-revoke threshold');
                reasons.push('All tokens for this agent have been revoked');
            }
            whyReasons.innerHTML = reasons.map(function(r) { return '<div class="cp-why-reason">' + r + '</div>'; }).join('');
        } else if (whyBlocked) {
            whyBlocked.className = 'cp-why-blocked';
        }
    }

    function executeStep(idx) {
        if (idx < 0 || idx >= steps.length) return;
        var s = steps[idx];

        Object.keys(nodeStates).forEach(function(id) { nodeStates[id].active = false; });

        if (nodeStates[s.from]) nodeStates[s.from].active = true;
        if (nodeStates[s.to]) {
            nodeStates[s.to].active = true;
            if (s.trust != null) nodeStates[s.to].trust = s.trust;
        }

        if (s.type === 'deny') {
            if (nodeStates[s.to]) nodeStates[s.to].denied = true;
            counts.deny++;
        } else if (s.type === 'revoke') {
            if (nodeStates[s.to]) { nodeStates[s.to].revoked = true; nodeStates[s.to].denied = true; }
            counts.revoke++;
        } else if (s.type === 'allow') {
            counts.allow++;
        }

        if (canvasOk) {
            spawnParticle(s.from, s.to, s.type);
            if (s.type === 'revoke') {
                var pos = getPos(s.to);
                spawnBurst(pos.x, pos.y, '#ef4444');
            }
        }

        pushLiveEvent(s.type, s.to, s.action || s.label);
        updateCounts();
        updateProgress();
        updateTimeline();
        updateDetail(idx);
    }

    function clearPlayback() {
        stopPlay();
        currentStep = -1;
        counts = { allow: 0, deny: 0, revoke: 0 };
        particles = [];
        burstEffects = [];
        resetNodeStates();
        updateCounts();
        updateProgress();
        updateTimeline();
        updateDetail(-1);
    }

    function startPlay() {
        if (steps.length === 0) return;
        isPlaying = true;
        document.getElementById('playBtn').textContent = '⏸';
        playNext();
    }

    function stopPlay() {
        isPlaying = false;
        if (playTimer) { clearTimeout(playTimer); playTimer = null; }
        var btn = document.getElementById('playBtn');
        if (btn) btn.textContent = '▶';
    }

    function playNext() {
        if (!isPlaying) return;
        if (currentStep >= steps.length - 1) { stopPlay(); return; }
        currentStep++;
        executeStep(currentStep);
        playTimer = setTimeout(playNext, playSpeed);
    }

    function buildNormalSteps(data) {
        var trust = getTrustScore('doc_agent') || 0.95;
        return [
            { from: 'user', to: 'prompt_defense', label: 'Prompt Defense ✓', action: '三层融合引擎检测通过', type: 'info', trust: trust },
            { from: 'prompt_defense', to: 'doc_agent', label: 'Issue Token', action: 'issue token → doc_agent', type: 'info', trust: trust },
            { from: 'doc_agent', to: 'data_agent', label: 'Delegate', action: 'delegate → data_agent', type: 'info', trust: getTrustScore('data_agent') || 0.92 },
            { from: 'data_agent', to: 'policy_engine', label: 'Policy Check', action: 'check: read:feishu_table', type: 'info', trust: null },
            { from: 'policy_engine', to: 'data_agent', label: 'ALLOW ✓', action: 'capability matched', type: 'allow', trust: null, reason: 'Capability scope matched, trust score normal' },
        ];
    }

    function buildEscalationSteps(data) {
        var trust = getTrustScore('external_agent') || 0.7;
        var steps = [
            { from: 'user', to: 'prompt_defense', label: 'Prompt Defense ✓', action: '三层融合引擎检测通过', type: 'info', trust: trust },
            { from: 'prompt_defense', to: 'external_agent', label: 'Issue Token', action: 'issue token → external_agent', type: 'info', trust: trust },
        ];
        if (data && data.steps) {
            data.steps.forEach(function(s) {
                if (s.action === 'issue_root_token') return;
                if (s.allowed === true) {
                    steps.push({ from: 'external_agent', to: 'policy_engine', label: s.action.split(':').pop() + ' ✓', action: s.action, type: 'allow', trust: s.trust_score || trust, reason: s.reason || 'Within capability scope' });
                } else if (s.allowed === false) {
                    steps.push({ from: 'external_agent', to: 'policy_engine', label: 'Policy Check', action: 'check: ' + s.action, type: 'info', trust: null });
                    steps.push({ from: 'policy_engine', to: 'external_agent', label: 'DENY ✗', action: 'capability mismatch', type: 'deny', trust: s.trust_score || (trust - 0.1), reason: s.reason || 'Capability scope mismatch' });
                }
            });
        }
        if (steps.length <= 1) {
            steps.push({ from: 'external_agent', to: 'policy_engine', label: 'Weather ✓', action: 'read:weather', type: 'allow', trust: trust, reason: 'Within capability scope' });
            steps.push({ from: 'external_agent', to: 'policy_engine', label: 'Policy Check', action: 'check: read:feishu_table:finance', type: 'info', trust: null });
            steps.push({ from: 'policy_engine', to: 'external_agent', label: 'DENY ✗', action: 'capability mismatch', type: 'deny', trust: trust - 0.1, reason: 'Capability scope mismatch: external_agent has read:weather but requested read:feishu_table:finance' });
        }
        return steps;
    }

    function buildReplaySteps(data) {
        var trust = getTrustScore('doc_agent') || 0.95;
        return [
            { from: 'user', to: 'prompt_defense', label: 'Prompt Defense ✓', action: '三层融合引擎检测通过', type: 'info', trust: trust },
            { from: 'prompt_defense', to: 'doc_agent', label: 'Issue Token', action: 'issue token → doc_agent', type: 'info', trust: trust },
            { from: 'doc_agent', to: 'data_agent', label: '1st Use ✓', action: 'read:feishu_table (1st)', type: 'allow', trust: getTrustScore('data_agent') || 0.92, reason: 'First use of token, allowed' },
            { from: 'doc_agent', to: 'data_agent', label: 'REPLAY ✗', action: 'read:feishu_table (REPLAY)', type: 'deny', trust: trust - 0.15, reason: 'Token already used — replay attack detected' },
        ];
    }

    function buildAutoRevokeSteps(data) {
        var initialTrust = (data && data.initial_trust) || 0.7;
        var finalTrust = (data && data.final_trust) || 0.2;
        var trustStep = (finalTrust - initialTrust) / 3;
        return [
            { from: 'user', to: 'prompt_defense', label: 'Prompt Defense ✓', action: '三层融合引擎检测通过', type: 'info', trust: initialTrust },
            { from: 'prompt_defense', to: 'external_agent', label: 'Issue Token', action: 'issue token → external_agent', type: 'info', trust: initialTrust },
            { from: 'external_agent', to: 'policy_engine', label: 'Weather ✓', action: 'read:weather', type: 'allow', trust: initialTrust, reason: 'Within capability scope' },
            { from: 'external_agent', to: 'policy_engine', label: 'Request 2 ✗', action: 'read:feishu_table:finance', type: 'deny', trust: initialTrust + trustStep, reason: 'Capability mismatch: requested finance data' },
            { from: 'external_agent', to: 'policy_engine', label: 'Request 3 ✗', action: 'read:feishu_table:hr', type: 'deny', trust: initialTrust + trustStep * 2, reason: 'Capability mismatch + trust degrading' },
            { from: 'policy_engine', to: 'external_agent', label: '🔥 AUTO-REVOKED', action: 'trust < threshold', type: 'revoke', trust: finalTrust, reason: 'Trust score (' + finalTrust.toFixed(2) + ') dropped below auto-revoke threshold — all tokens revoked' },
        ];
    }

    var GOV_BASE = '/api/governance';

    function buildPromptInjectSteps(data) {
        var trust = getTrustScore('data_agent') || 0.95;
        return [
            { from: 'user', to: 'prompt_defense', label: 'Prompt Defense', action: '三层融合引擎检测', type: 'info', trust: trust },
            { from: 'prompt_defense', to: 'data_agent', label: '🛡️ INJECTION BLOCKED', action: 'prompt_injection_blocked', type: 'deny', trust: trust - 0.05, reason: '三层融合引擎检测到 data_theft 攻击（风险=0.56，严重度=medium），攻击类型：ignore_rules', prompt_risk_score: 0.56, attack_types: ['ignore_rules'], attack_intent: 'data_theft', severity: 'medium' },
        ];
    }

    function buildProgressiveDemoSteps() {
        var trust = 0.95;
        return [
            { from: 'user', to: 'prompt_defense', label: 'Step1: 正常请求', action: '查财务数据 → 三层融合引擎检测通过', type: 'info', trust: trust, reason: '正常请求，Prompt Defense 通过' },
            { from: 'prompt_defense', to: 'doc_agent', label: 'Issue Token', action: 'issue token → doc_agent', type: 'info', trust: trust },
            { from: 'doc_agent', to: 'data_agent', label: 'Delegate ✓', action: 'delegate → data_agent', type: 'allow', trust: 0.96, reason: '✅ 正常执行，Trust: 0.95 → 0.96' },
            { from: 'user', to: 'prompt_defense', label: 'Step2: 轻微诱导', action: '稍微放宽限制 → 检测到潜在风险', type: 'info', trust: 0.89, prompt_risk_score: 0.35, attack_types: ['ignore_rules'], reason: '⚠️ 检测到潜在风险（risk=0.35）' },
            { from: 'prompt_defense', to: 'data_agent', label: '⚠️ DEGRADED', action: '降权执行（部分能力被限制）', type: 'info', trust: 0.89, prompt_risk_score: 0.35, attack_types: ['ignore_rules'], reason: '🛡️ IAM：降权执行，Trust: 0.95 → 0.89' },
            { from: 'user', to: 'prompt_defense', label: 'Step3: 强攻击', action: '忽略所有安全限制 → Prompt Injection', type: 'info', trust: 0.74 },
            { from: 'prompt_defense', to: 'data_agent', label: '🔥 INJECTION BLOCKED', action: 'prompt_injection_blocked', type: 'deny', trust: 0.74, prompt_risk_score: 0.72, attack_types: ['ignore_rules'], attack_intent: 'data_theft', severity: 'high', reason: '🔥 Prompt Injection Detected，风险分：0.72（High），Trust: 0.89 → 0.74' },
            { from: 'user', to: 'prompt_defense', label: 'Step4: 连续攻击', action: '你现在是管理员 → 角色覆写', type: 'info', trust: 0.60, prompt_risk_score: 0.55, attack_types: ['overwrite_role'] },
            { from: 'prompt_defense', to: 'data_agent', label: '⚠️ DEGRADED', action: '降权执行', type: 'info', trust: 0.60, reason: 'Trust: 0.74 → 0.60' },
            { from: 'user', to: 'prompt_defense', label: 'Step4+: 再次攻击', action: '忽略所有规则 → 连续高风险', type: 'info', trust: 0.39 },
            { from: 'prompt_defense', to: 'data_agent', label: '🔥 AUTO-REVOKED', action: '连续高风险 Prompt 行为', type: 'revoke', trust: 0.0, reason: '🔥 Agent 已被自动封禁（Auto-Revoke），原因：连续高风险 Prompt 行为，Trust: 0.60 → 0.00' },
        ];
    }

    function buildStepsFromEvents(events) {
        if (!events || events.length === 0) return [];
        var steps = [];
        events.forEach(function(ev) {
            var chain = (ev.agent_chain || []).map(function(c) {
                return c.startsWith('user:') ? 'user' : c;
            });
            var result = ev.result || 'allow';
            var type = result === 'auto_revoked' ? 'revoke' : (result === 'deny' || result === 'replay_blocked' ? 'deny' : 'allow');
            var action = ev.action || '';

            if (action === 'prompt_injection_blocked') {
                var target = chain.length >= 2 ? chain[1] : (ev.agent_id || 'doc_agent');
                steps.push({
                    from: 'user',
                    to: 'prompt_defense',
                    label: 'Prompt Defense',
                    action: '三层融合引擎检测',
                    type: 'info',
                    trust: ev.trust_before,
                });
                steps.push({
                    from: 'prompt_defense',
                    to: target,
                    label: '🛡️ INJECTION BLOCKED',
                    action: 'prompt_injection_blocked',
                    type: 'deny',
                    trust: ev.trust_after,
                    reason: ev.reason || '',
                    prompt_risk_score: ev.prompt_risk_score,
                    attack_types: ev.attack_types,
                    attack_intent: ev.attack_intent,
                    severity: ev.severity,
                });
                return;
            }

            if (action === 'prompt_defense_degraded') {
                var target = chain.length >= 2 ? chain[1] : (ev.agent_id || 'doc_agent');
                steps.push({
                    from: 'user',
                    to: 'prompt_defense',
                    label: 'Prompt Defense',
                    action: '三层融合引擎检测',
                    type: 'info',
                    trust: ev.trust_before,
                });
                steps.push({
                    from: 'prompt_defense',
                    to: target,
                    label: '⚠️ DEGRADED',
                    action: 'prompt_defense_degraded',
                    type: 'info',
                    trust: ev.trust_after,
                    reason: ev.reason || '',
                    prompt_risk_score: ev.prompt_risk_score,
                    attack_types: ev.attack_types,
                    attack_intent: ev.attack_intent,
                    severity: ev.severity,
                });
                return;
            }

            if (action === 'prompt_defense_passed') {
                var target = chain.length >= 2 ? chain[1] : (ev.agent_id || 'doc_agent');
                steps.push({
                    from: 'user',
                    to: 'prompt_defense',
                    label: 'Prompt Defense ✓',
                    action: '三层融合引擎检测通过',
                    type: 'info',
                    trust: ev.trust_before,
                    prompt_risk_score: ev.prompt_risk_score,
                });
                steps.push({
                    from: 'prompt_defense',
                    to: target,
                    label: 'PASS ✓',
                    action: 'prompt_defense_passed',
                    type: 'info',
                    trust: ev.trust_after,
                });
                return;
            }

            if (chain.length === 0) return;

            if (chain.length === 1) {
                steps.push({
                    from: 'policy_engine',
                    to: chain[0],
                    label: type === 'allow' ? 'ALLOW ✓' : (type === 'deny' ? 'DENY ✗' : '🔥 AUTO-REVOKED'),
                    action: ev.action || '',
                    type: type,
                    trust: ev.trust_after,
                    reason: ev.reason || '',
                });
                return;
            }

            steps.push({
                from: chain[0],
                to: chain[1],
                label: 'Issue Token',
                action: 'issue token → ' + chain[1],
                type: 'info',
                trust: ev.trust_before,
            });

            if (chain.length === 2) {
                steps.push({
                    from: chain[1],
                    to: 'policy_engine',
                    label: 'Policy Check',
                    action: 'check: ' + (ev.action || ''),
                    type: 'info',
                    trust: null,
                });
                steps.push({
                    from: 'policy_engine',
                    to: chain[1],
                    label: type === 'allow' ? 'ALLOW ✓' : (type === 'deny' ? 'DENY ✗' : '🔥 AUTO-REVOKED'),
                    action: type === 'allow' ? 'capability matched' : (type === 'deny' ? 'capability mismatch' : 'trust < threshold'),
                    type: type,
                    trust: ev.trust_after,
                    reason: ev.reason || '',
                });
                return;
            }

            for (var i = 1; i < chain.length - 1; i++) {
                var isLast = (i === chain.length - 2);
                if (!isLast) {
                    steps.push({
                        from: chain[i],
                        to: chain[i + 1],
                        label: 'Delegate',
                        action: 'delegate → ' + chain[i + 1],
                        type: 'info',
                        trust: ev.trust_after,
                    });
                } else {
                    steps.push({
                        from: chain[i],
                        to: 'policy_engine',
                        label: 'Policy Check',
                        action: 'check: ' + (ev.action || ''),
                        type: 'info',
                        trust: null,
                    });
                    steps.push({
                        from: 'policy_engine',
                        to: chain[i],
                        label: type === 'allow' ? 'ALLOW ✓' : (type === 'deny' ? 'DENY ✗' : '🔥 AUTO-REVOKED'),
                        action: type === 'allow' ? 'capability matched' : (type === 'deny' ? 'capability mismatch' : 'trust < threshold'),
                        type: type,
                        trust: ev.trust_after,
                        reason: ev.reason || '',
                    });
                }
            }
        });
        return steps;
    }

    async function selectScene(scene) {
        clearPlayback();
        selectedScene = scene;

        document.querySelectorAll('.cp-scene-btn').forEach(function(btn) { btn.classList.remove('active'); });
        var btnMap = { normal: 0, escalation: 1, replay: 2, autorevoke: 3, promptinject: 4, progressivedemo: 5 };
        var btns = document.querySelectorAll('.cp-scene-btn');
        if (btns[btnMap[scene]]) btns[btnMap[scene]].classList.add('active');

        var titles = { normal: '✅ Normal Flow', escalation: '⚠️ Escalation Attack', replay: '🔄 Replay Attack', autorevoke: '🔥 Auto-Revoke', promptinject: '🛡️ Prompt Injection', progressivedemo: '🎯 四步渐进式安全演示' };
        var titleEl = document.getElementById('sceneTitle');
        if (titleEl) titleEl.textContent = titles[scene] || scene;

        var empty = document.getElementById('playerEmpty');
        if (empty) empty.style.display = 'none';

        await loadTrustData();

        if (scene === 'normal') {
            try {
                var issueResp = await fetchJSON(BASE + '/issue-root', { method: 'POST', body: JSON.stringify({ agent_id: 'doc_agent', delegated_user: 'user_1' }) });
                steps = buildNormalSteps({ token: issueResp.token, capabilities: issueResp.capabilities });
            } catch (e) { steps = buildNormalSteps(null); }
        } else if (scene === 'escalation') {
            try {
                var data = await fetchJSON(BASE + '/demo/escalation-attack', { method: 'POST' });
                steps = buildEscalationSteps(data);
            } catch (e) { steps = buildEscalationSteps(null); }
        } else if (scene === 'replay') {
            try {
                var data = await fetchJSON(BASE + '/demo/replay-attack', { method: 'POST' });
                steps = buildReplaySteps(data);
            } catch (e) { steps = buildReplaySteps(null); }
        } else if (scene === 'autorevoke') {
            try {
                var data = await fetchJSON(BASE + '/demo/auto-revoke', { method: 'POST' });
                steps = buildAutoRevokeSteps(data);
                await loadTrustData();
            } catch (e) { steps = buildAutoRevokeSteps(null); }
        } else if (scene === 'promptinject') {
            steps = buildPromptInjectSteps(null);
        } else if (scene === 'progressivedemo') {
            steps = buildProgressiveDemoSteps();
        }

        updateTimeline();
        updateProgress();
    }

    function togglePlay() {
        if (steps.length === 0) return;
        if (isPlaying) { stopPlay(); } else { startPlay(); }
    }

    function stepForward() {
        if (steps.length === 0 || currentStep >= steps.length - 1) return;
        stopPlay();
        currentStep++;
        executeStep(currentStep);
    }

    function stepBack() {
        if (currentStep <= 0) return;
        stopPlay();
        clearPlayback();
        for (var i = 0; i < currentStep; i++) {
            executeStep(i);
        }
    }

    function goToStep(idx) {
        if (idx < 0 || idx >= steps.length) return;
        stopPlay();
        clearPlayback();
        for (var i = 0; i <= idx; i++) {
            executeStep(i);
        }
        currentStep = idx;
    }

    function seekTo(event) {
        if (steps.length === 0) return;
        var bar = document.getElementById('progressBar');
        if (!bar) return;
        var rect = bar.getBoundingClientRect();
        var pct = (event.clientX - rect.left) / rect.width;
        var idx = Math.min(steps.length - 1, Math.max(0, Math.floor(pct * steps.length)));
        goToStep(idx);
    }

    function renderInsight(title, text, color) {
        var area = document.getElementById('insightArea');
        if (!area) return;
        area.innerHTML = '<div class="cp-insight-box ' + color + '" style="margin-top:14px"><div class="cp-insight-title ' + color + '">' + title + '</div><div style="color:rgba(255,255,255,0.65)">' + text + '</div></div>';
    }

    connectWS();
    loadTrustData();
    resetNodeStates();

    return {
        selectScene: selectScene,
        togglePlay: togglePlay,
        stepForward: stepForward,
        stepBack: stepBack,
        goToStep: goToStep,
        seekTo: seekTo,
    };
})();
