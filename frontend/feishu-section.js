﻿﻿﻿(function() {
    var BASE = '/api/feishu';
    var userId = 'feishu_demo_' + Math.random().toString(36).substring(2, 6);
    var counts = { events: 0, denied: 0 };
    var currentTrust = 0.92;

    function fetchJSON(url, opts) {
        opts = opts || {};
        return fetch(url, Object.assign({}, opts, { headers: Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {}) })).then(function(r) { return r.json(); });
    }

    function getTag(status, attackType) {
        if (status === 'success') return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.55rem;font-weight:700;background:rgba(52,211,153,0.15);color:#34d399;margin-left:6px">✅ ALLOW</span>';
        if (status === 'auto_revoked') return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.55rem;font-weight:700;background:rgba(239,68,68,0.2);color:#ef4444;margin-left:6px;animation:fsTagPulse 1s infinite">🔥 REVOKED</span>';
        if (attackType === 'replay') return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.55rem;font-weight:700;background:rgba(251,191,36,0.15);color:#fbbf24;margin-left:6px">🔁 REPLAY</span>';
        if (status === 'denied') return '<span style="display:inline-block;padding:1px 8px;border-radius:10px;font-size:0.55rem;font-weight:700;background:rgba(239,68,68,0.15);color:#ef4444;margin-left:6px">❌ DENY</span>';
        return '';
    }

    function getBubbleStyle(status, attackType) {
        if (status === 'success') return 'background:rgba(52,211,153,0.06);border-color:rgba(52,211,153,0.2)';
        if (status === 'auto_revoked') return 'background:rgba(239,68,68,0.1);border-color:rgba(239,68,68,0.3)';
        if (attackType === 'replay') return 'background:rgba(251,191,36,0.06);border-color:rgba(251,191,36,0.2)';
        if (status === 'denied') return 'background:rgba(239,68,68,0.06);border-color:rgba(239,68,68,0.2)';
        return '';
    }

    function addMsg(type, content, resultData) {
        var chat = document.getElementById('feishuChatArea');
        if (!chat) return;
        var div = document.createElement('div');
        div.style.cssText = 'margin-bottom:12px;animation:fsSlide 0.3s ease';

        if (type === 'user') {
            div.style.cssText += ';display:flex;justify-content:flex-end';
            div.innerHTML = '<div style="max-width:75%"><div style="font-size:0.5rem;color:rgba(255,255,255,0.2);text-align:right;margin-bottom:3px">👤 ' + userId + '</div><div style="background:rgba(0,113,227,0.15);border:1px solid rgba(0,113,227,0.25);border-radius:12px 12px 4px 12px;padding:9px 13px;font-size:0.8rem;color:#f5f5f7">' + escapeHtml(content) + '</div></div>';
        } else {
            var status = resultData ? resultData.status : 'unknown';
            var attackType = resultData ? resultData.attack_type : null;
            var bubbleStyle = getBubbleStyle(status, attackType);
            var tag = getTag(status, attackType);
            var expandId = 'fexp_' + Date.now() + '_' + Math.random().toString(36).substring(2, 6);

            div.style.cssText += ';display:flex;flex-direction:column';
            var html = '<div style="font-size:0.5rem;color:rgba(255,255,255,0.2);margin-bottom:3px">🤖 AgentPass Security ' + tag + '</div>';
            html += '<div id="' + expandId + '_bubble" style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.06);border-radius:12px 12px 12px 4px;padding:9px 13px;max-width:88%;font-size:0.8rem;color:rgba(255,255,255,0.7);white-space:pre-wrap;cursor:pointer;' + bubbleStyle + '" onclick="document.getElementById(\'' + expandId + '\').classList.toggle(\'open\')">' + escapeHtml(content) + '</div>';

            if (resultData) {
                var chainStr = (resultData.chain || []).join(' → ') || '—';
                var trustStr = resultData.trust_score != null ? resultData.trust_score.toFixed(2) : '—';
                html += '<div id="' + expandId + '" style="display:none;margin-top:6px;padding:8px 10px;background:rgba(0,0,0,0.3);border-radius:6px;border:1px solid rgba(255,255,255,0.04);font-size:0.65rem;color:rgba(255,255,255,0.4);font-family:var(--mono)">';
                html += '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.03)"><span>Chain</span><span style="color:rgba(255,255,255,0.6);font-weight:600">' + chainStr + '</span></div>';
                html += '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.03)"><span>Capability</span><span style="color:rgba(255,255,255,0.6);font-weight:600">' + (resultData.capability || '—') + '</span></div>';
                html += '<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid rgba(255,255,255,0.03)"><span>Trust</span><span style="color:rgba(255,255,255,0.6);font-weight:600">' + trustStr + '</span></div>';
                html += '<div style="display:flex;justify-content:space-between;padding:2px 0"><span>Policy</span><span style="color:rgba(255,255,255,0.6);font-weight:600">' + (resultData.blocked_at ? 'BLOCKED at ' + resultData.blocked_at : 'PASS ✓') + '</span></div>';
                html += '</div>';
            }

            if (resultData && resultData.status !== 'info') {
                var explainData = {
                    agent_id: (resultData.chain || ['']).slice(-1)[0] || '',
                    action: resultData.capability || '',
                    decision: resultData.status === 'success' ? 'allow' : 'deny',
                    reason: resultData.reason || '',
                    trust_score: resultData.trust_score,
                    risk_score: resultData.platform_risk || 0,
                    chain_detail: resultData.chain || [],
                    blocked_at: resultData.blocked_at || '',
                    auto_revoked: resultData.auto_revoked || false,
                    prompt_risk_score: resultData.prompt_risk_score,
                    attack_types: resultData.attack_types || [],
                    attack_intent: resultData.attack_intent || '',
                    severity: resultData.severity || '',
                };
                html += '<div style="margin-top:6px">' + IAM_EXPLAIN.makeBtn('🧠 Explain Decision', explainData) + '</div>';
            }

            div.innerHTML = html;
        }

        chat.appendChild(div);
        chat.scrollTop = chat.scrollHeight;
    }

    function escapeHtml(str) {
        var d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    function updateTrustUI(trustScore) {
        if (trustScore == null) return;
        var oldTrust = currentTrust;
        currentTrust = trustScore;

        var el = document.getElementById('feishuTrust');
        var bar = document.getElementById('feishuTrustBar');
        if (el) {
            el.textContent = trustScore.toFixed(2);
            el.style.color = trustScore >= 0.7 ? '#34d399' : (trustScore >= 0.5 ? '#fbbf24' : '#ef4444');
            if (trustScore < oldTrust) {
                el.style.transform = 'scale(1.2)';
                setTimeout(function() { el.style.transform = 'scale(1)'; }, 300);
            }
        }
        if (bar) {
            var pct = Math.max(0, Math.min(100, trustScore * 100));
            bar.style.width = pct + '%';
            bar.style.background = trustScore >= 0.7 ? '#34d399' : (trustScore >= 0.5 ? '#fbbf24' : '#ef4444');
        }
    }

    function updateCounts() {
        var e;
        e = document.getElementById('feishuEvents'); if (e) e.textContent = counts.events;
        e = document.getElementById('feishuDenied'); if (e) e.textContent = counts.denied;
    }

    function showGlobalAlert(msg) {
        var existing = document.querySelector('.fs-global-alert');
        if (existing) existing.remove();
        var alert = document.createElement('div');
        alert.className = 'fs-global-alert';
        alert.textContent = msg;
        document.body.appendChild(alert);
        setTimeout(function() {
            alert.style.transition = 'opacity 0.4s';
            alert.style.opacity = '0';
            setTimeout(function() { alert.remove(); }, 400);
        }, 4000);
    }

    async function send() {
        var input = document.getElementById('feishuMsgInput');
        if (!input) return;
        var msg = input.value.trim();
        if (!msg) return;
        input.value = '';

        addMsg('user', msg);

        try {
            var result = await fetchJSON(BASE + '/test', {
                method: 'POST',
                body: JSON.stringify({ user_id: userId, message: msg }),
            });

            addMsg('bot', result.content || '处理完成', result);

            if (result.trust_score != null) updateTrustUI(result.trust_score);

            counts.events++;
            if (result.status === 'denied' || result.status === 'auto_revoked') counts.denied++;
            updateCounts();

            if (result.auto_revoked) {
                showGlobalAlert('🔥 Agent 已被系统自动封禁 — 异常行为触发 Auto-Revoke');
            }
        } catch (e) {
            addMsg('bot', '❌ 系统错误：' + e.message, { status: 'error' });
        }
    }

    function quickSend(msg) {
        var input = document.getElementById('feishuMsgInput');
        if (input) input.value = msg;
        send();
    }

    var style = document.createElement('style');
    style.textContent = '@keyframes fsSlide{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}@keyframes fsTagPulse{0%,100%{opacity:1}50%{opacity:0.5}}.fs-global-alert{position:fixed;top:0;left:0;right:0;z-index:9999;padding:16px 24px;text-align:center;background:linear-gradient(135deg,rgba(239,68,68,0.9),rgba(220,38,38,0.9));color:white;font-weight:700;font-size:1rem;backdrop-filter:blur(12px);animation:fsSlide 0.4s ease}.open{display:block!important}';
    document.head.appendChild(style);

    window.FSI = {
        send: send,
        quickSend: quickSend,
    };
})();
