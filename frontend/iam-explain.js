﻿﻿﻿﻿﻿﻿﻿﻿﻿var IAM_EXPLAIN = (function() {

    var styleInjected = false;

    function injectStyles() {
        if (styleInjected) return;
        styleInjected = true;
        var s = document.createElement('style');
        s.textContent = [
            '@keyframes iamExpIn{from{opacity:0}to{opacity:1}}',
            '@keyframes iamStepIn{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:translateX(0)}}',
            '@keyframes iamBtnPulse{0%,100%{box-shadow:0 0 4px rgba(139,92,246,0.3)}50%{box-shadow:0 0 12px rgba(139,92,246,0.6)}}',
            '@keyframes iamBtnGlow{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}',
            '@keyframes iamResultFlash{0%,100%{opacity:1}50%{opacity:0.6}}',
            '.iam-explain-btn{display:inline-flex;align-items:center;gap:5px;padding:5px 14px;border-radius:8px;border:1px solid rgba(139,92,246,0.4);background:linear-gradient(135deg,rgba(139,92,246,0.2),rgba(167,139,250,0.12));color:#c4b5fd;font-size:0.72rem;font-weight:700;cursor:pointer;transition:all 0.2s;letter-spacing:0.02em;animation:iamBtnPulse 2.5s ease infinite}',
            '.iam-explain-btn:hover{background:linear-gradient(135deg,rgba(139,92,246,0.35),rgba(167,139,250,0.25));transform:translateY(-1px);box-shadow:0 4px 16px rgba(139,92,246,0.3)}',
            '.iam-explain-btn-lg{padding:8px 20px;font-size:0.82rem;border-radius:10px}',
            '.iam-explain-btn-sm{padding:3px 8px;font-size:0.6rem;border-radius:5px;gap:3px;animation:none}',
            '.iam-explain-overlay{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,0.75);z-index:10000;display:flex;align-items:center;justify-content:center;backdrop-filter:blur(6px);animation:iamExpIn 0.3s ease}',
            '.iam-explain-modal{background:linear-gradient(180deg,#ffffff 0%,#f5f5f7 100%);border:1px solid rgba(139,92,246,0.15);border-radius:18px;width:92%;max-width:580px;max-height:82vh;overflow-y:auto;padding:0;box-shadow:0 24px 80px rgba(0,0,0,0.15),0 0 40px rgba(139,92,246,0.05)}',
            '.iam-explain-modal::-webkit-scrollbar{width:4px}',
            '.iam-explain-modal::-webkit-scrollbar-thumb{background:rgba(139,92,246,0.3);border-radius:2px}',
            '.iam-explain-header{padding:20px 24px 16px;border-bottom:1px solid rgba(0,0,0,0.06)}',
            '.iam-explain-body{padding:20px 24px 24px}',
            '.iam-explain-step{display:flex;gap:10px;align-items:flex-start;padding:10px 14px;margin-bottom:6px;border-radius:10px;opacity:0;animation:iamStepIn 0.35s ease forwards}',
            '.iam-explain-step-pass{background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.12)}',
            '.iam-explain-step-fail{background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.12)}',
            '.iam-explain-section{margin-bottom:14px;padding:12px 14px;background:rgba(0,0,0,0.02);border-radius:10px;border:1px solid rgba(0,0,0,0.04)}',
            '.iam-explain-section-title{font-size:0.75rem;font-weight:700;margin-bottom:6px}',
            '.iam-explain-section-body{font-size:0.7rem;color:#636366;line-height:1.6}',
        ].join('\n');
        document.head.appendChild(s);
    }

    function showExplainModal(explanation) {
        injectStyles();

        var existing = document.getElementById('iamExplainOverlay');
        if (existing) existing.remove();

        var decision = explanation.decision || 'unknown';
        var isAllow = decision === 'allow';
        var isAutoRevoked = explanation.auto_revoked || false;
        var isPromptBlocked = explanation.blocked_at === 'prompt_defense';
        var summaryColor = isAllow ? '#34d399' : (isAutoRevoked ? '#f87171' : (isPromptBlocked ? '#a78bfa' : '#ef4444'));
        var summaryIcon = isAllow ? '✅' : (isAutoRevoked ? '🔥' : (isPromptBlocked ? '🛡️' : '❌'));
        var headerBg = isAllow ? 'rgba(52,211,153,0.15)' : (isAutoRevoked ? 'rgba(239,68,68,0.2)' : (isPromptBlocked ? 'rgba(167,139,250,0.15)' : 'rgba(239,68,68,0.15)'));
        var headerBorder = isAllow ? 'rgba(52,211,153,0.3)' : (isAutoRevoked ? 'rgba(239,68,68,0.4)' : (isPromptBlocked ? 'rgba(167,139,250,0.3)' : 'rgba(239,68,68,0.3)'));

        var html = '';

        html += '<div class="iam-explain-header">';
        html += '<div style="display:flex;justify-content:space-between;align-items:center">';
        html += '<div style="display:flex;align-items:center;gap:10px">';
        html += '<div style="width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.2rem;background:' + headerBg + ';border:1px solid ' + headerBorder + '">' + summaryIcon + '</div>';
        html += '<div>';
        html += '<div style="font-size:1rem;font-weight:800;color:' + summaryColor + '">IAM 决策解释</div>';
        html += '<div style="font-size:0.65rem;color:#8e8e93">Explainable IAM · Agent ' + (explanation.agent_id || '—') + '</div>';
        html += '</div></div>';
        html += '<button onclick="document.getElementById(\'iamExplainOverlay\').remove()" style="width:32px;height:32px;border-radius:8px;background:rgba(0,0,0,0.04);border:1px solid rgba(0,0,0,0.08);color:#636366;font-size:1rem;cursor:pointer;display:flex;align-items:center;justify-content:center">✕</button>';
        html += '</div>';

        html += '<div style="margin-top:12px;padding:12px 14px;border-radius:10px;background:' + (isAllow ? 'rgba(52,211,153,0.08)' : (isAutoRevoked ? 'rgba(239,68,68,0.12)' : (isPromptBlocked ? 'rgba(167,139,250,0.08)' : 'rgba(239,68,68,0.08)'))) + ';border:1px solid ' + (isAllow ? 'rgba(52,211,153,0.2)' : (isAutoRevoked ? 'rgba(239,68,68,0.3)' : (isPromptBlocked ? 'rgba(167,139,250,0.2)' : 'rgba(239,68,68,0.2)'))) + '">';
        html += '<div style="font-size:0.85rem;font-weight:800;color:' + summaryColor + '">' + (explanation.summary || '') + '</div>';
        html += '<div style="font-size:0.65rem;color:#8e8e93;margin-top:4px">Action: ' + (explanation.action || '—') + '</div>';
        html += '</div></div>';

        html += '<div class="iam-explain-body">';

        html += '<div style="font-size:0.78rem;font-weight:700;color:#1d1d1f;margin-bottom:10px;display:flex;align-items:center;gap:6px"><span style="font-size:1rem">🔍</span> 决策步骤</div>';
        var steps = explanation.steps || [];
        steps.forEach(function(s, i) {
            var isPass = s.result === 'pass';
            var stepClass = isPass ? 'iam-explain-step-pass' : 'iam-explain-step-fail';
            html += '<div class="iam-explain-step ' + stepClass + '" style="animation-delay:' + (i * 0.12) + 's">';
            html += '<span style="font-size:1rem;flex-shrink:0;line-height:1">' + (s.icon || (isPass ? '✔️' : '❌')) + '</span>';
            html += '<div style="flex:1;min-width:0">';
            html += '<div style="display:flex;justify-content:space-between;align-items:center">';
            html += '<span style="font-size:0.76rem;font-weight:700;color:#1d1d1f">' + s.step + '</span>';
            html += '<span style="font-size:0.6rem;padding:2px 8px;border-radius:6px;font-weight:700;background:' + (isPass ? 'rgba(52,211,153,0.15)' : 'rgba(239,68,68,0.15)') + ';color:' + (isPass ? '#34d399' : '#ef4444') + '">' + s.result.toUpperCase() + '</span>';
            html += '</div>';
            html += '<div style="font-size:0.66rem;color:#636366;margin-top:3px;line-height:1.4">' + (s.detail || '') + '</div>';
            html += '</div></div>';
        });

        var sections = [
            { title: '⚠️ 风险分析', content: explanation.risk_analysis, color: '#fbbf24', bg: 'rgba(251,191,36,0.04)', border: 'rgba(251,191,36,0.1)' },
            { title: '🏆 信任分析', content: explanation.trust_analysis, color: '#64d2ff', bg: 'rgba(100,210,255,0.04)', border: 'rgba(100,210,255,0.1)' },
            { title: isAllow ? '✅ 最终决策' : '❌ 最终决策', content: explanation.final_reason, color: summaryColor, bg: isAllow ? 'rgba(52,211,153,0.06)' : 'rgba(239,68,68,0.06)', border: isAllow ? 'rgba(52,211,153,0.15)' : 'rgba(239,68,68,0.15)' },
            { title: '💡 建议', content: explanation.suggestion, color: '#c4b5fd', bg: 'rgba(196,181,253,0.04)', border: 'rgba(196,181,253,0.1)' },
        ];
        sections.forEach(function(sec) {
            if (!sec.content) return;
            html += '<div class="iam-explain-section" style="background:' + sec.bg + ';border-color:' + sec.border + '">';
            html += '<div class="iam-explain-section-title" style="color:' + sec.color + '">' + sec.title + '</div>';
            html += '<div class="iam-explain-section-body">' + sec.content + '</div>';
            html += '</div>';
        });

        html += '</div>';

        var overlay = document.createElement('div');
        overlay.id = 'iamExplainOverlay';
        overlay.className = 'iam-explain-overlay';
        overlay.innerHTML = '<div class="iam-explain-modal">' + html + '</div>';
        overlay.addEventListener('click', function(e) { if (e.target === overlay) overlay.remove(); });
        document.body.appendChild(overlay);
    }

    var _btnDataMap = {};
    var _btnCounter = 0;

    function makeBtn(text, data, cls) {
        var id = '_iamExp_' + (++_btnCounter);
        _btnDataMap[id] = data;
        return '<button class="iam-explain-btn ' + (cls || '') + '" data-iam-explain-id="' + id + '" onclick="IAM_EXPLAIN.explainByResult(IAM_EXPLAIN._getData(this.getAttribute(\'data-iam-explain-id\')))" >' + (text || '🧠 Explain') + '</button>';
    }

    function _getData(id) {
        return _btnDataMap[id] || {};
    }

    function explainByResult(data) {
        fetch('/api/explain/result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        })
        .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(showExplainModal)
        .catch(function(e) { alert('Explain failed: ' + e.message); });
    }

    function explainByMessage(message) {
        fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: message }),
        })
        .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(showExplainModal)
        .catch(function(e) { alert('Explain failed: ' + e.message); });
    }

    function explainByAgent(agentId, action) {
        fetch('/api/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ agent_id: agentId, action: action }),
        })
        .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(showExplainModal)
        .catch(function(e) { alert('Explain failed: ' + e.message); });
    }

    injectStyles();

    return {
        showExplainModal: showExplainModal,
        explainByResult: explainByResult,
        explainByMessage: explainByMessage,
        explainByAgent: explainByAgent,
        makeBtn: makeBtn,
        _getData: _getData,
    };
})();
