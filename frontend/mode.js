﻿var APP_MODE = (function() {
    var _mode = 'real';

    function init() {
        var params = new URLSearchParams(window.location.search);
        _mode = params.get('mode') === 'demo' ? 'demo' : 'real';
        renderBadge();
    }

    function getMode() {
        return _mode;
    }

    function isReal() {
        return _mode === 'real';
    }

    function isDemo() {
        return _mode === 'demo';
    }

    function toggle() {
        _mode = _mode === 'real' ? 'demo' : 'real';
        var params = new URLSearchParams(window.location.search);
        params.set('mode', _mode);
        var newUrl = window.location.pathname + '?' + params.toString();
        window.history.replaceState({}, '', newUrl);
        renderBadge();
    }

    function renderBadge() {
        var existing = document.getElementById('appModeBadge');
        if (existing) existing.remove();

        var badge = document.createElement('div');
        badge.id = 'appModeBadge';
        badge.style.cssText = 'position:fixed;top:12px;right:12px;z-index:9999;padding:6px 14px;border-radius:20px;font-size:0.72rem;font-weight:700;cursor:pointer;user-select:none;backdrop-filter:blur(8px);transition:all 0.3s';
        if (_mode === 'real') {
            badge.style.background = 'rgba(52,211,153,0.12)';
            badge.style.color = '#059669';
            badge.style.border = '1px solid rgba(52,211,153,0.3)';
            badge.textContent = '🟢 REAL DATA';
        } else {
            badge.style.background = 'rgba(217,119,6,0.12)';
            badge.style.color = '#b45309';
            badge.style.border = '1px solid rgba(217,119,6,0.3)';
            badge.textContent = '🟡 DEMO MODE';
        }
        badge.title = '点击切换模式 (当前: ' + _mode + ')';
        badge.onclick = toggle;
        document.body.appendChild(badge);
    }

    init();

    return {
        getMode: getMode,
        isReal: isReal,
        isDemo: isDemo,
        toggle: toggle,
    };
})();
