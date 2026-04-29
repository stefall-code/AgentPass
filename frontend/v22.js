// v2.5 前端逻辑

// WebSocket 心跳 — 防止服务器空闲自动关闭
let _v22Ws = null;
function initV22WebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;
    try {
        _v22Ws = new WebSocket(url);
        _v22Ws.onopen = () => {
            console.log('[v22] WebSocket connected');
        };
        _v22Ws.onmessage = (e) => {
            try {
                const msg = JSON.parse(e.data);
                if (msg.type === 'audit_event') {
                    v22PushAuditEvent(msg);
                }
            } catch {}
        };
        _v22Ws.onerror = () => {};
        _v22Ws.onclose = () => {
            _v22Ws = null;
            setTimeout(initV22WebSocket, 5000);
        };
    } catch (e) {
        console.warn('[v22] WebSocket init failed:', e);
    }
}
function startV22Heartbeat() {
    initV22WebSocket();
    setInterval(() => {
        if (_v22Ws && _v22Ws.readyState === WebSocket.OPEN) {
            _v22Ws.send('ping');
        }
    }, 8000);
}

// 标签页切换
function setupTabs() {
    const tabs = document.querySelectorAll('.v22-tab');
    const sections = document.querySelectorAll('.v22-section');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const sectionId = tab.dataset.section;
            
            // 更新标签状态
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            // 更新内容区域
            sections.forEach(section => {
                section.classList.remove('active');
                if (section.id === sectionId) {
                    section.classList.add('active');
                }
            });
            
            // 加载对应数据
            loadSectionData(sectionId);
        });
    });
}

// 加载各区域数据
async function loadSectionData(sectionId) {
    switch (sectionId) {
        case 'overview':
            await loadDashboard();
            break;
        case 'platforms':
            await loadPlatforms();
            break;
        case 'approvals':
            await loadApprovals();
            break;
        case 'risk':
            await loadRisk();
            break;
        case 'cost':
            await loadCost();
            break;
        case 'dlp':
            break;
        case 'demo':
            // Demo 页面不需要自动加载数据
            break;
    }
}

// 加载仪表盘数据
async function loadDashboard() {
    try {
        const summaryResponse = await fetchWithTimeout('/api/v2/dashboard/summary');
        const summary = await summaryResponse.json();
        
        document.getElementById('connected-platforms').textContent = summary.connected_platforms;
        document.getElementById('today-events').textContent = summary.total_requests;
        document.getElementById('pending-approvals').textContent = summary.pending_approvals;
        document.getElementById('high-risk-events').textContent = summary.high_risk_events;
        document.getElementById('today-cost').textContent = `$${summary.total_cost.toFixed(2)}`;
        
        const trendsResponse = await fetchWithTimeout('/api/v2/dashboard/trends');
        const trends = await trendsResponse.json();
        
        document.getElementById('dashboard-summary').innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>总调用量</h4>
                    <p>${summary.total_requests}</p>
                </div>
                <div>
                    <h4>平台分布</h4>
                    <p>中国: ${summary.platform_distribution.cn}, 美国: ${summary.platform_distribution.us}</p>
                </div>
                <div>
                    <h4>高风险事件</h4>
                    <p>${summary.high_risk_events}</p>
                </div>
                <div>
                    <h4>总成本</h4>
                    <p>$${summary.total_cost.toFixed(2)}</p>
                </div>
            </div>
        `;
        
        document.getElementById('dashboard-trends').innerHTML = `
            <h4>7天趋势</h4>
            <div style="margin-top: 20px;">
                <p>调用量: ${trends.requests[0].value} (今日)</p>
                <p>风险评分: ${trends.risk[0].value.toFixed(2)}</p>
                <p>成本: $${trends.cost[0].value.toFixed(2)}</p>
            </div>
        `;
    } catch (error) {
        console.error('加载仪表盘数据失败:', error);
        document.getElementById('dashboard-summary').innerHTML = '<p>加载仪表盘数据失败</p>';
    }
}

// 加载平台数据
async function loadPlatforms() {
    try {
        const response = await fetchWithTimeout('/api/v2/platforms');
        const platforms = await response.json();
        
        const platformsList = document.getElementById('platforms-list');
        platformsList.innerHTML = '';
        
        platforms.forEach(platform => {
            const card = document.createElement('div');
            card.className = 'v22-platform-card';
            card.innerHTML = `
                <div class="v22-platform-region ${platform.region}">${platform.region === 'cn' ? '中国' : '美国'}</div>
                <h4>${platform.platform}</h4>
                <p>状态: ${platform.connected ? '在线' : '离线'}</p>
                <p>模式: ${platform.mock ? '模拟' : '真实'}</p>
            `;
            platformsList.appendChild(card);
        });
    } catch (error) {
        console.error('加载平台数据失败:', error);
        document.getElementById('platforms-list').innerHTML = '<p>加载平台失败</p>';
    }
}

// 加载审批数据
async function loadApprovals() {
    try {
        const response = await fetchWithTimeout('/api/v2/approvals/pending');
        const approvals = await response.json();
        
        const approvalsList = document.getElementById('approvals-list');
        
        if (approvals.length === 0) {
            approvalsList.innerHTML = '<p>无待审批项</p>';
            return;
        }
        
        let html = '<table class="v22-table">';
        html += '<tr><th>ID</th><th>平台</th><th>用户</th><th>操作</th><th>风险评分</th><th>操作</th></tr>';
        
        approvals.forEach(approval => {
            html += `
                <tr>
                    <td>${approval.id}</td>
                    <td>${approval.platform}</td>
                    <td>${approval.user}</td>
                    <td>${approval.action}</td>
                    <td>${approval.risk_score.toFixed(2)}</td>
                    <td>
                        <button class="v22-btn v22-btn-primary" onclick="approveApproval('${approval.id}')">批准</button>
                        <button class="v22-btn v22-btn-secondary" onclick="rejectApproval('${approval.id}')">拒绝</button>
                    </td>
                </tr>
            `;
        });
        
        html += '</table>';
        approvalsList.innerHTML = html;
    } catch (error) {
        console.error('加载审批数据失败:', error);
        document.getElementById('approvals-list').innerHTML = '<p>加载审批失败</p>';
    }
}

// 批准审批
async function approveApproval(approvalId) {
    try {
        const response = await fetch(`/api/v2/approvals/${approvalId}/approve`, {
            method: 'POST'
        });
        const result = await response.json();
        alert(result.message);
        loadApprovals();
    } catch (error) {
        console.error('批准审批失败:', error);
    }
}

// 拒绝审批
async function rejectApproval(approvalId) {
    try {
        const response = await fetch(`/api/v2/approvals/${approvalId}/reject`, {
            method: 'POST'
        });
        const result = await response.json();
        alert(result.message);
        loadApprovals();
    } catch (error) {
        console.error('拒绝审批失败:', error);
    }
}

// 加载风险数据
async function loadRisk() {
    try {
        // 加载高风险事件
        const eventsResponse = await fetch('/api/v2/risk/events');
        const events = await eventsResponse.json();
        
        const riskEvents = document.getElementById('risk-events');
        if (events.length === 0) {
            riskEvents.innerHTML = '<p>无高风险事件</p>';
        } else {
            let html = '<table class="v22-table">';
            html += '<tr><th>平台</th><th>用户</th><th>操作</th><th>风险评分</th><th>已阻断</th><th>原因</th></tr>';
            
            events.forEach(event => {
                html += `
                    <tr>
                        <td>${event.platform}</td>
                        <td>${event.user}</td>
                        <td>${event.action}</td>
                        <td>${event.risk.toFixed(2)}</td>
                        <td>${event.blocked ? '是' : '否'}</td>
                        <td>${event.reason || '-'}</td>
                    </tr>
                `;
            });
            
            html += '</table>';
            riskEvents.innerHTML = html;
        }
        
        // 加载高风险用户
        const usersResponse = await fetch('/api/v2/risk/top-users');
        const users = await usersResponse.json();
        
        const topRiskUsers = document.getElementById('top-risk-users');
        let usersHtml = '<table class="v22-table">';
        usersHtml += '<tr><th>用户</th><th>风险评分</th><th>事件数</th><th>阻断次数</th></tr>';
        
        users.forEach(user => {
            usersHtml += `
                <tr>
                    <td>${user.user}</td>
                    <td>${user.risk_score.toFixed(2)}</td>
                    <td>${user.event_count}</td>
                    <td>${user.blocked_count}</td>
                </tr>
            `;
        });
        
        usersHtml += '</table>';
        topRiskUsers.innerHTML = usersHtml;
        
        // 加载高风险平台
        const platformsResponse = await fetch('/api/v2/risk/top-platforms');
        const platforms = await platformsResponse.json();
        
        const topRiskPlatforms = document.getElementById('top-risk-platforms');
        let platformsHtml = '<table class="v22-table">';
        platformsHtml += '<tr><th>平台</th><th>风险评分</th><th>高风险次数</th></tr>';
        
        platforms.forEach(platform => {
            platformsHtml += `
                <tr>
                    <td>${platform.platform}</td>
                    <td>${platform.risk_score.toFixed(2)}</td>
                    <td>${platform.high_risk_count}</td>
                </tr>
            `;
        });
        
        platformsHtml += '</table>';
        topRiskPlatforms.innerHTML = platformsHtml;
    } catch (error) {
        console.error('加载风险数据失败:', error);
    }
}

// 加载成本数据
async function loadCost() {
    try {
        // 加载成本摘要
        const summaryResponse = await fetch('/api/v2/cost/summary');
        const summary = await summaryResponse.json();
        
        const costSummary = document.getElementById('cost-summary');
        costSummary.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;">
                <div>
                    <h4>总成本</h4>
                    <p>$${summary.total_cost.toFixed(2)}</p>
                </div>
                <div>
                    <h4>预算预警</h4>
                    <p>${summary.budget_alert ? '是' : '否'}</p>
                </div>
            </div>
        `;
        
        // 加载平台成本
        const platformsResponse = await fetch('/api/v2/cost/platforms');
        const platforms = await platformsResponse.json();
        
        const costPlatforms = document.getElementById('cost-platforms');
        let platformsHtml = '<table class="v22-table">';
        platformsHtml += '<tr><th>平台</th><th>成本</th><th>Token 使用量</th></tr>';
        
        platforms.forEach(platform => {
            platformsHtml += `
                <tr>
                    <td>${platform.platform}</td>
                    <td>$${platform.cost.toFixed(2)}</td>
                    <td>${platform.token_usage}</td>
                </tr>
            `;
        });
        
        platformsHtml += '</table>';
        costPlatforms.innerHTML = platformsHtml;
        
        // 加载用户成本
        const usersResponse = await fetch('/api/v2/cost/users');
        const users = await usersResponse.json();
        
        const costUsers = document.getElementById('cost-users');
        let usersHtml = '<table class="v22-table">';
        usersHtml += '<tr><th>用户</th><th>成本</th><th>请求次数</th></tr>';
        
        users.forEach(user => {
            usersHtml += `
                <tr>
                    <td>${user.user}</td>
                    <td>$${user.cost.toFixed(2)}</td>
                    <td>${user.requests}</td>
                </tr>
            `;
        });
        
        usersHtml += '</table>';
        costUsers.innerHTML = usersHtml;
    } catch (error) {
        console.error('加载成本数据失败:', error);
    }
}

// 生成演示数据
async function generateDemoData() {
    const resultDiv = document.getElementById('demo-result');
    resultDiv.innerHTML = '<p>生成演示数据中...</p>';
    
    try {
        // 模拟生成演示数据
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // 串行加载数据，避免并发请求过多导致断连
        const loadFunctions = [
            { name: '仪表盘', func: loadDashboard },
            { name: '平台', func: loadPlatforms },
            { name: '审批', func: loadApprovals },
            { name: '风险', func: loadRisk },
            { name: '成本', func: loadCost }
        ];
        
        for (const { name, func } of loadFunctions) {
            try {
                resultDiv.innerHTML = `<p>加载${name}数据中...</p>`;
                await func();
            } catch (error) {
                console.error(`加载${name}数据失败:`, error);
                // 继续执行其他加载，不中断整个流程
            }
        }
        
        resultDiv.innerHTML = '<p>演示数据生成成功！</p>';
    } catch (error) {
        console.error('生成演示数据失败:', error);
        resultDiv.innerHTML = '<p>生成演示数据失败</p>';
    }
}

// DLP 检测
async function dlpCheck() {
    const input = document.getElementById('dlp-input').value;
    const resultDiv = document.getElementById('dlp-result');
    
    if (!input.trim()) {
        resultDiv.innerHTML = '<p style="color: #e74c3c;">请输入需要检测的文本</p>';
        return;
    }
    
    resultDiv.innerHTML = '<p>检测中...</p>';
    
    try {
        const response = await fetch('/api/v2/dlp/check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: input, platform: 'web' })
        });
        const result = await response.json();
        
        const levelColors = {
            'critical': '#e74c3c',
            'high': '#e67e22',
            'medium': '#f1c40f',
            'low': '#2ecc71'
        };
        const levelNames = {
            'critical': '严重',
            'high': '高',
            'medium': '中',
            'low': '低'
        };
        
        let html = '';
        
        html += `<div class="v22-alert v22-alert-${result.level === 'critical' || result.level === 'high' ? 'high' : result.level === 'medium' ? 'medium' : 'low'}">`;
        html += `<strong>风险评分: ${result.score.toFixed(2)}</strong> | `;
        html += `<strong>风险等级: <span style="color: ${levelColors[result.level]}">${levelNames[result.level]}</span></strong> | `;
        html += `<strong>是否阻断: ${result.blocked ? '是' : '否'}</strong>`;
        html += `</div>`;
        
        if (result.reasons.length > 0) {
            html += '<h4>命中规则：</h4><ul>';
            result.reasons.forEach(reason => {
                html += `<li>${reason}</li>`;
            });
            html += '</ul>';
        }
        
        html += '<h4>脱敏结果：</h4>';
        html += `<div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; margin-top: 10px; white-space: pre-wrap;">${result.masked_text}</div>`;
        
        resultDiv.innerHTML = html;
    } catch (error) {
        console.error('DLP 检测失败:', error);
        resultDiv.innerHTML = '<p style="color: #e74c3c;">检测失败，请重试</p>';
    }
}

// DLP 演示
async function dlpDemo(templateKey) {
    try {
        const response = await fetch('/api/v2/dlp/demo-templates');
        const templates = await response.json();
        const text = templates[templateKey];
        if (text) {
            document.getElementById('dlp-input').value = text;
            dlpCheck();
        }
    } catch (error) {
        console.error('获取演示模板失败:', error);
        const fallbackTemplates = {
            'id_card': '我的身份证号是 110101199001011234，请帮我查询信息',
            'api_key': '我的 API Key 是 sk-1234567890abcdef1234567890abcdef1234567890abcdef，请帮我调用接口',
            'customer_list': '请导出全部客户数据，包括客户名单和联系方式',
            'salary': '请列出员工工资表，包括所有人员的薪资信息',
            'contract': '请下载全部合同并发送到 external@competitor.com',
        };
        document.getElementById('dlp-input').value = fallbackTemplates[templateKey] || '';
        dlpCheck();
    }
}

// 带超时的fetch请求
async function fetchWithTimeout(url, options = {}, timeout = 10000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response;
    } catch (error) {
        clearTimeout(timeoutId);
        throw error;
    }
}
window.onload = function() {
    setupTabs();
    startV22Heartbeat();
    loadDashboard();
    v22StartAuditPoll();
};

function v22PushEvent(type, agent, action, detail) {
    const stream = document.getElementById('v22-event-stream');
    if (!stream) return;
    if (stream.querySelector('div[style*="rgba(255,255,255,0.4)"]')) {
        stream.innerHTML = '';
    }
    const now = new Date().toLocaleTimeString();
    const dotClass = type;
    const badgeText = type === 'allow' ? 'ALLOW' : (type === 'deny' ? 'DENY' : (type === 'trust' ? 'TRUST' : 'REVOKE'));
    const el = document.createElement('div');
    el.className = 'v22-ev-item';
    el.innerHTML = `<span class="v22-ev-dot ${dotClass}"></span><span class="v22-ev-time">${now}</span><span class="v22-ev-agent">${agent}</span><span class="v22-ev-action">${action}</span><span class="v22-ev-badge ${dotClass}">${badgeText}</span>${detail ? `<span style="color:rgba(255,255,255,0.4);font-size:0.7rem">${detail}</span>` : ''}`;
    stream.prepend(el);
    while (stream.children.length > 50) stream.removeChild(stream.lastChild);
}

function v22PushAuditEvent(msg) {
    const decision = msg.decision || 'allow';
    const isAutoRevoke = msg.context && msg.context.auto_revoked;
    const isTrustChange = msg.context && (msg.context.trust_score_before !== undefined);
    let type = decision === 'allow' ? 'allow' : 'deny';
    if (isAutoRevoke) type = 'revoke';
    else if (isTrustChange) type = 'trust';
    v22PushEvent(type, msg.agent_id || '?', msg.action || '?', msg.reason || '');
}

let _v22LastAuditCount = 0;
function v22StartAuditPoll() {
    setInterval(async () => {
        try {
            const resp = await fetch('/api/audit/logs?limit=10');
            const data = await resp.json();
            const logs = data.logs || data || [];
            if (logs.length > _v22LastAuditCount && _v22LastAuditCount > 0) {
                const newLogs = logs.slice(0, logs.length - _v22LastAuditCount);
                newLogs.reverse().forEach(log => {
                    const decision = log.decision || 'allow';
                    const isAutoRevoke = log.context && log.context.auto_revoked;
                    const isTrustChange = log.context && (log.context.trust_score_before !== undefined);
                    let type = decision === 'allow' ? 'allow' : 'deny';
                    if (isAutoRevoke) type = 'revoke';
                    else if (isTrustChange) type = 'trust';
                    v22PushEvent(type, log.agent_id || '?', log.action || '?', log.reason || '');
                });
            }
            _v22LastAuditCount = logs.length;
        } catch {}
    }, 3000);
}

async function v22SimulateAttack() {
    v22PushEvent('trust', 'system', '🔥 模拟攻击启动', 'external_agent 连续越权...');
    try {
        const resp = await fetch('/api/delegate/demo/auto-revoke', { method: 'POST', headers: { 'Content-Type': 'application/json' } });
        const data = await resp.json();

        data.steps.forEach((step, idx) => {
            setTimeout(() => {
                if (step.status === 'AUTO_REVOKED') {
                    v22PushEvent('revoke', 'external_agent', '🔥 AUTO REVOKED', step.message || '');
                } else if (step.auto_revoked) {
                    v22PushEvent('revoke', 'external_agent', step.action || '', step.reason || '');
                } else if (step.trust_score_before !== undefined) {
                    const type = step.allowed ? 'allow' : 'deny';
                    v22PushEvent(type, 'external_agent', step.action || '', `trust: ${step.trust_score_before} → ${step.trust_score_after}`);
                    if (step.trust_delta < 0) {
                        v22PushEvent('trust', 'external_agent', 'trust change', `${step.trust_score_before} → ${step.trust_score_after} (${step.trust_delta})`);
                    }
                }
            }, idx * 400);
        });

        if (data.auto_revoked_triggered) {
            setTimeout(() => {
                v22PushEvent('revoke', 'system', '🔥 Agent 已被自动封禁', `trust=${data.final_trust} < threshold=${data.auto_revoke_threshold}`);
            }, data.steps.length * 400 + 200);
        }
    } catch (e) {
        v22PushEvent('deny', 'system', '模拟攻击失败', e.message);
    }
}
