# Changelog

All notable changes to AgentPass are documented here.

## \[2.5.0] — 2026-04-30

### 🏆 评委级展示体系重构 — 从"能做出来"到"可信地很强"

本次更新聚焦于竞赛评委视角的展示优化，核心目标：让评委3秒内看到"这个系统是真的"。

#### 叙事驱动页面结构

- **3个主入口按钮**（替换原来的12个平铺按钮）：
  - 🔥 **为什么需要 AgentPass** — 三句话杀手页（问题→方案→创新）
  - ⚔️ **运行真实攻击测试** — 压倒性瞬间（脉冲动画按钮）
  - 🔍 **可验证性证明** — 评委验证
- 辅助按钮（四步演示/核心创新/四级撤销/OWASP/P2/OAuth/MCP-A2A/凭证经纪人/对齐检查）降级为 ghost 样式

#### 三句话杀手锏

1. Agent 身份不可验证 → A2A Token + 签名信任链
2. 权限会被绕过 → 能力约束 + 动态收缩
3. AI 会被攻击 → 把 Prompt 风险写入 IAM

核心流程：`Prompt → Risk → Trust → Capability → Decision`

#### 对抗式视觉证据

- **证据从"列表"改为"对抗流程"**：
  ```
  [ATTACK] No Token
      ↓
  [IAM CHECK]
      ↓
  [RESULT] 拒绝
  ```
- 每种攻击类型有独立中间层标签（IAM CHECK / SIGNATURE VERIFY / RISK→TRUST→CAPABILITY）
- 每条攻击标注 `来源: /api/xxx (真实后端)` — 证明不是前端动画
- 每条结果标注 `(403)` — HTTP 状态码，不是模拟

#### 压倒性瞬间 — 运行真实攻击测试

- 按钮文案：`⚔️ 运行真实攻击测试` + 小字 `调用后端 API，非前端模拟`
- 3种攻击实时执行：External Agent / Forged Token / Prompt Injection
- 每次攻击实时显示后端 API 来源 + BLOCKED 结果
- 毫秒级审计日志连续打印（`[04:39:52.182] external_agent → data_agent → DENY (312ms)`）
- 收尾一击：`攻击会改变权限结构` + 传统 IAM vs AgentPass 对比锚点

#### 结论卡片 + 可复现入口

- 3张结论卡片：外部攻击不可绕过(最硬) / 信任链不可伪造(架构级) / Prompt注入改变权限(核心创新)
- 每张卡片底部：`[运行测试] scripts/attack_bypass_test.py` / `[查看验证] /api/p2/judge/verify-all`
- 评委看到"这不是展示，是可以验证的"

#### 对比锚点

- 传统 IAM：`Request → Permission → Allow/Deny`（请求是静态的）
- AgentPass：`Prompt → Risk → Trust → Capability → Decision`（请求本身会改变权限）
- 差异总结：`差异不是"流程更多"，而是"权限由行为动态决定"`

#### 系统边界声明（工程声明）

- 适用：多 Agent 委派调用链 / 所有请求经过 IAM Gateway / 使用签名委派 Token（逐跳验证）
- 不覆盖：单 Agent 本地执行（无身份传播）/ Agent 内部逻辑漏洞（非身份问题）/ 非 Token 通道调用（绕过接入层）
- 结论：`本系统保证"身份与权限不可被外部绕过"，不保证"业务逻辑绝对安全"`
- 验证：`该边界可通过攻击测试脚本复现验证`

#### 指标精简 — 只保留3个最硬指标

- ❌ 删除：`7/7 PROVEN`、`10/10 OWASP`、`87.5% 兼容`、`<50ms 延迟(4个metrics)`
- ✅ 保留：外部攻击不可绕过 / 信任链不可伪造 / Prompt→权限变化
- 辅助信息降级为一行横排标签，不允许展开

#### 表达克制化

- `87.5%` → `7/8 fields aligned with industry standards`
- `7/7 PROVEN` → `核心声明全部有证据支撑`
- `fully implemented and running` → `core pipeline implemented`
- 删除"不是检测攻击/而是"等铺垫语
- 删除 q4 架构行、key_statement_cn 重复展示
- 辅助验证项从4行详情 → 一行横排短名

### 后端改动

- **`app/security/standard_hitl.py`** — KILLER_SUMMARY 重构为 three_core_claims 结构（含 evidence_items）；删除 chain_integrity/adversarial_agent/performance 独立区块；新增 supplementary 区块；兼容率从 `87.5%` 改为 `7/8 fields aligned`
- **`app/security/judge_verify.py`** — summary 从 `proven/total` 改为 `core_claims/core_claims_cn/all_core_proven`；verify_honest_capability_framing 表达克制化
- **`scripts/attack_bypass_test.py`** — 新增 Test [7] Chain tampering（MITM on delegation chain）

### 前端改动

- **`frontend/feishu.html`** — 按钮布局重构为3主入口+辅助；新增 fsAttackPulse 动画
- **`frontend/feishu.js`** — runKillerSummary 重写（对抗流程+结论卡片+边界声明+对比锚点）；runAttackDemo 新增（实时攻击+毫秒日志+收尾一击）；runJudgeVerify 重写（3核心声明+辅助验证项简化）

### 版本号更新

- 全量 `v2.4` → `v2.5`（main.py 4处 / 前端 HTML 8处 / v22.js / platforms.py / README.md）

## \[2.4.1] — 2026-04-27

### 🔗 链路打通修复 — 飞书请求完整进入 IAM 审计体系

**问题**：飞书 webhook 已能触发 orchestrator.run_task() 并返回真实数据，但请求不出现在 audit logs 和前端 Audit Center。

**根因**：
1. `app/adapters/__init__.py` — agentpass 模块缺失导致整个适配器层 ImportError，审计日志写入失败
2. `app/audit.py:log_event` — WebSocket 推送事件不包含 context 字段，前端实时收不到 platform 信息
3. `app/orchestrator/orchestrator.py` — 只传 trust_score，未传 trust_score_before/after

**修复**：
- `app/adapters/__init__.py` — try/except ImportError 优雅降级，提供轻量级 AuditAdapter
- `app/audit.py` — WebSocket emit_audit 事件增加 context 字段
- `app/orchestrator/orchestrator.py` — run_task 入口记录 trust_before，_log 辅助函数自动注入 trust_score_before/after
- `frontend/audit.js` — buildLogEntry 提取 platform/entry_point，renderRowCells 显示 💬 Feishu 标签，renderDetailContent 展示平台来源信息

### 🔄 真实数据 + 演示数据双模式驱动

**问题**：系统是"后端真实 + 前端假数据拼接" — Chain Viewer 用 demo API，Feishu 页面事件流是 mock，安全链路未从真实执行结果构建。

**改造**：建立统一真实事件流（Single Source of Truth）

- **统一 Governance Events API** — `GET /api/governance/events` 从数据库读取标准事件，支持 platform/agent_id/decision/action 过滤
  - 标准事件格式：event_id, timestamp, platform, entry_point, user_id, agent_id, agent_chain, action, result, trust_before, trust_after, risk_score, auto_revoked 等
- **Chain Viewer 改造** — selectScene 优先从 governance events 获取真实事件构建链路，buildStepsFromEvents 从 agent_chain 生成节点和边
- **Feishu 页面改造** — refresh 从 `/api/governance/events?platform=feishu` 获取真实事件流，渲染事件列表+安全链路+信任评分
- **模式切换** — 新增 `frontend/mode.js`，URL 参数 `?mode=real/demo`，右上角浮动徽章 🟢 REAL DATA / 🟡 DEMO MODE

### 🌐 external_agent 真实职责化

**问题**：external_agent 只有 `write:doc:public`，没有任何真实执行逻辑，是"废的"。

**改造**：external_agent → 外部信息代理（天气/搜索/API）

- **新增 capability** — `read:external_api:weather` + `read:web`（统一 engine.py 和 orchestrator.py）
- **天气路由** — `_parse_intent` 新增 "天气/weather/气温/温度/下雨/晴" → external_agent + read:external_api:weather
- **真实天气 API** — `_execute_external_agent` → `_query_weather()` → 调用 wttr.in 免费 API，支持 12 个中国城市
- **run_task 签发修复** — root token 按 target_agent 签发（而非硬编码 doc_agent），external_agent 直接 check + execute

### 🛡️ Prompt Defense 接入真实调用链

**问题**：Prompt Defense 是独立模块，仅在 /api/prompt-defense/analyze 和 /api/openclaw/check 被调用，Orchestrator/飞书 webhook/IAM check 完全不经过 prompt 注入检测。

**改造**：在 orchestrator.run_task() 中，IAM check 之前插入 prompt 注入检测

- **新增 `_check_prompt_injection()`** — 调用 PromptDefense 三层融合引擎，检测到注入则阻断并记录审计日志
- **调用链**：飞书/前端消息 → _parse_intent → **_check_prompt_injection** → secure_agent_call → _execute_agent
- **阻断行为**：记录 prompt_injection_blocked 审计日志，信任分惩罚，返回攻击类型和风险分数
- **_humanize_block 新增 prompt_injection 阶段**

### 🐛 Bug 修复

- **`'list' object has no attribute 'items'`** — `_generate_report` 对 data 调用 .items()，但 _execute_data_agent 多条记录时 data 是 list。修复：isinstance 分支处理 list/dict
- **飞书天气请求第三次报错** — `_process_feishu_message` 中同步调用 run_task（含 urllib.request.urlopen 阻塞 IO），多请求并发卡死事件循环。修复：`await asyncio.to_thread(run_task, ...)`
- **Chain Viewer user:xxx 替换错误** — `'user:e2e_verify'.replace('user:', 'user')` → `'usere2e_verify'`。修复：`c.startsWith('user:') ? 'user' : c`
- **Chain Viewer 缺少 policy_engine 步骤** — 真实 chain 只有 agent 节点，缺少 IAM 策略检查可视化。修复：自动插入 policy_engine 步骤
- **Escalation demo token 复用** — engine.check() 消费 token 后第二次 check 被当重放攻击。修复：为每次 check 签发独立 token
- **Demo 端点 capability 不一致** — external_agent 从 write:doc:public 改为 read:external_api:weather 后，6 个位置的旧 capability 未同步。修复：统一所有 escalation_actions 和 capabilities 引用
- **feishu/router.py union_id 错误消息** — 错误分支中 send_message 传了 reply_content 而非 error_msg

### 新增文件

- `frontend/mode.js` — 全局模式管理器（?mode=real/demo 切换 + 右上角浮动徽章）

### 修改文件

- `app/adapters/__init__.py` — agentpass ImportError 优雅降级
- `app/audit.py` — WebSocket emit_audit 增加 context 字段
- `app/delegation/engine.py` — external_agent capabilities 改为 read:external_api:weather + read:web
- `app/feishu/router.py` — asyncio.to_thread 修复阻塞 + demo capabilities 同步 + union_id 错误消息修复
- `app/orchestrator/orchestrator.py` — trust_score_before/after 传递 + _check_prompt_injection 接入 + _execute_external_agent 天气 API + _generate_report list/dict 兼容 + run_task 按 target_agent 签发 token + _parse_intent 天气路由 + _humanize_block prompt_injection 阶段
- `app/routers/delegation.py` — escalation-attack 改为 check weather ✓ → check finance ✗（独立 token）+ auto-revoke/trust-degrade escalation_actions 同步
- `app/routers/governance.py` — 重写 /events 端点从数据库读取标准事件 + /overview 从数据库计算平台统计
- `frontend/audit.js` — buildLogEntry 提取 platform/entry_point + Feishu 标签 + 详情展示平台信息
- `frontend/chain.js` — buildStepsFromEvents 从 agent_chain 构建链路 + selectScene 直接调用 demo API + buildEscalationSteps 解析 demo 数据 + buildAutoRevokeSteps 更新 capability
- `frontend/feishu.js` — refresh 从 governance events 获取真实数据 + 渲染事件流/安全链路/决策过程
- `frontend/trust.js` — demoDegrade/demoAutoRevoke 改回调用后端 demo API
- `frontend/risk.js` — simulateAttack 改回调用后端 demo API
- `frontend/index.html` — 引入 mode.js
- `frontend/chain.html` — 引入 mode.js
- `frontend/audit.html` — 引入 mode.js
- `frontend/trust.html` — 引入 mode.js
- `frontend/risk.html` — 引入 mode.js
- `frontend/feishu.html` — 引入 mode.js
- `frontend/governance.html` — 引入 mode.js
- `frontend/gateway.html` — 引入 mode.js
- `frontend/v22.html` — 引入 mode.js

## \[2.4.0] — 2026-04-26

### 🛡️ Feishu Security Gateway（飞书安全网关）

- **IAMTransport 拦截器** — 继承 `httpx.AsyncBaseTransport`，所有飞书 API 请求必须经过 IAM 授权
  - 请求路径自动映射为 IAM action（`mapRequestToAction`）
  - 认证路径（`/auth/v3/tenant_access_token/internal`）自动豁免
  - 允许时注入 Header：`X-Agent-ID` / `X-Trust-Score` / `X-Risk-Score`
  - 阻断时返回 403 + 详细原因（越权/低信任/撤销/动态策略违反）
  - **Fail-closed**：IAM 不可达时默认拒绝所有请求
- **mapRequestToAction(path, method)** — HTTP 路径 → IAM action 映射
  - `/im/v1/messages` → `write:feishu_message`
  - `/docx/v1/documents` → `write:doc`
  - `/calendar/v4/calendars` → `write:calendar`
  - `/bitable/v1/apps` → `read:bitable`
  - `/sheets/v3/spreadsheets` → `write:sheet`
  - `/drive/v1/files` → `read:drive`
  - `/wiki/v2/spaces` → `read:wiki`
  - `/contact/v3/users` → `read:contact`
  - 未匹配路径自动按 `method:resource` 生成
- **callIAMCheck(agent_id, action)** — 调用本地 IAM 校验
  - 先通过 `/api/delegate/issue-root` 签发 token
  - 再通过 `/api/delegate/check` 校验权限
  - 自动推断 `blocked_at`（capability_check / token_revocation / trust_check / replay_check / iam_check）
- **logAudit()** — 网关审计记录
  - 记录 agent / action / decision / latency / trust_score / risk_score / path / method
  - 内存存储，最多 500 条
- **Gateway 管理页面** — `/gateway`
  - 实时统计（请求数/允许/拒绝/自动撤销/平均延迟/拒绝率）
  - IAM Check 手动测试（支持 agent+action 或 path+method）
  - Path → Action 映射表
  - 攻击演示（Escalation / Bypass Attempt）
  - 审计日志流（自动刷新）
- **FeishuClient 改造** — 集成 IAM Gateway
  - Mock 模式下也经过 IAM 校验
  - 支持 `IAM_GATEWAY_ENABLED` 环境变量控制开关
  - 新增 `set_feishu_agent()` 切换 Agent 身份

### 🧠 Explainable IAM 增强（全页面覆盖）

- **Trust Dashboard** — Agent 详情面板 + 演示结果添加 Explain 按钮
- **Risk Dashboard** — 最高风险 Agent 列表 + 实时攻击流添加 Explain 按钮
- **Audit Center** — 表格行直接显示 🧠 小按钮（无需展开）+ 执行结果面板添加大号按钮
- **Chain Viewer** — 步骤详情使用 `makeBtn` + 大号样式
- **index.html 飞书交互区** — 消息气泡添加 Explain 按钮
- **新增样式** — `iam-explain-btn-sm`（紧凑型）/ `iam-explain-btn-lg`（醒目型）

### 🎨 前端升级

#### Gateway 管理页面（`/gateway`）

- **Hero 区域** — 渐变标题 + 安全网关流程图（Feishu CLI → IAM Gateway → Allow/Deny）
- **6 格统计面板** — Total / Allowed / Denied / Auto-Revoked / Avg Latency / Deny Rate，实时刷新
- **IAM Check 面板** — 支持 agent+action 直接检查，也支持 path+method 映射后检查
- **Path → Action 映射表** — 8 条预定义映射可视化展示
- **攻击演示** — Escalation Attack（4 级越权）/ Bypass Attempt（5 条路径绕过），结果逐行展示
- **审计日志表格** — Time / Agent / Action / Decision / Reason / Latency / Path，5 秒自动刷新
- **导航栏** — 紫色渐变 Logo + SECURITY GATEWAY 徽章 + 快速跳转链接

#### Explainable IAM 模态框增强

- **按钮动画** — 脉冲呼吸动画（`@keyframes iam-btn-pulse`），吸引注意力
- **三种尺寸** — `iam-explain-btn`（标准）/ `iam-explain-btn-sm`（紧凑，3px 8px）/ `iam-explain-btn-lg`（醒目，8px 20px）
- **统一生成函数** — `IAM_EXPLAIN.makeBtn(text, data, cls)` 替代各页面手写按钮 HTML
- **模态框视觉升级** — 渐变边框 + 毛玻璃遮罩 + 步骤逐条淡入动画 + 决策颜色编码（绿=Allow / 红=Deny）
- **7 页面全覆盖** — index / feishu / governance / audit / chain / trust / risk

#### 飞书交互区增强

- **消息气泡内嵌 Explain** — 每条安全消息直接显示 🧠 Explain Decision 按钮
- **feishu-section.js** — 首页飞书交互区同步支持 Explain
- **feishu.js** — 消息标签旁直接显示 Explain，无需展开详情

#### 审计中心增强

- **表格行内 Explain** — 每行审计记录直接显示 🧠 小按钮，无需展开
- **执行结果面板 Explain** — 操作执行后大号 Explain 按钮
- **展开详情 Explain** — 使用 `makeBtn` 统一样式

#### 信任/风险面板增强

- **Trust Dashboard** — Agent 详情面板大号 Explain + 降级演示 Explain + 自动撤销演示 Explain
- **Risk Dashboard** — 最高风险 Agent 列表 Explain + 实时攻击流紧凑型 Explain（🧠 图标）

#### 版本号全量更新

- 所有页面标题 `v2.3` → `v2.4`（governance / feishu / chain / audit / risk / trust / index / v22）
- `main.py` 4 处版本号更新
- `platforms.py` API 标签更新
- `v22.js` 注释更新

### 🌐 跨平台统一治理（延续 v2.3）

- **Platform Adapter** — 飞书/Web/API 多入口请求归一化
- **Orchestrator 升级** — 支持平台元数据注入 Token
- **动态策略增强** — 平台风险权重 + 企业数据访问规则
- **跨平台审计日志** — 平台/风险/自动撤销字段扩展
- **统一治理控制台** — `/governance` 页面，平台流量可视化 + 实时事件流 + Agent 控制

### 新增文件

- `app/feishu/iam_gateway.py` — IAM 安全网关核心模块
- `app/routers/gateway.py` — Gateway API 端点
- `frontend/gateway.html` — Gateway 管理页面

### 修改文件

- `app/feishu/client.py` — 集成 IAM Gateway Transport
- `app/feishu/__init__.py` — 导出 Gateway 模块
- `main.py` — 注册 Gateway 路由 + 页面 + 版本号 v2.4
- `frontend/iam-explain.js` — 新增 `iam-explain-btn-sm` 样式
- `frontend/trust.js` — Agent 详情 + 演示结果添加 Explain
- `frontend/risk.js` — 威胁列表 + 攻击流添加 Explain
- `frontend/audit.js` — 表格行 + 执行结果添加 Explain
- `frontend/chain.js` — 步骤详情使用 makeBtn
- `frontend/feishu-section.js` — 消息气泡添加 Explain
- `frontend/index.html` — 版本号 + iam-explain.js 引用
- `frontend/governance.html` — 版本号更新
- `frontend/feishu.html` — 版本号更新
- `frontend/chain.html` — 版本号更新
- `frontend/audit.html` — 版本号更新
- `frontend/risk.html` — 版本号更新
- `frontend/trust.html` — 版本号更新
- `frontend/v22.html` — 版本号更新
- `frontend/v22.js` — 版本号更新
- `app/routers/platforms.py` — API 标签版本号更新

## \[2.3.1] — 2026-04-24

### 产品定位升级

- **品牌重塑** — "Agent IAM" → **"AgentPass — AI Agent 安全治理平台"**
- **四大治理原则** — 主页新增 4 个治理原则卡片，每个可点击跳转对应页面：
  - 🛡 IAM 校验：每个请求都经授权检查
  - 🔍 决策解释：每个决策都有原因追溯
  - 📋 审计记录：每个行为都有不可篡改日志
  - 🏆 动态信任：每个 Agent 都有实时信任分

### Audit Center（审计中心）— 完全重写

- **8 列专业审计表格** — Timestamp / Agent / Action / Result / Reason / Risk / Trust Δ / 展开
- **⚡ 执行操作按钮** — 选择 Agent → 输入 Action → 自动签发 Token → 调用 /delegate/check → 展示决策过程
- **决策过程可视化** — Token 签发 → 信任评分 → Capability 匹配 → 最终决策（每步 ✓/✗ 标记）
- **日志实时追加（append 模式）** — 新日志插入表格顶部，带金色闪烁动画
- **筛选功能** — 按 Agent ID / Result (ALLOW/DENY) / Action 筛选
- **点击展开详情** — Token / JTI / Chain / Capabilities / Revoke Status / Auto-Revoke 详情 / Policy Trace
- **Trust Score 变化动画** — `0.80 → 0.70 (▼ -0.10)` 红色闪烁
- **risk_score > 0.7 → 红色加粗**
- **实时事件流** — WebSocket 连接状态指示灯（绿色脉冲/红色断开）
- **哈希链完整性验证** — 调用 /audit/integrity 端点
- **审计日志导出** — JSON/CSV 格式
- **新增后端 API（无需认证）**：
  - `GET /api/delegate/audit/logs` — 审计日志查询
  - `GET /api/delegate/audit/export` — 日志导出
  - `GET /api/delegate/audit/integrity` — 哈希链完整性验证

### Chain Viewer（调用链可视化）— 完全重写

- **SVG 调用链路图** — 纵向自动布局，节点显示 agent name / capability / trust_score / status
- **节点状态** — 👤 ISSUER / 🟢 ACTIVE / 🔴 DENIED / 🔥 AUTO-REVOKED
- **边标签** — action + ALLOW/DENY，Allow=绿色实线，Deny=红色虚线
- **点击节点 → 右侧详情面板** — Agent / Action / Capability / Trust Score / Status / Position / Risk Score / Reason
- **4 个 Demo 按钮**：
  - ✅ 正常流程 — 签发 Token → 查询链路 → 展示完整委托链
  - ⚠️ 越权攻击 — external_agent 越权访问 finance 数据
  - 🔄 重放攻击 — 同一 Token 重复使用被拒绝
  - 🔥 Auto Revoke — 信任分持续下降触发自动撤销
- **步骤日志 + 安全洞察** — 每个场景展示完整决策步骤和 Insight Box
- **Auto Revoke 特效** — 🔥 火焰闪烁动画 + 攻击横幅 Banner + 节点红色脉冲光晕

### Trust Dashboard（信任评分系统）— 完全重写

- **排行榜形式展示** — 按信任分从高到低排序
- **四级状态系统** — 🟢 SAFE (>0.7) / 🟡 WARNING (0.5~0.7) / 🔴 DANGER (<0.5) / 🔥 AUTO-REVOKED (<0.3)
- **进度条显示 Trust** — 渐变色填充 + 底部标注关键阈值
- **分数变化动画** — ▼ -0.10 红色闪烁 / ▲ +0.01 绿色上升
- **Auto Revoke 特效** — 整行红色脉冲光晕 + Badge 闪烁 + 进度条闪烁
- **顶部统计面板** — Agents / Safe / Warning / Danger / Revoked 计数
- **阈值显示栏** — Trust Threshold + Auto-Revoke Threshold
- **场景演示** — 📉 信任降级演示 + 🔥 自动撤销演示
- **已撤销 Agent 列表** — 显示被自动撤销的 Agent 及原因

### Risk Dashboard（风险分析）— 完全重写

- **6 个数字卡片** — Total Requests / Denied / Deny Rate / High Risk / Auto-Revoke / Revoked Agents
- **点击卡片 → 跳转 Audit Center 过滤结果**
- **决策分布环形图** — Canvas 绘制，Allow/Deny/Auto-Revoke 三段
- **Agent 风险等级分布条形图** — 🔥 Critical / 🔴 High / 🟡 Medium / 🟢 Low
- **风险事件时间线** — 最近 50 条事件，高风险标红
- **已撤销/已封禁列表** — AUTO-REVOKED Agent + Revoked Token
- **并发 API 加载** — 同时请求 4 个 API 优化性能

### 前端架构优化

- **品牌统一** — 所有页面标题从 "Agent IAM" 更新为 "AgentPass"
- **导航栏更新** — More 下拉菜单中 4 个子页面链接加粗高亮
- **内联暗色变量兜底** — 每个子页面内联 `html[data-theme="dark"]` CSS 变量，防止外部 CSS 加载失败导致白屏
- **CSS 变量 fallback** — `var(--bg-alt, #1c1c1e)` 双重保险

### Mac 适配

- **字体栈跨平台兼容** — 添加 `"Microsoft YaHei"`, `"Segoe UI"`, `"Noto Sans SC"`, `"Fira Code"`, `"Courier New"` fallback
- **清理未使用的 `import subprocess`** — main.py 中移除
- **确认兼容项** — 路径处理全部使用 pathlib.Path、数据库 URL 使用 .as_posix()、webbrowser.open() 跨平台兼容、无 Windows 特有命令

### 新增文件

- `frontend/risk.html` — 风险分析页面
- `frontend/risk.js` — 风险分析逻辑

### 修改文件

- `frontend/audit.html` — 审计中心完全重写
- `frontend/audit.js` — 审计中心逻辑完全重写
- `frontend/chain.html` — 调用链可视化完全重写
- `frontend/chain.js` — 调用链逻辑完全重写
- `frontend/trust.html` — 信任评分完全重写
- `frontend/trust.js` — 信任评分逻辑完全重写
- `frontend/index.html` — 品牌重塑 + 四大治理原则卡片 + 导航更新
- `frontend/styles.css` — 字体栈跨平台兼容
- `app/routers/delegation.py` — 新增 /audit/logs、/audit/export、/audit/integrity 端点
- `main.py` — 移除未使用的 subprocess import

## \[2.3.0] — 2026-04-24

### 产品定位升级

- **AI Agent 安全治理控制台** — 从"管理后台"升级为"安全可观测平台"
- 核心理念：**这个系统不是 UI，而是 Agent 行为治理系统**

### 核心新增：风险驱动自动撤销机制（Auto-Revoke）

- **自动撤销引擎** — 当 Agent 信任分低于阈值（0.3）时自动撤销其所有 Token
- **信任分动态管理** — 每次授权检查根据结果动态调整信任分（allow +0.02, deny -0.15）
- **AUTO_REVOKED_AGENTS** — 记录被自动撤销的 Agent、原因、撤销时信任分、时间戳
- **API 端点**：
  - `GET /api/delegate/trust` — 查询所有 Agent 信任分
  - `POST /api/delegate/trust/reset` — 重置信任分
  - `GET /api/delegate/auto-revoke/list` — 查看自动撤销记录
  - `POST /api/delegate/auto-revoke/clear` — 清除自动撤销记录
  - `POST /api/delegate/demo/auto-revoke` — 自动撤销演示端点

### 核心新增：4 大安全可观测页面

1. **Audit Center（审计中心）** — `/audit`
   - 实时审计事件流（WebSocket `/ws/audit`）
   - 审计日志查询（支持 Agent/Decision/Action 过滤）
   - 哈希链完整性验证
   - 审计日志导出（JSON/CSV）
   - 事件统计面板（总事件/允许/拒绝/撤销/哈希链状态）

2. **Chain Viewer（调用链）** — `/chain`（已有，大幅增强）
   - 4 种攻击场景演示：正常流程、越权攻击、重放攻击、Token 撤销
   - SVG 链路可视化 + 步骤日志
   - Token 状态实时查询（introspect API）
   - 决策过程可视化（每步 ALLOW/DENY 展示）

3. **Trust Dashboard（信任评分）** — `/trust`
   - Agent 信任评分矩阵（动态卡片，颜色编码）
   - 实时授权检查面板（触发 `/delegate/check`，展示决策过程）
   - 自动撤销记录管理
   - 信任分重置功能

4. **Risk Dashboard（风险分析）** — `/risk`（开发中）
   - 风险概览统计
   - Agent 风险等级矩阵
   - 风险事件时间线
   - 撤销 Token 管理

### 前端架构重构

- **统一导航栏** — 5 页面互通导航（Console / Audit / Chain / Trust / Risk）
- **glassmorphism 风格统一** — 所有新页面保持一致的毛玻璃风格
- **独立 JS 模块** — 每个页面独立 JS 文件，可独立刷新
- **缓存破坏** — 所有 CSS/JS 引用添加版本号参数

### 新增文件

- `frontend/audit.html` — 审计中心页面
- `frontend/audit.js` — 审计中心逻辑
- `frontend/trust.html` — 信任评分页面
- `frontend/trust.js` — 信任评分逻辑

### 修改文件

- `app/delegation/engine.py` — 新增 AUTO_REVOKE_THRESHOLD、AUTO_REVOKED_AGENTS、auto_revoke_agent()、check() 中集成信任分调整和自动撤销
- `app/routers/delegation.py` — 新增 trust/auto-revoke/demo 端点，修复 demo 端点状态清理和 capability scope 问题
- `frontend/chain.html` — 增强导航、缓存破坏
- `frontend/chain.js` — 重写 demo 函数（正确调用后端 demo 端点）、新增 clearDemoLogs()、renderChain() 支持 tokenForStatus 参数
- `frontend/index.html` — 新增实时安全事件流面板、模拟攻击按钮
- `frontend/app.js` — 新增事件流和模拟攻击功能
- `main.py` — 新增 /audit、/trust、/risk 页面路由

### Bug 修复

- **Token Status 一直 Loading** — renderChain() 不再无条件显示 "Loading..."，改为根据是否有 token 参数决定；demo 函数正确传递 token 给 loadTokenStatus()
- **Demo 端点状态残留** — 所有 demo 端点（chain-visualization/normal-flow/escalation-attack/replay-attack）现在都调用 clear_revoked() 和 reset_trust_scores()
- **Capability scope 不匹配** — demo 中的 action 从 `read:feishu_table` 改为 `read:feishu_table:finance`，与 data_agent 的 capabilities 匹配
- **Revoke demo Step 6 缺失** — agent 级撤销后 delegate 失败时也记录步骤
- **重放攻击 demo 首次 check 失败** — 修复 revoked 状态未清理导致的级联失败
- **越权攻击 demo 逻辑丢失** — 重写为调用后端 /demo/escalation-attack 端点
- **浏览器缓存** — 所有静态资源添加版本号参数

### 待完成

- ~~Risk Dashboard 页面（risk.html + risk.js）~~ → 已在 2.3.1 完成
- ~~index.html 首页导航入口更新~~ → 已在 2.3.1 完成
- ~~main.py 新页面路由注册~~ → 已在 2.3.0 完成

## \[2.2.0] — 2026-04-22

### 产品定位

- **Enterprise AI Governance Platform**
- **Unified Governance for China & US AI Agent Platforms**

### 核心目标

在保留现有系统全部功能与前端风格的前提下，新增：

- 强 DLP（数据泄露防护）
- 多平台 AI Agent 接入
- 跨平台统一审批
- 跨平台统一风控
- 跨平台统一审计
- 跨平台统一成本治理
- 独立 v2.2 新页面（避免旧首页过长）

### 总体要求

1. 保持现有项目结构，增量开发，不重构旧功能
2. 保持现有 glassmorphism UI 风格
3. 保持现有登录、治理中心、Prompt Defense 等功能可用
4. 所有新功能支持 mock 数据演示
5. 所有 API 使用 FastAPI
6. 前端与后端版本升级为 v2.2
7. 代码结构清晰，可继续扩展
8. 不删除旧页面，新增独立页面承载 v2.2
9. 完成后自动修复报错直到可运行

### 页面架构

- **保留原页面**：/（继续保留旧功能）
- **新增独立页面**：/v22（承载 AgentPass v2.2 全部新能力）

### 导航要求

- 在旧首页新增入口按钮：Enter AgentPass v2.2（跳转至 /v22）
- 在 /v22 页面顶部新增返回按钮：← Back to Classic Console（返回至 /）

### 新页面文件

- frontend/v22.html
- frontend/v22.js
- frontend/v22.css（如需要）

### Connectors 多平台接入架构

- **新增目录**：app/connectors/
- **connector 基类**：connect(), fetch\_events(), fetch\_cost(), fetch\_pending\_approvals(), health\_check()
- **接入平台**：
  - 【中国】Feishu、Qwen、DeepSeek、Doubao、ERNIE Bot、Kimi
  - 【美国】ChatGPT、Grok、Gemini、Claude
- **支持 mock 模式**：返回模拟的调用事件、风险事件、成本数据、待审批数据

### 统一事件模型

- **UnifiedEvent 结构**：id, timestamp, platform, region, user, team, action, resource, prompt, output, risk, risk\_level, approval\_required, approval\_status, cost, token\_usage, blocked, reason

### 强 DLP 引擎

- **新增文件**：app/security/dlp.py
- **检测能力**：
  1. 敏感信息识别（手机号、邮箱、身份证号、银行卡号、API Key 等）
  2. 企业敏感数据识别（confidential、internal only、salary 等关键词）
  3. 语义泄露识别（导出全部客户数据、给我数据库内容等意图）
  4. 输出阻断（命中高风险内容）
  5. 脱敏输出（如 13812345678 → 138\*\*\*\*5678）

### 统一审批引擎

- **新增文件**：app/approval/engine.py
- **规则**：risk > 0.75 → 审批；action == export\_data → 审批；cost > 50 → 财务审批等
- **状态**：pending, approved, rejected, expired
- **模拟审批人**：<manager@corp.com>, <finance@corp.com>, <admin@corp.com>

### 统一风控引擎

- **新增文件**：app/risk/unified.py
- **综合评分**：Prompt Injection 风险、DLP 风险、用户异常行为、高频调用、多平台连续攻击、深夜访问、高成本异常

### 统一成本治理

- **新增文件**：app/cost/engine.py
- **统计**：request\_count, token\_usage, usd\_cost, cny\_cost
- **维度**：今日、本周、本月、平台排行、用户排行、团队排行
- **预算规则**：monthly\_cost > budget → alert = true

### 后端 API

- **页面路由**：GET /v22
- **平台管理**：GET /api/v2/platforms, GET /api/v2/platforms/health, GET /api/v2/platforms/events
- **Dashboard**：GET /api/v2/dashboard/summary, GET /api/v2/dashboard/trends
- **审批中心**：GET /api/v2/approvals/pending, POST /api/v2/approvals/{id}/approve, POST /api/v2/approvals/{id}/reject
- **风险中心**：GET /api/v2/risk/events, GET /api/v2/risk/top-users, GET /api/v2/risk/top-platforms
- **成本中心**：GET /api/v2/cost/summary, GET /api/v2/cost/platforms, GET /api/v2/cost/users
- **DLP**：POST /api/v2/dlp/check

### 前端页面内容（/v22）

- **顶部 Hero**：AgentPass v2.2, Unified Governance for China & US AI Platforms
- **首屏 KPI 卡片**：已接入平台数、今日事件数、待审批数、高风险事件数、今日成本
- **Tab 导航**：Overview, Platforms, Approvals, Risk, Cost, DLP, Demo

### 各模块页面

1. **Platform Hub**：平台卡片、CN/US 标签、在线状态、今日调用量、风险值、成本
2. **Unified Dashboard**：总调用量、平台分布、风险趋势、审批趋势、成本趋势
3. **Approval Center**：待审批列表、来源平台、用户、操作、风险分、Approve/Reject 按钮
4. **Risk Center**：高风险事件、被拦截操作、Top Risk Users、Top Risk Platforms
5. **Cost Center**：平台成本排行、用户成本排行、趋势图、预算预警
6. **DLP Center**：输入框测试内容、风险评分、命中规则、是否阻断、脱敏结果、泄露类型标签
7. **Demo Center**：一键生成演示数据（多平台事件、高风险攻击、审批记录、成本数据、DLP 拦截案例）

### 性能优化

1. 原首页不加载 v2.2 JS
2. v22 页面按需请求 API
3. 图表延迟渲染
4. Demo 数据异步生成
5. 组件化代码结构

### 版本信息

- **系统版本**：AgentPass v2.2
- **页脚**：Unified Governance for China & US AI Platforms

### 交付输出

1. 新增文件列表
2. 修改文件列表
3. 新增 API 清单
4. 新页面访问地址
5. 新功能说明
6. Demo 入口位置
7. 已修复报错说明

### 编码风格

1. 保持现有代码风格
2. 专业商业化产品级 UI
3. 不删旧功能
4. 模块化
5. 可继续扩展
6. 可直接运行

## \[2.1.1] — 2026-04-20

### Added

- **三层融合引擎** — Prompt Defense 完全重构为三层架构：规则层(Rule Engine) + 语义层(Semantic Engine, TF-IDF+余弦相似度) + 行为层(Behavioral Engine, 滑动窗口+渐进式注入检测)
- **9种攻击类型** — 新增 `goal_hijacking`(目标劫持) 和 `prompt_leaking`(提示词套取) 两种攻击类型
- **Token走私专项检测** — Unicode零宽字符、混淆编码、Base64混入自然语言检测，命中强制 final\_score >= 0.85
- **语义意图识别** — TF-IDF向量与预设攻击意图库对比，识别 data\_theft/role\_override/jailbreak/indirect/goal\_hijack/prompt\_leak 六种意图
- **渐进式注入检测** — 基于滑动窗口(最近10条)计算风险趋势，检测突然拔高和连续高风险
- **DELETE /api/prompt-defense/history** — 清空指定 user\_id 的对话历史端点
- **前端完整重构** — 9个攻击pill按钮、三层进度条可视化、综合评分仪表盘、SVG风险趋势折线图、渐进式注入历史区、触发规则卡片列表

### Fixed

- **治理中心显示Bug** — 移除所有治理中心模块的 admin 会话硬限制，改为尝试请求+403友好提示，治理中心现在任何角色均可访问
- **规则评分逻辑** — 修复 `max(weighted_scores)` 为加权平均，修复 `matched_count/total_rules` 为 `0.4 + matched_count * 0.2`
- **过于宽泛的中文规则** — 删除单独的 `r"敏感数据"` 等，改为 `r"导出.*敏感数据"` 等带动作前缀的规则
- **外部模块依赖** — 完全移除对 `.ai_detector`、`.nlp_detector`、`.user_profile`、`.context_analyzer` 的依赖，所有能力内聚到单文件

### Improved

- **Prompt Defense 自包含** — 零外部依赖，import 即可用
- **正则性能优化** — 所有正则在 `__init__` 时预编译并缓存，使用 `IGNORECASE | UNICODE` 标志
- **TF-IDF预计算** — 意图库向量在初始化时预计算，analyze 时只需计算输入向量

## \[2.1.0] — 2026-04-19

### Added

- **Frontend: v1.3 Visual Migration** — Hero orbs, glassmorphism cards, reveal animations, prefers-color-scheme auto-detection
- **Frontend: Chart Color Fix** — All chart components (delegation graph, risk dashboard, threat map) now use CSS variables, correctly updating on theme switch
- **Frontend: Hamburger Menu** — Responsive mobile navigation
- **Human-in-the-Loop Approval** — `approval_requests` table, WebSocket-based approval flow, Feishu card integration, auto-timeout
- **Agent Reputation System** — ReputationEngine with KL-divergence consistency scoring, suspicious pattern detection, hourly recalculation
- **Semantic Role Drift Detection** — TF-IDF based drift detector, cosine distance tracking, injection turn identification
- **Zero-Trust Context Isolation** — AES-256 session encryption, delegation field filtering, cross-agent information leak detection
- **Prompt Defense Enhancement** — 3 new attack types (jailbreak\_roleplay, indirect\_injection, token\_smuggling), weighted scoring, multi-turn context window
- **Open Source Infrastructure** — Dockerfile, docker-compose.yml, CI/CD (GitHub Actions), CONTRIBUTING.md, CHANGELOG.md, LICENSE, .env.example, .gitignore, ADR docs
- **Permission Suggestions API** — `/api/insights/permission-suggestions` — least-privilege analysis with unused resource detection
- **Access Heatmap API** — `/api/insights/access-heatmap` — daily access pattern visualization
- **Heartbeat Shutdown** — Server auto-exits 15s after browser disconnects (replaces connection-count check)
- **Prompt Defense: 4 Semantic Detection Layers** — semantic probing, encoding obfuscation, mixed-language attack, context manipulation

### Fixed

- Policy engine time restriction bug (affected 3 demo scenarios)
- renderDashboard undefined variable in app.js
- Hardcoded chart colors not updating on theme switch
- **Page blank bug** — `initReveal()` now checks viewport immediately, lowers threshold to 5%, adds 300ms fallback, `@media (prefers-reduced-motion)` support, `init()` try/catch with error banner
- **Server not closing after browser shutdown** — replaced `ws_manager.count` check with heartbeat timeout (`_last_ping_at` + 15s idle → SIGINT), added 8s WebSocket ping from frontend, removed `reload=True` and `watch_browser` thread
- **Frontend-backend field alignment** — delegate `to_agent_id`→`target_agent_id`, integration `integration_name`→`service_name`, policyTrace `step.label`→`step.name`, verifyIntegrity `data.verified`→`data.valid`, delegationGraph `n.id`→`n.agent_id`, promptDefense `r.rule`→`r.injection_type`, OpenClaw stats aggregation
- **Governance center not loading** — added `isAdminSession()` check, auto-load governance data after admin login, friendly prompts for non-admin users
- **approval.py** — replaced `ws_manager.active_connections` with `ws_manager.broadcast()`
- **`/api/overview`** — added `policies` and `stats` fields
- **Ruff check** — all 11 lint errors fixed (unused variables, bare except, E402 imports)

## \[2.0.0] — 2026-04-18

### Added

- Prompt Injection Defense module (4 attack types)
- OpenClaw Integration API
- Security Dashboard frontend
- Prompt Defense demo panel
- SDK v0.2.1 published to PyPI
- Ngrok public URL support
- Hash chain audit integrity verification
- WebSocket real-time audit push
- SQLAlchemy ORM + connection pool optimization

### Fixed

- Audit log function call errors
- Frontend filter field errors
- Risk score storage/read field errors

## \[1.3.0] — 2026-04-17

### Fixed

- Scenario execution timeline status bug
- API key derivation logic
- Token constraints IP binding
- Introspect token response handling
- Suspend/reactivate button permissions
- Resource sensitivity check in access\_resource endpoint
- List documents endpoint field names
- Introspect endpoint revoked token handling

### Changed

- Default theme to light mode
- Auto-detect system theme preference
- Frontend version alignment with backend

## \[1.2.0] — 2026-04-16

### Added

- Animation transitions and micro-interactions
- Audit log export (JSON/CSV)
- Sidebar keyboard shortcuts
- One-click login all Demo Agents
- Pydantic settings management
- FastAPI dependency injection
- SQLAlchemy ORM integration
- WebSocket audit push
- Token IP binding and usage limits
- Auto risk-control suspension

## \[1.1.0] — 2026-04-15

### Added

- Agent registration and authentication
- JWT token management
- RBAC permission control
- ABAC attribute policies
- Audit logging
- Real-time WebSocket push
- Frontend management console

## \[1.0.0] — 2026-04-14

### Added

- Project initialization
- Base directory structure
- Core feature prototype
- Database schema design
- Frontend page framework

