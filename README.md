# Agent 身份与权限系统 v2.2

> Agent Identity & Authorization System — A2A Protocol, Heterogeneous Agent Orchestration, Delegated Authorization, Real-time Audit

## Quick Start

### 方式一：启动脚本（推荐）

```bash
# Windows
start.bat

# Linux/macOS
chmod +x start.sh && ./start.sh
```

### 方式二：手动启动

```bash
pip install -r requirements.txt
python main.py
```

服务器启动后访问 `http://127.0.0.1:8000`，浏览器自动打开。

### 环境变量（可选）

复制 `.env.example` 为 `.env`，填入飞书和LLM API配置：

```env
# 飞书应用凭证
FEISHU_APP_ID=cli_xxxxx
FEISHU_APP_SECRET=xxxxx

# LLM API Keys（至少配置一个）
OPENAI_API_KEY=sk-xxxxx
DEEPSEEK_API_KEY=sk-xxxxx
QWEN_API_KEY=sk-xxxxx
```

不配置 `.env` 也可运行，系统会自动降级为 Mock 模式。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户 / 飞书客户端                              │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │ Web / Feishu                     │ Feishu Webhook
               ▼                                  ▼
┌──────────────────────┐  ┌──────────────────────────────────────────┐
│   Platform Adapter   │  │         Feishu IAM Gateway               │
│  (Web/Feishu/API)    │  │  IAMTransport → mapRequestToAction       │
│  风险权重计算          │  │  → callIAMCheck → Decision               │
└──────────┬───────────┘  └──────────────┬───────────────────────────┘
           │                             │
           ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Orchestrator（编排器）                            │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │意图解析   │  │六层验证    │  │Agent路由   │  │异构LLM调用        │   │
│  │_parse_   │  │six_layer │  │doc_agent  │  │OpenAI/DeepSeek/  │   │
│  │intent    │  │_verify   │  │data_agent │  │Qwen/Gemini/...   │   │
│  └─────────┘  └──────────┘  │ext_agent  │  └──────────────────┘   │
│                              └───────────┘                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
           ┌───────────────────┼───────────────────────┐
           ▼                   ▼                       ▼
┌─────────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│ DelegationEngine│ │  Audit Logger    │ │  Prompt Defense      │
│ Token签发/验证   │ │  SHA-256哈希链    │ │  三层融合引擎          │
│ 能力交集计算     │ │  委派链记录       │ │  9种攻击类型检测       │
│ 信任评分管理     │ │  WebSocket推送    │ │  LLM语义分析降级      │
│ 自动封禁        │ │  完整性验证       │ │  社工/注入/重放检测    │
└─────────────────┘ └──────────────────┘ └──────────────────────┘
```

### 异构 Agent 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Registry                            │
├──────────────┬──────────────────┬───────────────────────────┤
│  📄 doc_agent │  📊 data_agent   │  🌐 external_agent        │
│  协调者       │  企业数据查询      │  外部信息检索              │
├──────────────┼──────────────────┼───────────────────────────┤
│ Model:       │ Model:           │ Model:                    │
│ GPT-4o-mini  │ DeepSeek Chat    │ Qwen Plus                 │
│ (US)         │ (CN)             │ (CN)                      │
├──────────────┼──────────────────┼───────────────────────────┤
│ Engine:      │ Engine:          │ Engine:                   │
│ OpenAI API   │ DeepSeek API     │ DashScope API             │
├──────────────┼──────────────────┼───────────────────────────┤
│ Toolset:     │ Toolset:         │ Toolset:                  │
│ feishu_      │ feishu_data      │ web_search                │
│ document     │ (bitable/sheet)  │ (search/scrape)           │
├──────────────┼──────────────────┼───────────────────────────┤
│ Capabilities:│ Capabilities:    │ Capabilities:             │
│ write:doc    │ read:feishu_     │ read:web                  │
│ delegate:*   │ table:*          │ (无飞书内部权限)            │
│ read:*       │ read:calendar    │                           │
└──────────────┴──────────────────┴───────────────────────────┘
```

### 三Agent协作委派链

```
👤 user
  ↓ 委派
📄 doc_agent（协调者 · OpenAI GPT-4o-mini）
  ├─ ↓ 委派 → 📊 data_agent（内部数据 · DeepSeek Chat）
  ├─ ↓ 委派 → 🌐 external_agent（外部搜索 · Qwen Plus）
  └─ ↓ 汇总 → 📄 doc_agent（生成报告 → 飞书文档）
```

---

## Access Token 字段说明

系统使用 JWT (JSON Web Token) 作为 Access Token，遵循 OAuth 2.0 + A2A 协议扩展。

### Token 结构

```
Header:  { "alg": "HS256", "typ": "JWT" }
Payload: { ...下方字段... }
```

### Payload 字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sub` | string | ✅ | Token 主体，格式为 `agent:{agent_id}` |
| `iss` | string | ✅ | 签发者，固定为 `agent-iam-system` |
| `aud` | string | ✅ | 受众，固定为 `agent-delegation` |
| `exp` | integer | ✅ | 过期时间（Unix timestamp），默认签发后 1 小时 |
| `iat` | integer | ✅ | 签发时间（Unix timestamp） |
| `jti` | string | ✅ | JWT ID，唯一标识，用于防重放检测 |
| `delegated_user` | string | ✅ | 被委托用户 ID，格式为 `user:{user_id}` |
| `capabilities` | string[] | ✅ | 能力声明列表，如 `["read:doc", "write:doc:public"]` |
| `chain_depth` | integer | ✅ | 委派链深度，根 Token 为 0，每委派一次 +1，最大 5 |
| `parent_jti` | string | ❌ | 父 Token 的 JTI，根 Token 无此字段 |
| `delegated_by` | string | ❌ | 委派者 Agent ID，根 Token 无此字段 |
| `scope` | string | ❌ | 有效权限范围，为 `用户权限 ∩ Agent能力` 的交集 |
| `platform` | string | ❌ | 来源平台，`web` / `feishu` |
| `ip_address` | string | ❌ | 绑定 IP 地址（可选约束） |
| `max_uses` | integer | ❌ | 最大使用次数，默认无限制 |
| `task_id` | string | ❌ | 关联任务 ID，用于一次性 Token |

### 能力声明格式 (Capability Statement)

```
action:resource[:scope]

action:   read | write | delete | delegate | export
resource: doc | feishu_table | feishu_table:finance | feishu_table:hr |
          bitable | sheet | calendar | task | contact | mail |
          vc | wiki | drive | approval | attendance | okr | web
scope:    public | internal | confidential (可选)
```

### 委派授权 (Delegated Authorization)

当 Agent A 委派给 Agent B 时：

1. 系统计算 **有效权限 = 用户权限 ∩ Agent A 能力 ∩ Agent B 能力**
2. 签发新 Token，`chain_depth` +1，`parent_jti` 指向父 Token
3. 若 `chain_depth > 5`，拒绝委派（防止无限委派链）
4. 若有效权限为空集，拒绝委派

### Token 生命周期

```
Issue → Use → Check → [Allow/Deny] → [Expire/Revoke/Auto-Revoke]
                ↓
           Replay Detection (jti 去重)
                ↓
           Trust Score Update (deny 时降低)
                ↓
           Auto-Revoke (trust < 0.3 时自动封禁)
```

---

## 审计日志字段说明

### 日志记录结构

每条审计日志记录以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | integer | 自增主键 |
| `agent_id` | string | 执行操作的 Agent ID |
| `action` | string | 操作类型，如 `read:feishu_table:finance` |
| `resource` | string | 资源标识，如 `feishu:web` / `web` |
| `decision` | string | 授权决策：`allow` / `deny` |
| `reason` | string | 决策原因描述 |
| `ip_address` | string | 请求来源 IP |
| `token_id` | string | 关联的 Token JTI |
| `created_at` | string | 创建时间（ISO 8601） |
| `context_json` | JSON | 扩展上下文（见下方） |

### context_json 扩展字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | string | 发起请求的用户 ID |
| `message` | string | 用户原始消息（截断至100字符） |
| `trust_score` | float | 当前信任分 |
| `platform` | string | 来源平台 |
| `delegation_chain` | string[] | 完整委派链，如 `["user:xxx", "doc_agent", "data_agent"]` |
| `delegation_chain_display` | string | 可读委派链，如 `"user:xxx → doc_agent → data_agent"` |
| `delegation_steps` | array | 委派步骤详情，每步包含 from/to/action/type |
| `_chain_hash` | string | SHA-256 哈希链值，用于完整性验证 |
| `risk_score` | float | 风险评分 |
| `trust_score_before` | float | 操作前信任分 |
| `trust_score_after` | float | 操作后信任分 |
| `auto_revoked` | boolean | 是否触发自动封禁 |
| `blocked_at` | string | 被阻断的安全层 |
| `prompt_risk_score` | float | Prompt 风险评分 |
| `attack_types` | string[] | 检测到的攻击类型 |
| `attack_intent` | string | 攻击意图 |
| `severity` | string | 严重度 |
| `six_layer` | object | 六层验证详情 |
| `full_chain` | string[] | 完整协作链（三Agent协作时） |
| `doc_url` | string | 生成的飞书文档 URL |

### 哈希链完整性验证

审计日志使用 SHA-256 哈希链保证不可篡改：

```
genesis → hash(log_1) → hash(log_2, prev_hash) → ... → hash(log_n, prev_hash)
```

- 每条日志的 `_chain_hash` = SHA-256(agent_id + action + resource + decision + reason + created_at + prev_hash)
- 第一条日志的 `prev_hash` = `"genesis"`
- 验证接口：`GET /api/delegate/audit/integrity`
- 重建接口：`POST /api/delegate/audit/rebuild-chain`

---

## A2A 协议文档

### 协议概述

本系统实现了 Agent-to-Agent (A2A) 认证与授权协议，基于 OAuth 2.0 扩展，支持：

- **Token 签发与验证**：为每个 Agent 建立唯一、可校验的身份
- **能力声明 (Capability Statement)**：细粒度操作权限定义
- **委托授权 (Delegated Authorization)**：用户权限 ∩ Agent能力 的有效权限交集
- **实时拦截**：Agent 调用另一个 Agent 时强制权限校验
- **审计可追溯**：完整上下文 + 哈希链完整性

### 协议流程

#### 1. 正常请求流程

```
User → Platform Adapter → Orchestrator
  → Intent Parse → Agent Route
  → Issue Root Token (doc_agent)
  → Six-Layer Verify → Prompt Defense
  → Execute Agent Task
  → [Optional] Delegate to sub-agent
  → Log Audit Event → Return Result
```

#### 2. 委派授权流程

```
doc_agent (with root token)
  → secure_agent_call(target=data_agent, action=read:feishu_table)
  → DelegationEngine.check():
      1. Verify parent token (signature, expiry, revocation)
      2. Check chain_depth < MAX_CHAIN_DEPTH (5)
      3. Compute effective capabilities = parent_caps ∩ target_caps
      4. Check action ∈ effective capabilities
      5. Evaluate dynamic policy (time, IP, platform risk)
      6. Check trust score ≥ threshold
      → ALLOW: Issue delegated token, execute
      → DENY: Record audit, update trust score
```

#### 3. 异常与边界流程

| 场景 | 处理方式 | HTTP 状态码 | 审计记录 |
|------|----------|------------|----------|
| Token 过期 | 拒绝请求 | 200 (decision=deny) | reason="token expired" |
| Token 已撤销 | 拒绝请求 | 200 (decision=deny) | reason="token revoked" |
| Token 重放 | 拒绝请求，信任分 -0.15 | 200 (decision=deny) | attack_type="replay" |
| 能力不匹配 | 拒绝请求，信任分 -0.10 | 200 (decision=deny) | reason="capability mismatch" |
| 委派链过深 (>5) | 拒绝委派 | 200 (decision=deny) | reason="chain depth exceeded" |
| 信任分低于阈值 (<0.3) | 自动封禁，撤销所有 Token | 200 (decision=deny) | auto_revoked=true |
| Prompt 注入检测 | 阻断请求 | 200 (decision=deny) | attack_type="prompt_injection" |
| 社工攻击检测 | 阻断请求 | 200 (decision=deny) | attack_type="social_engineering" |
| 越权访问 | 阻断请求，信任分 -0.20 | 200 (decision=deny) | attack_type="escalation" |
| IAM 不可达 | Fail-closed，拒绝所有请求 | 503 | reason="IAM unreachable" |
| 飞书 API 不可用 | 降级为 Mock 模式 | 200 | mock=true |
| LLM API 不可用 | 降级为规则匹配 | 200 | mode="degraded" |
| 并发审计写入 | 线程锁保证顺序 | - | hash chain 保证完整性 |
| 数据库写入失败 | 日志记录错误，不中断请求 | 200 | (可能丢失该条审计) |

#### 4. 动态授权策略

```
evaluate_dynamic_policy():
  1. 时间约束：工作时间 vs 非工作时间，敏感操作仅工作时间允许
  2. IP 约束：内网 IP 放行，外网 IP 加权
  3. 平台风险：web=1.0, feishu=1.2, api=0.8
  4. 信任分阈值：allow ≥ 0.5, degrade < 0.5, deny < 0.3
  5. 企业数据规则：finance/HR 数据需额外审批
```

#### 5. 信任分动态调整

| 事件 | 信任分变化 |
|------|-----------|
| 正常请求通过 | +0.01 |
| 能力不匹配被拒 | -0.10 |
| 越权访问被拒 | -0.20 |
| Token 重放被拒 | -0.15 |
| Prompt 注入被阻断 | -0.05 |
| 信任分 < 0.3 | 自动封禁（AUTO-REVOKE） |
| 信任分重置 | 恢复初始值 |

---

## 页面导航

| 路由 | 页面 | 说明 |
|------|------|------|
| `/` | 安全指挥中心 | 主面板 + 飞书交互 + 异构Agent架构展示 |
| `/feishu` | 飞书安全控制台 | 飞书消息交互 + 攻击演示 |
| `/gateway` | IAM 网关 | 飞书 API 安全网关控制台 |
| `/governance` | 统一治理 | 跨平台治理控制台 |
| `/audit` | 审计中心 | 9列表格 + 委派链 + 执行 + Explain |
| `/chain` | 链路查看器 | SVG 委派链可视化 + 攻击回放 |
| `/trust` | 信任面板 | 信任分排名 + 降权/封禁演示 |
| `/risk` | 威胁雷达 | 风险分析 + 决策分布图 |

## 核心 API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/delegate/issue-root` | POST | 签发根委派 Token |
| `/api/delegate/check` | POST | 校验委派权限 |
| `/api/delegate/trust` | GET | 获取所有 Agent 信任分 |
| `/api/delegate/trust/reset` | POST | 重置信任分 |
| `/api/delegate/agents` | GET | 获取 Agent 能力列表 |
| `/api/delegate/agents/heterogeneous` | GET | 获取异构 Agent 架构信息 |
| `/api/delegate/audit/logs` | GET | 查询审计日志 |
| `/api/delegate/audit/integrity` | GET | 验证哈希链完整性 |
| `/api/delegate/audit/rebuild-chain` | POST | 重建哈希链 |
| `/api/feishu/test` | POST | 飞书消息测试 |
| `/api/feishu/webhook` | POST | 飞书事件回调 |
| `/api/prompt-defense/analyze` | POST | Prompt 注入分析 |
| `/api/explain/result` | POST | IAM 决策解释 |

## 异构 Agent 信息

| Agent | 模型 | 推理引擎 | 工具集 | 区域 |
|-------|------|----------|--------|------|
| 📄 doc_agent | OpenAI GPT-4o-mini | OpenAI Chat Completions API | feishu_document | US |
| 📊 data_agent | DeepSeek Chat | DeepSeek Chat Completions API | feishu_data | CN |
| 🌐 external_agent | Qwen Plus | Qwen DashScope API | web_search | CN |

## License

MIT
