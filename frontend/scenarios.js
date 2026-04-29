// 场景执行模块 - 结构化测试框架
// 每个场景有明确的测试目标、预期结果和通过/失败判定
import { loginAgent, batchLogin, getAgentInfo, introspectToken, readDocument, writeDocument, executeTask, callIntegration, delegateTask, getAgents, resetDemo, authFetch } from '/static/api.js';
import state, { setActiveSession, upsertSession, addTimelineEntry, getAdminSession, setTestResult } from '/static/state.js';
import { pushToast, logConsole } from '/static/render.js';
import { parseErrorMessage } from '/static/utils.js';

// 演示预设
const DEMO_PRESETS = {
  agent_admin_demo: {
    agentId: "agent_admin_demo",
    label: "Admin Demo Agent",
    apiKey: "admin-demo-key",
    roleHint: "admin",
    source: "demo",
  },
  agent_operator_demo: {
    agentId: "agent_operator_demo",
    label: "Operator Demo Agent",
    apiKey: "operator-demo-key",
    roleHint: "operator",
    source: "demo",
  },
  agent_operator_peer_demo: {
    agentId: "agent_operator_peer_demo",
    label: "Operator Peer Demo Agent",
    apiKey: "operator-peer-demo-key",
    roleHint: "operator",
    source: "demo",
  },
  agent_editor_demo: {
    agentId: "agent_editor_demo",
    label: "Editor Demo Agent",
    apiKey: "editor-demo-key",
    roleHint: "editor",
    source: "demo",
  },
  agent_basic_demo: {
    agentId: "agent_basic_demo",
    label: "Basic Demo Agent",
    apiKey: "basic-demo-key",
    roleHint: "basic",
    source: "demo",
  },
};

// 场景定义 - 每个场景有明确的测试目标和预期结果
export const SCENARIO_DEFS = {
  "basic-access": {
    name: "基础访问控制",
    description: "验证 basic 角色只能访问公开文档，无法访问机密文档",
    category: "permission",
    steps: [
      { name: "登录 basic agent", expect: "success" },
      { name: "读取公开文档 public_brief", expect: "allow" },
      { name: "读取机密文档 admin_playbook", expect: "deny" },
    ],
  },
  "operator-flow": {
    name: "操作执行流程",
    description: "验证 operator 角色可以执行任务和调用API集成",
    category: "execution",
    steps: [
      { name: "登录 operator agent", expect: "success" },
      { name: "执行任务 summarize_logs", expect: "allow" },
      { name: "调用集成 knowledge_base", expect: "allow" },
    ],
  },
  "delegation": {
    name: "任务委派",
    description: "验证 operator 可以委派给同角色 peer，但无法委派给 editor（目标无执行权限）",
    category: "delegation",
    steps: [
      { name: "登录 operator agent", expect: "success" },
      { name: "委派任务给 operator peer", expect: "allow" },
      { name: "委派任务给 editor (应失败)", expect: "deny" },
    ],
  },
  "risk-lock": {
    name: "风控自动挂起",
    description: "验证重复越权请求触发自动挂起机制（3次拒绝后自动suspended）",
    category: "risk",
    steps: [
      { name: "登录目标 agent", expect: "success" },
      { name: "发起越权请求 x3", expect: "deny" },
      { name: "验证 agent 被自动挂起", expect: "suspended" },
    ],
  },
  "token-constraints": {
    name: "Token 约束",
    description: "验证 IP 绑定和调用次数限制的强制执行",
    category: "token",
    steps: [
      { name: "创建绑定错误IP的token", expect: "success" },
      { name: "使用错误IP访问 (应拒绝)", expect: "deny" },
      { name: "创建次数限制token (2次)", expect: "success" },
      { name: "耗尽次数后访问 (应拒绝)", expect: "deny" },
    ],
  },
  "editor-write": {
    name: "编辑者写入控制",
    description: "验证 editor 角色可以读写公开和内部文档，但无法访问机密文档",
    category: "permission",
    steps: [
      { name: "登录 editor agent", expect: "success" },
      { name: "读取内部文档 team_notes", expect: "allow" },
      { name: "写入内部文档 team_notes", expect: "allow" },
      { name: "读取机密文档 admin_playbook", expect: "deny" },
    ],
  },
};

/**
 * 判断测试步骤是否通过
 * @param {string} expectType 预期类型: success/allow/deny/suspended
 * @param {boolean} succeeded 操作是否成功（未抛出异常）
 * @param {string} [errorMessage] 错误信息
 * @param {*} [extraData] 额外数据
 * @returns {{ passed: boolean, detail: string }}
 */
function evaluateStep(expectType, succeeded, errorMessage, extraData) {
  switch (expectType) {
    case "success":
      return {
        passed: succeeded,
        detail: succeeded ? "操作成功" : `操作失败: ${errorMessage || "未知错误"}`,
      };
    case "allow":
      return {
        passed: succeeded,
        detail: succeeded ? "访问按预期放行" : `预期放行但被拒绝: ${errorMessage || "未知错误"}`,
      };
    case "deny":
      return {
        passed: !succeeded,
        detail: !succeeded ? `按预期被拒绝: ${errorMessage || "权限不足"}` : "预期拒绝但被放行，权限控制失效",
      };
    case "suspended":
      return {
        passed: !!extraData?.isSuspended,
        detail: extraData?.isSuspended
          ? "Agent 已被自动挂起，风控机制生效"
          : `Agent 状态为 ${extraData?.status || "unknown"}，风控未触发`,
      };
    default:
      return { passed: false, detail: `未知预期类型: ${expectType}` };
  }
}

/**
 * 确保指定Agent已登录，如果已登录则复用现有会话
 * @param {string} agentId Agent ID
 * @returns {Promise<object>} 会话对象
 */
async function ensureLoggedIn(agentId) {
  const existing = state.sessions[agentId];
  if (existing?.token) {
    try {
      await introspectToken(existing);
      return existing;
    } catch {
      // token已失效，需要重新登录
    }
  }
  const preset = DEMO_PRESETS[agentId];
  if (!preset) {
    throw new Error(`未找到预设: ${agentId}`);
  }
  const data = await loginAgent(agentId, preset.apiKey, {
    expiresInMinutes: 60,
    usageLimit: 30,
  });
  upsertSession({
    agentId,
    label: preset.label,
    apiKey: preset.apiKey,
    token: data.access_token,
    tokenType: data.token_type,
    expiresAt: data.expires_at,
    jti: data.jti,
    usageLimit: data.usage_limit,
    roleHint: preset.roleHint,
    source: preset.source,
  });
  try {
    const profile = await getAgentInfo(state.sessions[agentId]);
    state.sessions[agentId].profile = profile;
  } catch {
    // profile获取失败不影响登录
  }
  return state.sessions[agentId];
}

/**
 * 登录所有演示会话（使用批量API提升效率）
 */
export async function loginAllDemoSessions() {
  try {
    const results = await batchLogin(
      "agent_admin_demo",
      "admin-demo-key",
      Object.values(DEMO_PRESETS).map((preset) => ({
        agentId: preset.agentId,
        apiKey: preset.apiKey,
        expiresInMinutes: 60,
        usageLimit: 30,
      }))
    );
    for (const item of results) {
      if (item.success && item.token) {
        const preset = DEMO_PRESETS[item.agent_id];
        upsertSession({
          agentId: item.agent_id,
          label: preset?.label || item.agent_id,
          apiKey: preset?.apiKey || "",
          token: item.token.access_token,
          tokenType: item.token.token_type,
          expiresAt: item.token.expires_at,
          jti: item.token.jti,
          usageLimit: item.token.usage_limit,
          roleHint: preset?.roleHint || null,
          source: preset?.source || "demo",
        });
      } else {
        logConsole(`登录 ${item.agent_id} 失败`, item.error || "Unknown error", "error");
        pushToast(`登录失败`, `无法登录 ${item.agent_id}: ${item.error}`, "error");
      }
    }
    // 并行获取所有profile
    const profilePromises = Object.keys(state.sessions)
      .filter((id) => state.sessions[id]?.token && DEMO_PRESETS[id])
      .map(async (id) => {
        try {
          const profile = await getAgentInfo(state.sessions[id]);
          state.sessions[id].profile = profile;
        } catch {
          // profile获取失败不影响
        }
      });
    await Promise.all(profilePromises);
    pushToast("全部会话已准备好", `${results.filter((r) => r.success).length} 个 Demo Agent 已经登录并保存。`);
  } catch (error) {
    // 批量登录失败，回退到逐个登录
    logConsole("批量登录失败，回退到逐个登录", error.message, "warn");
    for (const preset of Object.values(DEMO_PRESETS)) {
      try {
        const data = await loginAgent(preset.agentId, preset.apiKey, {
          expiresInMinutes: 60,
          usageLimit: 30,
        });
        upsertSession({
          agentId: preset.agentId,
          label: preset.label,
          apiKey: preset.apiKey,
          token: data.access_token,
          tokenType: data.token_type,
          expiresAt: data.expires_at,
          jti: data.jti,
          usageLimit: data.usage_limit,
          roleHint: preset.roleHint,
          source: preset.source,
        });
        try {
          const profile = await getAgentInfo(state.sessions[preset.agentId]);
          state.sessions[preset.agentId].profile = profile;
        } catch {
          // profile获取失败不影响
        }
      } catch (err) {
        logConsole(`登录 ${preset.agentId} 失败`, err.message, "error");
        pushToast(`登录失败`, `无法登录 ${preset.agentId}`, "error");
      }
    }
    pushToast("全部会话已准备好", "5 个 Demo Agent 已经登录并保存。");
  }
}

/**
 * 运行基本访问场景
 * 测试目标: basic角色只能访问公开文档
 * @returns {{ passed: number, failed: number }}
 */
export async function runBasicAccessScenario() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["basic-access"];

  try {
    // 步骤1: 登录 basic agent
    const session = await ensureLoggedIn("agent_basic_demo");
    setActiveSession("agent_basic_demo");
    const r1 = evaluateStep("success", true);
    addTimelineEntry("Basic Access", `✓ ${def.steps[0].name}: ${r1.detail}`, "success");
    r1.passed ? results.passed++ : results.failed++;

    // 步骤2: 读取公开文档
    try {
      const publicDoc = await readDocument("public_brief", session);
      const r2 = evaluateStep("allow", true);
      addTimelineEntry("Basic Access", `✓ ${def.steps[1].name}: ${r2.detail} (sensitivity=${publicDoc.sensitivity})`, "success");
      results.passed++;
    } catch (error) {
      const r2 = evaluateStep("allow", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Basic Access", `✗ ${def.steps[1].name}: ${r2.detail}`, "error");
      results.failed++;
    }

    // 步骤3: 读取机密文档（应被拒绝）
    try {
      await readDocument("admin_playbook", session);
      const r3 = evaluateStep("deny", true);
      addTimelineEntry("Basic Access", `✗ ${def.steps[2].name}: ${r3.detail}`, "error");
      results.failed++;
    } catch (error) {
      const r3 = evaluateStep("deny", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Basic Access", `✓ ${def.steps[2].name}: ${r3.detail}`, "success");
      results.passed++;
    }
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("basic-access", results);
  return results;
}

/**
 * 运行操作员流程场景
 * 测试目标: operator角色可以执行任务和调用API
 * @returns {{ passed: number, failed: number }}
 */
export async function runOperatorFlowScenario() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["operator-flow"];

  try {
    const session = await ensureLoggedIn("agent_operator_demo");
    setActiveSession("agent_operator_demo");
    addTimelineEntry("Operator Flow", `✓ ${def.steps[0].name}: 登录成功`, "success");
    results.passed++;

    // 执行任务
    try {
      const task = await executeTask({
        task_name: "summarize_logs",
        resource: "sandbox",
        parameters: { source: "scenario-runner" },
      }, session);
      const r = evaluateStep("allow", true);
      addTimelineEntry("Operator Flow", `✓ ${def.steps[1].name}: ${r.detail} (exec_id=${task.execution_id})`, "success");
      results.passed++;
    } catch (error) {
      const r = evaluateStep("allow", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Operator Flow", `✗ ${def.steps[1].name}: ${r.detail}`, "error");
      results.failed++;
    }

    // 调用集成
    try {
      const apiCall = await callIntegration({
        service_name: "knowledge_base",
        payload: { source: "scenario-runner" },
      }, session);
      const r = evaluateStep("allow", true);
      addTimelineEntry("Operator Flow", `✓ ${def.steps[2].name}: ${r.detail} (${apiCall.service_name})`, "success");
      results.passed++;
    } catch (error) {
      const r = evaluateStep("allow", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Operator Flow", `✗ ${def.steps[2].name}: ${r.detail}`, "error");
      results.failed++;
    }
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("operator-flow", results);
  return results;
}

/**
 * 运行委派场景
 * 测试目标: operator可以委派给peer，但无法委派给editor
 * @returns {{ passed: number, failed: number }}
 */
export async function runDelegationScenario() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["delegation"];

  try {
    const operatorSession = await ensureLoggedIn("agent_operator_demo");
    await ensureLoggedIn("agent_operator_peer_demo");
    setActiveSession("agent_operator_demo");
    addTimelineEntry("Delegation", `✓ ${def.steps[0].name}: 登录成功`, "success");
    results.passed++;

    // 委派给 operator peer（应成功）
    try {
      const success = await delegateTask({
        target_agent_id: "agent_operator_peer_demo",
        task_name: "prepare_report",
        resource: "sandbox",
      }, operatorSession);
      const r = evaluateStep("allow", true);
      addTimelineEntry("Delegation", `✓ ${def.steps[1].name}: ${r.detail} (delegation_id=${success.delegation_id})`, "success");
      results.passed++;
    } catch (error) {
      const r = evaluateStep("allow", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Delegation", `✗ ${def.steps[1].name}: ${r.detail}`, "error");
      results.failed++;
    }

    // 委派给 editor（应失败 - editor无execute_task权限）
    try {
      await delegateTask({
        target_agent_id: "agent_editor_demo",
        task_name: "prepare_report",
        resource: "sandbox",
      }, operatorSession);
      const r = evaluateStep("deny", true);
      addTimelineEntry("Delegation", `✗ ${def.steps[2].name}: ${r.detail}`, "error");
      results.failed++;
    } catch (error) {
      const r = evaluateStep("deny", false, parseErrorMessage(error.data || error.message || error));
      addTimelineEntry("Delegation", `✓ ${def.steps[2].name}: ${r.detail}`, "success");
      results.passed++;
    }
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("delegation", results);
  return results;
}

/**
 * 运行风控触发场景
 * 测试目标: 重复越权请求触发自动挂起
 * @returns {{ passed: number, failed: number }}
 */
export async function runRiskTrigger() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["risk-lock"];

  const targetAgentId = document.getElementById("riskTargetAgent").value;
  const attempts = Number(document.getElementById("riskAttempts").value);
  const deniedDoc = document.getElementById("riskDeniedDoc").value;

  try {
    const session = await ensureLoggedIn(targetAgentId);
    addTimelineEntry("Risk Demo", `✓ ${def.steps[0].name}: ${targetAgentId} 登录成功`, "success");
    results.passed++;

    // 发起越权请求
    let denyCount = 0;
    for (let index = 0; index < attempts; index += 1) {
      try {
        await readDocument(deniedDoc, session);
        addTimelineEntry("Risk Demo", `✗ 第 ${index + 1} 次越权请求意外放行`, "error");
      } catch (error) {
        denyCount += 1;
        addTimelineEntry("Risk Demo", `✓ 第 ${index + 1} 次请求被拒绝: ${parseErrorMessage(error.data || error.message || error)}`, "success");
      }
    }

    // 评估越权步骤
    if (denyCount > 0) {
      results.passed++;
    } else {
      results.failed++;
    }

    // 检查Agent状态
    const adminSession = await ensureLoggedIn("agent_admin_demo");
    const agents = await getAgents(adminSession);
    const target = agents.find((item) => item.agent_id === targetAgentId);
    const isSuspended = target?.status === "suspended";

    const r3 = evaluateStep("suspended", false, null, { isSuspended, status: target?.status });
    addTimelineEntry(
      "Risk Demo 完成",
      `${isSuspended ? "✓" : "✗"} ${def.steps[2].name}: ${r3.detail} (${denyCount}次拒绝, 状态=${target?.status || "unknown"})`,
      r3.passed ? "success" : "warn",
    );
    r3.passed ? results.passed++ : results.failed++;
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("risk-lock", results);
  return results;
}

/**
 * 运行Token约束场景
 * 测试目标: IP绑定和调用次数限制的强制执行
 * @returns {{ passed: number, failed: number }}
 */
export async function runTokenConstraints() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["token-constraints"];

  try {
    addTimelineEntry("Token Constraints", "开始演示 IP 绑定和调用次数限制。", "warn");

    // 步骤1: 创建绑定错误IP的token
    const ipMismatchToken = await loginAgent("agent_admin_demo", DEMO_PRESETS.agent_admin_demo.apiKey, {
      boundIp: "10.10.10.10",
      usageLimit: 5,
      expiresInMinutes: 60,
    });
    addTimelineEntry("Token Constraints", `✓ ${def.steps[0].name}: token已创建 (bound_ip=10.10.10.10)`, "success");
    results.passed++;

    // 步骤2: 使用错误IP访问（应被拒绝）
    try {
      await authFetch("/me", { token: ipMismatchToken.access_token, tokenType: "Bearer" });
      const r = evaluateStep("deny", true);
      addTimelineEntry("IP Binding", `✗ ${def.steps[1].name}: ${r.detail}`, "error");
      results.failed++;
    } catch (error) {
      const r = evaluateStep("deny", false, parseErrorMessage(error.data || error.message));
      addTimelineEntry("IP Binding", `✓ ${def.steps[1].name}: ${r.detail}`, "success");
      results.passed++;
    }

    // 步骤3: 创建次数限制token
    const limitedToken = await loginAgent("agent_basic_demo", DEMO_PRESETS.agent_basic_demo.apiKey, {
      usageLimit: 2,
      expiresInMinutes: 60,
    });
    const limitedSession = { token: limitedToken.access_token, tokenType: "Bearer" };
    addTimelineEntry("Token Constraints", `✓ ${def.steps[2].name}: token已创建 (usage_limit=2)`, "success");
    results.passed++;

    // 消耗2次
    await authFetch("/me", limitedSession);
    await authFetch("/me", limitedSession);

    // 步骤4: 第3次应被拒绝
    try {
      await authFetch("/me", limitedSession);
      const r = evaluateStep("deny", true);
      addTimelineEntry("Usage Limit", `✗ ${def.steps[3].name}: ${r.detail}`, "error");
      results.failed++;
    } catch (error) {
      const r = evaluateStep("deny", false, parseErrorMessage(error.data || error.message));
      addTimelineEntry("Usage Limit", `✓ ${def.steps[3].name}: ${r.detail}`, "success");
      results.passed++;
    }
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("token-constraints", results);
  return results;
}

/**
 * 运行编辑者写入控制场景
 * 测试目标: editor角色可以读写公开和内部文档，但无法访问机密文档
 * @returns {{ passed: number, failed: number }}
 */
export async function runEditorWriteScenario() {
  const results = { passed: 0, failed: 0 };
  const def = SCENARIO_DEFS["editor-write"];

  try {
    const session = await ensureLoggedIn("agent_editor_demo");
    setActiveSession("agent_editor_demo");
    addTimelineEntry("Editor Write", `✓ ${def.steps[0].name}: 登录成功`, "success");
    results.passed++;

    // 读取内部文档
    try {
      await readDocument("team_notes", session);
      addTimelineEntry("Editor Write", `✓ ${def.steps[1].name}: 读取成功`, "success");
      results.passed++;
    } catch (error) {
      addTimelineEntry("Editor Write", `✗ ${def.steps[1].name}: ${parseErrorMessage(error.data || error.message || error)}`, "error");
      results.failed++;
    }

    // 写入内部文档
    try {
      await writeDocument("team_notes", { content: "Updated by editor test", sensitivity: "internal" }, session);
      addTimelineEntry("Editor Write", `✓ ${def.steps[2].name}: 写入成功`, "success");
      results.passed++;
    } catch (error) {
      addTimelineEntry("Editor Write", `✗ ${def.steps[2].name}: ${parseErrorMessage(error.data || error.message || error)}`, "error");
      results.failed++;
    }

    // 读取机密文档（应被拒绝）
    try {
      await readDocument("admin_playbook", session);
      addTimelineEntry("Editor Write", `✗ ${def.steps[3].name}: 预期拒绝但被放行`, "error");
      results.failed++;
    } catch (error) {
      addTimelineEntry("Editor Write", `✓ ${def.steps[3].name}: ${parseErrorMessage(error.data || error.message || error)}`, "success");
      results.passed++;
    }
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    results.failed++;
  }

  setTestResult("editor-write", results);
  return results;
}

/**
 * 运行完整答辩流程
 * 依次执行所有测试场景，汇总结果
 * @returns {{ passed: number, failed: number, scenarios: object }}
 */
export async function runJudgeWalkthrough() {
  const totalResults = { passed: 0, failed: 0, scenarios: {} };

  addTimelineEntry("Judge Walkthrough", "▶ 开始运行全部测试场景...", "warn");

  await loginAllDemoSessions();

  // 依次执行各场景
  const scenarioRunners = [
    { id: "basic-access", runner: runBasicAccessScenario },
    { id: "operator-flow", runner: runOperatorFlowScenario },
    { id: "delegation", runner: runDelegationScenario },
    { id: "risk-lock", runner: runRiskTrigger },
    { id: "token-constraints", runner: runTokenConstraints },
    { id: "editor-write", runner: runEditorWriteScenario },
  ];

  for (const { id, runner } of scenarioRunners) {
    const def = SCENARIO_DEFS[id];
    addTimelineEntry(def.name, `▶ 开始执行: ${def.description}`, "warn");
    const result = await runner();
    totalResults.scenarios[id] = result;
    totalResults.passed += result.passed;
    totalResults.failed += result.failed;
    const status = result.failed === 0 ? "全部通过" : `${result.failed} 项失败`;
    addTimelineEntry(def.name, `◆ 完成: ${result.passed} 通过, ${result.failed} 失败 — ${status}`, result.failed === 0 ? "success" : "error");
  }

  // 恢复admin会话
  setActiveSession("agent_admin_demo");

  // 汇总
  const summaryStatus = totalResults.failed === 0 ? "success" : "warn";
  addTimelineEntry(
    "Judge Walkthrough",
    `◆ 全部场景执行完毕: ${totalResults.passed} 通过, ${totalResults.failed} 失败`,
    summaryStatus,
  );

  return totalResults;
}

/**
 * 运行指定场景
 * @param {string} name 场景名称
 * @returns {Promise<{ passed: number, failed: number }>}
 */
export async function runScenarioByName(name) {
  try {
    let result;
    if (name === "basic-access") {
      result = await runBasicAccessScenario();
    } else if (name === "operator-flow") {
      result = await runOperatorFlowScenario();
    } else if (name === "delegation") {
      result = await runDelegationScenario();
    } else if (name === "risk-lock") {
      result = await runRiskTrigger();
    } else if (name === "token-constraints") {
      result = await runTokenConstraints();
    } else if (name === "editor-write") {
      result = await runEditorWriteScenario();
    } else if (name === "judge-walkthrough") {
      result = await runJudgeWalkthrough();
      return result;
    } else {
      addTimelineEntry("未知场景", `未找到场景: ${name}`, "error");
      return { passed: 0, failed: 1 };
    }

    // 单场景结果摘要
    const def = SCENARIO_DEFS[name];
    if (def) {
      const status = result.failed === 0 ? "全部通过" : `${result.failed} 项失败`;
      addTimelineEntry(def.name, `◆ 完成: ${result.passed} 通过, ${result.failed} 失败 — ${status}`, result.failed === 0 ? "success" : "error");
    }
    return result;
  } catch (error) {
    addTimelineEntry("Scenario Failed", parseErrorMessage(error.data || error.message || error), "error");
    pushToast("场景执行失败", parseErrorMessage(error.data || error.message || error), "error");
    return { passed: 0, failed: 1 };
  }
}

// 导出演示预设
export { DEMO_PRESETS };
