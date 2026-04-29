const STORAGE_KEY = 'iam_sessions';
const ACTIVE_KEY = 'iam_active_id';

export const state = {
  listSessions() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch { return []; }
  },

  getSession(agentId) {
    return this.listSessions().find(s => s.agent_id === agentId) || null;
  },

  saveSession(agentId, data) {
    const sessions = this.listSessions().filter(s => s.agent_id !== agentId);
    sessions.push({ ...data, agent_id: agentId });
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  },

  removeSession(agentId) {
    const sessions = this.listSessions().filter(s => s.agent_id !== agentId);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    if (this.getActiveId() === agentId) {
      const next = sessions[0]?.agent_id || null;
      localStorage.setItem(ACTIVE_KEY, next || '');
    }
  },

  getActiveId() {
    return localStorage.getItem(ACTIVE_KEY) || null;
  },

  setActiveId(agentId) {
    localStorage.setItem(ACTIVE_KEY, agentId);
  },

  clearAll() {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(ACTIVE_KEY);
  },
};
