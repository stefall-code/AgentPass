export function $(selector) {
  return document.querySelector(selector);
}

export function $$(selector) {
  return document.querySelectorAll(selector);
}

const ESCAPE_MAP = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

export function escapeHtml(str) {
  if (typeof str !== 'string') return '';
  return str.replace(/[&<>"']/g, c => ESCAPE_MAP[c]);
}

export function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function parseErrorMessage(err) {
  if (!err) return 'Unknown error';
  if (typeof err === 'string') return err;
  return err.message || err.detail || 'Unknown error';
}

export function generateId(prefix = 'id') {
  return prefix + '_' + Math.random().toString(36).substring(2, 10);
}

export function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

export function formatTime(iso) {
  if (!iso) return '—';
  try { return new Date(iso).toLocaleTimeString(); } catch { return iso; }
}
