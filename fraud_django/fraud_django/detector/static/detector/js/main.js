// main.js — shared utilities

function getCookie(name) {
  let val = null;
  document.cookie.split(';').forEach(c => {
    const [k, v] = c.trim().split('=');
    if (k === name) val = decodeURIComponent(v);
  });
  return val;
}
window.CSRF = getCookie('csrftoken') || '';

function fmtPct(v)  { return (v * 100).toFixed(1) + '%'; }
function fmtAmt(v)  { return '$' + parseFloat(v).toFixed(2); }
function fmtNum(v)  { return parseInt(v).toLocaleString(); }

function riskClass(riskLevel) {
  return { CRITICAL:'badge-critical', HIGH:'badge-high', MEDIUM:'badge-medium', LOW:'badge-low' }[riskLevel] || 'badge-medium';
}
function riskColor(riskLevel) {
  return { CRITICAL:'#ef4444', HIGH:'#f97316', MEDIUM:'#3b82f6', LOW:'#22c55e' }[riskLevel] || '#3b82f6';
}

window.utils = { getCookie, fmtPct, fmtAmt, fmtNum, riskClass, riskColor };
