// Small display helpers shared by the UI modules.

/** 75.3 -> "01:15.30" (or "1:01:15.30" past an hour). */
export function formatTime(seconds, withFraction = true) {
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const secStr = withFraction ? sec.toFixed(2).padStart(5, "0") : String(Math.floor(sec)).padStart(2, "0");
  const mm = String(m).padStart(2, "0");
  return h ? `${h}:${mm}:${secStr}` : `${mm}:${secStr}`;
}

export function formatDuration(seconds) {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

export function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

export function toast(message, kind = "info", ms = 3500) {
  const box = document.getElementById("toasts");
  const node = el("div", { className: `toast ${kind}`, textContent: message });
  box.append(node);
  setTimeout(() => node.remove(), ms);
}
