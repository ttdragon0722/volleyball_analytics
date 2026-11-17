// Wires the video element, hold recorder, timeline and panels together.
// Edits live in memory (two layers: model + manual) until the user presses Ctrl+S.
import { api } from "./api.js";
import { HoldRecorder } from "./recorder.js";
import { Timeline } from "./timeline.js";
import { Editor, Picker, SegmentList } from "./panels.js";
import { formatTime, toast } from "./format.js";
import { History } from "./history.js";
import {
  compose, deleteSegment, editSegment, kindOf, normalize, paint, sameRange, splitLayers,
} from "./ops.js";
import { createModelJob, createProxyJob } from "./jobs.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const KEY_STATES = Object.freeze({ "1": "service", "2": "play", "3": "no-play" });
const JUMP_S = 3;
const STORAGE_KEY = "videomae-labeler:v1";
const EMPTY_LAYERS = Object.freeze({ model: Object.freeze([]), manual: Object.freeze([]) });

let info = null;              // {path, fps, frame_count, width, height}
let history = new History(EMPTY_LAYERS);
let view = Object.freeze([]); // composed timeline (what is shown and saved)
let savedKey = "";
let selected = null;
let lastFrame = 0;

// ------------------------------------------------------------------ prefs (per-viewer convenience only)
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {}; } catch { return {}; }
}
function savePrefs(patch) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...loadPrefs(), ...patch })); } catch { /* unavailable */ }
}
function rememberPosition() {
  if (!info) return;
  savePrefs({ lastPath: info.path, positions: { ...(loadPrefs().positions ?? {}), [info.path]: currentFrame() } });
}

// ------------------------------------------------------------------ frames & seeking
function currentFrame() {
  if (!info) return 0;
  return Math.min(info.frame_count - 1, Math.max(0, Math.floor(video.currentTime * info.fps + 1e-4)));
}
function seekTo(frame) {
  if (!info) return;
  const to = Math.min(info.frame_count - 1, Math.max(0, Math.round(frame)));
  if (!video.paused) recorder.seek(currentFrame(), to);
  video.currentTime = (to + 0.5) / info.fps;
  timeline.setPlayhead(to);
}
const seekBy = (seconds) => info && seekTo(currentFrame() + seconds * info.fps);

// ------------------------------------------------------------------ document state
const keyOf = (segments) => JSON.stringify(segments);
const isDirty = () => !!info && keyOf(view) !== savedKey;

function render() {
  view = info ? compose(history.current.model, history.current.manual, info.frame_count) : Object.freeze([]);
  selected = view.find((s) => sameRange(s, selected)) ?? null;
  timeline.setSegments(view);
  timeline.setSelected(selected);
  editor.show(selected, info?.fps ?? 30);
  list.render({ segments: view, selected, fps: info?.fps ?? 30 });
  $("btn-undo").disabled = !history.canUndo;
  $("btn-redo").disabled = !history.canRedo;
  const dirty = isDirty();
  $("btn-save").disabled = !info;
  $("btn-save").classList.toggle("dirty", dirty);
  setStatus(dirty ? "● 有未儲存的變更" : $("save-status").dataset.saved || "—", dirty ? "dirty" : "");
  renderHud();
}
function update(layers) {
  history = history.push(layers);
  render();
}
function setStatus(text, cls = "") {
  const node = $("save-status");
  node.textContent = text;
  node.className = `save-status ${cls}`;
}

async function save() {
  if (!info) return;
  const path = info.path, snapshot = view;
  setStatus("儲存中…", "saving");
  try {
    await api.save(path, snapshot);
    if (info?.path !== path) return;
    savedKey = keyOf(snapshot);
    $("save-status").dataset.saved = `已儲存 · ${new Date().toLocaleTimeString()}`;
    render();
    toast("已儲存到資料庫");
  } catch (err) {
    setStatus("儲存失敗", "error");
    toast(`儲存失敗：${err.message}`, "error", 8000);
  }
}

// ------------------------------------------------------------------ hold-to-label
const recorder = new HoldRecorder({
  commit: (start, end, state) => {
    const layers = history.current;
    update({ ...layers, manual: paint(layers.manual, start, end, state) });
  },
  onChange: renderHud,
});

function renderHud() {
  const active = recorder.active;
  const rec = $("hud-rec"), hud = $("hud-state");
  hud.hidden = !active;
  if (active) { hud.textContent = active.toUpperCase(); hud.className = `hud-state ${active}`; }
  rec.className = `hud-rec${recorder.recording ? " live" : ""}`;
  rec.textContent = recorder.recording ? "● 標註中" : active ? "按住中 · 播放後開始標註" : "按住 1 / 2 / 3 標註";
  document.querySelectorAll(".state-btn").forEach((b) => b.classList.toggle("active", b.dataset.state === active));
  $("btn-play").textContent = video.paused ? "▶" : "❚❚";
}
function renderUnder(frame) {
  const s = view.find((x) => x.start_frame <= frame && frame <= x.end_frame);
  const label = { manual: "手動", model: "模型", default: "預設" };
  $("hud-under").textContent = s ? `時間軸：${s.state}（${label[kindOf(s)]}）` : "";
}

function pausePlayback() {
  if (!info || video.paused) return;
  const f = currentFrame();
  video.pause();
  recorder.pause(f);
  video.currentTime = (f + 0.5) / info.fps;   // decoder may run past f; snap back
}

// ------------------------------------------------------------------ selection & editing
function select(seg, { seek = false } = {}) {
  selected = seg;
  render();
  if (seg && seek) seekTo(seg.start_frame);
}
function applyEdit(seg, next) {
  update(editSegment(history.current, seg, next));
  select(view.find((s) => s.start_frame <= next.start && next.start <= s.end_frame) ?? null);
}
function deleteSelected(seg = selected) {
  if (!seg) return;
  const next = deleteSegment(history.current, seg);
  if (!next) { toast("預設 no-play 不需要刪除；要改狀態請按住 1 / 2 標註或在右側修改"); return; }
  update(next);
  select(null);
}
function undo() { if (history.canUndo) { history = history.undo(); render(); } }
function redo() { if (history.canRedo) { history = history.redo(); render(); } }

const timeline = new Timeline($("timeline"), $("overview"), {
  onSeek: seekTo,
  onSelect: (seg) => select(seg),
  onResize: (seg, start, end) => applyEdit(seg, { start, end, state: seg.state }),
});
const editor = new Editor({ onApply: applyEdit, onDelete: deleteSelected, onJump: seekTo, currentFrame });
const list = new SegmentList({ onPick: (seg) => select(seg, { seek: true }) });
const picker = new Picker({ onPick: openVideo });

// ------------------------------------------------------------------ background jobs
// ------------------------------------------------------------------ model picker
const modelSelect = $("model-select");
const selectedModelLabel = () => modelSelect.selectedOptions[0]?.dataset.short ?? "模型";

async function loadModels() {
  try {
    const { models, default: fallback } = await api.models();
    const options = models.map((m) => {
      const f1 = m.macro_f1 != null ? ` · F1 ${m.macro_f1.toFixed(2)}` : "";
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = `${m.label}${f1}`;
      opt.dataset.short = m.label;
      opt.title = m.val_videos.length ? `${m.id}\n驗證影片：${m.val_videos.join("、")}` : m.id;
      return opt;
    });
    modelSelect.replaceChildren(...options);
    const saved = loadPrefs().model;
    modelSelect.value = models.some((m) => m.id === saved) ? saved : fallback ?? "";
    modelSelect.disabled = models.length === 0;
  } catch (err) {
    toast(`讀取模型清單失敗：${err.message}`, "error");
  }
}
modelSelect.addEventListener("change", () => savePrefs({ model: modelSelect.value }));

const modelJob = createModelJob({
  api, current: () => info?.path, model: () => modelSelect.value || null, modelLabel: selectedModelLabel,
  onProgress: (text) => { $("model-status").textContent = text; },
  onReady: (segments) => {
    update({ ...history.current, model: normalize(segments) });
    $("model-status").textContent = `${selectedModelLabel()} 預標完成（尚未儲存）`;
    toast("模型預標完成，你按住標註的部分不受影響。記得 Ctrl+S 儲存");
  },
  onError: (msg) => { $("model-status").textContent = ""; toast(msg, "error", 8000); },
});
let usingProxy = false;
const proxyJob = createProxyJob({
  api, current: () => info?.path,
  onProgress: (text) => showOverlay(text),
  onReady: () => { loadSource(true); toast("已改用相容播放檔"); },
  onError: (msg) => showOverlay(msg),
});

function showOverlay(text) {
  const box = $("empty");
  box.textContent = text;
  box.hidden = false;
}
function loadSource(proxy) {
  usingProxy = proxy;
  $("empty").hidden = true;
  const resume = loadPrefs().positions?.[info.path];
  video.addEventListener("loadedmetadata", () => { if (resume) seekTo(resume); }, { once: true });
  video.src = api.videoUrl(info.path, proxy);
}

// ------------------------------------------------------------------ opening a video
async function pickWithNativeDialog() {
  const btn = $("btn-open");
  btn.disabled = true;
  btn.textContent = "請在跳出的視窗選擇…";
  try {
    const { path } = await api.pick(loadPrefs().lastDir ?? "");
    if (path) await openVideo(path);
  } catch (err) {
    toast(err.message, "error", 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = "開啟影片";
  }
}

async function openVideo(path) {
  if (isDirty() && !window.confirm("目前影片有未儲存的變更，確定要放棄並開啟其他影片嗎？")) return;
  pausePlayback();
  recorder.releaseAll(currentFrame());
  try {
    const data = await api.open(path);
    info = data.video;
    history = new History(splitLayers(data.segments));
    selected = null;
    savedKey = keyOf(compose(history.current.model, history.current.manual, info.frame_count));
    $("save-status").dataset.saved = data.segments.length ? "已載入資料庫的標註" : "尚無標註";
    $("video-name").textContent = `${info.path}  ·  ${info.width}×${info.height} @ ${info.fps.toFixed(2)} fps`;
    $("time-total").textContent = formatTime(info.frame_count / info.fps);
    $("btn-model").disabled = !data.model_available;
    $("model-status").textContent = data.model_available ? ""
      : data.model_available === undefined ? "伺服器版本過舊，請重新啟動 videomae_labeler"
        : "找不到模型權重（src/weights/game_state/config.json），無法預標";
    timeline.setVideo(info.frame_count, info.fps);
    render();
    savePrefs({ lastPath: info.path, lastDir: info.path.split("/").slice(0, -1).join("/") });
    const proxy = await api.proxyStatus(info.path);
    loadSource(proxy.state === "ready");
  } catch (err) {
    toast(`開啟失敗：${err.message}`, "error", 6000);
  }
}

// ------------------------------------------------------------------ video events
video.addEventListener("play", () => { recorder.play(currentFrame()); renderHud(); });
video.addEventListener("pause", () => {
  recorder.pause(video.ended && info ? info.frame_count - 1 : currentFrame());
  rememberPosition();
  renderHud();
});
video.addEventListener("seeked", rememberPosition);
video.addEventListener("error", () => {
  if (!info || !video.getAttribute("src")) return;
  if (!usingProxy) { proxyJob.run(); return; }
  toast("相容播放檔也無法播放，請確認影片檔案沒有損壞", "error", 8000);
});

function tick() {
  if (info) {
    const f = currentFrame();
    if (f !== lastFrame || !video.paused) {
      $("time-cur").textContent = formatTime(video.currentTime);
      $("frame-cur").textContent = f;
      timeline.setPlayhead(f, { follow: !video.paused });
      renderUnder(f);
      lastFrame = f;
    }
    timeline.setPreview(recorder.preview(f));
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

window.addEventListener("beforeunload", (e) => {
  rememberPosition();
  if (isDirty() || recorder.recording) { e.preventDefault(); e.returnValue = ""; }
});
window.addEventListener("blur", () => recorder.releaseAll(currentFrame()));

// ------------------------------------------------------------------ controls & keyboard
function togglePlay() {
  if (!info) return;
  if (video.paused) video.play().catch((err) => toast(`無法播放：${err.message}`, "error"));
  else pausePlayback();
}
function shiftSpeed(dir) {
  const sel = $("speed");
  sel.selectedIndex = Math.min(sel.options.length - 1, Math.max(0, sel.selectedIndex + dir));
  video.playbackRate = Number(sel.value);
}

$("btn-open").addEventListener("click", pickWithNativeDialog);
$("btn-recent").addEventListener("click", () => picker.open());
$("btn-play").addEventListener("click", togglePlay);
$("btn-start").addEventListener("click", () => seekTo(0));
$("btn-undo").addEventListener("click", undo);
$("btn-redo").addEventListener("click", redo);
$("btn-save").addEventListener("click", save);
$("btn-model").addEventListener("click", () => modelJob.run({ reuseFinished: false }));
$("btn-help").addEventListener("click", () => $("help").showModal());
$("speed").addEventListener("change", (e) => { video.playbackRate = Number(e.target.value); });
video.addEventListener("click", togglePlay);
document.querySelectorAll(".state-btn").forEach((b) => {
  const state = b.dataset.state;
  b.addEventListener("pointerdown", (e) => { b.setPointerCapture(e.pointerId); info && recorder.keyDown(state, currentFrame()); });
  const up = () => info && recorder.keyUp(state, currentFrame());
  b.addEventListener("pointerup", up);
  b.addEventListener("pointercancel", up);
});

function isTyping(e) {
  return e.target.closest?.("input, select, textarea") && e.target.type !== "checkbox" && e.target.type !== "radio";
}

document.addEventListener("keydown", (e) => {
  if (document.querySelector("dialog[open]")) return;
  const key = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  if ((e.ctrlKey || e.metaKey) && key === "s") { e.preventDefault(); save(); return; }
  if (isTyping(e)) { if (e.key === "Escape") e.target.blur(); return; }
  if (e.ctrlKey || e.metaKey) {
    if (key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
    else if (key === "y" || (key === "z" && e.shiftKey)) { e.preventDefault(); redo(); }
    return;
  }
  if (e.altKey) return;
  if (KEY_STATES[e.key]) {
    e.preventDefault();
    if (!e.repeat && info) recorder.keyDown(KEY_STATES[e.key], currentFrame());
    return;
  }
  const actions = {
    " ": togglePlay,
    a: () => seekBy(-JUMP_S), ArrowLeft: () => seekBy(-JUMP_S),
    d: () => seekBy(JUMP_S), ArrowRight: () => seekBy(JUMP_S),
    ",": () => seekTo(currentFrame() - 1), ".": () => seekTo(currentFrame() + 1),
    Home: () => seekTo(0),
    "[": () => shiftSpeed(-1), "]": () => shiftSpeed(1),
    Delete: () => deleteSelected(), Backspace: () => deleteSelected(),
    Escape: () => select(null),
    "?": () => $("help").showModal(),
  };
  if (actions[key]) { e.preventDefault(); actions[key](); document.activeElement?.blur?.(); }
});
document.addEventListener("keyup", (e) => {
  if (KEY_STATES[e.key] && info) recorder.keyUp(KEY_STATES[e.key], currentFrame());
});

render();
loadModels();
const last = loadPrefs().lastPath;
if (last) openVideo(last);
