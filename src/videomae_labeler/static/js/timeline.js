// Zoomable canvas timeline + whole-video overview strip.
import { formatTime } from "./format.js";
import { kindOf } from "./ops.js";

export const COLORS = Object.freeze({ service: "#d946ef", play: "#22c55e", "no-play": "#ef4444" });
const BG = "#171a21", GRID = "#2a2f3a", TEXT = "#8a92a3", PLAYHEAD = "#f8fafc";
const LANE_TOP = 22, LANE_H = 52, EDGE_PX = 7, DRAG_PX = 4;
const TICK_STEPS_S = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const DEFAULT_VIEW_S = 60, MIN_VIEW_S = 2;
const ALPHA = Object.freeze({ manual: 0.95, model: 0.55, default: 0.22 });

export class Timeline {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {HTMLCanvasElement} overview
   * @param {{onSeek:(f:number)=>void, onSelect:(seg:object|null)=>void,
   *          onResize:(seg:object, start:number, end:number)=>void}} handlers
   */
  constructor(canvas, overview, handlers) {
    this.canvas = canvas;
    this.overview = overview;
    this.handlers = handlers;
    this.segments = [];
    this.frameCount = 1;
    this.fps = 30;
    this.playhead = 0;
    this.selected = null;
    this.preview = null;
    this.draft = null;          // {seg, start, end} while an edge is being dragged
    this.view = { start: 0, span: 1 };
    this.drag = null;
    this.dirty = true;
    this.#bindEvents();
    new ResizeObserver(() => this.#invalidate()).observe(canvas);
    const loop = () => { if (this.dirty) this.#render(); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  }

  setVideo(frameCount, fps) {
    this.frameCount = Math.max(1, frameCount);
    this.fps = fps;
    this.view = { start: 0, span: Math.min(this.frameCount, Math.round(DEFAULT_VIEW_S * fps)) };
    this.#invalidate();
  }

  setSegments(segments) { this.segments = segments; this.#invalidate(); }
  setSelected(seg) { this.selected = seg; this.#invalidate(); }
  setPreview(preview) { this.preview = preview; this.#invalidate(); }

  setPlayhead(frame, { follow = false } = {}) {
    this.playhead = frame;
    const { start, span } = this.view;
    if (frame < start || frame > start + span) this.#scrollTo(frame - span * 0.15);
    else if (follow && frame > start + span * 0.85) this.#scrollTo(frame - span * 0.15);
    this.#invalidate();
  }

  // ---------------------------------------------------------------- geometry
  #width() { return this.canvas.clientWidth || 1; }
  #x(frame) { return ((frame - this.view.start) / this.view.span) * this.#width(); }
  #frameAt(x) {
    const f = Math.floor(this.view.start + (x / this.#width()) * this.view.span);
    return Math.min(this.frameCount - 1, Math.max(0, f));
  }
  #scrollTo(start) {
    const maxStart = Math.max(0, this.frameCount - this.view.span);
    this.view = { ...this.view, start: Math.min(maxStart, Math.max(0, start)) };
    this.#invalidate();
  }
  #zoom(factor, anchorX) {
    const anchor = this.view.start + (anchorX / this.#width()) * this.view.span;
    const minSpan = Math.max(10, MIN_VIEW_S * this.fps);
    const span = Math.min(this.frameCount, Math.max(minSpan, this.view.span * factor));
    this.view = { start: this.view.start, span };
    this.#scrollTo(anchor - (anchorX / this.#width()) * span);
  }
  #segmentAt(frame) {
    return this.segments.find((s) => s.start_frame <= frame && frame <= s.end_frame) ?? null;
  }
  #invalidate() { this.dirty = true; }

  // ---------------------------------------------------------------- events
  #bindEvents() {
    const c = this.canvas;
    c.addEventListener("wheel", (e) => {
      e.preventDefault();
      const x = e.offsetX;
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const delta = e.shiftKey ? e.deltaY : e.deltaX;
        this.#scrollTo(this.view.start + (delta / this.#width()) * this.view.span);
      } else {
        this.#zoom(e.deltaY > 0 ? 1.25 : 0.8, x);
      }
    }, { passive: false });

    c.addEventListener("pointerdown", (e) => {
      c.setPointerCapture(e.pointerId);
      const edge = this.#edgeHit(e.offsetX);
      this.drag = edge
        ? { mode: "resize", edge, x0: e.offsetX }
        : { mode: "press", x0: e.offsetX, start0: this.view.start };
      if (edge) this.draft = { seg: this.selected, start: this.selected.start_frame, end: this.selected.end_frame };
    });

    c.addEventListener("pointermove", (e) => {
      if (!this.drag) { c.style.cursor = this.#edgeHit(e.offsetX) ? "ew-resize" : "crosshair"; return; }
      const dx = e.offsetX - this.drag.x0;
      if (this.drag.mode === "press" && Math.abs(dx) > DRAG_PX) this.drag = { ...this.drag, mode: "pan" };
      if (this.drag.mode === "pan") {
        this.#scrollTo(this.drag.start0 - (dx / this.#width()) * this.view.span);
        c.style.cursor = "grabbing";
      } else if (this.drag.mode === "resize") {
        const f = this.#frameAt(e.offsetX);
        const { seg } = this.draft;
        this.draft = this.drag.edge === "start"
          ? { seg, start: Math.min(f, seg.end_frame), end: seg.end_frame }
          : { seg, start: seg.start_frame, end: Math.max(f, seg.start_frame) };
        this.#invalidate();
      }
    });

    const finish = (e) => {
      const drag = this.drag;
      this.drag = null;
      c.style.cursor = "crosshair";
      if (!drag) return;
      if (drag.mode === "press") {
        const frame = this.#frameAt(e.offsetX);
        this.handlers.onSeek(frame);
        this.handlers.onSelect(this.#segmentAt(frame));
      } else if (drag.mode === "resize") {
        const { seg, start, end } = this.draft;
        this.draft = null;
        if (start !== seg.start_frame || end !== seg.end_frame) this.handlers.onResize(seg, start, end);
        this.#invalidate();
      }
    };
    c.addEventListener("pointerup", finish);
    c.addEventListener("pointercancel", () => { this.drag = null; this.draft = null; this.#invalidate(); });

    this.overview.addEventListener("pointerdown", (e) => {
      const frame = Math.floor((e.offsetX / (this.overview.clientWidth || 1)) * this.frameCount);
      this.#scrollTo(frame - this.view.span / 2);
      this.handlers.onSeek(Math.min(this.frameCount - 1, Math.max(0, frame)));
    });
  }

  #edgeHit(x) {
    const s = this.selected;
    if (!s) return null;
    if (Math.abs(x - this.#x(s.start_frame)) <= EDGE_PX) return "start";
    if (Math.abs(x - this.#x(s.end_frame + 1)) <= EDGE_PX) return "end";
    return null;
  }

  // ---------------------------------------------------------------- drawing
  #prepare(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, w, h);
    return { ctx, w, h };
  }

  #render() {
    this.dirty = false;
    this.#renderMain();
    this.#renderOverview();
  }

  #renderMain() {
    const { ctx, w, h } = this.#prepare(this.canvas);
    this.#drawTicks(ctx, w, h);
    const visible = this.segments.filter(
      (s) => s.end_frame >= this.view.start && s.start_frame <= this.view.start + this.view.span);
    for (const seg of visible) {
      if (this.draft && seg === this.draft.seg) continue;
      this.#drawSegment(ctx, seg.start_frame, seg.end_frame, seg.state, { kind: kindOf(seg) });
    }
    if (this.draft) this.#drawSegment(ctx, this.draft.start, this.draft.end, this.draft.seg.state, { selected: true });
    else if (this.selected) this.#drawSegment(ctx, this.selected.start_frame, this.selected.end_frame, this.selected.state,
      { selected: true, kind: kindOf(this.selected) });
    if (this.preview) this.#drawSegment(ctx, this.preview.start, this.preview.end, this.preview.state, { live: true });
    this.#drawPlayhead(ctx, h);
  }

  #drawTicks(ctx, w, h) {
    const pxPerSec = (w / this.view.span) * this.fps;
    const step = TICK_STEPS_S.find((s) => s * pxPerSec >= 80) ?? TICK_STEPS_S.at(-1);
    const startS = Math.floor(this.view.start / this.fps / step) * step;
    ctx.font = "11px ui-monospace, Consolas, monospace";
    ctx.textBaseline = "top";
    for (let t = startS; t * this.fps <= this.view.start + this.view.span; t += step) {
      const x = Math.round(this.#x(t * this.fps)) + 0.5;
      ctx.strokeStyle = GRID;
      ctx.beginPath(); ctx.moveTo(x, 14); ctx.lineTo(x, h); ctx.stroke();
      ctx.fillStyle = TEXT;
      ctx.fillText(formatTime(t, step < 1), x + 3, 3);
    }
  }

  #drawSegment(ctx, start, end, state, { kind = "manual", selected = false, live = false } = {}) {
    const x0 = Math.max(-2, this.#x(start)), x1 = Math.min(this.#width() + 2, this.#x(end + 1));
    const wpx = Math.max(1, x1 - x0);
    const short = kind !== "manual" && !live;
    const y = short ? LANE_TOP + LANE_H * 0.3 : LANE_TOP, hpx = short ? LANE_H * 0.7 : LANE_H;
    ctx.globalAlpha = live ? 0.8 : ALPHA[kind];
    ctx.fillStyle = COLORS[state];
    ctx.fillRect(x0, y, wpx, hpx);
    ctx.globalAlpha = 1;
    if (wpx > 2) { ctx.fillStyle = BG; ctx.fillRect(x1 - 1, y, 1, hpx); }
    if (live) {
      ctx.setLineDash([4, 3]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 + 0.75, y + 0.75, wpx - 1.5, hpx - 1.5); ctx.setLineDash([]);
    }
    if (selected) {
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1, y + 1, wpx - 2, hpx - 2);
      ctx.fillStyle = "#fff";
      ctx.fillRect(this.#x(start) - 2, y + hpx / 2 - 9, 4, 18);
      ctx.fillRect(this.#x(end + 1) - 2, y + hpx / 2 - 9, 4, 18);
    }
    if (wpx > 70) {
      ctx.fillStyle = "#fff";
      ctx.font = "600 12px 'Segoe UI', sans-serif";
      ctx.textBaseline = "middle";
      const label = `${state}${kind === "model" ? " (模型)" : kind === "default" ? " (預設)" : ""}${live ? " ●" : ""}`;
      ctx.fillText(label, Math.max(x0, 0) + 6, y + hpx / 2);
    }
  }

  #drawPlayhead(ctx, h) {
    const x = Math.round(this.#x(this.playhead + 0.5)) + 0.5;
    ctx.strokeStyle = PLAYHEAD; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = PLAYHEAD;
    ctx.beginPath(); ctx.moveTo(x - 5, 12); ctx.lineTo(x + 5, 12); ctx.lineTo(x, 18); ctx.fill();
  }

  #renderOverview() {
    const { ctx, w, h } = this.#prepare(this.overview);
    const scale = w / this.frameCount;
    for (const seg of this.segments) {
      ctx.globalAlpha = ALPHA[kindOf(seg)];
      ctx.fillStyle = COLORS[seg.state];
      ctx.fillRect(seg.start_frame * scale, 3, Math.max(1, (seg.end_frame - seg.start_frame + 1) * scale), h - 6);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#93c5fd"; ctx.lineWidth = 1.5;
    ctx.strokeRect(this.view.start * scale + 0.75, 0.75, Math.max(3, this.view.span * scale) - 1.5, h - 1.5);
    ctx.fillStyle = PLAYHEAD;
    ctx.fillRect(this.playhead * scale - 1, 0, 2, h);
  }
}
