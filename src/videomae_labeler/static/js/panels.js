// Side panels (segment editor, segment list) and the open-video dialog.
import { api } from "./api.js";
import { COLORS } from "./timeline.js";
import { el, formatDuration, formatTime, toast } from "./format.js";
import { kindOf, sameRange } from "./ops.js";

const $ = (id) => document.getElementById(id);
const KIND_LABEL = Object.freeze({ manual: "手動", model: "模型", default: "預設" });

export class Editor {
  /** @param {{onApply:(seg:object, next:{start:number,end:number,state:string})=>void,
   *          onDelete:(seg:object)=>void, onJump:(frame:number)=>void, currentFrame:()=>number}} h */
  constructor(h) {
    this.h = h;
    this.seg = null;
    this.fps = 30;
    this.form = $("editor-form");
    this.start = $("ed-start");
    this.end = $("ed-end");
    this.form.addEventListener("submit", (e) => { e.preventDefault(); this.#apply(); });
    this.form.addEventListener("input", () => this.#refreshTimes());
    $("ed-delete").addEventListener("click", () => this.seg && h.onDelete(this.seg));
    $("ed-jump").addEventListener("click", () => this.seg && h.onJump(this.seg.start_frame));
    this.form.querySelectorAll("[data-set]").forEach((btn) => btn.addEventListener("click", () => {
      (btn.dataset.set === "start" ? this.start : this.end).value = h.currentFrame();
      this.#refreshTimes();
    }));
  }

  show(seg, fps) {
    this.seg = seg;
    this.fps = fps;
    $("editor-empty").hidden = !!seg;
    this.form.hidden = !seg;
    if (!seg) return;
    this.start.value = seg.start_frame;
    this.end.value = seg.end_frame;
    this.form.querySelector(`input[name=state][value="${seg.state}"]`).checked = true;
    $("ed-source").textContent = {
      manual: "手動標註 · 刪除後變成預設 no-play",
      model: `模型預測（信心 ${seg.confidence.toFixed(2)}）· 修改後會變成手動標註，刪除後變成預設 no-play`,
      default: "預設 no-play（沒有模型預測也沒有手動標註）· 修改後會變成手動標註",
    }[kindOf(seg)];
    this.#refreshTimes();
  }

  #refreshTimes() {
    const dur = (Number(this.end.value) - Number(this.start.value) + 1) / this.fps;
    $("ed-start-t").textContent = formatTime(Number(this.start.value) / this.fps);
    $("ed-end-t").textContent = `${formatTime((Number(this.end.value) + 1) / this.fps)} · 長度 ${formatDuration(Math.max(0, dur))}`;
  }

  #apply() {
    const start = Number(this.start.value), end = Number(this.end.value);
    const state = this.form.querySelector("input[name=state]:checked")?.value;
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < start) {
      toast("起點／終點幀不正確：必須是整數，且終點 ≥ 起點", "error");
      return;
    }
    this.h.onApply(this.seg, { start, end, state });
  }
}

export class SegmentList {
  /** @param {{onPick:(seg:object)=>void}} h */
  constructor(h) {
    this.h = h;
    this.list = $("seg-list");
    this.filter = $("list-filter");
    this.filter.addEventListener("change", () => this.render(this.last));
    this.last = { segments: [], selected: null, fps: 30 };
  }

  render({ segments, selected, fps }) {
    this.last = { segments, selected, fps };
    this.#stats(segments, fps);
    const mode = this.filter.value;
    const shown = segments.filter((s) => mode === "all" || kindOf(s) === mode);
    const rows = shown.map((seg) => {
      const kind = kindOf(seg);
      const li = el("li", { className: `${kind}${sameRange(seg, selected) ? " selected" : ""}` },
        el("span", { className: "dot", style: `background:${COLORS[seg.state]}` }),
        el("span", { textContent: seg.state }),
        el("span", { className: "range",
          textContent: `${formatTime(seg.start_frame / fps)} → ${formatTime((seg.end_frame + 1) / fps)}` }),
        el("span", { className: "src",
          textContent: `${KIND_LABEL[kind]} · ${formatDuration((seg.end_frame - seg.start_frame + 1) / fps)}` }));
      li.addEventListener("click", () => this.h.onPick(seg));
      return li;
    });
    this.list.replaceChildren(...rows);
    this.list.querySelector(".selected")?.scrollIntoView({ block: "nearest" });
  }

  #stats(segments, fps) {
    const parts = ["service", "play", "no-play"].map((state) => {
      const segs = segments.filter((s) => s.state === state);
      const secs = segs.reduce((acc, s) => acc + (s.end_frame - s.start_frame + 1), 0) / fps;
      return el("span", {}, el("span", { className: "dot", style: `background:${COLORS[state]}` }),
        `${state} `, el("b", { textContent: `${segs.length}` }), ` · ${formatDuration(secs)}`);
    });
    $("stats").replaceChildren(...parts);
  }
}

export class Picker {
  /** Dialog listing videos already in the DB, plus a free-text path box. @param {{onPick:(path:string)=>void}} h */
  constructor(h) {
    this.dialog = $("picker");
    this.input = $("pick-path");
    this.dialog.addEventListener("close", () => {
      if (this.dialog.returnValue === "open" && this.input.value.trim()) h.onPick(this.input.value.trim());
    });
  }

  async open() {
    this.dialog.showModal();
    this.input.focus();
    const ul = $("pick-media");
    try {
      const media = await api.media();
      const items = media.map((m) => {
        const li = el("li", { className: "vid" }, el("span", { textContent: m.path }),
          el("span", { className: "muted", textContent: `手動 ${m.manual} / 共 ${m.segments} 段` }));
        li.addEventListener("click", () => { this.input.value = m.path; this.dialog.close("open"); });
        return li;
      });
      ul.replaceChildren(...(items.length ? items : [el("li", { className: "empty-item", textContent: "（還沒有）" })]));
    } catch (err) {
      ul.replaceChildren(el("li", { className: "empty-item", textContent: err.message }));
    }
  }
}
