// Pure, immutable timeline operations (no DOM). Frames are inclusive on both ends.
//
// The working document has two layers:
//   model  – the model's play / service predictions (source "model")
//   manual – what the user labelled by holding 1 / 2 / 3 (source "manual")
// The timeline that is shown and saved is compose(model, manual, frameCount):
// manual wins wherever it overlaps the model, and every frame not covered by
// either layer becomes a DEFAULT no-play segment (source "model", confidence 0).
// Finally, touching segments with the same state are merged into one: the result
// is "manual" if any part was manual (manual overrides model), and its confidence
// is the length-weighted mean of the parts. The server applies the same rule.

export const STATES = Object.freeze(["service", "play", "no-play"]);
export const ACTIVE_MODEL_STATES = Object.freeze(["service", "play"]);
export const DEFAULT_CONFIDENCE = 0;

/** @typedef {{start_frame:number, end_frame:number, state:string, source:string, confidence:number}} Segment */

/** @returns {Segment} */
export function seg(start, end, state, source = "manual", confidence = source === "manual" ? 1 : 0.5) {
  return Object.freeze({ start_frame: start, end_frame: end, state, source, confidence });
}

export const isDefault = (s) => s.source === "model" && s.state === "no-play" && s.confidence === DEFAULT_CONFIDENCE;
export const kindOf = (s) => (s.source === "manual" ? "manual" : isDefault(s) ? "default" : "model");
export const sameRange = (a, b) => !!a && !!b && a.start_frame === b.start_frame && a.end_frame === b.end_frame;

const byStart = (a, b) => a.start_frame - b.start_frame || a.end_frame - b.end_frame;

/** Sort, reject overlaps, merge touching manual segments with the same state. */
export function normalize(segments) {
  const out = [];
  for (const s of [...segments].sort(byStart)) {
    const prev = out.at(-1);
    if (prev && s.start_frame <= prev.end_frame) {
      throw new Error(`overlapping segments [${prev.start_frame},${prev.end_frame}] and [${s.start_frame},${s.end_frame}]`);
    }
    if (prev && prev.source === "manual" && s.source === "manual" && prev.state === s.state
        && prev.end_frame + 1 === s.start_frame) {
      out[out.length - 1] = seg(prev.start_frame, s.end_frame, s.state, "manual", prev.confidence);
    } else {
      out.push(s);
    }
  }
  return Object.freeze(out);
}

/** Remove every covered frame in [start, end], trimming / splitting overlaps. */
export function clearRange(segments, start, end) {
  const [a, b] = start <= end ? [start, end] : [end, start];
  const out = [];
  for (const s of segments) {
    if (s.end_frame < a || s.start_frame > b) { out.push(s); continue; }
    if (s.start_frame < a) out.push({ ...s, end_frame: a - 1 });
    if (s.end_frame > b) out.push({ ...s, start_frame: b + 1 });
  }
  return normalize(out.map(Object.freeze));
}

/** Paint [start, end] as a manual `state` on a layer, overwriting what was there. */
export function paint(layer, start, end, state) {
  if (!STATES.includes(state)) throw new Error(`unknown state ${state}`);
  const [a, b] = start <= end ? [start, end] : [end, start];
  return normalize([...clearRange(layer, a, b), seg(a, b, state, "manual", 1)]);
}

/** Keep only frames in [0, frameCount). */
export function clamp(segments, frameCount) {
  return clearRange(segments, frameCount, Number.MAX_SAFE_INTEGER);
}

/** Fill every uncovered frame in [0, frameCount) with a default no-play segment. */
export function fillGaps(segments, frameCount) {
  const out = [];
  let next = 0;
  for (const s of normalize(segments)) {
    if (s.start_frame > next) out.push(seg(next, s.start_frame - 1, "no-play", "model", DEFAULT_CONFIDENCE));
    out.push(s);
    next = s.end_frame + 1;
  }
  if (next < frameCount) out.push(seg(next, frameCount - 1, "no-play", "model", DEFAULT_CONFIDENCE));
  return Object.freeze(out);
}

const length = (s) => s.end_frame - s.start_frame + 1;

/** Merge two touching same-state segments (see the rule at the top of this file). */
function mergePair(a, b) {
  const source = a.source === "manual" || b.source === "manual" ? "manual" : "model";
  const confidence = (a.confidence * length(a) + b.confidence * length(b)) / (length(a) + length(b));
  return seg(a.start_frame, b.end_frame, a.state, source, confidence);
}

/** Merge every run of touching segments that share a state (input sorted, non-overlapping). */
export function mergeSameState(segments) {
  const out = [];
  for (const s of segments) {
    const prev = out.at(-1);
    if (prev && prev.state === s.state && prev.end_frame + 1 === s.start_frame) out[out.length - 1] = mergePair(prev, s);
    else out.push(s);
  }
  return Object.freeze(out);
}

/** The visible / saved timeline: manual over model, gaps as default no-play, same states merged. */
export function compose(model, manual, frameCount) {
  let base = model.filter((s) => ACTIVE_MODEL_STATES.includes(s.state));
  for (const m of manual) base = clearRange(base, m.start_frame, m.end_frame);
  return mergeSameState(fillGaps(clamp([...base, ...manual], frameCount), frameCount));
}

/**
 * Make possibly-overlapping segments non-overlapping: in start order, each segment
 * overwrites the frames it shares with earlier ones (so the later-starting one wins).
 * Older vb_annotator versions could store overlaps (e.g. play extended 1 s backwards).
 */
export function resolveOverlaps(segments) {
  let out = Object.freeze([]);
  for (const s of [...segments].sort(byStart)) {
    const prev = out.at(-1);
    out = prev && s.start_frame <= prev.end_frame
      ? normalize([...clearRange(out, s.start_frame, s.end_frame), s])
      : normalize([...out, s]);
  }
  return out;
}

/**
 * Split segments loaded from the DB back into layers (default fills are dropped).
 * Overlaps are resolved per layer; manual beats model later in compose().
 */
export function splitLayers(segments) {
  return {
    model: resolveOverlaps(segments.filter((s) => s.source === "model" && ACTIVE_MODEL_STATES.includes(s.state))),
    manual: resolveOverlaps(segments.filter((s) => s.source === "manual")),
  };
}

/**
 * Delete what the user clicked: its whole range becomes default no-play, in both
 * layers. Returns null when there is nothing to delete (a default no-play segment).
 */
export function deleteSegment(layers, target) {
  if (kindOf(target) === "default") return null;
  const { start_frame: a, end_frame: b } = target;
  return { model: clearRange(layers.model, a, b), manual: clearRange(layers.manual, a, b) };
}

/** Change range and/or state of a visible segment; the result is always a manual label. */
export function editSegment(layers, target, { start, end, state }) {
  const removed = deleteSegment(layers, target) ?? layers;
  return { ...removed, manual: paint(removed.manual, start, end, state) };
}
