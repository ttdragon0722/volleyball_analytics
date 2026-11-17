// Run: node --test tests/videomae_labeler/
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  compose, deleteSegment, editSegment, fillGaps, kindOf, mergeSameState, normalize, paint, seg, splitLayers,
} from "../../src/videomae_labeler/static/js/ops.js";

const triples = (segs) => segs.map((s) => [s.start_frame, s.end_frame, s.state, kindOf(s)]);
const M = (s, e, st, c = 0.8) => seg(s, e, st, "model", c);
const U = (s, e, st) => seg(s, e, st, "manual", 1);

test("empty video is all default no-play", () => {
  assert.deepEqual(triples(compose([], [], 100)), [[0, 99, "no-play", "default"]]);
});

test("model play/service kept, model no-play and gaps become default no-play", () => {
  const model = [M(0, 9, "no-play"), M(10, 19, "service"), M(20, 49, "play")];
  assert.deepEqual(triples(compose(model, [], 60)), [
    [0, 9, "no-play", "default"], [10, 19, "service", "model"], [20, 49, "play", "model"], [50, 59, "no-play", "default"]]);
});

test("user labels win over overlapping model segments", () => {
  const model = [M(10, 49, "play")];
  const manual = [U(30, 39, "service"), U(45, 55, "no-play")];
  assert.deepEqual(triples(compose(model, manual, 60)), [
    [0, 9, "no-play", "default"], [10, 29, "play", "model"], [30, 39, "service", "manual"],
    [40, 44, "play", "model"], [45, 59, "no-play", "manual"]]);   // 45-55 manual + 56-59 default merged
});

test("touching same-state segments merge: manual wins, confidence length-weighted", () => {
  const model = [M(0, 89, "service", 0.6)];
  const manual = [U(21, 39, "service")];
  const out = compose(model, manual, 100);
  assert.deepEqual(triples(out), [[0, 89, "service", "manual"], [90, 99, "no-play", "default"]]);
  assert.ok(Math.abs(out[0].confidence - (0.6 * 71 + 1 * 19) / 90) < 1e-9);
  const modelOnly = mergeSameState([M(0, 9, "play", 0.5), M(10, 29, "play", 0.8)]);
  assert.deepEqual(triples(modelOnly), [[0, 29, "play", "model"]]);
  assert.ok(Math.abs(modelOnly[0].confidence - 0.7) < 1e-9);
});

test("compose clamps to frame count", () => {
  assert.deepEqual(triples(compose([M(5, 500, "play")], [], 10)), [[0, 4, "no-play", "default"], [5, 9, "play", "model"]]);
});

test("paint overwrites and merges touching manual labels", () => {
  let layer = paint([], 0, 9, "play");
  layer = paint(layer, 10, 19, "play");
  layer = paint(layer, 5, 12, "service");
  assert.deepEqual(triples(layer), [[0, 4, "play", "manual"], [5, 12, "service", "manual"], [13, 19, "play", "manual"]]);
});

test("delete turns the whole selected range into default no-play; default cannot be deleted", () => {
  const layers = { model: [M(0, 49, "play")], manual: [U(10, 19, "service")] };
  const visible = compose(layers.model, layers.manual, 60);
  const afterManual = deleteSegment(layers, visible[1]);
  assert.deepEqual(triples(compose(afterManual.model, afterManual.manual, 60)), [
    [0, 9, "play", "model"], [10, 19, "no-play", "default"], [20, 49, "play", "model"], [50, 59, "no-play", "default"]]);
  const afterModel = deleteSegment(layers, visible[0]);
  assert.deepEqual(triples(compose(afterModel.model, afterModel.manual, 60)), [
    [0, 9, "no-play", "default"], [10, 19, "service", "manual"], [20, 49, "play", "model"], [50, 59, "no-play", "default"]]);
  assert.equal(deleteSegment(layers, visible[3]), null);
});

test("deleting a merged manual+model segment clears both layers", () => {
  const layers = { model: [M(0, 89, "service")], manual: [U(21, 39, "service")] };
  const merged = compose(layers.model, layers.manual, 90)[0];
  const after = deleteSegment(layers, merged);
  assert.deepEqual(triples(compose(after.model, after.manual, 90)), [[0, 89, "no-play", "default"]]);
});

test("edit turns any segment into a manual label", () => {
  const layers = { model: [M(10, 29, "play")], manual: [] };
  const target = compose(layers.model, layers.manual, 40)[1];
  const next = editSegment(layers, target, { start: 12, end: 25, state: "service" });
  assert.deepEqual(triples(compose(next.model, next.manual, 40)), [
    [0, 11, "no-play", "default"], [12, 25, "service", "manual"], [26, 39, "no-play", "default"]]);
});

test("splitLayers round-trips a saved timeline", () => {
  const saved = compose([M(10, 29, "play")], [U(20, 24, "service")], 40);
  const { model, manual } = splitLayers(saved);
  assert.deepEqual(compose(model, manual, 40), saved);
});

test("normalize rejects overlaps and inputs are not mutated", () => {
  assert.throws(() => normalize([U(0, 10, "play"), U(5, 15, "play")]));
  const layer = Object.freeze([U(0, 9, "play")]);
  paint(layer, 2, 3, "service");
  assert.deepEqual(triples(layer), [[0, 9, "play", "manual"]]);
  assert.deepEqual(triples(fillGaps([], 0)), []);
});

test("overlapping segments from older vb_annotator data are resolved on load", () => {
  const db = [
    M(8824, 9268, "play", 1), M(9240, 9479, "play", 1),          // model play overlapping model play
    U(14104, 14343, "service"), U(14343, 14652, "play"),          // manual overlap by one frame
    U(18845, 19504, "no-play"), M(18900, 19019, "play", 0.99),    // manual vs model
  ];
  const { model, manual } = splitLayers(db);
  const out = compose(model, manual, 20000);
  assert.doesNotThrow(() => normalize(out));
  const at = (f) => out.find((s) => s.start_frame <= f && f <= s.end_frame);
  assert.equal(at(9250).state, "play");
  assert.equal(at(14343).state, "play");                          // later-starting manual wins
  assert.equal(at(14342).state, "service");
  assert.equal(at(18950).state, "no-play");                       // manual beats model
  assert.equal(kindOf(at(18950)), "manual");
});
