// Run: node --test tests/videomae_labeler/
import { test } from "node:test";
import assert from "node:assert/strict";
import { HoldRecorder } from "../../src/videomae_labeler/static/js/recorder.js";

function make() {
  const commits = [];
  const rec = new HoldRecorder({ commit: (s, e, st) => commits.push([s, e, st]) });
  return { rec, commits };
}

test("labels only while a key is held during playback", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("service", 30);
  rec.keyUp("service", 59);
  rec.keyDown("play", 100);
  rec.keyUp("play", 299);
  rec.pause(400);
  assert.deepEqual(commits, [[30, 59, "service"], [100, 299, "play"]]);
});

test("nothing is recorded while paused", () => {
  const { rec, commits } = make();
  rec.keyDown("play", 10);
  rec.keyUp("play", 10);
  assert.deepEqual(commits, []);
});

test("holding a second key switches state; releasing it falls back", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("service", 10);
  rec.keyDown("play", 40);
  rec.keyUp("play", 69);
  rec.keyUp("service", 79);
  assert.deepEqual(commits, [[10, 39, "service"], [40, 69, "play"], [70, 79, "service"]]);
});

test("releasing a non-active key does not interrupt the active run", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("service", 0);
  rec.keyDown("play", 10);
  rec.keyUp("service", 20);
  rec.keyUp("play", 30);
  assert.deepEqual(commits, [[0, 9, "service"], [10, 30, "play"]]);
});

test("pause closes the run and resume while still held continues it", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("play", 5);
  rec.pause(20);
  rec.play(20);
  rec.keyUp("play", 30);
  assert.deepEqual(commits, [[5, 20, "play"], [20, 30, "play"]]);
});

test("key held before play starts recording on play", () => {
  const { rec, commits } = make();
  rec.keyDown("service", 0);
  rec.play(50);
  rec.keyUp("service", 60);
  assert.deepEqual(commits, [[50, 60, "service"]]);
});

test("seek while holding splits the run", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("play", 0);
  rec.seek(10, 500);
  rec.keyUp("play", 505);
  assert.deepEqual(commits, [[0, 10, "play"], [500, 505, "play"]]);
});

test("repeat keydown is ignored and releaseAll closes", () => {
  const { rec, commits } = make();
  rec.play(0);
  rec.keyDown("play", 1);
  rec.keyDown("play", 3);
  rec.releaseAll(9);
  assert.deepEqual(commits, [[1, 9, "play"]]);
  assert.equal(rec.active, null);
});

test("preview reflects the open run", () => {
  const { rec } = make();
  rec.play(0);
  assert.equal(rec.preview(5), null);
  rec.keyDown("service", 3);
  assert.deepEqual(rec.preview(9), { start: 3, end: 9, state: "service" });
});
