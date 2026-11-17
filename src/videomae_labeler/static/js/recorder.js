// Hold-to-label state machine. No DOM access, so it can be unit tested with Node.
//
// A label is only recorded while playback runs AND a state key (1/2/3) is held.
// The run covers every frame shown while the key is down, inclusive:
//   key down at frame a ... key up at frame b  ->  commit(a, b, state)
// Holding a second key switches state at that frame (the newest key wins);
// releasing it falls back to the key still held. Pausing or seeking closes the
// run; if the key is still held when playback continues, a new run starts.

export const STATES = Object.freeze(["service", "play", "no-play"]);

export class HoldRecorder {
  /** @param {{commit:(start:number, end:number, state:string)=>void, onChange?:()=>void}} deps */
  constructor({ commit, onChange = () => {} }) {
    this.commit = commit;
    this.onChange = onChange;
    this.held = Object.freeze([]);   // held state keys, newest last
    this.playing = false;
    this.runStart = null;
  }

  /** The state being labelled right now (newest held key), or null. */
  get active() { return this.held.at(-1) ?? null; }
  get recording() { return this.runStart !== null; }

  keyDown(state, frame) {
    if (!STATES.includes(state)) throw new Error(`unknown state ${state}`);
    if (this.held.includes(state)) return;
    this.#close(frame - 1);
    this.held = Object.freeze([...this.held, state]);
    this.#open(frame);
    this.onChange();
  }

  keyUp(state, frame) {
    if (!this.held.includes(state)) return;
    const wasActive = this.active === state;
    if (wasActive) this.#close(frame);
    this.held = Object.freeze(this.held.filter((s) => s !== state));
    if (wasActive) this.#open(frame + 1);
    this.onChange();
  }

  releaseAll(frame) {
    this.#close(frame);
    this.held = Object.freeze([]);
    this.onChange();
  }

  play(frame) {
    this.playing = true;
    this.#open(frame);
    this.onChange();
  }

  /** Playback paused / ended at `frame` (the last frame that was shown). */
  pause(frame) {
    this.#close(frame);
    this.playing = false;
    this.onChange();
  }

  /** Seek while playing: close at the last shown frame, reopen at the new one. */
  seek(fromFrame, toFrame) {
    this.#close(fromFrame);
    this.#open(toFrame);
    this.onChange();
  }

  /** Open run for a live preview: {start, end, state} or null. */
  preview(frame) {
    if (!this.recording || frame < this.runStart) return null;
    return { start: this.runStart, end: frame, state: this.runState };
  }

  #open(frame) {
    if (this.playing && this.active && this.runStart === null) {
      this.runStart = frame;
      this.runState = this.active;
    }
  }

  #close(endFrame) {
    if (this.runStart === null) return;
    if (endFrame >= this.runStart) this.commit(this.runStart, endFrame, this.runState);
    this.runStart = null;
  }
}
