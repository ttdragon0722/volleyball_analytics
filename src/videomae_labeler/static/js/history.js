// Immutable undo / redo history of the working layers.
const MAX_DEPTH = 200;

export class History {
  constructor(current, past = [], future = []) {
    this.current = current;
    this.past = Object.freeze(past);
    this.future = Object.freeze(future);
    Object.freeze(this);
  }

  get canUndo() { return this.past.length > 0; }
  get canRedo() { return this.future.length > 0; }

  push(next) {
    if (next === this.current) return this;
    return new History(next, [...this.past, this.current].slice(-MAX_DEPTH), []);
  }

  undo() {
    if (!this.canUndo) return this;
    return new History(this.past.at(-1), this.past.slice(0, -1), [this.current, ...this.future]);
  }

  redo() {
    if (!this.canRedo) return this;
    return new History(this.future[0], [...this.past, this.current], this.future.slice(1));
  }
}
