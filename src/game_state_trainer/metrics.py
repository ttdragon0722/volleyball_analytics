"""Classification metrics (pure numpy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Report:
    classes: tuple[str, ...]
    confusion: tuple[tuple[int, ...], ...]   # rows = true, cols = predicted
    precision: tuple[float, ...]
    recall: tuple[float, ...]
    f1: tuple[float, ...]
    accuracy: float
    macro_f1: float
    support: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy, "macro_f1": self.macro_f1,
            "per_class": {c: {"precision": p, "recall": r, "f1": f, "support": s}
                          for c, p, r, f, s in zip(self.classes, self.precision, self.recall, self.f1, self.support)},
            "confusion": {"rows_true_cols_pred": list(self.classes), "matrix": [list(r) for r in self.confusion]},
        }


def evaluate(y_true, y_pred, classes: tuple[str, ...]) -> Report:
    t, p = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if t.shape != p.shape:
        raise ValueError("y_true and y_pred must have the same length")
    k = len(classes)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (t, p), 1)
    tp = np.diag(cm).astype(float)
    pred_tot, true_tot = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_tot, out=np.zeros(k), where=pred_tot > 0)
    recall = np.divide(tp, true_tot, out=np.zeros(k), where=true_tot > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    present = true_tot > 0
    return Report(
        classes=tuple(classes),
        confusion=tuple(tuple(int(x) for x in row) for row in cm),
        precision=tuple(round(float(x), 4) for x in precision),
        recall=tuple(round(float(x), 4) for x in recall),
        f1=tuple(round(float(x), 4) for x in f1),
        accuracy=round(float(tp.sum() / max(1, cm.sum())), 4),
        macro_f1=round(float(f1[present].mean()) if present.any() else 0.0, 4),
        support=tuple(int(x) for x in true_tot),
    )


def format_report(rep: Report, title: str) -> str:
    lines = [f"{title}  accuracy {rep.accuracy:.3f}  macro-F1 {rep.macro_f1:.3f}",
             f"  {'class':8} {'prec':>6} {'recall':>6} {'f1':>6} {'n':>6}"]
    for c, p, r, f, s in zip(rep.classes, rep.precision, rep.recall, rep.f1, rep.support):
        lines.append(f"  {c:8} {p:6.3f} {r:6.3f} {f:6.3f} {s:6d}")
    header = "  true\\pred " + " ".join(f"{c:>8}" for c in rep.classes)
    lines.append(header)
    for c, row in zip(rep.classes, rep.confusion):
        lines.append(f"  {c:9} " + " ".join(f"{x:8d}" for x in row))
    return "\n".join(lines)
