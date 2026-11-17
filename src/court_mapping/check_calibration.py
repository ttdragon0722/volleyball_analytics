"""
Check a saved court calibration before running the full pipeline.

Prints per-point reprojection error (acceptance: < 5 px at 1080p) and writes a
diagnostic image: the calibration frame with annotated (red) vs reprojected
(green) points, next to a bird's-eye warp of the frame.

Usage (from src/):
    python -m court_mapping.check_calibration --db ../db/calibration.db --video path/to/video.mp4
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from court_mapping.calibration import provider_from_calibration
from court_mapping.court_geometry import COURT_LENGTH_M, COURT_WIDTH_M, GROUND_POINTS_M
from court_mapping.homography import CalibrationError, CourtHomography
from court_mapping.minimap import MinimapStyle, court_to_minimap_px, render_minimap

ACCEPTABLE_ERROR_PX = 5.0
REFERENCE_HEIGHT_PX = 1080

logger = logging.getLogger(__name__)


def error_report(homography: CourtHomography, frame_height: int) -> tuple[str, bool]:
    """Human-readable error table; threshold scales with resolution relative to 1080p."""
    threshold = ACCEPTABLE_ERROR_PX * frame_height / REFERENCE_HEIGHT_PX
    lines = [f"{'point':<24}{'error_px':>10}  inlier"]
    for name, err in homography.reprojection_errors_px.items():
        lines.append(f"{name:<24}{err:>10.2f}  {'yes' if name in homography.inlier_names else 'NO'}")
    passed = homography.max_error_px < threshold
    lines.append(f"mean={homography.mean_error_px:.2f}px max={homography.max_error_px:.2f}px "
                 f"threshold={threshold:.2f}px -> {'PASS' if passed else 'FAIL'}")
    return "\n".join(lines), passed


def render_diagnostic(frame: np.ndarray, points: dict, homography: CourtHomography,
                      style: MinimapStyle = MinimapStyle()) -> np.ndarray:
    annotated = frame.copy()
    for name, (x_m, y_m) in GROUND_POINTS_M.items():
        if name in points:
            cv2.circle(annotated, tuple(int(v) for v in points[name]), 8, (0, 0, 255), 2)
        rx, ry = homography.to_image(np.array([[x_m, y_m]]))[0]
        cv2.drawMarker(annotated, (int(rx), int(ry)), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)

    court_template = render_minimap((), style)
    h, w = court_template.shape[:2]
    # image -> minimap pixels = (court metres -> minimap pixels) @ image_to_court
    s, p = style.pixels_per_metre, style.padding_m
    metres_to_px = np.array([[s, 0, p * s], [0, s, p * s], [0, 0, 1]], dtype=np.float64)
    warped = cv2.warpPerspective(frame, metres_to_px @ homography.image_to_court, (w, h))
    _draw_court_outline(warped, style)

    scale = annotated.shape[0] / h
    warped_resized = cv2.resize(warped, (int(w * scale), annotated.shape[0]))
    return np.hstack([annotated, warped_resized])


def _draw_court_outline(img: np.ndarray, style: MinimapStyle) -> None:
    cv2.rectangle(img, court_to_minimap_px(0, 0, style),
                  court_to_minimap_px(COURT_WIDTH_M, COURT_LENGTH_M, style), (0, 255, 255), 1)


def _read_frame(video_path: str, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} from {video_path}")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", required=True, help="CalibrationDB sqlite path")
    parser.add_argument("--video", required=True, help="video_name as stored by the calibration GUI")
    parser.add_argument("--out", help="diagnostic image path (default: <video>_calibration_check.png)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from vb_gui.calibration.database import CalibrationDB  # heavy import (sqlalchemy) only for the CLI

    record = CalibrationDB(db_path=args.db).load(args.video)
    if record is None:
        logger.error("No calibration stored for %s in %s", args.video, args.db)
        return 2
    try:
        homography = provider_from_calibration(record).homography
    except CalibrationError as exc:
        logger.error("Invalid calibration: %s", exc)
        return 2

    frame = _read_frame(args.video, int(record.get("frame_index", 0)))
    report, passed = error_report(homography, frame.shape[0])
    logger.info(report)

    out = Path(args.out or Path(args.video).with_suffix("").as_posix() + "_calibration_check.png")
    cv2.imwrite(out.as_posix(), render_diagnostic(frame, record["points"], homography))
    logger.info("Diagnostic image: %s", out)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
