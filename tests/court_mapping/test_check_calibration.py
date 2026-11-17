import numpy as np

from court_mapping.check_calibration import error_report, render_diagnostic
from court_mapping.homography import fit_homography


def test_error_report_pass_and_fail(calibration_points):
    good = fit_homography(calibration_points)
    report, passed = error_report(good, 1080)
    assert passed and "PASS" in report

    # Off by ~20 px but within a loose RANSAC threshold -> still an inlier, fails acceptance.
    shifted = {**calibration_points, "middle_line_left": [calibration_points["middle_line_left"][0] + 20,
                                                          calibration_points["middle_line_left"][1]]}
    report, passed = error_report(fit_homography(shifted, ransac_threshold_px=100), 1080)
    assert not passed and "FAIL" in report


def test_render_diagnostic_shape(calibration_points):
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    out = render_diagnostic(frame, calibration_points, fit_homography(calibration_points))
    assert out.shape[0] == 1080 and out.shape[1] > 1920
    assert frame.mean() == 128
