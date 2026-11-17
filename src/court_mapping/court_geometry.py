"""Court plane coordinate system (metres). Origin: far end line, left side (as seen in the image)."""
from types import MappingProxyType

COURT_WIDTH_M = 9.0
COURT_LENGTH_M = 18.0
NET_Y_M = COURT_LENGTH_M / 2
ATTACK_LINE_OFFSET_M = 3.0

# Only points lying on the floor may be used for the homography.
# The calibration GUI's net_top_* / net_bottom_* points are in the air and are excluded.
GROUND_POINTS_M = MappingProxyType({
    "far_end_line_left": (0.0, 0.0),
    "far_end_line_right": (COURT_WIDTH_M, 0.0),
    "far_attack_line_left": (0.0, NET_Y_M - ATTACK_LINE_OFFSET_M),
    "far_attack_line_right": (COURT_WIDTH_M, NET_Y_M - ATTACK_LINE_OFFSET_M),
    "middle_line_left": (0.0, NET_Y_M),
    "middle_line_right": (COURT_WIDTH_M, NET_Y_M),
    "near_attack_line_left": (0.0, NET_Y_M + ATTACK_LINE_OFFSET_M),
    "near_attack_line_right": (COURT_WIDTH_M, NET_Y_M + ATTACK_LINE_OFFSET_M),
    "near_end_line_left": (0.0, COURT_LENGTH_M),
    "near_end_line_right": (COURT_WIDTH_M, COURT_LENGTH_M),
})

MIN_POINTS_FOR_HOMOGRAPHY = 4
