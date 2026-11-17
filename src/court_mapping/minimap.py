"""2D bird's-eye court rendering and overlay onto video frames."""
from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from court_mapping.court_geometry import ATTACK_LINE_OFFSET_M, COURT_LENGTH_M, COURT_WIDTH_M, NET_Y_M
from court_mapping.projection import PlayerPosition

BGR = tuple[int, int, int]


@dataclass(frozen=True)
class MinimapStyle:
    pixels_per_metre: float = 20.0
    padding_m: float = 1.5
    floor_color: BGR = (60, 110, 200)
    surround_color: BGR = (40, 70, 40)
    line_color: BGR = (255, 255, 255)
    net_color: BGR = (40, 40, 40)
    far_team_color: BGR = (255, 160, 0)
    near_team_color: BGR = (0, 0, 230)
    player_radius_px: int = 7
    held_opacity: float = 0.4               # players not detected this frame (last known position)
    airborne_ring_color: BGR = (0, 230, 255)  # ring around jumping players (frozen position)
    opacity: float = 0.85
    frame_margin_px: int = 20


def render_minimap(players: Iterable[PlayerPosition], style: MinimapStyle) -> np.ndarray:
    """Top-down court (far end at the top) with one labelled dot per player."""
    court = _draw_court(style)
    for player in players:
        _draw_player(court, player, style)
    return court


def overlay_minimap(frame: np.ndarray, minimap: np.ndarray, style: MinimapStyle = MinimapStyle()) -> np.ndarray:
    """Return a copy of `frame` with the minimap blended into the top-right corner."""
    fh, fw = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    m = style.frame_margin_px
    if mh + 2 * m > fh or mw + 2 * m > fw:
        raise ValueError(f"Minimap {mw}x{mh} does not fit into frame {fw}x{fh}; reduce pixels_per_metre.")

    out = frame.copy()
    y0, x0 = m, fw - mw - m
    region = out[y0:y0 + mh, x0:x0 + mw]
    out[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(minimap, style.opacity, region, 1 - style.opacity, 0)
    return out


def court_to_minimap_px(x_m: float, y_m: float, style: MinimapStyle) -> tuple[int, int]:
    s, p = style.pixels_per_metre, style.padding_m
    return int(round((x_m + p) * s)), int(round((y_m + p) * s))


def _draw_court(style: MinimapStyle) -> np.ndarray:
    s, p = style.pixels_per_metre, style.padding_m
    height = int(round((COURT_LENGTH_M + 2 * p) * s))
    width = int(round((COURT_WIDTH_M + 2 * p) * s))
    img = np.full((height, width, 3), style.surround_color, dtype=np.uint8)

    top_left = court_to_minimap_px(0, 0, style)
    bottom_right = court_to_minimap_px(COURT_WIDTH_M, COURT_LENGTH_M, style)
    cv2.rectangle(img, top_left, bottom_right, style.floor_color, thickness=-1)
    cv2.rectangle(img, top_left, bottom_right, style.line_color, thickness=2)

    for y_m in (NET_Y_M - ATTACK_LINE_OFFSET_M, NET_Y_M + ATTACK_LINE_OFFSET_M):
        cv2.line(img, court_to_minimap_px(0, y_m, style), court_to_minimap_px(COURT_WIDTH_M, y_m, style),
                 style.line_color, 1)
    cv2.line(img, court_to_minimap_px(-0.5, NET_Y_M, style), court_to_minimap_px(COURT_WIDTH_M + 0.5, NET_Y_M, style),
             style.net_color, 3)
    return img


def _draw_player(img: np.ndarray, player: PlayerPosition, style: MinimapStyle) -> None:
    """Draws onto `img`, which is always a canvas freshly created by render_minimap."""
    team_color = style.far_team_color if player.side == "far" else style.near_team_color
    alpha = 1.0 if player.observed else style.held_opacity
    color = _mix(team_color, style.floor_color, alpha)
    outline = _mix((255, 255, 255), style.floor_color, alpha)
    center = court_to_minimap_px(player.x_m, player.y_m, style)

    cv2.circle(img, center, style.player_radius_px, color, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(img, center, style.player_radius_px, outline, thickness=1, lineType=cv2.LINE_AA)
    if player.airborne:
        cv2.circle(img, center, style.player_radius_px + 4, style.airborne_ring_color, 2, cv2.LINE_AA)
    cv2.putText(img, str(player.tracker_id), (center[0] + style.player_radius_px, center[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, outline, 1, cv2.LINE_AA)


def _mix(color: BGR, background: BGR, alpha: float) -> BGR:
    """Colour of `color` drawn at `alpha` opacity over `background`."""
    return tuple(int(round(alpha * c + (1 - alpha) * b)) for c, b in zip(color, background))
