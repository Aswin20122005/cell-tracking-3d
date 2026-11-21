"""Link per-frame detections into tracks across time.

Optimal (Hungarian) nearest-neighbour assignment between consecutive frames, capped
by a max link distance and a max gap, followed by short-track removal to drop
transient false positives.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def link_frames(detections_per_frame, max_distance=10.0, min_track_length=3, max_gap=1):
    """Link detections across frames into tracks.

    Parameters
    ----------
    detections_per_frame : list[np.ndarray]
        One (N_t, 3) array of (z, y, x) detections per timepoint.
    max_gap : int
        How many consecutive frames a track may miss (a cell not detected) and still
        be continued. Tracks that have been missing longer are closed, so a new cell
        appearing near where an old one vanished long ago isn't wrongly joined to it.

    Returns
    -------
    list[dict] : each track as {"id": int, "frames": [t...], "coords": [(z,y,x)...]}
    """
    tracks = []
    next_id = 0

    for t, dets in enumerate(detections_per_frame):
        dets = np.asarray(dets).reshape(-1, 3)
        # Only tracks seen within the allowed gap can be extended.
        active = [i for i, tr in enumerate(tracks) if t - tr["frames"][-1] <= max_gap + 1]
        prev_coords = np.array([tracks[i]["coords"][-1] for i in active]).reshape(-1, 3)
        assigned_active, assigned_dets = _assign(prev_coords, dets, max_distance)

        for ai, di in zip(assigned_active, assigned_dets):
            tracks[active[ai]]["frames"].append(t)
            tracks[active[ai]]["coords"].append(tuple(dets[di]))

        # unmatched detections start new tracks
        matched = set(assigned_dets)
        for di in range(len(dets)):
            if di not in matched:
                tracks.append({"id": next_id, "frames": [t], "coords": [tuple(dets[di])]})
                next_id += 1

    return [tr for tr in tracks if len(tr["frames"]) >= min_track_length]


def _assign(prev_coords, dets, max_distance):
    if len(prev_coords) == 0 or len(dets) == 0:
        return [], []
    cost = np.linalg.norm(prev_coords[:, None, :] - dets[None, :, :], axis=2)
    cost[cost > max_distance] = 1e6
    rows, cols = linear_sum_assignment(cost)
    keep_t, keep_d = [], []
    for r, c in zip(rows, cols):
        if cost[r, c] < 1e6:
            keep_t.append(r)
            keep_d.append(c)
    return keep_t, keep_d
