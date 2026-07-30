"""Link per-frame detections into tracks across time.

Greedy/optimal nearest-neighbour assignment between consecutive frames, capped by
a max link distance, followed by short-track removal to drop transient false
positives.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def link_frames(detections_per_frame, max_distance=10.0, min_track_length=3):
    """Link detections across frames into tracks.

    Parameters
    ----------
    detections_per_frame : list[np.ndarray]
        One (N_t, 3) array of (z, y, x) detections per timepoint.

    Returns
    -------
    list[dict] : each track as {"id": int, "frames": [t...], "coords": [(z,y,x)...]}
    """
    tracks = []            # active tracks: list of dicts
    next_id = 0

    for t, dets in enumerate(detections_per_frame):
        if t == 0:
            for d in dets:
                tracks.append({"id": next_id, "frames": [t], "coords": [tuple(d)]})
                next_id += 1
            continue

        prev_coords = np.array([tr["coords"][-1] for tr in tracks]) if tracks else np.empty((0, 3))
        assigned_tracks, assigned_dets = _assign(prev_coords, dets, max_distance)

        for ti, di in zip(assigned_tracks, assigned_dets):
            tracks[ti]["frames"].append(t)
            tracks[ti]["coords"].append(tuple(dets[di]))

        # unmatched detections start new tracks
        for di in range(len(dets)):
            if di not in assigned_dets:
                tracks.append({"id": next_id, "frames": [t], "coords": [tuple(dets[di])]})
                next_id += 1

    tracks = [tr for tr in tracks if len(tr["frames"]) >= min_track_length]
    return tracks


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
