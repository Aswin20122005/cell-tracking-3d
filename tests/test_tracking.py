import numpy as np

from src.tracking.linker import link_frames


def test_two_cells_moving_slowly_give_two_tracks():
    frames = [np.array([[5, 10, 10], [5, 40, 40]]) + t for t in range(4)]
    tracks = link_frames(frames, max_distance=5, min_track_length=2)
    assert len(tracks) == 2
    assert all(tr["frames"] == [0, 1, 2, 3] for tr in tracks)


def test_far_jump_starts_a_new_track():
    frames = [np.array([[5, 10, 10]]), np.array([[5, 50, 50]])]
    tracks = link_frames(frames, max_distance=5, min_track_length=1)
    assert len(tracks) == 2


def test_short_tracks_are_filtered():
    frames = [np.array([[5, 10, 10]]), np.empty((0, 3)), np.empty((0, 3))]
    assert link_frames(frames, max_distance=5, min_track_length=2) == []


def test_one_missed_frame_is_bridged():
    frames = [np.array([[5, 10, 10]]), np.empty((0, 3)), np.array([[5, 11, 10]])]
    tracks = link_frames(frames, max_distance=5, min_track_length=2, max_gap=1)
    assert len(tracks) == 1 and tracks[0]["frames"] == [0, 2]


def test_track_missing_too_long_is_not_resumed():
    frames = [np.array([[5, 10, 10]])] + [np.empty((0, 3))] * 5 + [np.array([[5, 10, 10]])]
    tracks = link_frames(frames, max_distance=5, min_track_length=1, max_gap=1)
    assert len(tracks) == 2


def test_assignment_is_one_to_one():
    frames = [np.array([[5, 10, 10], [5, 12, 10]]), np.array([[5, 11, 10]])]
    tracks = link_frames(frames, max_distance=5, min_track_length=2)
    assert len(tracks) == 1
