import numpy as np

from jester_data import build_streams, fit_real_timing, scenario_of, synthesize_tracks, target_of
from sequence_utils import add_motion_features, motion_sequence, sliding_windows


def test_scenarios_and_targets():
    assert scenario_of("Swiping Left") == "intentional"
    assert target_of("Swiping Left") == "previous"
    assert scenario_of("No gesture") == "background"
    assert scenario_of("Thumb Up") == "near_miss"
    assert target_of("Thumb Up") == "none"
    assert scenario_of("Drumming Fingers") is None


def test_synthesized_track_reproduces_displacement():
    rng = np.random.default_rng(0)
    pose = rng.uniform(0.2, 0.8, size=(1, 21, 3)).astype(np.float32)
    motion = np.array([[0.3, -0.1, 0.0, 0.25, -0.05, 0.0, 0.0]], dtype=np.float32)
    track = synthesize_tracks(pose, motion, 16)[0].reshape(16, 21, 3)
    np.testing.assert_allclose(track[-1, 0] - track[0, 0], motion[0, 0:3], atol=1e-6)
    np.testing.assert_allclose(track[-1, 8] - track[0, 8], motion[0, 3:6], atol=1e-6)


def test_no_hand_video_stays_empty():
    track = synthesize_tracks(np.zeros((1, 21, 3), np.float32), np.ones((1, 7), np.float32), 8)
    assert not track.any()


def test_vectorised_motion_matches_realtime_rule():
    rng = np.random.default_rng(1)
    frames = rng.uniform(0.1, 0.9, size=(6, 63)).astype(np.float32)
    frames[2] = 0.0  # hand lost for one frame
    expected = []
    previous = None
    for frame in frames:
        expected.append(add_motion_features(frame, previous))
        previous = frame
    np.testing.assert_allclose(motion_sequence(frames), np.stack(expected), atol=1e-6)
    assert not motion_sequence(frames)[3, 63:].any()  # re-entry creates no fake jump


def test_sliding_windows_end_at_each_frame():
    features = np.arange(5, dtype=np.float32).reshape(1, 5, 1) + 1
    windows = sliding_windows(features, 3)
    assert windows.shape == (1, 5, 3, 1)
    assert windows[0, 0, :, 0].tolist() == [0, 0, 1]
    assert windows[0, 4, :, 0].tolist() == [3, 4, 5]


def test_fit_real_timing_keeps_speed():
    frames = np.arange(5, dtype=np.float32).reshape(5, 1)
    assert fit_real_timing(frames, 8)[:, 0].tolist() == [0, 1, 2, 3, 4, 4, 4, 4]
    long = np.arange(20, dtype=np.float32).reshape(20, 1)
    assert fit_real_timing(long, 16)[:, 0].tolist() == list(range(2, 18))


def test_frame_level_csv_is_detected(tmp_path):
    from jester_data import LANDMARK_COLUMNS, MOTION_COLUMNS, load_videos

    rows = []
    for video_id, raw in [(1, "Swiping Left"), (2, "No gesture")]:
        for frame_idx in range(3):
            row = {c: 0.1 * (frame_idx + 1) for c in LANDMARK_COLUMNS}
            row.update({c: 0.0 for c in MOTION_COLUMNS})
            row.update(raw_label=raw, split="train", video_id=video_id, frame_idx=frame_idx * 2)
            rows.append(row)
    path = tmp_path / "frames.csv"
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)
    videos = load_videos(str(path), 4)
    assert videos.tracks.shape == (2, 4, 63)
    assert videos.meta["target"].tolist() == ["previous", "none"]
    np.testing.assert_allclose(videos.tracks[0, :, 0], [0.1, 0.2, 0.3, 0.3], atol=1e-6)


def test_stream_window_at_matches_full_windows():
    from jester_data import stream_window_at, stream_windows

    tracks = np.random.default_rng(3).uniform(0.1, 0.9, size=(4, 6, 63)).astype(np.float32)
    ends = np.array([0, 5, 9, 13])
    full = stream_windows(tracks, 6, 3, 5, 0.0, np.random.default_rng(0))
    picked = stream_window_at(tracks, ends, 6, 3, 5, 0.0, np.random.default_rng(0))
    np.testing.assert_allclose(picked, full[np.arange(4), ends])


def test_stream_layout():
    tracks = np.ones((2, 4, 63), np.float32)
    stream = build_streams(tracks, pre_frames=3, post_frames=2, jitter=0.0, rng=np.random.default_rng(0))
    assert stream.shape == (2, 9, 63)
    assert not stream[:, :3].any()
    assert stream[:, 3:].all()
