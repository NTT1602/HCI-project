import numpy as np

from jester_data import build_streams, scenario_of, synthesize_tracks, target_of
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


def test_stream_layout():
    tracks = np.ones((2, 4, 63), np.float32)
    stream = build_streams(tracks, pre_frames=3, post_frames=2, jitter=0.0, rng=np.random.default_rng(0))
    assert stream.shape == (2, 9, 63)
    assert not stream[:, :3].any()
    assert stream[:, 3:].all()
