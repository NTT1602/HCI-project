"""Abstention mới (C3/C4), heuristic của app và gộp lớp lệnh — cùng hành vi với HCI-Slideshow-Interaction/tests/test_ambiguity_policies.py."""
import pandas as pd

from evaluate_policies import restrict_commands
from policies import DecisionPolicy

BASE = dict(base_threshold=0.8, hold_frames=2, cooldown_frames=5, clear_threshold=0.9,
            conflict_floor=0.5, conflict_frames=4, abstain_wait_frames=4, margin_threshold=0.25)


def frame(p_next, p_prev):
    probs = {"next": p_next, "previous": p_prev, "none": round(1 - p_next - p_prev, 6)}
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[0][0], ranked[0][1], ranked[0][1] - ranked[1][1], probs


def run(policy, frames):
    events, statuses = [], []
    for i, (pn, pp) in enumerate(frames):
        label, conf, margin, probs = frame(pn, pp)
        d = policy.update(label, conf, margin, probs)
        statuses.append(d.status)
        if d.event:
            events.append((i, d.event))
    return events, statuses


def test_clear_swipe_not_delayed_by_abstention():
    frames = [(0.97, 0.01)] * 6
    fixed, _ = run(DecisionPolicy("fixed", **BASE), frames)
    abst, _ = run(DecisionPolicy("fixed_abstention", abstain=True, **BASE), frames)
    assert fixed == abst == [(1, "next")]


def test_lead_in_wrong_direction_suppressed():
    frames = [(0.05, 0.85)] * 3 + [(0.99, 0.005)] * 10   # đoạn lấy đà đọc thành previous, rồi cú vuốt thật
    rb = dict(BASE, reverse_block_frames=15, repeat_block_frames=16)  # như app: chặn chiều ngược + reset GRU
    fixed, _ = run(DecisionPolicy("fixed", **rb), frames)
    abst, statuses = run(DecisionPolicy("fixed_abstention", abstain=True, **rb), frames)
    assert [e for _, e in fixed] == ["previous"]
    assert [e for _, e in abst] == ["next"]
    assert "abstain" in statuses


def test_gray_zone_waits_then_rejects_once():
    abst, statuses = run(DecisionPolicy("fixed_abstention", abstain=True, **BASE), [(0.75, 0.02)] * 12)
    assert abst == [] and statuses.count("abstain_reject") == 1


def test_repeat_block_emulates_model_reset():
    frames = [(0.97, 0.01)] * 14
    plain, _ = run(DecisionPolicy("fixed", **BASE), frames)
    blocked, _ = run(DecisionPolicy("fixed", repeat_block_frames=16, **BASE), frames)
    assert len(plain) > 1 and len(blocked) == 1


def test_restrict_commands_merges_zoom_into_none():
    df = pd.DataFrame([{
        "trial_id": "t", "trial_type": "intentional", "intended_action": "zoom_in", "frame_idx": 0,
        "predicted_label": "zoom_in", "confidence": 0.7, "margin": 0.5,
        "p_next": 0.1, "p_none": 0.0, "p_previous": 0.0, "p_zoom_in": 0.7, "p_zoom_out": 0.2,
    }])
    out = restrict_commands(df, ["next", "previous"])
    row = out.iloc[0]
    assert row.predicted_label == "none" and abs(row.confidence - 0.9) < 1e-9 and abs(row.margin - 0.8) < 1e-9
    assert row.trial_type == "near_miss" and row.intended_action == "none"
