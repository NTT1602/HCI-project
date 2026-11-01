import pandas as pd

from evaluate_policies import evaluate_trial, summarize
from policies import DecisionPolicy, make_policies


def test_fixed_policy_requires_hold_and_cooldown():
    policy = DecisionPolicy("fixed", hold_frames=3, cooldown_frames=3)
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event == "next"
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event == "next"


def test_abstention_rejects_ambiguous_prediction():
    policy = DecisionPolicy("fixed_abstention", hold_frames=1, margin_threshold=0.2, abstain=True)
    decision = policy.update("next", 0.9, 0.1)
    assert decision.event is None
    assert decision.status == "abstain"


def test_adaptive_policy_can_accept_clear_stable_gesture():
    policy = DecisionPolicy("adaptive", adaptive=True, hold_frames=3)
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event is None
    assert policy.update("next", 0.9, 0.8).event == "next"


def test_evaluator_counts_background_trigger():
    rows = []
    for trial_id, trial_type, intended, labels in [
        ("intentional-1", "intentional", "next", ["next", "next", "next"]),
        ("background-1", "background", "none", ["next", "next", "next"]),
    ]:
        for frame_idx, label in enumerate(labels):
            rows.append({
                "trial_id": trial_id,
                "trial_type": trial_type,
                "intended_action": intended,
                "frame_idx": frame_idx,
                "predicted_label": label,
                "confidence": 0.9,
                "margin": 0.8,
            })
    data = pd.DataFrame(rows)
    policy = DecisionPolicy("fixed", hold_frames=3, cooldown_frames=2)
    results = [evaluate_trial(group, policy, 30)[0] for _, group in data.groupby("trial_id")]
    summary = summarize(pd.DataFrame(results).assign(policy="fixed"))
    assert summary.loc[0, "command_success_rate"] == 1.0
    assert summary.loc[0, "false_trigger_events"] == 1


def test_make_policies_returns_four_configurations():
    assert set(make_policies()) == {"fixed", "adaptive", "fixed_abstention", "adaptive_abstention"}
