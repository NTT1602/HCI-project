import csv
from gate import GestureGate, gesture_to_slide_action


def test_gate_requires_hold_and_cooldown(tmp_path):
    log = tmp_path / "events.csv"
    gate = GestureGate(threshold=0.8, hold_frames=3, cooldown_frames=3, log_path=str(log))
    assert gate.update("next", 0.9) is None
    assert gate.update("next", 0.9) is None
    assert gate.update("next", 0.9) == "next"
    assert gate.update("next", 0.9) is None
    assert gate.update("next", 0.9) is None
    assert gate.update("next", 0.9) is None
    with log.open(newline="", encoding="utf-8") as file:
        assert len(list(csv.reader(file))) == 2


def test_gate_rejects_low_confidence():
    gate = GestureGate(hold_frames=1, log_path="data/test-events.csv")
    assert gate.update("next", 0.79) is None
    assert gesture_to_slide_action("next") == "NEXT SLIDE"
