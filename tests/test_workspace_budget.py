import json
from pathlib import Path

import torch

from bilab.models.factory import build_model

ROOT = Path(__file__).parents[1]


def test_final_persistent_states_are_exactly_4096_bytes() -> None:
    config = json.loads((ROOT / "experiments/cognitive_core_v0/configs/final.json").read_text())
    core = build_model("cognitive_core", config)
    episodic = build_model("episodic", config)
    assert core.state_nbytes(core.initial_state(1)) == 4096
    assert episodic.state_nbytes(episodic.initial_state(1)) == 4096
    assert episodic.capacity == 511


def test_state_size_is_constant_through_ten_thousand_observations(
    tiny_core_config: dict,
) -> None:
    core = build_model("cognitive_core", tiny_core_config).eval()
    episodic = build_model("episodic", tiny_core_config).eval()
    observation = torch.tensor([[0, 1, 0, 4, 0, 0, 0]])
    target = torch.tensor([2])
    lengths = {10, 100, 1_000, 10_000}
    with torch.no_grad():
        core_state = core.initial_state(1)
        episodic_state = episodic.initial_state(1)
        expected_core = core.state_nbytes(core_state)
        expected_episodic = episodic.state_nbytes(episodic_state)
        for step in range(1, 10_001):
            _, core_state, _ = core.step(observation, core_state, target)
            _, episodic_state, _ = episodic.step(observation, episodic_state, target)
            if step in lengths:
                assert core.state_nbytes(core_state) == expected_core
                assert episodic.state_nbytes(episodic_state) == expected_episodic
                assert episodic_state.count.item() <= episodic.capacity


def test_no_memory_baseline_has_no_state_or_history(tiny_core_config: dict) -> None:
    model = build_model("no_memory", tiny_core_config).eval()
    observation = torch.tensor([[0, 1, 0, 4, 0, 0, 0]])
    with torch.no_grad():
        before, state, _ = model.step(observation, None, torch.tensor([1]))
        model.step(torch.tensor([[3, 4, 2, 4, 1, 9, 0]]), None, torch.tensor([3]))
        after, state_after, _ = model.step(observation, None, torch.tensor([1]))
    assert state is None and state_after is None
    assert torch.equal(before, after)
