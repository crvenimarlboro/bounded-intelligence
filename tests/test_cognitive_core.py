import torch

from bilab.models.cognitive_core import CognitiveCore
from bilab.models.factory import build_model, count_parameters


def _observation(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[0, 1, 0, 4, 0, 3, 0]] * batch)


def test_recurrent_block_reuses_one_parameter_set(tiny_core_config: dict) -> None:
    model = CognitiveCore(tiny_core_config)
    calls = 0

    def count_call(*_args) -> None:
        nonlocal calls
        calls += 1

    hook = model.thought_block.register_forward_hook(count_call)
    model.step(_observation(), model.initial_state(1), torch.tensor([2]))
    hook.remove()
    assert calls == tiny_core_config["model"]["thought_cycles"]
    assert len({id(parameter) for parameter in model.thought_block.parameters()}) == len(
        list(model.thought_block.parameters())
    )


def test_prediction_error_changes_workspace_update(tiny_core_config: dict) -> None:
    torch.manual_seed(4)
    model = CognitiveCore(tiny_core_config).eval()
    state = model.initial_state(1)
    with torch.no_grad():
        _, full_state, _ = model.step(_observation(), state.clone(), torch.tensor([1]), mode="full")
        _, no_error_state, _ = model.step(
            _observation(), state.clone(), torch.tensor([1]), mode="no_prediction_error"
        )
    assert not torch.equal(full_state, no_error_state)


def test_future_loss_reaches_workspace_update_parameters(tiny_core_config: dict) -> None:
    torch.manual_seed(5)
    model = CognitiveCore(tiny_core_config)
    state = model.initial_state(1)
    _, state, _ = model.step(_observation(), state, torch.tensor([1]))
    logits, _, _ = model.step(_observation(), state, torch.tensor([2]))
    torch.nn.functional.cross_entropy(logits, torch.tensor([2])).backward()
    assert model.workspace_candidate.weight.grad is not None
    assert model.workspace_gate.weight.grad is not None
    assert model.workspace_candidate.weight.grad.abs().sum() > 0
    assert model.workspace_gate.weight.grad.abs().sum() > 0


def test_trainable_parameter_counts_are_within_two_percent(tiny_core_config: dict) -> None:
    counts = {
        variant: count_parameters(build_model(variant, tiny_core_config))
        for variant in ("no_memory", "episodic", "cognitive_core")
    }
    assert (max(counts.values()) - min(counts.values())) / max(counts.values()) <= 0.02


def test_workspace_reset_prevents_cross_world_state_leakage(tiny_core_config: dict) -> None:
    model = CognitiveCore(tiny_core_config).eval()
    initial = model.initial_state(1)
    with torch.no_grad():
        _, adapted, _ = model.step(_observation(), initial, torch.tensor([3]))
        reset = model.initial_state(1)
        first_logits, _, _ = model.step(_observation(), reset, None)
        reference_logits, _, _ = model.step(_observation(), model.initial_state(1), None)
    assert not torch.equal(adapted, reset)
    assert torch.equal(first_logits, reference_logits)


def test_frozen_workspace_stays_unchanged(tiny_core_config: dict) -> None:
    model = CognitiveCore(tiny_core_config)
    state = model.initial_state(1)
    _, next_state, _ = model.step(
        _observation(), state.clone(), torch.tensor([1]), mode="workspace_frozen"
    )
    assert torch.equal(state, next_state)
