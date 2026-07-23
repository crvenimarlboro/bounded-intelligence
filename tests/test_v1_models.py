import copy

import torch
from torch.nn import functional as F

from bilab.models.v1 import (
    EpisodicControl,
    FactorizedStateCore,
    GRUStateCore,
    NoMemoryControl,
    PredictiveStateCore,
    StepPrediction,
    V1ModelConfig,
    build_v1_model,
    count_trainable_parameters,
)


def _step(model: torch.nn.Module, public: torch.Tensor, state: torch.Tensor, outcome: int):
    prediction = model.predict(public, state)
    target = torch.tensor([outcome])
    return prediction, model.update(public, state, target, prediction)


def test_float_workspace_size_reset_and_batch_isolation() -> None:
    torch.manual_seed(1)
    model = FactorizedStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    initial = model.initial_state(2)
    assert initial.shape == (2, 4)
    assert initial.numel() * initial.element_size() // 2 == model.persistent_bytes == 16
    public = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    prediction = model.predict(public, initial)
    update = model.update(public, initial, torch.tensor([0, 1]), prediction)
    assert update.state.shape == initial.shape
    assert not torch.equal(update.state[0], update.state[1])
    assert torch.equal(model.initial_state(2), initial)


def test_state_shape_remains_constant_through_100_000_updates() -> None:
    torch.manual_seed(2)
    model = FactorizedStateCore(V1ModelConfig(hidden_dim=8, state_dim=4))
    public = torch.tensor([[1.0, 1.0]])
    state = model.initial_state(1)
    with torch.no_grad():
        prediction = model.predict(public, state)
        for _ in range(100_000):
            state = model.update(public, state, torch.tensor([1]), prediction).state
    assert state.shape == (1, 4)
    assert state.numel() * state.element_size() == 16
    assert not state.requires_grad


def test_episodic_control_uses_exact_budget_and_deterministic_eviction() -> None:
    model = EpisodicControl(budget_bytes=16, hidden_dim=16)
    state = model.initial_state(1)
    assert state.dtype == torch.uint8
    assert len(model.canonical_bytes(state)) == 16
    for index in range(100):
        public = torch.tensor([[float(index % 2), float(index > 0)]])
        prediction = model.predict(public, state)
        state = model.update(public, state, torch.tensor([index % 2]), prediction).state
        assert len(model.canonical_bytes(state)) == 16
    assert int(state[0, 0].item()) >> 4 == 15
    assert int(state[0, 0].item()) & 15 == 10


def test_no_memory_control_has_no_cross_step_state() -> None:
    model = NoMemoryControl(hidden_dim=16)
    state = model.initial_state(3)
    assert state.shape == (3, 0)
    public = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    prediction = model.predict(public, state)
    updated = model.update(public, state, torch.tensor([0, 1, 1]), prediction).state
    assert updated.shape == (3, 0)
    assert torch.equal(model.predict(public, state).logits, model.predict(public, updated).logits)


def test_shared_thought_parameters_are_reused_across_cycles() -> None:
    model = GRUStateCore(V1ModelConfig(hidden_dim=16, state_dim=4, thought_cycles=3))
    thought_parameter_ids = {id(parameter) for parameter in model.thought_block.parameters()}
    prediction = model.predict(torch.tensor([[0.0, 0.0]]), model.initial_state(1))
    assert len(prediction.thought_states) == 3
    assert thought_parameter_ids == {
        id(parameter) for parameter in model.thought_block.parameters()
    }
    assert not any(name.startswith("thought_blocks") for name, _ in model.named_modules())


def test_prediction_error_changes_writer_pathway() -> None:
    torch.manual_seed(3)
    plain = GRUStateCore(V1ModelConfig(hidden_dim=16, state_dim=4, feedback_mode="outcome_only"))
    error_aware = GRUStateCore(
        V1ModelConfig(hidden_dim=16, state_dim=4, feedback_mode="detached_error")
    )
    error_aware.load_state_dict(copy.deepcopy(plain.state_dict()))
    public = torch.tensor([[1.0, 0.0]])
    state = plain.initial_state(1)
    forced = StepPrediction(
        logits=torch.tensor([[4.0, -4.0]]), hidden=torch.zeros(1, 16), thought_states=()
    )
    outcome = torch.tensor([1])
    plain_update = plain.update(public, state, outcome, forced)
    error_update = error_aware.update(public, state, outcome, forced)
    assert plain_update.error_signal.item() == 0.0
    assert error_update.error_signal.item() > 0.99
    assert not torch.allclose(plain_update.state, error_update.state)


def test_delayed_query_gradient_reaches_first_writer_and_reader() -> None:
    torch.manual_seed(4)
    model = GRUStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    state = model.initial_state(1)
    evidence = torch.tensor([[0.0, 0.0]])
    evidence_prediction = model.predict(evidence, state)
    first_state = model.update(evidence, state, torch.tensor([1]), evidence_prediction).state
    first_state.retain_grad()
    current = first_state
    for value in (1.0, 0.0, 1.0):
        gap = torch.tensor([[value, 1.0]])
        gap_prediction = model.predict(gap, current)
        current = model.update(gap, current, torch.tensor([int(value)]), gap_prediction).state
    query = model.predict(torch.tensor([[1.0, 1.0]]), current)
    F.cross_entropy(query.logits, torch.tensor([0])).backward()
    writer_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.writer.parameters()
        if parameter.grad is not None
    )
    reader_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.state_reader.parameters()
        if parameter.grad is not None
    )
    assert first_state.grad is not None and first_state.grad.norm().item() > 0
    assert writer_gradient > 0
    assert reader_gradient > 0


def test_supervised_state_loss_reaches_writer_parameters() -> None:
    torch.manual_seed(5)
    model = GRUStateCore(V1ModelConfig(hidden_dim=16, state_dim=4))
    public = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    state = model.initial_state(2)
    prediction = model.predict(public, state)
    updated = model.update(public, state, torch.tensor([0, 0]), prediction).state
    F.cross_entropy(model.probe(updated), torch.tensor([0, 1])).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in model.writer.parameters()
    )


def test_all_required_families_build_with_bounded_counts() -> None:
    config = V1ModelConfig(hidden_dim=32, state_dim=4)
    models = {
        family: build_v1_model(family, config) for family in ("gru", "predictive", "factorized")
    }
    models["no_memory"] = build_v1_model("no_memory", config)
    models["episodic"] = build_v1_model("episodic", config, budget_bytes=16)
    assert isinstance(models["predictive"], PredictiveStateCore)
    for model in models.values():
        assert 1_000 < count_trainable_parameters(model) < 500_000


def test_factorized_context_writer_updates_only_selected_slots() -> None:
    torch.manual_seed(6)
    model = FactorizedStateCore(V1ModelConfig(hidden_dim=16, state_dim=4, context_count=2))
    state = model.initial_state(1)
    public = torch.tensor([[1.0, 0.0, 0.0]])
    prediction = model.predict(public, state)
    updated = model.update(public, state, torch.tensor([1]), prediction).state
    assert not torch.equal(updated[:, [0, 2]], state[:, [0, 2]])
    assert torch.equal(updated[:, [1, 3]], state[:, [1, 3]])


def test_composition_query_does_not_overwrite_two_rule_slots() -> None:
    torch.manual_seed(7)
    model = FactorizedStateCore(
        V1ModelConfig(hidden_dim=16, state_dim=2, context_count=3, rule_count=2)
    )
    state = torch.tensor([[0.25, -0.5]])
    public = torch.tensor([[1.0, 2.0, 1.0]])
    prediction = model.predict(public, state)
    updated = model.update(public, state, torch.tensor([0]), prediction).state
    assert torch.equal(updated, state)


def test_marked_non_event_changes_neither_factorized_nor_episodic_state() -> None:
    factorized = FactorizedStateCore(
        V1ModelConfig(hidden_dim=16, state_dim=2, context_count=4, rule_count=2)
    )
    episodic = EpisodicControl(budget_bytes=8, hidden_dim=16, context_count=4)
    public = torch.tensor([[1.0, 3.0, 1.0]])
    for model in (factorized, episodic):
        state = model.initial_state(1)
        prediction = model.predict(public, state)
        updated = model.update(public, state, torch.tensor([1]), prediction).state
        assert torch.equal(updated, state)
