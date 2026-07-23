import torch

from bilab.models.v1 import StepPrediction
from bilab.v3.models import (
    V3_RAW_INPUT_CONTRACT,
    RawBilinearOverwriteCore,
    RawDiscreteOverwriteCore,
    RawHardRouterCore,
    RelationAttractorCore,
    RelationHardSkipCore,
    RelationHardSkipSoftTrainCore,
    V3ModelConfig,
    validate_v3_raw_trace,
)


def _prediction(batch_size: int, hidden_dim: int) -> StepPrediction:
    return StepPrediction(
        logits=torch.zeros(batch_size, 2),
        hidden=torch.zeros(batch_size, hidden_dim),
        thought_states=(),
    )


def test_v3_raw_contract_excludes_relation_slot_mask_and_future() -> None:
    prohibited = set(V3_RAW_INPUT_CONTRACT["prohibited"])
    assert {
        "input_xor_outcome",
        "input_equals_outcome",
        "signed_rule_relation",
        "correct_state_slot",
        "semantic_write_or_no_write_target",
        "future_observation",
    } <= prohibited
    assert "previous_state" in V3_RAW_INPUT_CONTRACT["fields"]


def test_v3a_raw_writer_captures_only_public_fields_and_bounded_state() -> None:
    model = RawBilinearOverwriteCore(V3ModelConfig(family="raw_bilinear_overwrite", hidden_dim=16))
    public = torch.tensor([[1.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    outcome = torch.tensor([0, 1])
    state = model.initial_state(2)
    update = model.update(public, state, outcome, model.predict(public, state))
    validate_v3_raw_trace(update.trace, public, outcome, state)
    assert update.trace.derived_fields == ()
    assert update.trace.writer_input.shape == (2, 8)
    assert update.trace.previous_state.shape == (2, 2)


def test_relation_scaffold_exists_only_in_v3b_isolation() -> None:
    raw = RawHardRouterCore(V3ModelConfig(family="raw_hard_router", hidden_dim=16))
    scaffolded = RelationHardSkipCore(V3ModelConfig(family="relation_hard_skip", hidden_dim=16))
    public = torch.tensor([[1.0, 0.0, 1.0]])
    outcome = torch.tensor([0])
    raw_state = raw.initial_state(1)
    scaffolded_state = scaffolded.initial_state(1)
    raw_update = raw.update(
        public,
        raw_state,
        outcome,
        raw.predict(public, raw_state),
    )
    scaffolded_update = scaffolded.update(
        public,
        scaffolded_state,
        outcome,
        scaffolded.predict(public, scaffolded_state),
    )
    assert raw_update.trace.derived_fields == ()
    assert scaffolded_update.trace.derived_fields == ("signed_rule_relation",)
    assert raw.relation_scaffold is False
    assert scaffolded.relation_scaffold is True


def test_v3a_fixed_route_is_explicit_scaffold_and_exact_nonwrite() -> None:
    model = RawDiscreteOverwriteCore(V3ModelConfig(family="raw_discrete_overwrite", hidden_dim=16))
    state = torch.tensor([[0.25, -0.75]])
    public = torch.tensor([[1.0, 3.0, 1.0]])
    outcome = torch.tensor([0])
    update = model.update(public, state, outcome, model.predict(public, state))
    assert model.fixed_routing is True
    assert update.trace.routing_mode == "fixed_research_route"
    assert torch.equal(update.state, state)
    assert torch.equal(update.gate, torch.zeros_like(update.gate))


def test_learned_hard_skip_is_exact_identity_without_semantic_branch() -> None:
    model = RelationHardSkipCore(V3ModelConfig(family="relation_hard_skip", hidden_dim=16))
    with torch.no_grad():
        model.write_controller[-1].weight.zero_()
        model.write_controller[-1].bias.fill_(-20.0)
    model.eval()
    state = torch.tensor([[0.25, -0.75]])
    public = torch.tensor([[1.0, 3.0, 1.0]])
    outcome = torch.tensor([0])
    update = model.update(public, state, outcome, model.predict(public, state))
    assert torch.equal(update.state, state)
    assert torch.equal(update.trace.write_strength, torch.zeros(1, 1))
    assert bool(update.trace.exact_skip.all())


def test_hard_gate_can_still_write_after_contradictory_feedback() -> None:
    model = RelationHardSkipCore(V3ModelConfig(family="relation_hard_skip", hidden_dim=16))
    with torch.no_grad():
        model.write_controller[-1].weight.zero_()
        model.write_controller[-1].bias.fill_(20.0)
    model.eval()
    state = torch.tensor([[0.25, -0.75]])
    public = torch.tensor([[0.0, 0.0, 1.0]])
    outcome = torch.tensor([1])
    update = model.update(public, state, outcome, model.predict(public, state))
    assert bool((update.gate > 0).any())
    assert not torch.equal(update.state, state)


def test_soft_train_hard_skip_is_exact_during_evaluation() -> None:
    model = RelationHardSkipSoftTrainCore(
        V3ModelConfig(family="relation_hard_skip_softtrain", hidden_dim=16)
    )
    model.eval()
    state = torch.tensor([[0.25, -0.5]])
    public = torch.tensor([[0.0, 2.0, 0.0]])
    with torch.no_grad():
        model.write_controller[-1].weight.zero_()
        model.write_controller[-1].bias.fill_(-10.0)
        update = model.update(
            public,
            state,
            torch.tensor([1]),
            model.predict(public, state),
        )
    assert torch.equal(update.state, state)
    assert bool(update.trace.exact_skip.all())


def test_attractor_projects_every_coordinate_to_a_learned_code() -> None:
    model = RelationAttractorCore(V3ModelConfig(family="relation_attractor", hidden_dim=16))
    model.eval()
    state = torch.tensor([[0.25, -0.75]])
    public = torch.tensor([[1.0, 3.0, 1.0]])
    outcome = torch.tensor([0])
    update = model.update(public, state, outcome, model.predict(public, state))
    codes = torch.tanh(model.state_codebook)
    assert all(torch.isclose(value, codes).any() for value in update.state.flatten())
    assert update.trace.code_indices is not None


def test_state_size_is_constant_through_100_000_exact_skips() -> None:
    model = RelationHardSkipCore(V3ModelConfig(family="relation_hard_skip", hidden_dim=8))
    with torch.no_grad():
        model.write_controller[-1].weight.zero_()
        model.write_controller[-1].bias.fill_(-20.0)
    model.eval()
    state = model.initial_state(1)
    public = torch.tensor([[0.0, 3.0, 1.0]])
    outcome = torch.tensor([1])
    prediction = _prediction(1, 8)
    with torch.no_grad():
        for _ in range(100_000):
            state = model.update(public, state, outcome, prediction).state
    assert state.shape == (1, 2)
    assert state.numel() * state.element_size() == 8
    assert model.persistent_bytes == 8
    assert len(model.canonical_state_bytes(state)) == 8
    assert state.grad_fn is None


def test_v3_configuration_rejects_extra_state_or_recurrence() -> None:
    config = V3ModelConfig(family="raw_hard_router")
    for changes in (
        {"state_dim": 4},
        {"thought_cycles": 2},
    ):
        try:
            V3ModelConfig(**{**config.__dict__, **changes})
        except ValueError:
            pass
        else:
            raise AssertionError("invalid V3 state/recurrence configuration was accepted")
