from dataclasses import replace

import torch

from bilab.models.v1 import StepPrediction
from bilab.v2.models import (
    RAW_WRITER_FIELD_NAMES,
    RAW_WRITER_INPUT_CONTRACT,
    BilinearRouterCore,
    RawFixedRoutingCore,
    RawGRUCore,
    RawRouterCore,
    RelationRouterCore,
    V2ModelConfig,
    raw_writer_input,
    validate_raw_writer_trace,
)


def _prediction(batch: int, hidden: int) -> StepPrediction:
    return StepPrediction(
        logits=torch.zeros(batch, 2),
        hidden=torch.zeros(batch, hidden),
        thought_states=(),
    )


def test_raw_writer_contract_contains_no_sufficient_statistic() -> None:
    names = {field["name"] for field in RAW_WRITER_INPUT_CONTRACT["fields"]}
    prohibited = set(RAW_WRITER_INPUT_CONTRACT["prohibited"])
    assert names == {
        "current_input",
        "operation_one_hot",
        "phase",
        "observed_outcome_one_hot",
    }
    assert {"input_xor_outcome", "correct_state_slot", "write_or_no_write_target"} <= prohibited
    assert V2ModelConfig(family="raw_router").rule_count == 2


def test_v2c_writer_receives_only_raw_public_fields() -> None:
    model = RawRouterCore(V2ModelConfig(family="raw_router", hidden_dim=16))
    public = torch.tensor([[1.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    outcome = torch.tensor([0, 1])
    state = model.initial_state(2)
    update = model.update(public, state, outcome, model.predict(public, state))
    validate_raw_writer_trace(update.trace, public, outcome)
    assert update.trace.field_names == RAW_WRITER_FIELD_NAMES
    assert update.trace.derived_fields == ()
    assert update.trace.routing_mode == "learned_soft_router"
    assert update.trace.writer_input.shape == (2, 8)


def test_bilinear_router_still_receives_only_raw_public_fields() -> None:
    model = BilinearRouterCore(V2ModelConfig(family="bilinear_router", hidden_dim=16))
    public = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    outcome = torch.tensor([1, 0])
    state = model.initial_state(2)
    update = model.update(public, state, outcome, model.predict(public, state))

    validate_raw_writer_trace(update.trace, public, outcome)
    assert update.trace.derived_fields == ()


def test_relation_scaffold_is_explicit_and_confined_to_v2b() -> None:
    raw = RawRouterCore(V2ModelConfig(family="raw_router", hidden_dim=16))
    scaffolded = RelationRouterCore(V2ModelConfig(family="relation_router", hidden_dim=16))
    public = torch.tensor([[1.0, 1.0, 1.0]])
    outcome = torch.tensor([0])
    raw_update = raw.update(
        public, raw.initial_state(1), outcome, raw.predict(public, raw.initial_state(1))
    )
    state = scaffolded.initial_state(1)
    scaffolded_update = scaffolded.update(public, state, outcome, scaffolded.predict(public, state))
    assert raw_update.trace.derived_fields == ()
    assert scaffolded_update.trace.derived_fields == ("signed_rule_relation",)
    assert raw.relation_scaffold is False
    assert scaffolded.relation_scaffold is True


def test_fixed_routing_exists_only_in_v2a() -> None:
    fixed = RawFixedRoutingCore(V2ModelConfig(family="raw_fixed", hidden_dim=16))
    generic = RawRouterCore(V2ModelConfig(family="raw_router", hidden_dim=16))
    public = torch.tensor([[0.0, 1.0, 1.0]])
    outcome = torch.tensor([1])
    fixed_state = fixed.initial_state(1)
    generic_state = generic.initial_state(1)
    fixed_update = fixed.update(public, fixed_state, outcome, fixed.predict(public, fixed_state))
    generic_update = generic.update(
        public, generic_state, outcome, generic.predict(public, generic_state)
    )
    assert fixed.fixed_routing is True
    assert fixed_update.trace.routing_mode == "fixed_context_mask"
    assert torch.equal(fixed_update.trace.route, torch.tensor([[0.0, 1.0]]))
    assert generic.fixed_routing is False
    assert generic_update.trace.routing_mode == "learned_soft_router"
    assert torch.all((generic_update.trace.route > 0) & (generic_update.trace.route < 1))


def test_batch_elements_update_independently_and_reset_is_exact() -> None:
    model = RawGRUCore(V2ModelConfig(family="raw_gru", hidden_dim=16))
    public = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    outcomes = torch.tensor([0, 1])
    initial = model.initial_state(2)
    update = model.update(public, initial, outcomes, model.predict(public, initial))
    assert update.state.shape == (2, 2)
    assert not torch.equal(update.state[0], update.state[1])
    assert torch.equal(model.initial_state(2), torch.zeros(2, 2))


def test_state_size_is_constant_through_100_000_raw_feedback_updates() -> None:
    model = RawRouterCore(V2ModelConfig(family="raw_router", hidden_dim=8))
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


def test_quantizer_runs_after_every_update_and_counts_all_runtime_bytes() -> None:
    config = V2ModelConfig(family="raw_router", hidden_dim=16, quantization_bits=4)
    model = RawRouterCore(config)
    state = model.initial_state(1)
    public = torch.tensor([[1.0, 0.0, 0.0]])
    outcome = torch.tensor([0])
    updated = model.update(public, state, outcome, model.predict(public, state)).state
    levels = torch.linspace(-1.0, 1.0, 16)
    assert all(torch.isclose(value, levels).any() for value in updated.flatten())
    assert model.quantization_events == 1
    assert model.persistent_bytes == 1
    assert len(model.canonical_state_bytes(updated)) == 1

    float_model = RawRouterCore(replace(config, quantization_bits=32))
    assert float_model.persistent_bytes == 8


def test_raw_input_encoding_is_categorical_not_relational() -> None:
    public = torch.tensor([[1.0, 2.0, 1.0]])
    outcome = torch.tensor([0])
    encoded = raw_writer_input(public, outcome)
    assert torch.equal(
        encoded,
        torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]]),
    )
