from __future__ import annotations

import torch

from bilab.v3.models import V3ModelConfig, build_v3_model
from bilab.v3.training import compose_v3c_staged_state_dict


def _public(operation: int = 0, phase: int = 0, input_value: int = 0) -> torch.Tensor:
    return torch.tensor([[float(input_value), float(operation), float(phase)]])


def test_joint_families_use_raw_fields_and_eight_bytes() -> None:
    for family in (
        "raw_soft_router_hard_skip_softtrain",
        "raw_hard_router_softtrain",
    ):
        model = build_v3_model(V3ModelConfig(family=family))
        model.eval()
        state = model.initial_state(1)
        public = _public()
        update = model.update(public, state, torch.tensor([1]), model.predict(public, state))

        assert model.stage == "v3c"
        assert model.relation_scaffold is False
        assert model.fixed_routing is False
        assert model.hard_preservation is True
        assert model.persistent_bytes == 8
        assert update.trace.derived_fields == ()
        assert update.trace.writer_input.shape[-1] == 8


def test_soft_router_joint_candidate_has_exact_evaluation_skip() -> None:
    model = build_v3_model(V3ModelConfig(family="raw_soft_router_hard_skip_softtrain"))
    with torch.no_grad():
        for parameter in model.write_controller.parameters():
            parameter.zero_()
        model.write_controller[-1].bias.fill_(-2.0)

    state = torch.tensor([[0.25, -0.75]])
    public = _public(operation=3, phase=1, input_value=1)
    outcome = torch.tensor([0])

    model.train()
    training_update = model.update(public, state, outcome, model.predict(public, state))
    assert 0.0 < float(training_update.trace.write_strength.detach()) < 0.5

    model.eval()
    evaluation_update = model.update(public, state, outcome, model.predict(public, state))
    assert torch.equal(
        evaluation_update.trace.write_strength,
        torch.zeros_like(evaluation_update.trace.write_strength),
    )
    assert torch.equal(evaluation_update.state, state)
    assert bool(evaluation_update.trace.exact_skip.all())


def test_hard_router_is_soft_during_training_and_deterministic_at_evaluation() -> None:
    model = build_v3_model(V3ModelConfig(family="raw_hard_router_softtrain"))
    with torch.no_grad():
        for parameter in model.router.parameters():
            parameter.zero_()

    state = model.initial_state(1)
    public = _public(operation=0, phase=0, input_value=0)
    outcome = torch.tensor([0])

    model.train()
    training_route = model.update(public, state, outcome, model.predict(public, state)).trace.route
    assert torch.allclose(training_route, torch.tensor([[0.5, 0.5]]))

    model.eval()
    evaluation_route = model.update(
        public, state, outcome, model.predict(public, state)
    ).trace.route
    assert torch.equal(evaluation_route.sum(dim=-1), torch.ones(1))
    assert bool(torch.all((evaluation_route == 0) | (evaluation_route == 1)))


def test_staged_composition_uses_v3a_writer_and_v3b_memory_subsystems() -> None:
    raw = build_v3_model(V3ModelConfig(family="raw_bilinear_overwrite"))
    preservation = build_v3_model(V3ModelConfig(family="relation_hard_skip_softtrain"))
    target = build_v3_model(V3ModelConfig(family="raw_soft_router_hard_skip_softtrain"))

    with torch.no_grad():
        for parameter in raw.writer_encoder.parameters():
            parameter.fill_(1.25)
        for parameter in preservation.parameters():
            parameter.fill_(-0.75)
        for parameter in preservation.writer_encoder.parameters():
            parameter.fill_(9.0)

    staged, metadata = compose_v3c_staged_state_dict(target, raw, preservation)
    target.load_state_dict(staged, strict=False)

    for parameter in target.writer_encoder.parameters():
        assert torch.equal(parameter, torch.full_like(parameter, 1.25))
    for module in (target.value_candidate, target.router, target.write_controller):
        for parameter in module.parameters():
            assert torch.equal(parameter, torch.full_like(parameter, -0.75))

    assert metadata["raw_relation_family"] == "raw_bilinear_overwrite"
    assert metadata["preservation_family"] == "relation_hard_skip_softtrain"
    assert all(
        source == "v3a_raw_relation"
        for name, source in metadata["parameter_source"].items()
        if name.startswith("writer_encoder.")
    )
    assert all(
        source == "v3b_preservation"
        for name, source in metadata["parameter_source"].items()
        if name.startswith(("router.", "write_controller.", "value_candidate."))
    )
