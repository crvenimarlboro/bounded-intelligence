import torch

from bilab.models.v1 import FactorizedStateCore, V1ModelConfig
from bilab.v2.scaffold_audit import (
    SCAFFOLD_INTERVENTIONS,
    V1ScaffoldAuditCore,
    v1_scaffold_inventory,
)


def _update(model: FactorizedStateCore, public: torch.Tensor, outcome: int) -> torch.Tensor:
    state = model.initial_state(1)
    prediction = model.predict(public, state)
    return model.update(public, state, torch.tensor([outcome]), prediction).state


def test_v1_scaffold_inventory_identifies_every_prohibited_path() -> None:
    inventory = v1_scaffold_inventory()
    names = {item["name"] for item in inventory if item.get("final_v2_prohibited")}
    assert names == {
        "signed_input_outcome_relation",
        "fixed_context_to_slot_mask",
        "fixed_nonwrite_for_composition",
        "fixed_nonwrite_for_distractors",
    }
    composition = next(item for item in inventory if item["name"] == "learned_composition_reader")
    assert composition["final_v2_prohibited"] is False


def test_scaffold_audit_model_is_state_dict_compatible_and_complete() -> None:
    config = V1ModelConfig(hidden_dim=16, state_dim=2, context_count=4, rule_count=2)
    source = FactorizedStateCore(config)
    for intervention in SCAFFOLD_INTERVENTIONS:
        audited = V1ScaffoldAuditCore(config, intervention)
        audited.load_state_dict(source.state_dict(), strict=True)


def test_relation_and_routing_interventions_change_only_declared_pathways() -> None:
    config = V1ModelConfig(hidden_dim=16, state_dim=2, context_count=4, rule_count=2)
    source = FactorizedStateCore(config)
    public = torch.tensor([[1.0, 0.0, 1.0]])
    native = V1ScaffoldAuditCore(config, "native")
    native.load_state_dict(source.state_dict())
    zero = V1ScaffoldAuditCore(config, "relation_zero")
    zero.load_state_dict(source.state_dict())
    swapped = V1ScaffoldAuditCore(config, "routing_swapped")
    swapped.load_state_dict(source.state_dict())
    native_state = _update(native, public, 0)
    zero_state = _update(zero, public, 0)
    swapped_state = _update(swapped, public, 0)
    assert not torch.equal(native_state, zero_state)
    assert native_state[0, 1] == 0
    assert swapped_state[0, 0] == 0
    assert swapped_state[0, 1] != 0


def test_native_composition_and_distractor_are_hard_nonwrites() -> None:
    config = V1ModelConfig(hidden_dim=16, state_dim=2, context_count=4, rule_count=2)
    source = FactorizedStateCore(config)
    for context in (2.0, 3.0):
        public = torch.tensor([[1.0, context, 1.0]])
        assert torch.equal(_update(source, public, 0), source.initial_state(1))
    enabled = V1ScaffoldAuditCore(config, "distractor_writes_enabled")
    enabled.load_state_dict(source.state_dict())
    public = torch.tensor([[1.0, 3.0, 1.0]])
    assert not torch.equal(_update(enabled, public, 0), enabled.initial_state(1))
