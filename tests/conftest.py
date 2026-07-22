import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.fixture
def tiny_core_config() -> dict:
    config = json.loads((ROOT / "experiments/cognitive_core_v0/configs/pilot.json").read_text())
    config = copy.deepcopy(config)
    config["workspace_bytes"] = 128
    config["environment"].update(
        {
            "train_worlds": 4,
            "train_episode_length": 8,
            "validation_worlds": 2,
            "evaluation_worlds_per_category": 2,
            "evaluation_episode_length": 10,
            "rule_change_episode_length": 12,
            "rule_change_step": 5,
        }
    )
    config["model"].update(
        {
            "hidden_dim": 24,
            "symbol_embedding_dim": 6,
            "operation_embedding_dim": 4,
            "kind_embedding_dim": 4,
            "noise_embedding_dim": 3,
            "marker_embedding_dim": 2,
            "outcome_embedding_dim": 6,
            "error_embedding_dim": 6,
            "thought_cycles": 2,
            "episodic_retrieval_records": 4,
        }
    )
    config["training"].update({"seeds": [3], "epochs": 1, "batch_size": 2, "torch_threads": 1})
    config["evaluation"]["adaptation_checkpoints"] = [0, 1, 2, 4, 8]
    config["evaluation"]["recovery_windows"] = [[0, 1], [2, 3], [4, 6]]
    config["success_criteria"]["minimum_positive_seeds"] = 1
    return config
