import json
from copy import deepcopy
from pathlib import Path

import pytest

from bilab.manifest import REQUIRED, ManifestError, load_manifest, validate_manifest

ROOT = Path(__file__).parents[1]


def _valid() -> dict[str, object]:
    return json.loads((ROOT / "experiments/smoke/manifest.json").read_text())


def test_checked_in_manifest_is_valid() -> None:
    manifest = load_manifest(ROOT / "experiments/smoke/manifest.json")
    assert manifest["experiment_id"] == "smoke.pipeline.v1"


def test_physical_schema_and_semantic_validator_require_same_fields() -> None:
    schema = json.loads((ROOT / "schemas/experiment-manifest.schema.json").read_text())
    assert set(schema["required"]) == REQUIRED
    assert set(schema["properties"]) == REQUIRED


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"experiment_id": "NO"}, "experiment_id"),
        ({"repetition_count": 2}, "number of seeds"),
        ({"seeds": [7, 7, 41]}, "unique"),
        ({"status": "completed"}, "final_conclusion"),
        ({"evidence_classification": "certain"}, "evidence_classification"),
    ],
)
def test_invalid_manifests_are_rejected(mutation: dict[str, object], message: str) -> None:
    manifest = deepcopy(_valid())
    manifest.update(mutation)
    with pytest.raises(ManifestError, match=message):
        validate_manifest(manifest)


def test_unknown_fields_are_rejected() -> None:
    manifest = _valid()
    manifest["surprise"] = True
    with pytest.raises(ManifestError, match="unknown fields"):
        validate_manifest(manifest)
