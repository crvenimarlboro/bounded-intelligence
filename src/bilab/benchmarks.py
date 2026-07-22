"""Normalize native llama-bench JSONL/CSV without changing the sources."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
REQUIRED_FIELDS = {
    "build_commit",
    "build_number",
    "cpu_info",
    "gpu_info",
    "backends",
    "model_filename",
    "model_type",
    "model_size",
    "model_n_params",
    "n_batch",
    "n_ubatch",
    "n_threads",
    "cpu_mask",
    "cpu_strict",
    "poll",
    "type_k",
    "type_v",
    "n_gpu_layers",
    "n_cpu_moe",
    "split_mode",
    "main_gpu",
    "no_kv_offload",
    "devices",
    "tensor_split",
    "tensor_buft_overrides",
    "use_mmap",
    "use_direct_io",
    "embeddings",
    "no_op_offload",
    "no_host",
    "fit_target",
    "fit_min_ctx",
    "flash_attn",
    "n_prompt",
    "n_gen",
    "n_depth",
    "test_time",
    "avg_ns",
    "stddev_ns",
    "avg_ts",
    "stddev_ts",
}


class BenchmarkError(ValueError):
    """A source is not a valid native llama-bench result."""


def _integer(raw: dict[str, Any], key: str) -> int:
    try:
        return int(raw[key])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkError(f"{key} must be an integer") from error


def _number(raw: dict[str, Any], key: str) -> float:
    try:
        return float(raw[key])
    except (KeyError, TypeError, ValueError) as error:
        raise BenchmarkError(f"{key} must be numeric") from error


def _boolean(raw: dict[str, Any], key: str) -> bool:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"1", "true"}:
        return True
    if str(value).strip().lower() in {"0", "false"}:
        return False
    raise BenchmarkError(f"{key} must be boolean or 0/1")


def _text(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if value is None or not str(value).strip():
        raise BenchmarkError(f"{key} must be non-empty")
    return str(value).strip()


@dataclass(frozen=True)
class BenchmarkRecord:
    schema_version: str
    record_id: str
    source_file: str
    source_row: int
    build_commit: str
    build_number: int
    model_filename: str
    model_type: str
    model_size_bytes: int
    model_parameters: int
    backend: str
    device: str
    cpu_info: str
    gpu_info: str
    gpu_layers: int
    threads: int
    batch_size: int
    micro_batch_size: int
    key_cache_type: str
    value_cache_type: str
    flash_attention: int
    runtime_options: dict[str, object]
    phase: str
    input_tokens: int
    output_tokens: int
    timestamp_utc: str
    average_nanoseconds: int
    stddev_nanoseconds: int
    tokens_per_second: float
    stddev_tokens_per_second: float
    sample_nanoseconds: tuple[int, ...]
    sample_tokens_per_second: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["sample_nanoseconds"] = list(self.sample_nanoseconds)
        value["sample_tokens_per_second"] = list(self.sample_tokens_per_second)
        return value

    def compatibility_conditions(self) -> dict[str, object]:
        """Conditions that must match when GPU-layer offload is the intervention."""

        return {
            "build_commit": self.build_commit,
            "build_number": self.build_number,
            "model_type": self.model_type,
            "model_filename": self.model_filename,
            "model_size_bytes": self.model_size_bytes,
            "model_parameters": self.model_parameters,
            "backend": self.backend,
            "cpu_info": self.cpu_info,
            "gpu_info": self.gpu_info,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "micro_batch_size": self.micro_batch_size,
            "key_cache_type": self.key_cache_type,
            "value_cache_type": self.value_cache_type,
            "flash_attention": self.flash_attention,
            "runtime_options": self.runtime_options,
            "phase": self.phase,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def normalize_record(raw: dict[str, Any], source: Path, row: int) -> BenchmarkRecord:
    missing = sorted(REQUIRED_FIELDS - raw.keys())
    if missing:
        raise BenchmarkError(f"{source.name}:{row}: missing fields: {', '.join(missing)}")
    n_prompt = _integer(raw, "n_prompt")
    n_gen = _integer(raw, "n_gen")
    if (n_prompt > 0) == (n_gen > 0):
        raise BenchmarkError(
            f"{source.name}:{row}: exactly one of n_prompt and n_gen must be positive"
        )
    phase = "prompt_processing" if n_prompt > 0 else "token_generation"
    identity_input = {
        key: raw[key]
        for key in sorted(REQUIRED_FIELDS)
        if key not in {"avg_ns", "stddev_ns", "avg_ts", "stddev_ts"}
    }
    identity_input["source_file"] = source.name
    identity_input["source_row"] = row
    record_id = hashlib.sha256(
        json.dumps(identity_input, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    sample_ns = tuple(int(value) for value in raw.get("samples_ns", []))
    sample_ts = tuple(float(value) for value in raw.get("samples_ts", []))
    return BenchmarkRecord(
        schema_version=SCHEMA_VERSION,
        record_id=record_id,
        source_file=source.name,
        source_row=row,
        build_commit=_text(raw, "build_commit"),
        build_number=_integer(raw, "build_number"),
        model_filename=_text(raw, "model_filename"),
        model_type=_text(raw, "model_type"),
        model_size_bytes=_integer(raw, "model_size"),
        model_parameters=_integer(raw, "model_n_params"),
        backend=_text(raw, "backends"),
        device=_text(raw, "devices"),
        cpu_info=_text(raw, "cpu_info"),
        gpu_info=_text(raw, "gpu_info"),
        gpu_layers=_integer(raw, "n_gpu_layers"),
        threads=_integer(raw, "n_threads"),
        batch_size=_integer(raw, "n_batch"),
        micro_batch_size=_integer(raw, "n_ubatch"),
        key_cache_type=_text(raw, "type_k"),
        value_cache_type=_text(raw, "type_v"),
        flash_attention=_integer(raw, "flash_attn"),
        runtime_options={
            "cpu_mask": _text(raw, "cpu_mask"),
            "cpu_strict": _boolean(raw, "cpu_strict"),
            "poll": _integer(raw, "poll"),
            "cpu_moe_layers": _integer(raw, "n_cpu_moe"),
            "split_mode": _text(raw, "split_mode"),
            "main_gpu": _integer(raw, "main_gpu"),
            "kv_offload": not _boolean(raw, "no_kv_offload"),
            "tensor_split": _text(raw, "tensor_split"),
            "tensor_buffer_overrides": _text(raw, "tensor_buft_overrides"),
            "memory_map": _boolean(raw, "use_mmap"),
            "direct_io": _boolean(raw, "use_direct_io"),
            "embeddings": _boolean(raw, "embeddings"),
            "operation_offload_disabled": _integer(raw, "no_op_offload"),
            "host_buffer_disabled": _boolean(raw, "no_host"),
            "fit_target": _integer(raw, "fit_target"),
            "fit_minimum_context": _integer(raw, "fit_min_ctx"),
            "depth": _integer(raw, "n_depth"),
        },
        phase=phase,
        input_tokens=n_prompt,
        output_tokens=n_gen,
        timestamp_utc=_text(raw, "test_time"),
        average_nanoseconds=_integer(raw, "avg_ns"),
        stddev_nanoseconds=_integer(raw, "stddev_ns"),
        tokens_per_second=_number(raw, "avg_ts"),
        stddev_tokens_per_second=_number(raw, "stddev_ts"),
        sample_nanoseconds=sample_ns,
        sample_tokens_per_second=sample_ts,
    )


def _raw_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            for row, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise BenchmarkError(f"{path.name}:{row}: invalid JSON: {error}") from error
                    if not isinstance(value, dict):
                        raise BenchmarkError(f"{path.name}:{row}: record must be an object")
                    yield row, value
        return
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row, value in enumerate(reader, start=2):
                yield row, dict(value)
        return
    raise BenchmarkError(f"unsupported benchmark extension: {path.suffix}")


def parse_file(path: Path) -> list[BenchmarkRecord]:
    records = [normalize_record(raw, path, row) for row, raw in _raw_rows(path)]
    if not records:
        raise BenchmarkError(f"{path}: no records")
    return records


def compatibility_differences(
    left: BenchmarkRecord, right: BenchmarkRecord
) -> dict[str, tuple[object, object]]:
    first = left.compatibility_conditions()
    second = right.compatibility_conditions()
    return {key: (first[key], second[key]) for key in first if first[key] != second[key]}


def compare_records(left: BenchmarkRecord, right: BenchmarkRecord) -> dict[str, Any]:
    differences = compatibility_differences(left, right)
    if differences:
        return {
            "status": "rejected_incompatible",
            "left": left.record_id,
            "right": right.record_id,
            "differences": differences,
            "tokens_per_second_ratio": None,
        }
    return {
        "status": "comparable",
        "left": left.record_id,
        "right": right.record_id,
        "differences": {},
        "tokens_per_second_ratio": right.tokens_per_second / left.tokens_per_second,
    }


def _signature(record: BenchmarkRecord) -> str:
    return json.dumps(record.compatibility_conditions(), sort_keys=True, separators=(",", ":"))


def normalize_sources(paths: Iterable[Path]) -> dict[str, Any]:
    records: list[BenchmarkRecord] = []
    skipped: list[str] = []
    for path in paths:
        try:
            records.extend(parse_file(path))
        except BenchmarkError as error:
            skipped.append(str(error))
    if not records:
        detail = "; ".join(skipped) if skipped else "no input paths"
        raise BenchmarkError(f"no valid llama-bench records: {detail}")

    groups: dict[tuple[str, int], list[BenchmarkRecord]] = {}
    signatures_by_phase: dict[str, set[str]] = {}
    for record in records:
        signature = _signature(record)
        groups.setdefault((signature, record.gpu_layers), []).append(record)
        signatures_by_phase.setdefault(record.phase, set()).add(signature)
    summaries: list[dict[str, Any]] = []
    for (signature_text, gpu_layers), members in groups.items():
        summaries.append(
            {
                "condition_id": hashlib.sha256(signature_text.encode()).hexdigest()[:16],
                "phase": members[0].phase,
                "gpu_layers": gpu_layers,
                "run_count": len(members),
                "mean_tokens_per_second": statistics.fmean(
                    member.tokens_per_second for member in members
                ),
                "record_ids": [member.record_id for member in members],
            }
        )
    summaries.sort(key=lambda item: (item["condition_id"], item["gpu_layers"]))
    comparisons: list[dict[str, Any]] = []
    for condition_id in sorted({item["condition_id"] for item in summaries}):
        cohort = [item for item in summaries if item["condition_id"] == condition_id]
        reference = min(cohort, key=lambda item: item["gpu_layers"])
        for candidate in cohort:
            if candidate is reference:
                continue
            comparisons.append(
                {
                    "status": "comparable",
                    "condition_id": condition_id,
                    "phase": candidate["phase"],
                    "reference_gpu_layers": reference["gpu_layers"],
                    "candidate_gpu_layers": candidate["gpu_layers"],
                    "tokens_per_second_ratio": (
                        candidate["mean_tokens_per_second"] / reference["mean_tokens_per_second"]
                    ),
                }
            )
    warnings = [f"skipped non-native or invalid input: {message}" for message in skipped]
    for phase, signatures in signatures_by_phase.items():
        if len(signatures) > 1:
            warnings.append(
                f"{phase} contains {len(signatures)} incompatible condition cohorts; "
                "derived comparisons remain separated"
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "records": [record.to_dict() for record in records],
        "summaries": sorted(summaries, key=lambda item: (item["phase"], item["gpu_layers"])),
        "comparisons": comparisons,
        "warnings": warnings,
    }


def render_summary(document: dict[str, Any]) -> str:
    lines = [
        "llama-bench normalized summary",
        f"records: {len(document['records'])}",
        f"compatible derived comparisons: {len(document['comparisons'])}",
    ]
    for item in document["summaries"]:
        lines.append(
            f"- {item['phase']}: gpu_layers={item['gpu_layers']}, "
            f"runs={item['run_count']}, mean={item['mean_tokens_per_second']:.2f} tokens/s"
        )
    if document["warnings"]:
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in document["warnings"])
    return "\n".join(lines)
