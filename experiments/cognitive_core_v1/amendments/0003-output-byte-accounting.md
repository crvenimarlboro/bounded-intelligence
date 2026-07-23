# Post-run correction 0003: normalized-output byte accounting

Status: reporting-only correction after confirmatory execution.

The confirmatory runner initially measured `output_bytes` immediately before writing
`results.json`. The recorded value (4,571,352 bytes) therefore excluded the normalized result
itself. This did not affect training, evaluation, checkpoint selection, seeds, thresholds, or any
capability metric.

The runner now writes the normalized result, measures the complete output directory, and refreshes
the field until its serialized size is stable. The original pre-result measurement is retained as
`output_bytes_before_normalized_result`. Existing normalized results are corrected only through the
documented `bilab v1 report --refresh-resource-accounting` command. Raw metrics and checkpoints are
unchanged.
