# Pre-registration correction 0002: pilot generation-seed overlap

Status: corrected during development before confirmatory registration.

The first 800-step reversal pilot used 16 groups per optimizer step, consuming generation seeds
30,000--42,799 despite nominally reserving validation at 35,000 and evaluation at 38,000. Those
results are invalid and retained under the ignored pilot artifact directory.

The corrected reversal pilot used eight groups, consuming 30,000--36,399, validation at
37,000--37,063, and evaluation at 38,000--38,127. All three corrected seeds passed. Executable
range validation and a regression test now reject inclusive train/validation/evaluation overlap.
