# Pre-registration correction 0001: reversal schedule leakage

Status: corrected during development before confirmatory registration.

The first reversal environment always changed context 0 at step 4. A pilot model predicted the new
rule with 100% accuracy before receiving contradictory feedback, proving that recurrent state had
learned the schedule rather than detected a change. The result was rejected.

The correction randomizes reversal across context-0 queries at steps 4, 5, 8, or 11 and includes
no-change training worlds. Paired constructions can expose identical public histories with opposite
step-4 targets. The corrected pilot predicted the unpredictable change itself at 0--1.95%, then
recovered to 100% one feedback step later. A regression test preserves the ambiguity property.
