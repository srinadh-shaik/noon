#!/usr/bin/env bash
# Raw data -> output/{matching_results,candidate_pairs}.tsv. Every stage exits non-zero when one
# of its HARD verify checks fails, which stops the run. Resume with FROM=<step>, e.g. FROM=9 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
FROM=${FROM:-1}

STEPS=(
  "s0_harness.py build"
  "s1_normalise.py"
  "s2_knowledge.py"
  "s3_retrieve.py curve"
  "s4_candidates.py full"
  "s4_candidates.py test"
  "s5_features.py full"
  "s5_features.py test"
  "s6_score.py full"
  "s6_score.py test"
  "s8_decide.py tune full"
  "s8_decide.py apply test full"
  "s9_output.py write test"
  "s9_output.py summary"
)

T0=$SECONDS
for i in "${!STEPS[@]}"; do
  n=$((i + 1))
  (( n < FROM )) && continue
  echo "=== step $n/${#STEPS[@]}: ${STEPS[$i]}"
  t=$SECONDS
  # shellcheck disable=SC2086  # word-split the step into script + args
  $PY src/${STEPS[$i]}
  echo "=== step $n done in $((SECONDS - t))s (total $((SECONDS - T0))s)"   # U6 wall-clock
done
