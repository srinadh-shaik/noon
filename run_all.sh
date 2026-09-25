#!/usr/bin/env bash
# Raw data -> output/{matching_results,candidate_pairs}.tsv. Every stage exits non-zero when one
# of its HARD verify checks fails, which stops the run. Resume with FROM=<step>, e.g. FROM=9 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
FROM=${FROM:-1}

# step 16 ships the owner-or-none model only if gate G5 (step 15) kept it; evaluated when the step runs
APPLY='s8_decide.py apply test $(grep -q "^| G5 |.*✅" reports/gates.md && echo "full_owner --owner-model" || echo full)'
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
  "s7_owner_model.py fit full"
  "s7_owner_model.py predict test full"
  "s8_decide.py tune full --owner-model"
  "s8_decide.py tune full"
  "s7_owner_model.py gate full"
  "$APPLY"
  "s9_output.py write test"
  "s9_output.py summary"
)

T0=$SECONDS
for i in "${!STEPS[@]}"; do
  n=$((i + 1))
  (( n < FROM )) && continue
  step=$(eval echo "${STEPS[$i]}")
  echo "=== step $n/${#STEPS[@]}: $step"
  t=$SECONDS
  # shellcheck disable=SC2086  # word-split the step into script + args
  $PY src/$step
  echo "=== step $n done in $((SECONDS - t))s (total $((SECONDS - T0))s)"   # U6 wall-clock
done
