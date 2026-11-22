#!/usr/bin/env bash
# Same-machine independent rerun of the selected configuration (fresh process, same seed/config/data), stored under
# results/phase1/reproduction/<run>/ so that select_model does not treat it as a candidate run.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python}; CFG=${1:?config}; SEL=${2:?selected run name}
export FETALVISION_DATA=${FETALVISION_DATA:-/workspace/data}
$PY -m fetalvision.classification.train --config "$CFG" > results/phase1/logs/reproduction_${SEL}.log 2>&1
mkdir -p results/phase1/reproduction && rm -rf "results/phase1/reproduction/${SEL}" && mv "results/phase1/${SEL}-rerun" "results/phase1/reproduction/${SEL}"
$PY - "$SEL" <<'PYEOF'
import json, sys
sel = sys.argv[1]
a = json.load(open(f"results/phase1/{sel}/metrics.json")); b = json.load(open(f"results/phase1/reproduction/{sel}/metrics.json"))
b["reproduction_of"] = sel; b["config_name"] = f"phase1_{sel.replace('p1v2-','').replace('-','_')}.yaml"
rows = {k: (a["metrics"][k], b["metrics"][k]) for k in ["accuracy", "macro_f1", "ece_10bins"]}
rows["abdomen_recall"] = (a["metrics"]["per_class"]["Abdomen"]["recall"], b["metrics"]["per_class"]["Abdomen"]["recall"])
b["comparison_pp"] = {k: round((v[1] - v[0]) * 100, 2) for k, v in rows.items()}
json.dump(b, open(f"results/phase1/reproduction/{sel}/metrics.json", "w"), indent=2)
print("reproduction comparison (pp):", b["comparison_pp"])
PYEOF
echo REPRO_DONE
