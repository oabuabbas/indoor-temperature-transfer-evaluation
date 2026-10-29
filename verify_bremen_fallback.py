"""Check frozen Bremen artifacts and arithmetic, without rerunning predictions."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
out = root / "outputs"
summary = json.loads((out / "bremen_fallback_v1_summary.json").read_text(encoding="utf8"))
manifest = json.loads((out / "bremen_manifest.json").read_text(encoding="utf8"))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(root / "protocol_bremen_fallback_v1.md") == summary["protocol_sha256"]
for f in manifest["files"]:
    assert sha(root / "data" / "bremen" / f["file"]) == f["sha256_local"]
    assert sha(out / "bremen_hourly_v1" / f["hourly_file"]) == f["hourly_sha256_local"]
months = pd.read_csv(out / "bremen_fallback_v1_months.csv")
sensors = pd.read_csv(out / "bremen_fallback_v1_sensors.csv")
branches = pd.read_csv(out / "bremen_fallback_v1_branches.csv")
common = pd.read_csv(out / "bremen_fallback_v1_common.csv")
choices = pd.read_csv(out / "bremen_fallback_v1_choices.csv")
assert choices.shape[0] == 40 and choices.sensor.nunique() == 4 and (choices.history_n >= 24).all()
assert not months.duplicated(["sensor", "month", "model"]).any()
assert (months.groupby(["sensor", "month"]).model.nunique() == 6).all()
assert (months.scored <= months.ready).all() and (months.ready <= months.opportunities).all()
calc = months.groupby(["sensor", "model"])[["opportunities", "ready", "scored", "abs_error_sum"]].sum().sort_index()
saved = sensors.set_index(["sensor", "model"])[calc.columns].sort_index()
assert np.allclose(calc, saved)
fallback = months[months.model == "fallback"].set_index(["sensor", "month"])
b = branches.groupby(["sensor", "month"])[["opportunities", "scored", "abs_error_sum"]].sum().sort_index()
f = fallback.loc[b.index]
assert np.array_equal(b.opportunities, f.opportunities)
assert np.array_equal(b.scored, f.scored)
assert np.allclose(b.abs_error_sum, f.abs_error_sum)
for model, d in sensors.groupby("model"):
    x = summary["methods_own_support"][model]
    assert int(d.opportunities.sum()) == x["opportunities"]
    assert int(d.ready.sum()) == x["ready"] and int(d.scored.sum()) == x["scored"]
    assert np.isclose(d.mae_c.mean(), x["equal_sensor_mean_own_support_mae_c"])
assert common.n.sum() == 29022
for model in ("fallback", "persistence"):
    assert np.allclose(common[model + "_mae_c"], common[model + "_abs_sum"] / common.n)
assert (common.fallback_mae_c < common.persistence_mae_c).all()
selector = months[months.model == "rolling_selector"].set_index(["sensor", "month"])
source = months[months.model == "source_daily_ridge"].set_index(["sensor", "month"])
assert np.array_equal(selector.sort_index().scored, source.sort_index().scored)
print("PASS: protocol and input hashes, monthly aggregation, branches, coverage, common-support arithmetic and identical source/selector score counts")
print("Checks do not independently recompute predictions, verify timing semantics, or prove masks identical merely from counts.")
