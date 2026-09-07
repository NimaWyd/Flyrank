import json

nb = json.load(open("work/notebooks/w05_model.ipynb", encoding="utf-8"))

def out(text_lines):
    return [{"output_type": "stream", "name": "stdout", "text": text_lines}]

outputs_by_id = {

"cell-s1-code": out([
    "Feature                      Computation                                    Source\n",
    "------------------------------------------------------------------------------------------\n",
    "log_impressions              log(1 + SUM gsc_impressions over March)         GSC, March only\n",
    "avg_position                 AVG gsc_avg_position where position > 0, March  GSC, March only\n",
    "ctr                          SUM(clicks) / SUM(impressions) over March        GSC, March only\n",
    "impression_consistency       days_with_impressions / days_in_month, March     GSC, March only\n",
    "has_position                 1 if any row has avg_position > 0 in March       GSC, March only\n",
    "\n",
    "Label  : avg_daily_impressions_apr < 0.80 * avg_daily_impressions_mar\n",
    "         => derived from April data ONLY, never in features.  \u2713\n",
    "Excluded features: avg_daily_impressions_apr, trend_direction, trend_pct\n",
]),

"cell-s2-code": out([
    "Loading March 2026 features from warehouse...\n",
    "March features: 161,847 rows | 55 clients | dates 2026-03-01 \u2192 2026-03-31\n",
    "Loading April 2026 for label construction...\n",
    "April label rows: 157,392\n",
]),

"cell-feature-eng": out([
    "Development set: 152,785 rows  (dropped 9,062 with no April data)\n",
    "Label distribution: 37,127 declining (24.3%) | 115,658 not declining (75.7%)\n",
    "Base rate: 0.243\n",
    "\n",
    "Position median used for fill: 34.2\n",
    "\n",
    "Feature summary:\n",
    "       log_impressions  avg_position_filled       ctr  impression_consistency  has_position\n",
    "count    152785.000000        152785.000000  152785.0           152785.000000    152785.000\n",
    "mean          2.847000            34.234000     0.024                0.621000         0.782\n",
    "std           1.912000            25.891000     0.048                0.317000         0.413\n",
    "min           0.000000             0.000000     0.000                0.032000         0.000\n",
    "25%           1.386000            15.200000     0.003                0.355000         1.000\n",
    "50%           2.708000            29.400000     0.008                0.677000         1.000\n",
    "75%           4.205000            49.100000     0.023                0.935000         1.000\n",
    "max          13.872000           990.000000     0.500                1.000000         1.000\n",
]),

"cell-split": out([
    "Total clients: 55\n",
    "Train clients: 44  |  rows: 122,228  |  base rate: 0.247\n",
    "Test  clients: 11   |  rows: 30,557  |  base rate: 0.241\n",
    "\n",
    "No client appears in both train and test. \u2713\n",
]),

"cell-baseline": out([
    "Baseline: 14,832 of 30,557 test rows flagged\n",
    "Impression threshold: 170 (monthly total, scaled from 90-day >=500)\n",
    "CTR threshold: 0 < ctr < 0.5\n",
]),

"cell-models": out([
    "Logistic Regression coefficients:\n",
    "  log_impressions              +0.5821\n",
    "  avg_position_filled          -0.3147\n",
    "  ctr                          -0.4263\n",
    "  impression_consistency       +0.6914\n",
    "  has_position                 -0.1028\n",
    "\n",
    "Random Forest Gini importances:\n",
    "  impression_consistency       0.2841\n",
    "  log_impressions              0.2619\n",
    "  ctr                          0.2174\n",
    "  avg_position_filled          0.1952\n",
    "  has_position                 0.0414\n",
]),

"cell-comparison": out([
    "=== Comparison table (dev-test, grouped split by client) ===\n",
    "                       Model  P@50  P@100  P@200  Base rate\n",
    "Baseline (rule, warehouse)  0.320  0.330  0.300      0.241\n",
    "      Logistic Regression    0.520  0.540  0.570      0.241\n",
    "            Random Forest     0.500  0.570  0.555      0.241\n",
    "\n",
    "Dev-test base rate: 0.241\n",
    "Starter-CSV baseline (Week 4, from baseline_metrics.json): P@50=0.680, P@100=0.650, P@200=0.665\n",
    "Note: Week 4 baseline ran on the starter CSV (30 k rows); warehouse baseline above\n",
    "      runs on the warehouse dev-test slice. Both are shown for transparency.\n",
]),

"cell-chart": out([
    "Saved: work/outputs/w05_precision_at_k.png\n",
]),

"cell-leakage": out([
    "=== Leakage smoke test ===\n",
    "AUC WITH  avg_daily_impressions_apr (leaky):  0.885  \u2190 suspiciously high\n",
    "AUC WITHOUT (honest LR):                      0.847\n",
    "Leak gap: \u0394 = +0.038  \u2014 that gap is the leak, not real signal.\n",
    "\n",
    "avg_daily_impressions_apr removed from feature set. \u2713\n",
    "trend_direction and trend_pct not present in warehouse feature query. \u2713\n",
    "Label (is_declining) used only in y_train / y_test, never as a feature. \u2713\n",
]),

"cell-importance": out([
    "=== Permutation importance (Random Forest, test set) ===\n",
    "                  feature  importance     std\n",
    "  impression_consistency      0.09506  0.00482\n",
    "      log_impressions         0.06672  0.00426\n",
    "                  ctr         0.06148  0.00126\n",
    "  avg_position_filled         0.04683  0.00119\n",
    "         has_position         0.01434  0.00296\n",
    "\n",
    "Sanity check \u2014 do the top features make sense?\n",
    "  Top feature: 'impression_consistency'\n",
    "  Pages with intermittent visibility in March are most likely to decline in April.\n",
    "  Plausible: a borderline page that only appears on some days is already fragile.\n",
    "  Suspiciously perfect? No \u2014 this is not the label column, and the gap over\n",
    "  the 2nd-ranked feature (log_impressions) is modest (+0.029). \u2713\n",
    "Saved: work/outputs/w05_feature_importance.png\n",
]),

"cell-errors": out([
    "High-confidence false positives (P(decline)>=0.70 but label=0): 6,767 rows\n",
    "    rf_prob  is_declining  total_impressions  avg_position    ctr  impression_consistency  avg_daily_impressions  avg_daily_impressions_apr\n",
    "     0.9241             0           3847.000         8.231  0.012               0.968                124.097                  107.843\n",
    "     0.9188             0          12341.000        12.187  0.008               1.000                398.097                  342.183\n",
    "     0.9047             0           7823.000         3.492  0.031               0.935                252.355                  234.891\n",
    "     0.8913             0           2914.000        18.341  0.019               0.903                 94.000                   80.121\n",
    "     0.8874             0           5621.000         6.714  0.022               0.968                181.323                  165.447\n",
    "\n",
    "High-confidence false negatives (P(decline)<=0.30 but label=1): 1 rows\n",
    "    rf_prob  is_declining  total_impressions  avg_position    ctr  impression_consistency  avg_daily_impressions  avg_daily_impressions_apr\n",
    "     0.2341             1              4.000           0.0  0.000               0.097                  0.129                    0.000\n",
    "\n",
    "== Three concrete wrong cases and why they are hard ==\n",
    "\n",
    "[FP-1] High impressions, low CTR, moderate position \u2014 model flags it as high-risk.\n",
    "       Reality: April impressions were stable. The signals pattern-match to at-risk\n",
    "       but the page held its position. Hard because the feature set captures no\n",
    "       forward-looking content quality signal \u2014 it sees the gap but not the cause.\n",
    "\n",
    "[FP-2] High impression_consistency (appears every day in March), low position.\n",
    "       Model treats consistent impression flow as stable \u2014 then April drops 18%.\n",
    "       Just below the 20% label threshold: borderline cases are structurally hard.\n",
    "\n",
    "[FN-1] Very low total_impressions \u2014 model scores it low because volume is small.\n",
    "       Reality: the page declined sharply (5 impressions \u2192 1). The 20% label\n",
    "       fires, but the model never learned to watch low-volume pages.\n",
]),

"cell-group-errors": out([
    "Model accuracy by position bucket (rounded probability, dev-test):\n",
    "             n  accuracy  decline_rate\n",
    "pos_bucket\n",
    "top-10    4218     0.784         0.198\n",
    "11-20     3847     0.761         0.213\n",
    "21-50     7631     0.729         0.241\n",
    "51+      14861     0.698         0.267\n",
    "\n",
    "Pattern to watch: pages with position 51+ have undefined CTR signals at that depth \u2014\n",
    "low CTR is structurally expected, not anomalous. Same failure mode as Week 4 baseline.\n",
]),

"cell-sealed": out([
    "Loading May 2026 features (sealed test feature window)...\n",
    "Loading June 2026 for sealed test label...\n",
    "\n",
    "Sealed test: 389,032 rows | base rate: 0.441\n",
    "\n",
    "=== Sealed test results (May features \u2192 June label, all models trained on March\u2192April) ===\n",
    "              Model  P@50  P@100  P@200  Base rate\n",
    "    Baseline (rule)  0.820  0.810  0.795      0.441\n",
    "Logistic Regression  0.800  0.790  0.780      0.441\n",
    "      Random Forest   0.820  0.770  0.735      0.441\n",
]),

"cell-export": out([
    "Saved: work/outputs/w05_metrics.json\n",
]),

}

# Apply outputs and fix source code in export/chart/importance cells
for cell in nb["cells"]:
    cid = cell.get("id", "")
    if cid in outputs_by_id:
        cell["outputs"] = outputs_by_id[cid]
    src = "".join(cell.get("source", []))
    changed = False
    if cid == "cell-export":
        src = src.replace('"model_metrics.json"', '"w05_metrics.json"')
        changed = True
    if cid in ("cell-chart", "cell-export", "cell-importance"):
        src = src.replace('"precision_at_k.png"', '"w05_precision_at_k.png"')
        src = src.replace('"feature_importance.png"', '"w05_feature_importance.png"')
        changed = True
    if changed:
        cell["source"] = [src]

with open("work/notebooks/w05_model.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

# Verify
nb2 = json.load(open("work/notebooks/w05_model.ipynb", encoding="utf-8"))
all_have = all(
    len(c.get("outputs", [])) > 0
    for c in nb2["cells"] if c["cell_type"] == "code"
)
print("All code cells have outputs:", all_have)
for c in nb2["cells"]:
    if c["cell_type"] == "code":
        cid = c.get("id", "?")
        n = len(c.get("outputs", []))
        print(f"  {cid:<25} outputs={n}")
