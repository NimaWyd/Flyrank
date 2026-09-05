"""
Full ranking signal analysis pipeline.
Pulls warehouse data, trains LR + RF, evaluates, saves metrics + charts.
Run with: HF_TOKEN="hf_..." python work/pipeline.py
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
assert HF_TOKEN, "HF_TOKEN not set"

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

BASE = "hf://datasets/FlyRank/internship-warehouse"
FACT_MAR = f"{BASE}/fact_content_daily_performance/month=2026-03/*.parquet"
FACT_APR = f"{BASE}/fact_content_daily_performance/month=2026-04/*.parquet"
FACT_MAY = f"{BASE}/fact_content_daily_performance/month=2026-05/*.parquet"
FACT_JUN = f"{BASE}/fact_content_daily_performance/month=2026-06/*.parquet"

# ── Download parquet files via HuggingFace Hub ───────────────────────────────
import tempfile
from huggingface_hub import hf_hub_download

REPO_ID = "FlyRank/internship-warehouse"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_parquet(month: str) -> str:
    path = f"fact_content_daily_performance/month={month}/data_0.parquet"
    local = os.path.join(CACHE_DIR, f"month={month}.parquet")
    if not os.path.exists(local):
        print(f"  Downloading {month}...")
        src = hf_hub_download(
            repo_id=REPO_ID, filename=path,
            repo_type="dataset", token=HF_TOKEN,
            local_dir=CACHE_DIR
        )
        import shutil
        shutil.copy(src, local)
    else:
        print(f"  Using cached {month}")
    return local

print("Connecting to warehouse (via huggingface_hub download)...")
con = duckdb.connect()

def pull_features(local_path: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            client_hash_id,
            content_hash_id,
            SUM(gsc_impressions)                                               AS total_impressions,
            AVG(gsc_impressions)                                               AS avg_daily_impressions,
            AVG(CASE WHEN gsc_avg_position > 0 THEN gsc_avg_position END)     AS avg_position,
            CASE WHEN SUM(gsc_impressions) > 0
                 THEN SUM(gsc_clicks) * 1.0 / SUM(gsc_impressions)
                 ELSE NULL END                                                 AS ctr,
            SUM(CASE WHEN gsc_impressions > 0 THEN 1 ELSE 0 END)              AS days_with_impressions,
            COUNT(DISTINCT report_date)                                        AS days_in_month
        FROM read_parquet('{local_path}')
        GROUP BY client_hash_id, content_hash_id
    """).df()

def pull_labels(local_path: str) -> pd.DataFrame:
    return con.execute(f"""
        SELECT
            client_hash_id,
            content_hash_id,
            AVG(gsc_impressions) AS avg_daily_impressions_label
        FROM read_parquet('{local_path}')
        GROUP BY client_hash_id, content_hash_id
    """).df()

# ── Pull data ────────────────────────────────────────────────────────────────
mar_path = get_parquet("2026-03")
apr_path = get_parquet("2026-04")
may_path = get_parquet("2026-05")
jun_path = get_parquet("2026-06")

print("Pulling March 2026 features...")
mar = pull_features(mar_path)
print(f"  March: {len(mar):,} rows")

print("Pulling April 2026 labels...")
apr = pull_labels(apr_path)
print(f"  April: {len(apr):,} rows")

print("Pulling May 2026 features (sealed)...")
may = pull_features(may_path)
print(f"  May:   {len(may):,} rows")

print("Pulling June 2026 labels (sealed)...")
jun = pull_labels(jun_path)
print(f"  June:  {len(jun):,} rows")

# ── Build dev frame ──────────────────────────────────────────────────────────
dev = mar.merge(apr, on=["client_hash_id", "content_hash_id"], how="inner")
dev["is_declining"] = (
    dev["avg_daily_impressions_label"] < dev["avg_daily_impressions"] * 0.80
).astype(int)

print(f"\nDev frame: {len(dev):,} rows, {dev['client_hash_id'].nunique()} clients")
base_rate_dev = dev["is_declining"].mean()
print(f"Dev base rate: {base_rate_dev:.4f}")

# ── Feature engineering ──────────────────────────────────────────────────────
def engineer(df):
    df = df.copy()
    df["log_impressions"] = np.log1p(df["total_impressions"])
    median_pos = df["avg_position"].median()
    df["avg_position_filled"] = df["avg_position"].fillna(median_pos)
    df["has_position"] = (df["avg_position"] > 0).astype(float)
    df["impression_consistency"] = df["days_with_impressions"] / df["days_in_month"]
    df["ctr"] = df["ctr"].fillna(0.0)
    return df

dev = engineer(dev)
FEATURE_COLS = ["log_impressions", "avg_position_filled", "ctr",
                "impression_consistency", "has_position"]

# ── Grouped split ────────────────────────────────────────────────────────────
clients = dev["client_hash_id"].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(dev, groups=clients))

train_clients = set(dev.iloc[train_idx]["client_hash_id"])
test_clients  = set(dev.iloc[test_idx]["client_hash_id"])
assert train_clients.isdisjoint(test_clients), "Client leakage!"
print(f"\nSplit: {len(train_idx):,} train / {len(test_idx):,} test rows")
print(f"  Train clients: {len(train_clients)} | Test clients: {len(test_clients)}")

X_train = dev.iloc[train_idx][FEATURE_COLS].values
X_test  = dev.iloc[test_idx][FEATURE_COLS].values
y_train = dev.iloc[train_idx]["is_declining"].values
y_test  = dev.iloc[test_idx]["is_declining"].values
g_test  = dev.iloc[test_idx]["client_hash_id"].values

base_rate_test = y_test.mean()
print(f"Test base rate: {base_rate_test:.4f}")

# ── Precision@K helper ───────────────────────────────────────────────────────
def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return y_true[idx].mean()

def eval_model(scores, y_true, name):
    p50  = precision_at_k(y_true, scores, 50)
    p100 = precision_at_k(y_true, scores, 100)
    p200 = precision_at_k(y_true, scores, 200)
    print(f"  {name:30s}  P@50={p50:.3f}  P@100={p100:.3f}  P@200={p200:.3f}")
    return {"p50": round(p50, 4), "p100": round(p100, 4), "p200": round(p200, 4)}

print("\n── Dev-test evaluation ──")

# Baseline rule (scaled: 170 monthly impression threshold)
IMPR_THRESHOLD = 170
baseline_mask = (
    (dev.iloc[test_idx]["total_impressions"] >= IMPR_THRESHOLD) &
    (dev.iloc[test_idx]["ctr"] > 0) &
    (dev.iloc[test_idx]["ctr"] < 0.5)
).values
baseline_score = np.where(
    baseline_mask,
    dev.iloc[test_idx]["total_impressions"].values / dev.iloc[test_idx]["ctr"].clip(lower=1e-6).values,
    0.0
)
baseline_metrics = eval_model(baseline_score, y_test, "Heuristic rule (scaled)")

# Logistic Regression
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)
lr = LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
lr.fit(X_tr_s, y_train)
lr_scores = lr.predict_proba(X_te_s)[:, 1]
lr_metrics = eval_model(lr_scores, y_test, "Logistic Regression")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=20,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
rf_scores = rf.predict_proba(X_test)[:, 1]
rf_metrics = eval_model(rf_scores, y_test, "Random Forest")

auc_lr = roc_auc_score(y_test, lr_scores)
auc_rf = roc_auc_score(y_test, rf_scores)
print(f"\n  AUC — LR: {auc_lr:.3f}  RF: {auc_rf:.3f}")

# ── Leakage smoke test ───────────────────────────────────────────────────────
print("\n── Leakage smoke test ──")
leaky_cols = FEATURE_COLS + ["avg_daily_impressions_label"]
work_leak = dev.iloc[train_idx][leaky_cols + ["is_declining"]].dropna()
X_leak = StandardScaler().fit_transform(work_leak[leaky_cols].values)
y_leak = work_leak["is_declining"].values
lr_leaky = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_leak, y_leak)
auc_leaky = roc_auc_score(y_leak, lr_leaky.predict_proba(X_leak)[:, 1])

work_clean = dev.iloc[train_idx][FEATURE_COLS + ["is_declining"]].dropna()
X_clean = StandardScaler().fit_transform(work_clean[FEATURE_COLS].values)
y_clean = work_clean["is_declining"].values
lr_clean = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_clean, y_clean)
auc_clean = roc_auc_score(y_clean, lr_clean.predict_proba(X_clean)[:, 1])

print(f"  AUC WITH leaky feature:    {auc_leaky:.3f}  ← inflated")
print(f"  AUC WITHOUT leaky feature: {auc_clean:.3f}")
print(f"  Leak delta: {auc_leaky - auc_clean:+.3f}")

# ── Permutation importance ───────────────────────────────────────────────────
print("\nComputing permutation importance...")
pi = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10, random_state=SEED, scoring="average_precision"
)
importance_means = pi.importances_mean
importance_stds  = pi.importances_std

fig, ax = plt.subplots(figsize=(8, 4))
idx_sorted = np.argsort(importance_means)
colors = ["#1a6feb" if m > 0 else "#dc2626" for m in importance_means[idx_sorted]]
ax.barh(
    [FEATURE_COLS[i] for i in idx_sorted],
    importance_means[idx_sorted],
    xerr=importance_stds[idx_sorted],
    color=colors, ecolor="#6b7280", capsize=3
)
ax.axvline(0, color="#374151", linewidth=0.8)
ax.set_xlabel("Mean drop in Average Precision (10 repeats)")
ax.set_title("Permutation Feature Importance — Random Forest (dev-test hold-out)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=150)
plt.close()
print("  Saved feature_importance.png")

# ── Precision@K curve ────────────────────────────────────────────────────────
ks = [10, 20, 30, 50, 75, 100, 150, 200]
fig, ax = plt.subplots(figsize=(9, 5))
for scores, label, color, ls in [
    (baseline_score, "Heuristic rule", "#9aa3b2", "--"),
    (lr_scores,      "Logistic Regression", "#f59e0b", ":"),
    (rf_scores,      "Random Forest", "#1a6feb", "-"),
]:
    pk = [precision_at_k(y_test, scores, k) for k in ks]
    ax.plot(ks, pk, label=label, color=color, linestyle=ls, linewidth=2.5, marker="o", markersize=4)

ax.axhline(base_rate_test, color="#b8860b", linewidth=1.4, linestyle="--", label=f"Base rate ({base_rate_test:.3f})")
ax.set_xlabel("K — queue rows reviewed")
ax.set_ylabel("Precision@K")
ax.set_title("Precision@K — Dev-test hold-out (client-grouped split)")
ax.legend(fontsize=10)
ax.set_ylim(0, 1)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "precision_at_k.png"), dpi=150)
plt.close()
print("  Saved precision_at_k.png")

# ── Error analysis ───────────────────────────────────────────────────────────
test_df = dev.iloc[test_idx].copy()
test_df["rf_score"] = rf_scores
test_df["y_true"]   = y_test

fp = test_df[(test_df["rf_score"] >= 0.70) & (test_df["y_true"] == 0)]
fn = test_df[(test_df["rf_score"] <= 0.30) & (test_df["y_true"] == 1)]
print(f"\n── Error analysis ──")
print(f"  False positives (score≥0.70, label=0): {len(fp):,}")
print(f"  False negatives (score≤0.30, label=1): {len(fn):,}")

bins = [0, 10, 20, 30, 50, 999]
labels_bins = ["1-10","11-20","21-30","31-50","51+"]
test_df["pos_bucket"] = pd.cut(
    test_df["avg_position_filled"], bins=bins, labels=labels_bins, right=True
)
pos_acc = test_df.groupby("pos_bucket", observed=True)["y_true"].mean()
print(f"\n  Decline rate by position bucket:")
print(pos_acc.to_string())

# ── Sealed test (May → June) ─────────────────────────────────────────────────
print("\n── Sealed test (May → June 2026) ──")
sealed = may.merge(jun, on=["client_hash_id", "content_hash_id"], how="inner")
sealed["is_declining"] = (
    sealed["avg_daily_impressions_label"] < sealed["avg_daily_impressions"] * 0.80
).astype(int)
sealed = engineer(sealed)

print(f"  Sealed rows: {len(sealed):,}, clients: {sealed['client_hash_id'].nunique()}")
base_rate_sealed = sealed["is_declining"].mean()
print(f"  Sealed base rate: {base_rate_sealed:.4f}")

X_sealed = sealed[FEATURE_COLS].values
y_sealed  = sealed["is_declining"].values

# Sealed baseline
sealed_mask = (
    (sealed["total_impressions"] >= IMPR_THRESHOLD) &
    (sealed["ctr"] > 0) &
    (sealed["ctr"] < 0.5)
).values
sealed_baseline_score = np.where(
    sealed_mask,
    sealed["total_impressions"].values / sealed["ctr"].clip(lower=1e-6).values,
    0.0
)
sealed_baseline = eval_model(sealed_baseline_score, y_sealed, "Sealed — heuristic rule")
sealed_rf_score = rf.predict_proba(X_sealed)[:, 1]
sealed_rf = eval_model(sealed_rf_score, y_sealed, "Sealed — Random Forest")

# ── Tier analysis (action playbook) ──────────────────────────────────────────
print("\n── Tier analysis ──")
starter_csv = os.path.join(os.path.dirname(__file__), "outputs", "baseline_action_score.csv")

def assign_tier(row):
    impr = row.get("impressions_90d", row.get("total_impressions", 0))
    pos  = row.get("avg_position", 0) or 0
    ctr  = row.get("ctr", 0) or 0
    if impr >= 500 and 0 < pos <= 20 and 0 < ctr < 0.5:
        return "ctr_fix_page1"
    if impr >= 500 and pos > 20 and 0 < ctr < 0.5:
        return "rank_first"
    if impr >= 200:
        return "monitor_stable"
    return "deprioritize"

# Use dev frame (warehouse) for tier analysis with scaled threshold
tier_df = dev.copy()
tier_df["impressions_90d"] = tier_df["total_impressions"]
tier_df["tier"] = tier_df.apply(assign_tier, axis=1)
tier_summary = (
    tier_df.groupby("tier")
    .agg(n=("is_declining","count"), decline_rate=("is_declining","mean"))
    .reset_index()
    .sort_values("decline_rate", ascending=False)
)
print(tier_summary.to_string(index=False))

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = {
    "dev": {
        "base_rate": round(float(base_rate_test), 4),
        "n_rows": int(len(y_test)),
        "n_clients": int(len(test_clients)),
        "baseline_rule": baseline_metrics,
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "auc_lr": round(float(auc_lr), 4),
        "auc_rf": round(float(auc_rf), 4),
        "leakage_auc_with_leak": round(float(auc_leaky), 4),
        "leakage_auc_without_leak": round(float(auc_clean), 4),
        "error_analysis": {
            "false_positives_n": int(len(fp)),
            "false_negatives_n": int(len(fn))
        }
    },
    "sealed": {
        "base_rate": round(float(base_rate_sealed), 4),
        "n_rows": int(len(y_sealed)),
        "n_clients": int(sealed["client_hash_id"].nunique()),
        "baseline_rule": sealed_baseline,
        "random_forest": sealed_rf
    },
    "tier_summary": tier_summary.to_dict(orient="records"),
    "permutation_importance": {
        col: {"mean": round(float(importance_means[i]), 5),
              "std":  round(float(importance_stds[i]), 5)}
        for i, col in enumerate(FEATURE_COLS)
    }
}

out_path = os.path.join(OUT, "model_metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved {out_path}")
print("\n=== DONE ===")
print(json.dumps(metrics, indent=2))
