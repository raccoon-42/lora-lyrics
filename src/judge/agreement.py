"""Inter-rater agreement (export seam, file 3 of 3).

Consumes ONLY the tidy long table written by runner.py
(results/judge/scores.jsonl): one row per (item_id, judge, criterion, target,
score). Computes agreement PER CRITERION, because the rating units differ:

  style_match  -> 100 single items
  coherence    -> all 140 items (widest pool -> most stable kappa)
  presence     -> 80 (blend item x target) units (thinnest pool -> least stable)

Agreement is LLM-vs-LLM here (no human rater yet): it measures reliability, not
validity, and cannot detect shared model bias -- disclose that in the writeup.

Stats per criterion:
  - mean pairwise quadratic-weighted Cohen's kappa  (primary)
  - percent agreement: exact + within-1             (kappa-paradox guard)
  - Krippendorff's alpha, ordinal                   (robustness)
  - ICC(2,k), two-way random, absolute agreement    (robustness)

Run from src/:  uv run python -m judge.agreement
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import krippendorff
import pingouin as pg
from sklearn.metrics import cohen_kappa_score

from config import RESULTS_DIR

SCORES_PATH = RESULTS_DIR / "judge" / "scores.jsonl"
AGREEMENT_PATH = RESULTS_DIR / "judge" / "agreement.json"


def load_scores(path=SCORES_PATH):
    rows = [json.loads(l) for l in path.open()]
    df = pd.DataFrame(rows)
    n_null = int(df["score"].isna().sum())
    df = df.dropna(subset=["score"]).copy()
    df["score"] = df["score"].astype(int)
    # target may be None (JSON null) -> a single unit dimension alongside item_id.
    df["target"] = df["target"].astype("string").fillna("_")
    df["unit"] = df["item_id"] + "|" + df["target"]
    return df, n_null


def _pivot(df_crit):
    """units x judges score matrix (NaN where a judge has no score for a unit)."""
    return df_crit.pivot_table(index="unit", columns="judge", values="score")


def pairwise_qwk(mat):
    """Mean quadratic-weighted Cohen's kappa over judge pairs (pairwise-complete)."""
    judges = list(mat.columns)
    per_pair = {}
    for a, b in combinations(judges, 2):
        both = mat[[a, b]].dropna()
        if len(both) < 2 or both[a].nunique() < 2 and both[b].nunique() < 2:
            per_pair[f"{a} vs {b}"] = None
            continue
        per_pair[f"{a} vs {b}"] = float(
            cohen_kappa_score(both[a], both[b], weights="quadratic", labels=[1, 2, 3, 4, 5])
        )
    vals = [v for v in per_pair.values() if v is not None]
    return (float(np.mean(vals)) if vals else None), per_pair


def pairwise_pct(mat):
    """Mean pairwise exact and within-1 agreement over judge pairs."""
    judges = list(mat.columns)
    exact, within1 = [], []
    for a, b in combinations(judges, 2):
        both = mat[[a, b]].dropna()
        if not len(both):
            continue
        d = (both[a] - both[b]).abs()
        exact.append(float((d == 0).mean()))
        within1.append(float((d <= 1).mean()))
    return (float(np.mean(exact)) if exact else None,
            float(np.mean(within1)) if within1 else None)


def krippendorff_ordinal(mat):
    # reliability_data = raters x units, NaN for missing.
    data = mat.T.to_numpy(dtype=float)
    if np.isnan(data).all():
        return None
    try:
        return float(krippendorff.alpha(reliability_data=data,
                                         level_of_measurement="ordinal"))
    except Exception:
        return None


def icc2k(df_crit):
    """ICC(2,k) two-way random, absolute agreement, average measures.
    pingouin needs balanced data -> restrict to units rated by every judge."""
    n_judges = df_crit["judge"].nunique()
    counts = df_crit.groupby("unit")["judge"].nunique()
    complete = counts[counts == n_judges].index
    sub = df_crit[df_crit["unit"].isin(complete)]
    if sub["unit"].nunique() < 2:
        return None, 0
    try:
        icc = pg.intraclass_corr(data=sub, targets="unit", raters="judge",
                                 ratings="score")
        # ICC(2,k) = two-way random, absolute agreement, average measures = pingouin "ICC(A,k)".
        row = icc[icc["Type"] == "ICC(A,k)"]
        val = float(row["ICC"].iloc[0]) if len(row) else None
    except Exception:
        val = None
    return val, int(len(complete))


def compute(df):
    out = {}
    for criterion in sorted(df["criterion"].unique()):
        dc = df[df["criterion"] == criterion]
        mat = _pivot(dc)
        qwk, per_pair = pairwise_qwk(mat)
        pct_exact, pct_within1 = pairwise_pct(mat)
        icc, n_complete = icc2k(dc)
        out[criterion] = {
            "n_units": int(mat.shape[0]),
            "n_judges": int(mat.shape[1]),
            "n_complete_units": n_complete,
            "mean_quadratic_kappa": qwk,
            "pairwise_kappa": per_pair,
            "pct_agreement_exact": pct_exact,
            "pct_agreement_within1": pct_within1,
            "krippendorff_alpha_ordinal": krippendorff_ordinal(mat),
            "icc_2k": icc,
        }
    return out


def _fmt(x):
    return "  n/a" if x is None else f"{x:5.2f}"


def main():
    df, n_null = load_scores()
    judges = sorted(df["judge"].unique())
    results = compute(df)

    print(f"judges ({len(judges)}): {', '.join(judges)}")
    if n_null:
        print(f"dropped {n_null} unparseable (null) score rows")
    print(f"\n{'criterion':<13}{'units':>6}{'QWK':>7}{'exact':>7}{'≤1':>7}{'Kalpha':>8}{'ICC2k':>7}")
    print("-" * 55)
    for crit, r in results.items():
        print(f"{crit:<13}{r['n_units']:>6}"
              f"{_fmt(r['mean_quadratic_kappa']):>7}"
              f"{_fmt(r['pct_agreement_exact']):>7}"
              f"{_fmt(r['pct_agreement_within1']):>7}"
              f"{_fmt(r['krippendorff_alpha_ordinal']):>8}"
              f"{_fmt(r['icc_2k']):>7}")

    AGREEMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGREEMENT_PATH.write_text(json.dumps(
        {"judges": judges, "n_null_dropped": n_null, "criteria": results}, indent=2))
    print(f"\nwrote {AGREEMENT_PATH}")


if __name__ == "__main__":
    main()
