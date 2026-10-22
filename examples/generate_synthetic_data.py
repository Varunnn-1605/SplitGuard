"""Generate reproducible synthetic datasets that exercise every SplitGuard detector.

Usage:
    python examples/generate_synthetic_data.py [--out examples/data] [--seed 7]

Every dataset is built with a fixed seed, so reruns produce identical files.
The generator only uses numpy and pandas; it does not import SplitGuard.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_TRAIN, N_TEST = 800, 200


def _classification(rng: np.random.Generator, n: int) -> pd.DataFrame:
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    region = rng.choice(["north", "south", "east", "west"], size=n)
    logit = 0.9 * x1 - 0.6 * x2 + (region == "north") * 0.4
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "x1": x1.round(6),
            "x2": x2.round(6),
            "region": region,
            "income": rng.lognormal(10, 0.5, n).round(2),
            "churn": y,
        }
    )


def _write(out: Path, name: str, **frames: pd.DataFrame) -> None:
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    for part, frame in frames.items():
        frame.to_csv(folder / f"{part}.csv", index=False)


def generate(out: Path, seed: int = 7) -> list[str]:
    rng = np.random.default_rng(seed)
    made = []

    # 1. Clean binary classification.
    _write(out, "01_clean", train=_classification(rng, N_TRAIN), test=_classification(rng, N_TEST))
    made.append("01_clean")

    # 2. Numerical feature copied directly from the target.
    train, test = _classification(rng, N_TRAIN), _classification(rng, N_TEST)
    for df in (train, test):
        df["churn_score_final"] = df["churn"].astype(float)
    _write(out, "02_target_copy", train=train, test=test)
    made.append("02_target_copy")

    # 3. Strong but legitimate relationship (regression; corr ~0.95, not a copy).
    def regression(n: int) -> pd.DataFrame:
        sqft = rng.uniform(500, 3500, n)
        age = rng.uniform(0, 60, n)
        price = 150 * sqft - 800 * age + rng.normal(0, 40_000, n)
        return pd.DataFrame({"sqft": sqft.round(1), "age": age.round(1), "price": price.round(0)})

    _write(out, "03_strong_legit", train=regression(N_TRAIN), test=regression(N_TEST))
    made.append("03_strong_legit")

    # 4. Exact duplicates crossing train/test (25 test rows copied from train).
    train, test = _classification(rng, N_TRAIN), _classification(rng, N_TEST)
    test.iloc[:25] = train.iloc[100:125].to_numpy()
    _write(out, "04_cross_duplicates", train=train, test=test.astype(train.dtypes.to_dict()))
    made.append("04_cross_duplicates")

    # 5. Repeated entities across partitions (random row split of multi-row customers).
    n = N_TRAIN + N_TEST
    data = _classification(rng, n)
    data.insert(0, "customer_id", [f"C{i:04d}" for i in rng.integers(0, 250, n)])
    is_test = rng.random(n) < 0.2
    _write(out, "05_repeated_entities", train=data[~is_test], test=data[is_test])
    made.append("05_repeated_entities")

    # 6. Post-outcome feature with explicit availability metadata.
    def events(n: int, start: str) -> pd.DataFrame:
        pred = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
        outcome = pred + pd.Timedelta(days=30)
        df = _classification(rng, n)
        # refund_amount is entered during follow-up: 15 days AFTER the outcome.
        df["refund_amount"] = (df["churn"] * rng.uniform(10, 50, n)).round(2)
        df["refund_amount_available_at"] = (outcome + pd.Timedelta(days=15)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        # tenure is snapshotted 1 day before prediction time.
        df["tenure_months"] = rng.integers(1, 72, n)
        df["tenure_available_at"] = (pred - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        df["prediction_time"] = pred.strftime("%Y-%m-%dT%H:%M:%SZ")
        df["outcome_time"] = outcome.strftime("%Y-%m-%dT%H:%M:%SZ")
        return df

    _write(
        out,
        "06_post_outcome_feature",
        train=events(N_TRAIN, "2024-01-01"),
        test=events(N_TEST, "2024-07-01"),
    )
    made.append("06_post_outcome_feature")

    # 7. Valid time-based dataset with no post-prediction features.
    def clean_events(n: int, start: str) -> pd.DataFrame:
        df = events(n, start).drop(columns=["refund_amount", "refund_amount_available_at"])
        return df

    _write(
        out,
        "07_temporal_clean",
        train=clean_events(N_TRAIN, "2024-01-01"),
        test=clean_events(N_TEST, "2024-07-01"),
    )
    made.append("07_temporal_clean")

    # 8. High-cardinality but legitimate identifier (unique, disjoint across partitions).
    train, test = _classification(rng, N_TRAIN), _classification(rng, N_TEST)
    ids = rng.permutation(10**6)[: N_TRAIN + N_TEST]
    train.insert(0, "transaction_id", [f"T{i:07d}" for i in ids[:N_TRAIN]])
    test.insert(0, "transaction_id", [f"T{i:07d}" for i in ids[N_TRAIN:]])
    _write(out, "08_high_cardinality_id", train=train, test=test)
    made.append("08_high_cardinality_id")

    # 9. Schema mismatch: test lacks 'income', adds 'promo_code', 'x2' becomes text.
    train, test = _classification(rng, N_TRAIN), _classification(rng, N_TEST)
    test = test.drop(columns=["income"])
    test["promo_code"] = rng.choice(["A", "B"], size=N_TEST)
    test["x2"] = test["x2"].map(lambda v: f"{v:.3f}").astype(object)
    test.loc[:4, "x2"] = "missing"
    _write(out, "09_schema_mismatch", train=train, test=test)
    made.append("09_schema_mismatch")

    # 10. Unsplit dataset needing split-strategy clarification.
    data = _classification(rng, 600)
    data.insert(0, "patient_id", [f"P{i:03d}" for i in rng.integers(0, 120, 600)])
    data["visit_date"] = (
        pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365, 600), unit="D")
    ).strftime("%Y-%m-%d")
    data["split"] = np.where(rng.random(600) < 0.2, "test", "train")
    _write(out, "10_unsplit", data=data.drop(columns=["split"]))
    _write(out, "10b_split_column", data=data)
    made += ["10_unsplit", "10b_split_column"]
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    for name in generate(args.out, args.seed):
        print(f"wrote {args.out / name}")


if __name__ == "__main__":
    main()
