# SplitGuard examples

`generate_synthetic_data.py` builds reproducible datasets (fixed seed, numpy/pandas
only) under `examples/data/`:

```bash
python examples/generate_synthetic_data.py
```

| Folder | Scenario | Expected result |
| --- | --- | --- |
| `01_clean` | Clean binary classification | No findings |
| `02_target_copy` | `churn_score_final` is a numeric copy of `churn` | TGT-001 critical (plus a TMP-005 name hint) |
| `03_strong_legit` | Regression, `sqft` strongly but legitimately drives `price` | Only TGT-005 medium / **low confidence** |
| `04_cross_duplicates` | 25 test rows copied from train | DUP-002 critical (25 rows, 12.5%) |
| `05_repeated_entities` | Customers appear in both partitions | DUP-003, high with `--scenario new_entities` |
| `06_post_outcome_feature` | `refund_amount` recorded 15 days after the outcome | TMP-001 high, TMP-002 critical |
| `07_temporal_clean` | Chronological split, features snapshotted before prediction | No findings |
| `08_high_cardinality_id` | Unique `transaction_id`, disjoint across partitions | ID-001 low only |
| `09_schema_mismatch` | Test drops `income`, adds `promo_code`, `x2` becomes text | SCH-001, SCH-002, SCH-003 |
| `10_unsplit` | One file, repeated patients, visit dates | `needs_user_input` until a scenario is given |
| `10b_split_column` | Same data with a `split` column | Mode B audit |

## Commands

```bash
D=examples/data

splitguard audit --train $D/02_target_copy/train.csv --test $D/02_target_copy/test.csv --target churn

splitguard audit --train $D/05_repeated_entities/train.csv --test $D/05_repeated_entities/test.csv \
  --target churn --entity-id customer_id --scenario new_entities

splitguard audit --train $D/06_post_outcome_feature/train.csv --test $D/06_post_outcome_feature/test.csv \
  --target churn --timestamp prediction_time --outcome-timestamp outcome_time \
  --availability refund_amount=refund_amount_available_at \
  --availability tenure_months=tenure_available_at \
  --scenario forecasting --output report.json --fail-on high      # exits 1

splitguard audit --data $D/10b_split_column/data.csv --split-column split --target churn --entity-id patient_id

splitguard inspect --data $D/10_unsplit/data.csv --target churn --entity-id patient_id --timestamp visit_date
splitguard inspect --data $D/10_unsplit/data.csv --target churn --entity-id patient_id --scenario new_entities

splitguard split --data $D/10_unsplit/data.csv --target churn --entity-id patient_id \
  --scenario new_entities --confirm-strategy group --out-dir examples/output/split
```

`sample_report.json` is the report from the `06_post_outcome_feature` command above,
generated with `--no-timestamp`.
