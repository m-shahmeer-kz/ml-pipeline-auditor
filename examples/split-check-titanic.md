# Train/test split check

> LEAKAGE LIKELY: test scores from this split are not trustworthy.

- Train rows: 982 · Test rows: 327
- Test rows identical to a training row: 0
- Target share max gap: 0.0241

- **[HIGH] PROXY**: Feature 'boat' alone predicts 'survived' almost perfectly (category purity = 0.9785). Check that it is known at prediction time and is not computed from the target.
