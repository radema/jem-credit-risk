# Future Feature Engineering Ideas

## 1. Join Flags (`*_joined_flag`)
- **Description:** When joining external tables (Depth-0, Depth-1, Depth-2) onto the base dataset, create a binary indicator feature (e.g., `*_joined_flag`).
- **Value:** `1` if the case_id had matching records in the joined table, `0` if it was entirely missing.
- **Rationale:** The presence or absence of data in specific tables (like credit bureau history) carries a strong predictive signal. This explicitly passes the availability signal to the model before missing columns are imputed.

## 2. Advanced Aggregations (Depth 1 & 2)
### Numeric Features
Instead of only calculating the `max` and `mean`, expand aggregations to capture the full distribution:
- `max`
- `min`
- `mean`
- `std` (standard deviation)
- `iqr` (Interquartile Range)

### Categorical / String Features
Expand beyond simply taking the `last` observed value:
- `mode` (most frequent)
- `last` (most recent)
- `first` (oldest)
- Number of times the `mode` appeared (frequency count of the mode)
- Number of times the `last` value appeared
- Number of times the `first` value appeared
- **Temporal Metric:** Time elapsed since the first occurrence of the `last` value (requires a date/time column associated with the records).
