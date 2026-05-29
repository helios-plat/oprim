# time_series_split

Split a date sequence into train / validation / out-of-sample segments with an optional gap.

## Usage

```python
from datetime import date, timedelta
from oprim import time_series_split

dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(365)]

result = time_series_split(
    dates=dates,
    train_pct=0.6,
    val_pct=0.2,
    gap_days=15,
)

print(result["train"])       # (date(2024, 1, 1), date(2024, 7, 13))
print(result["val"])         # (date(2024, 7, 29), ...)
print(result["oos"])         # (..., date(2024, 12, 30))
print(result["n_train"])     # 219
print(result["gap_days"])    # 15
```
