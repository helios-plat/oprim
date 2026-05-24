# equity_curve_segment_label

Label each row of an equity curve with its segment (train/gap/val/oos).

## Usage

```python
import pandas as pd
from oprim import time_series_split, equity_curve_segment_label

# Assume equity_df has columns [date, equity]
splits = time_series_split(dates=equity_df["date"].tolist(), train_pct=0.6, val_pct=0.2, gap_days=5)

labeled = equity_curve_segment_label(
    equity_curve=equity_df,
    split_dates=splits["split_dates"],
)

print(labeled.head())
#         date  equity segment
# 0 2024-01-01   100.0   train
# 1 2024-01-02   100.5   train
# ...

# Filter by segment
oos_data = labeled[labeled["segment"] == "oos"]
```
