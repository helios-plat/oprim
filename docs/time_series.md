# Time Series Operations

## ops.log_returns
Compute log returns from price series. `r_t = ln(P_t / P_{t-period})`

## ops.cumulative_returns
Compute equity curve from returns. Supports log/simple, compound/non-compound.

## ops.rolling_window_split
Generate (start, end) index pairs for rolling windows.

## ops.lag_forward_fill
Forward-fill with max gap limit + optional lag shift.

## ops.percentile_rank
Percentile rank: rolling, expanding, or cross-sectional.

## ops.ewma_smooth
Exponentially Weighted Moving Average with half_life/span/alpha parameterization.

## ops.realized_vol
Rolling volatility: close-to-close, Garman-Klass, Parkinson estimators.

## ops.zscore_normalize
Z-score normalization with rolling/expanding window + clip.

## ops.gap_detect
Detect and classify gaps in time series by severity.

## ops.resample_align
Align multiple DataFrames to common frequency/timezone.

## ops.purge_embargo_split
Purge-embargo CV split per López de Prado 2018 Ch 7.
