"""Regime-related atomic operations."""

from __future__ import annotations

import warnings
from itertools import groupby
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd

STABILITY_NEW = "experimental"  # for Sprint 0 additions only


def regime_filter_data(
    data: pd.DataFrame,
    regime_labels: pd.Series,
    target_regime: str | list[str],
    mode: Literal["hard", "soft"] = "hard",
    min_probability: float = 0.5,
) -> pd.DataFrame:
    """Filter data by regime labels.

    Parameters
    ----------
    data : pd.DataFrame
        Data to filter.
    regime_labels : pd.Series
        Regime labels aligned with data index.
    target_regime : str | list[str]
        Target regime(s) to keep.
    mode : {"hard", "soft"}
        Hard = string labels, soft = probability dicts.
    min_probability : float
        Minimum probability threshold (soft mode only).

    Returns
    -------
    pd.DataFrame
        Filtered data.
    """
    if isinstance(target_regime, str):
        target_regime = [target_regime]

    # Align indices
    common_idx = data.index.intersection(regime_labels.index)
    if common_idx.empty:
        raise ValueError("No overlapping indices between data and regime_labels")

    data_aligned = data.loc[common_idx]
    labels_aligned = regime_labels.loc[common_idx]

    if mode == "hard":
        # Check target exists
        unique_labels = labels_aligned.unique()
        for t in target_regime:
            if t not in unique_labels:
                raise ValueError(f"target_regime '{t}' not found in labels")
        mask = labels_aligned.isin(target_regime)
    else:  # soft
        # Check for non-dict elements in soft mode
        non_dict_mask = ~labels_aligned.apply(lambda x: isinstance(x, dict))
        if non_dict_mask.any():
            raise ValueError(
                f"mode='soft' requires dict values, found {non_dict_mask.sum()} non-dict elements"
            )
        
        # Vectorized: convert to DataFrame
        prob_df = pd.DataFrame(labels_aligned.tolist(), index=common_idx)
        mask = (prob_df[target_regime] >= min_probability).any(axis=1)

    return data_aligned[mask]


def regime_transition_matrix(
    regime_labels: pd.Series,
    states: list[str] | None = None,
    include_duration: bool = True,
) -> dict[str, Any]:
    """Estimate transition matrix and duration distribution from regime labels.

    Parameters
    ----------
    regime_labels : pd.Series
        Sequence of regime labels.
    states : list[str] | None
        Explicit state list. None = infer from data.
    include_duration : bool
        Whether to compute duration statistics.

    Returns
    -------
    dict with transition_matrix, duration_distribution, stationary_distribution, n_transitions.
    """
    labels = regime_labels.values
    if states is None:
        states = sorted(set(labels))

    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    # Count transitions and track unknown labels
    trans_counts = np.zeros((n_states, n_states))
    unknown_labels = set()
    skipped_transitions = 0
    
    for i in range(len(labels) - 1):
        if labels[i] in state_idx and labels[i + 1] in state_idx:
            trans_counts[state_idx[labels[i]], state_idx[labels[i + 1]]] += 1
        else:
            skipped_transitions += 1
            if labels[i] not in state_idx:
                unknown_labels.add(labels[i])
            if labels[i + 1] not in state_idx:
                unknown_labels.add(labels[i + 1])
    
    if skipped_transitions > 0 and states is not None:
        warnings.warn(
            f"{skipped_transitions} transitions skipped due to unknown labels: {unknown_labels}",
            stacklevel=2
        )

    # Normalize to probabilities
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_matrix = trans_counts / row_sums

    trans_df = pd.DataFrame(trans_matrix, index=states, columns=states)

    # Stationary distribution - handle non-ergodic chains
    try:
        eigenvalues, eigenvectors = np.linalg.eig(trans_matrix.T)
        # Find eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        # Handle non-ergodic chains: force non-negative
        stationary = np.abs(stationary)
        if stationary.sum() == 0 or np.isnan(stationary.sum()):
            stationary = np.ones(n_states) / n_states
        else:
            stationary = stationary / stationary.sum()
    except Exception:  # pragma: no cover
        stationary = np.ones(n_states) / n_states  # pragma: no cover

    stat_dist = pd.Series(stationary, index=states)

    # Duration distribution - vectorized with groupby
    duration_dist = {}
    if include_duration:
        for state in states:
            durations = [sum(1 for _ in g) for k, g in groupby(labels) if k == state]
            if durations:
                duration_dist[state] = {
                    "mean": float(np.mean(durations)),
                    "median": float(np.median(durations)),
                    "min": int(np.min(durations)),
                    "max": int(np.max(durations)),
                    "count": len(durations),
                }
            else:
                duration_dist[state] = {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}

    n_transitions = int(trans_counts.sum())

    return {
        "transition_matrix": trans_df,
        "duration_distribution": duration_dist,
        "stationary_distribution": stat_dist,
        "n_transitions": n_transitions,
    }


def regime_label_align(
    target_index: pd.DatetimeIndex,
    regime_labels: pd.Series,
    method: Literal["asof", "ffill"] = "asof",
    tolerance: pd.Timedelta | None = None,
) -> pd.Series:
    """Align regime labels to a target time index.

    Parameters
    ----------
    target_index : pd.DatetimeIndex
        Target timestamps to align to.
    regime_labels : pd.Series
        Regime labels with DatetimeIndex.
    method : {"asof", "ffill"}
        Alignment method.
    tolerance : pd.Timedelta | None
        Maximum time gap for alignment.

    Returns
    -------
    pd.Series
        Regime labels aligned to target_index.
    """
    if not isinstance(target_index, pd.DatetimeIndex):
        target_index = pd.DatetimeIndex(target_index)

    # Use merge_asof for both methods (ffill just differs in tolerance handling)
    target_df = pd.DataFrame(index=target_index)
    label_df = pd.DataFrame({"regime": regime_labels.values}, index=regime_labels.index)
    
    if method == "ffill" and tolerance is None:
        # ffill without tolerance: allow unlimited forward-fill
        merged = pd.merge_asof(
            target_df, label_df,
            left_index=True, right_index=True,
            direction="backward",
        )
    else:
        merged = pd.merge_asof(
            target_df, label_df,
            left_index=True, right_index=True,
            direction="backward",
            tolerance=tolerance,
        )
    
    return merged["regime"]


def markov_next_state_distribution(
    current_state: str,
    transition_matrix: dict[str, dict[str, float]],
    n_steps: int = 1,
    trend: Optional[Literal["up", "down", None]] = None,
) -> dict[str, float]:
    """Compute the probability distribution over future states n steps ahead.

    Given current state and a Markov transition matrix, returns the predicted
    distribution over states after n_steps transitions. If trend is provided,
    applies a trend bias to upward/downward transitions (states ordered
    alphabetically).

    Parameters
    ----------
    current_state : current discrete state name
    transition_matrix : {from_state: {to_state: probability}}
                        Each row must sum to 1.0 (within 1e-6 tolerance).
    n_steps : number of forward steps (default 1)
    trend : optional trend bias modifier ("up" biases toward higher-indexed
            states, "down" toward lower-indexed states, None = no bias)

    Returns
    -------
    {state_name: probability} mapping, sums to 1.0

    Raises
    ------
    ValueError : if current_state not in matrix, n_steps < 1, or row sums invalid

    Reference
    ---------
    Norris, J. R. (1997). Markov Chains. Cambridge University Press.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if current_state not in transition_matrix:
        raise ValueError(f"'{current_state}' not found in transition_matrix")

    for from_state, row in transition_matrix.items():
        total = sum(row.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Row '{from_state}' sums to {total:.8f}, must be 1.0 ± 1e-6"
            )

    states = sorted(transition_matrix.keys())
    n = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    M = np.array(
        [
            [transition_matrix.get(states[i], {}).get(states[j], 0.0) for j in range(n)]
            for i in range(n)
        ],
        dtype=np.float64,
    )

    if trend is not None:
        beta = 0.2
        for i in range(n):
            for j in range(n):
                if trend == "up" and j > i:
                    M[i, j] *= 1.0 + beta
                elif trend == "down" and j < i:
                    M[i, j] *= 1.0 + beta
            row_sum = M[i].sum()
            if row_sum > 0:
                M[i] /= row_sum

    Mn = np.linalg.matrix_power(M, n_steps)

    v = np.zeros(n)
    v[state_idx[current_state]] = 1.0

    result = v @ Mn

    return {states[j]: float(result[j]) for j in range(n)}
