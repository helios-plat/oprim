# oprim

Atomic operations library — Layer 1 meta-primitives for financial analysis.

## Overview

`oprim` provides 31 atomic, stateless, pure-function operations for quantitative finance. Each operation:

- Completes a single mathematical/statistical task independently
- Depends only on standard libraries (numpy, scipy, pandas, scikit-learn, statsmodels)
- Never calls another oprim operation internally

## Install

```bash
pip install oprim
```

## Development

```bash
pip install -e ".[dev]"
pytest --cov
ruff check .
```

## Architecture

```
Layer 1: oprim (this library) — atomic ops, mutually independent
Layer 0: numpy / scipy / pandas / sklearn / statsmodels
```

See [ADR-061](https://github.com/helios-plat/oprim) for full specification.
