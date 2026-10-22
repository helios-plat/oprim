# Cognitive Modeling Atomic Operations

Atomic operations for cognitive modeling, including Bayesian Knowledge Tracing (BKT) and Free Spaced Repetition Scheduler (FSRS).

## 1. Bayesian Knowledge Tracing (BKT)

BKT is a common algorithm for cognitive modeling, similar to Hidden Markov Models (HMM), used to track the hidden state of a learner's knowledge mastery.

### BKT Functions

#### `bkt_new_state`
Creates a new `KCState` with given prior parameters.

- **Parameters**: `kc_id`, `p_init`, `p_transit`, `p_guess`, `p_slip`.
- **Returns**: `KCState` instance.

#### `bkt_update`
Forgetting-aware BKT update.

- **Parameters**: `state`, `correct`, `current_ts` (optional), `halflife` (optional), `ema_alpha` (default 0.1), `mastery_cap` (default 0.97).
- **Update Logic**:
  1. Applies exponential decay if time has passed and halflife is provided.
  2. Bayesian update based on correctness.
  3. Mastery probability is capped at `mastery_cap`.
  4. `long_term_mastery` is updated via EMA.
- **Returns**: New `KCState`.

#### `bkt_classify_error`
Classifies if an incorrect response was a "careless" mistake or due to "dontknow".

- **Criteria**: Compares the probability of being a slip versus being in a non-mastery state.

#### `bkt_predict_correct`
Predicts the probability that the next response will be correct.

- **Formula**: $P(Correct) = P(Mastery) \cdot (1 - P(Slip)) + (1 - P(Mastery)) \cdot P(Guess)$

## 2. Spaced Repetition (FSRS)

FSRS is a modern spaced repetition algorithm for managing memory retention.

### FSRS Functions

#### `fsrs_new_card`
Creates a new FSRS card.

#### `fsrs_review`
Reviews a card and returns the updated card and review log.

- **Parameters**: `card`, `rating` (1-4), `current_ts` (optional), `weights` (optional).

#### `fsrs_retrievability`
Calculates the current retrievability $R$ (probability of recall).

#### `fsrs_map_rating`
Maps integer 1-4 to FSRS Rating enum.
- 1: Again
- 2: Hard
- 3: Good
- 4: Easy

#### `fsrs_due_date`
Returns the next due date for the card.

## 3. General Functions

#### `exp_forgetting`
Calculates exponential forgetting approximation.
- **Formula**: $R = 0.5^{(days/halflife)}$

## Data Structure: `KCState`

```python
@dataclass
class KCState:
    kc_id: str
    p_init: float = 0.20
    p_transit: float = 0.20
    p_guess: float = 0.15
    p_slip: float = 0.12
    p_mastery: Optional[float] = None
    long_term_mastery: Optional[float] = None
    last_interaction_ts: Optional[float] = None
    n_attempts: int = 0
```
