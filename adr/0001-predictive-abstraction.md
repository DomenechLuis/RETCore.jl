````markdown
# ADR 0001 - Predictive Abstraction Layer

**Status:** Proposed

**Date:** 2026-07-23

**Author:** Luis Doménech Ballester

---

# Context

The current implementation defines prediction-related functionality separately for each statistical model.

Examples include:

- `log_log_quantile`
- `exceedance_probability`
- parts of `plot_means!`

Although the mathematical details differ between models, the structure of the calculations is remarkably similar. The same predictive quantities (quantiles, cumulative probabilities, exceedance probabilities and graphical summaries) are repeatedly implemented in model-specific code.

As a consequence:

- significant code duplication exists between models;
- adding a new model requires implementing the same prediction logic repeatedly;
- future extensions become progressively harder to maintain.

The issue became evident while considering support for operations such as:

```julia
quantile(fo, [0.05, 0.5, 0.95], v)
````

and while examining opportunities to share computations between:

* `quantile`
* `exceedance_probability`
* `tolerance_limit`
* predictive visualisations

***

# Current Model Structure

The current models can all be interpreted as evaluating a probability distribution after a suitable transformation of the physical variables.

## Normal Model

```julia
μ = a + m*v

logpdf(
    Normal(μ, s),
    N
)
```

Equivalent representation:

```julia
Distribution: Normal(μ, s)

Parameters:
    A = (μ, s)

Support value:
    B = N
```

***

## Exponential Model

```julia
pred = a + m*v

d = pred - N

logpdf(
    Exponential(λ),
    d
)
```

Equivalent representation:

```julia
Distribution: Exponential(λ)

Parameters:
    A = (λ,)

Support value:
    B = pred - N
```

***

## Lognormal Model

```julia
pred  = a + m*v
zmode = exp(l - s^2)

z = pred + zmode - N

logpdf(
    LogNormal(l, s),
    z
)
```

Equivalent representation:

```julia
Distribution: LogNormal(l, s)

Parameters:
    A = (l, s)

Support value:
    B = pred + zmode - N
```

***

# Observation

All current models can be described as:

```julia
logpdf(
    D(A...),
    B
)
```

where:

* `D` is a known probability distribution;
* `A` are the distribution parameters;
* `B` is a model-specific transformation of the physical observation.

The differences between models are therefore primarily transformation-related rather than prediction-related.

***

# Proposed Design

Introduce a common predictive abstraction built around three concepts:

## 1. Predictive Distribution

```julia
predictive_distribution(state)
```

Returns the base distribution associated with the model.

Examples:

```julia
Normal(...)
```

```julia
Exponential(...)
```

```julia
LogNormal(...)
```

***

## 2. Support Transformation

```julia
support_transform(state, value)
```

Maps a physical quantity into the support of the predictive distribution.

### Normal

```julia
support_transform(state, N) = N
```

### Exponential

```julia
support_transform(state, N) =
    pred - N
```

### Lognormal

```julia
support_transform(state, N) =
    pred + zmode - N
```

***

## 3. Inverse Support Transformation

```julia
support_inverse(state, z)
```

Maps a value defined on the support of the predictive distribution back into the physical space.

### Normal

```julia
support_inverse(state, z) =
    z
```

### Exponential

```julia
support_inverse(state, z) =
    pred - z
```

### Lognormal

```julia
support_inverse(state, z) =
    pred + zmode - z
```

***

# Generic Prediction Operations

Once the abstraction is available, several functions become model-independent.

## Quantile Evaluation

Conceptually:

```julia
dist = predictive_distribution(state)

z = quantile(dist, p)

value = support_inverse(state, z)
```

***

## Probability Evaluation

Conceptually:

```julia
dist = predictive_distribution(state)

z = support_transform(state, value)

p = cdf(dist, z)
```

or

```julia
p = ccdf(dist, z)
```

depending on the convention used.

***

## Likelihood Evaluation

Conceptually:

```julia
dist = predictive_distribution(state)

z = support_transform(state, observation)

logpdf(dist, z)
```

or

```julia
logcdf(dist, z)
```

for censored observations.

***

# Expected Consequences

## Positive

* Significant reduction of code duplication.
* Cleaner implementation of prediction-related functionality.
* Easier addition of future models.
* Shared implementation of:
  * `quantile`
  * `exceedance_probability`
  * `tolerance_limit`
  * predictive visualisation tools.
* Possibility of future optimisations through reuse of common predictive state.
* Potential support for multiple probabilities evaluated simultaneously:

```julia
quantile(fo, [0.05, 0.5, 0.95], v)
```

without redesigning the API.

***

## Negative

* Requires a substantial refactor of prediction-related code.
* Existing tests may require updates.
* Additional abstraction layer increases conceptual complexity.
* The exact representation of the predictive state remains to be designed.

***

# Alternatives Considered

## Keep Current Design

Advantages:

* No refactoring required.
* Current implementation is straightforward.

Disadvantages:

* Prediction logic remains duplicated across models.
* Future model additions require repeated implementation effort.

***

## Predictive Parameters Only

A previous idea was to expose only:

```julia
predictive_parameters(...)
```

and build prediction functions directly on top of those parameters.

This approach simplifies some computations but does not fully capture the forward and inverse transformations required by all current models.

The support transformation abstraction appears to generalise better.

***

# Open Questions

* Should the abstraction be centred on:
  * a predictive state object,
  * predictive parameters,
  * or predictive distributions?
* Should the abstraction remain internal or become part of the public API?
* Can plotting routines be fully implemented on top of this interface?
* Can censored likelihoods be expressed entirely through the new abstraction?
* Is the additional abstraction justified by the expected number of future models?

***

# Decision

No implementation will be performed at this stage.

The current architecture will remain unchanged until:

1. The package API is stabilised.
2. Existing functionality is fully implemented.
3. The test suite passes consistently.

The proposal should be revisited in a dedicated experimental branch, for example:

```bash
git checkout -b feature/predictive-abstraction
```

before any production integration is attempted.

```
```
