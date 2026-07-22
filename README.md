# RETCore.jl

RETCore.jl provides statistical tools for the analysis and interpretation of Rain Erosion Test (RET) results.

It fits Bayesian linear models of cycle count `N` versus velocity `v` on the
log-log scale (Gaussian, log-normal or exponential scatter), handles censored
run-out data, and produces predictions such as quantiles, tolerance limits and
exceedance-probability maps.

Requires Julia ≥ 1.11.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/DomenechLuis/RETCore.jl")
```

## Example

```julia
using RETCore

# Observed (v, N) points; optionally censored run-outs too.
data = DataObject(v, N)

fo = FitObject(normal_linear_model, data)

prepare_data!(fo)   # log10 transform (+ optional centring)
fit!(fo)            # NUTS sampling

# Predictions
p   = exceedance_probability(fo, 130.0, 1e9)      # P(N_obs > 1e9 | v = 130)
q   = quantile(fo, 0.5, 130.0)                     # posterior median cycle count
lim = tolerance_limit(fo, 0.9, 0.95, 130.0)        # lower tolerance limit

# Plots
plot_means(fo)
plot_exceedance(fo)
```

Three models are available — `normal_linear_model`, `lognormal_linear_model` and
`exponential_linear_model` — each with several prior scenarios and a
`prior_sensitivity` helper to compare them.

## Disclaimer

This software is intended for research purposes only.

The software is provided "as is" and without warranty of any kind. No guarantee is made regarding the correctness, accuracy, reliability, completeness, or suitability of the software or any results obtained from its use.

## Author

Luis Doménech Ballester
