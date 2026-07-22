# RETCore.jl — Code Review

A review of RETCore.jl focused on three goals: **simplicity**, **readability**
and **reusability as a package** for future projects. This document records the
findings; the accompanying pull request implements the fixes and adds a test
suite and docstrings.

## Summary

The package has a sound design — dispatching on `FitObject{typeof(model)}` to
specialise behaviour per model is idiomatic and extensible. The main issues were
a handful of blocking bugs, a large amount of duplication across the three model
files, an environment that could not load on the pinned Julia version, and tests
that exercised almost none of the numeric logic.

## 1. Blocking bugs (the package could not run)

| # | Location | Problem | Fix |
|---|----------|---------|-----|
| 1 | `Project.toml` | `logrange` is only available in Julia ≥ 1.11, but there was no `[compat] julia` entry (and the default env was 1.10). | Added `julia = "1.11"`. |
| 2 | `prediction.jl`, `plotting.jl` (×3) | Default `v_range = (100.0, 160, 0)` — a 3-element tuple caused by `.0` mistyped as `, 0`; splatting into `logrange` raised `MethodError`. | Corrected to `(100.0, 160.0)`. |
| 3 | `plotting.jl` | `plot_values` forwarded keywords positionally (`plot_values!(p, fo, v, N, kwargs...)`), breaking on any kwarg. | Use `; kwargs...`. |
| 4 | `prediction.jl` | `quantile(fo, prob, v::Real)` used `exp10(...)` (no broadcast) on the per-draw matrix, so it errored on any fitted model; the vector method was broken the same way. | Broadcast and flatten to a per-draw `Vector`. |
| 5 | `Project.toml` | The `test` target referenced `Test` without an `[extras]` entry, so the environment failed to resolve on current Julia. | Added `[extras] Test`. |

## 2. Correctness / consistency

- **Duplicate method** — `exceedance_probability(fo, v::AbstractVector, N::AbstractVector)`
  was defined twice with the same signature; the first (dead) definition was removed.
- **`build_chain!` ignored `Nmean`** — centring of `N` was applied in
  `prepare_data!` but never undone, so `center_N = true` produced a wrong
  intercept. The back-transform is now the single, correct expression
  `b = a - m·vmean + Nmean`, which also collapses the two former branches into one.
- **Copy-paste error messages** — `prob` validation reported "v must be …"; fixed.
- **`plot_means(fo)` was broken for two models** — the wrapper forwarded a
  `kwargs_mean` keyword, but the log-normal and exponential `plot_means!`
  methods expect `kwargs_mode`, so the non-mutating call errored for them. The
  wrapper now forwards keywords transparently (`plot_means(fo; kwargs...)`).
- **`_validate_prediction_coordinates`** was defined and tested but never used;
  it is now the single validation entry point for `exceedance_probability`.
- **`capture_data`** silently assumed a sorted grid; it now validates `issorted`.
- **`DataObject` length validation could be bypassed** — when all four fields
  shared a type, Julia's auto-generated positional constructor was selected over
  the validating outer one. Validation moved into an inner constructor so it
  always runs.

## 3. Simplicity — removing duplication

The three model files were ~80% identical. The shared skeleton was factored out
while keeping each model's genuine maths in one obvious place:

- **`exceedance_probability`** — the validation + chain-averaging loop is now a
  single generic method in `prediction.jl`. Each model supplies only a small
  `_exceedance_closure(fo)` (2–5 lines) returning its per-draw kernel.
- **`prepare_data!`** — the three identical per-model methods became one generic
  `prepare_data!(::FitObject)`.
- **Priors** — the repeated intercept/slope literals moved into `_base_prior`;
  each `prior_*` now `merge`s that with its dispersion parameters, so the shared
  beliefs live in one place and per-model priors are one line each.

Together these removed several hundred lines without changing the public API.

## 4. Readability

- Docstrings added to the whole exported surface (`DataObject`, `FitObject`,
  `fit!`, `prepare_data!`, `exceedance_probability`, `quantile`,
  `tolerance_limit`, `exceedance_grid`, the plotting functions and each model).
- Removed dead commented-out code (old model in `normal.jl`, the manual test
  block in `data.jl`) and trimmed the large descriptive comment in
  `diagnostics.jl` to a concise roadmap note.
- Normalised the mixed tabs/spaces in the export list and moved the embedded
  `v_aeronordic` grid to a documented `const`.

## 5. Reusability as a package

- **`Manifest.toml` removed from version control** (and added to `.gitignore`).
  It had been resolved on Julia 1.10 and its pinned standard-library versions
  (MbedTLS_jll, LibGit2_jll, …) failed to install on 1.11 — a concrete instance
  of why libraries should not commit a Manifest. Compatibility now lives in
  `[compat]`.
- **`tolerance_limit` is now exported** (it was documented but not exported).
- **README** — fixed the placeholder install URL and expanded the example.
- **Continuous integration** — added `.github/workflows/CI.yml` running the test
  suite on Julia 1.11.

> Note: `FlexiChains` is registered in the General registry, but an **up-to-date
> local registry** is required to resolve it (an old copy will report it as
> unregistered). `Pkg.instantiate` after `Pkg.Registry.update()` is enough.

## 6. Testing

Tests previously covered only construction and argument validation. The suite is
reorganised into layers so the numeric logic is exercised **without** running
MCMC:

- **Unit** — data handling, priors, centring transforms, validation helpers.
- **Numeric** — `exceedance_probability`, `quantile`, `tolerance_limit` and
  `exceedance_grid` are checked against closed-form expressions by injecting
  constant "chains" (`test/testutils.jl`), so they run in milliseconds.
- **Plotting** — smoke tests that every entry point builds a `Plots.Plot`.
- **Integration** — one end-to-end MCMC fit, guarded by `RETCORE_RUN_MCMC=true`
  so the default (and CI) run stays fast.

## 7. Optional future work (not in this PR)

- Implement the diagnostics sketched in `diagnostics.jl` (WAIC, PSIS-LOO,
  posterior-predictive checks, residual/QQ plots).
- Parametrise `FitObject`'s `raw_data`/`chains` fields to remove type
  instability if prediction throughput ever matters.
- Loosen `[compat]` lower bounds once the package is tested against a range of
  dependency versions.
