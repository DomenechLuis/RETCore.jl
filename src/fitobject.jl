"""
    FitObject{M}

Mutable container that bundles everything needed to fit a Bayesian RET model
and to store its results.

# Fields
- `model`: the Turing model function (e.g. `normal_linear_model`). Its type `M`
  is used for dispatch, so each model can specialise `prepare_data!`,
  `exceedance_probability`, `plot_means!`, etc.
- `raw_data::DataObject`: the observations in physical units.
- `pre_data`: the transformed data actually passed to the sampler (log10 scale,
  optionally centred). `nothing` until [`prepare_data!`](@ref) is called.
- `center_v`, `center_N`: whether to centre the (log10) velocity / cycle counts
  before sampling. Centring improves sampler geometry; the intercept is mapped
  back to the original scale in [`build_chain!`](@ref).
- `n_warmup`, `n_samples`, `n_chains`: NUTS sampler settings.
- `target_accept`: NUTS target acceptance rate.
- `init`: optional initial parameters (`NamedTuple`); empty means automatic.
- `vmean`, `Nmean`: the means subtracted when centring (0 when not centring).
- `chains`: the posterior samples (a `FlexiChain`), or `nothing` before fitting.
- `diagnostics`: reserved for convergence diagnostics; `nothing` by default.

Construct one with [`FitObject(model, raw_data; kwargs...)`](@ref).
"""
mutable struct FitObject{M}

    model::M
    raw_data::DataObject
    pre_data::Union{Nothing,DataObject}

    center_v::Bool
    center_N::Bool

    n_warmup::Int
    n_samples::Int
    n_chains::Int

    target_accept::Float64

    init::NamedTuple

    vmean::Float64
    Nmean::Float64

    chains::Any

    diagnostics::Any
end


"""
    FitObject(model, raw_data; kwargs...)

Build a [`FitObject`](@ref) for `model` (a Turing model function such as
`normal_linear_model`) and a [`DataObject`](@ref) `raw_data`, leaving it ready
for [`prepare_data!`](@ref) and [`fit!`](@ref).

# Keyword arguments
- `center_v`, `center_N` (`false`): centre log10 velocity / cycle counts.
- `n_warmup` (`20_000`), `n_samples` (`5_000`), `n_chains` (`4`): NUTS settings.
- `target_accept` (`0.85`): NUTS target acceptance rate.
- `init` (`NamedTuple()`): optional initial parameters; empty means automatic.
- `pre_data`, `vmean`, `Nmean`: advanced/internal; normally left at defaults.

# Example
```julia
fo = FitObject(normal_linear_model, DataObject(v, N))
prepare_data!(fo)
fit!(fo)
```
"""
function FitObject(
    model,
    raw_data;
    pre_data = nothing,
    center_v = false,
    center_N = false,
    n_warmup = 20_000,
    n_samples = 5_000,
    n_chains = 4,
    target_accept = 0.85,
    init = NamedTuple(),
    vmean = 0.0,
    Nmean = 0.0,
)

    FitObject(
        model,
        raw_data,
        pre_data,
        center_v,
        center_N,
        n_warmup,
        n_samples,
        n_chains,
        target_accept,
        init,
        vmean,
        Nmean,
        nothing,
        nothing,
    )
end

function FitObject(fo::FitObject; copy_results = false)

    newfo = FitObject(
        fo.model,
        fo.raw_data;
        pre_data = fo.pre_data,
        center_v = fo.center_v,
        center_N = fo.center_N,
        n_warmup = fo.n_warmup,
        n_samples = fo.n_samples,
        n_chains = fo.n_chains,
        target_accept = fo.target_accept,
        init = fo.init,
        vmean = fo.vmean,
        Nmean = fo.Nmean,
    )

    if copy_results
        newfo.chains = deepcopy(fo.chains)
        newfo.diagnostics = deepcopy(fo.diagnostics)
    end

    return newfo

end

function preparedata_standard!(fo::FitObject)

    v = log10.(fo.raw_data.v)
    N = log10.(fo.raw_data.N)

    vc = log10.(fo.raw_data.v_censored)
    Nc = log10.(fo.raw_data.N_censored)

    if fo.center_v
        fo.vmean = mean(v)

        v .-= fo.vmean
        vc .-= fo.vmean
    else
        fo.vmean = 0.0
    end

    if fo.center_N
        fo.Nmean = mean(N)

        N .-= fo.Nmean
        Nc .-= fo.Nmean
    else
        fo.Nmean = 0.0
    end

    fo.pre_data = DataObject(v, N, vc, Nc)

    return fo
end


"""
    prepare_data!(fo::FitObject)

Transform `fo.raw_data` into the representation the sampler expects and store it
in `fo.pre_data`, returning `fo`.

The standard preparation takes base-10 logarithms of both velocity and cycle
counts and, when `fo.center_v` / `fo.center_N` are set, subtracts their means
(recorded in `fo.vmean` / `fo.Nmean`). All current models share this behaviour,
so this generic method handles them; a model may still specialise
`prepare_data!(::FitObject{typeof(mymodel)})` if it needs something different.

Must be called before [`fit!`](@ref).
"""
prepare_data!(fo::FitObject) = preparedata_standard!(fo)


"""
    _base_prior(scenario::Symbol) -> NamedTuple

Return the intercept (`a`) and slope (`m`) priors shared by every model for a
given `scenario` (`:default`, `:wide`, `:optimistic`, `:pessimistic` or
`:high_scatter`). Each model completes this with its own dispersion parameters
(`s`, `l`, …) via `merge`, which keeps the common location/slope beliefs defined
in a single place. Internal helper.
"""
function _base_prior(scenario::Symbol)
    a = scenario === :wide ? Normal(25, 20) : Normal(25, 10)

    m = if scenario === :wide
        truncated(Normal(-5.0, 5.0), -Inf, 0.0)
    elseif scenario === :optimistic
        truncated(Normal(-1.5, 1.0), -Inf, 0.0)
    elseif scenario === :pessimistic
        truncated(Normal(-12.0, 2.0), -Inf, 0.0)
    else # :default and :high_scatter share the same location/slope
        truncated(Normal(-5.0, 2.0), -Inf, 0.0)
    end

    return (a = a, m = m)
end


"""
    prior_default(model)

Recommended prior for production use.

This prior should represent the best currently available physical and
experimental knowledge for the selected model.

It is the prior that should normally be used when the objective is
parameter inference or prediction rather than prior sensitivity analysis.

Goals:
- represent the most realistic prior beliefs;
- incorporate domain expertise whenever available;
- serve as the reference scenario against which all other priors are
  compared.

Each model should implement its own version through multiple dispatch.
"""
function prior_default end


"""
    prior_wide(model)

Weakly informative prior.

This prior should express roughly the same central expectations as
`prior_default`, but with substantially larger uncertainty.

Goals:
- assess the influence of prior precision on the posterior;
- explore a broader region of the parameter space;
- verify that the inference is primarily driven by the data.

Each model should implement its own version through multiple dispatch.
"""
function prior_wide end


"""
    prior_optimistic(model)

Optimistic prior scenario.

This prior intentionally shifts one or more model parameters toward
favourable outcomes relative to those expected under `prior_default`.

It does not need to be the most realistic prior. Its purpose is to test
whether conclusions remain stable under optimistic assumptions.

Goals:
- assess sensitivity to favourable prior beliefs;
- stress-test the analysis from the optimistic side;
- detect excessive dependence on positive prior assumptions.

Each model should implement its own version through multiple dispatch.
"""
function prior_optimistic end


"""
    prior_pessimistic(model)

Pessimistic prior scenario.

This prior intentionally shifts one or more model parameters toward
unfavourable outcomes relative to those expected under `prior_default`.

It does not need to be the most realistic prior. Its purpose is to test
whether conclusions remain stable under conservative assumptions.

Goals:
- assess sensitivity to pessimistic prior beliefs;
- stress-test the analysis from the conservative side;
- detect excessive dependence on negative prior assumptions.

Each model should implement its own version through multiple dispatch.
"""
function prior_pessimistic end


"""
    prior_high_scatter(model)

High-variability prior scenario.

This prior should favour substantially larger variability than would
normally be assumed under `prior_default`.

Location and trend parameters may remain similar to the default prior,
while parameters controlling dispersion are deliberately shifted toward
larger values.

Goals:
- assess sensitivity to increased variability;
- evaluate robustness under high experimental scatter;
- determine whether predictive conclusions depend strongly on the
  assumed level of dispersion.

Each model should implement its own version through multiple dispatch.
"""
function prior_high_scatter end

"""
    fit!(fo::FitObject; prior = prior_default(fo.model)) -> fo

Run NUTS sampling for `fo.model` on the prepared data and store the posterior in
`fo.chains`, returning `fo`.

[`prepare_data!`](@ref) must have been called first (otherwise an error is
raised). Sampling uses `fo.n_chains` chains of `fo.n_samples` samples each, with
`fo.n_warmup` warmup steps and target acceptance `fo.target_accept`, run over
`MCMCThreads()`. If `fo.init` is non-empty it seeds every chain.

`prior` selects the prior scenario (see [`prior_default`](@ref) and the
`prior_wide` / `prior_optimistic` / `prior_pessimistic` / `prior_high_scatter`
variants); it defaults to the model's recommended prior.

After sampling, [`build_chain!`](@ref) adds the physical-scale intercept `b`.
"""
function fit!(fo::FitObject; prior = prior_default(fo.model))

    isnothing(fo.pre_data) && error("prepare_data! has not been called")

    vfit = fo.pre_data.v
    Nfit = fo.pre_data.N
    vfitc = fo.pre_data.v_censored
    Nfitc = fo.pre_data.N_censored

    sampler = NUTS(fo.n_warmup, fo.target_accept)

    if isempty(fo.init)
        fo.chains = sample(
            fo.model(vfit, Nfit, vfitc, Nfitc; prior=prior),
            sampler,
            MCMCThreads(),
            fo.n_samples,
            fo.n_chains,
        )
    else
        init_chain = InitFromParams(fo.init)
        fo.chains = sample(
            fo.model(vfit, Nfit, vfitc, Nfitc; prior=prior),
            sampler,
            MCMCThreads(),
            fo.n_samples,
            fo.n_chains;
            initial_params = fill(init_chain, fo.n_chains),
        )
    end
    build_chain!(fo)
    return fo
end

"""
    build_chain!(fo::FitObject) -> fo

Add the physical-scale intercept `b` to `fo.chains` from the sampled intercept
`a` and slope `m`, undoing any centring applied by [`prepare_data!`](@ref).

Because the model is fitted on centred data (`v' = v - vmean`, `N' = N - Nmean`),
the fitted relationship `N' = a + m·v'` maps back to physical scale as
`N = (a - m·vmean + Nmean) + m·v`, so the physical intercept is

    b = a - m·vmean + Nmean

When a coordinate is not centred its mean is `0`, so this single expression
covers every combination of `center_v` / `center_N`.
"""
function build_chain!(fo::FitObject)
    fo.chains = FlexiChains.transform_values(
        fo.chains,
        [@varname(a), @varname(m)] =>
            ((a, m) -> a - m * fo.vmean + fo.Nmean) => @varname(b),
    )
    return fo
end
