
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

function build_chain!(fo::FitObject)

    if !fo.center_v
        fo.chains = FlexiChains.transform_values(
            fo.chains,
            [@varname(a)] => ((a) -> a) => @varname(b)
        )
        return fo
    end

    fo.chains = FlexiChains.transform_values(
        fo.chains,
        [@varname(a), @varname(m)] => ((a, m) -> a - m*fo.vmean) => @varname(b)
    )
    return fo
end
