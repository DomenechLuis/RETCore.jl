# Roadmap (not yet implemented): goodness-of-fit and posterior-predictive
# diagnostics — information criteria (WAIC, PSIS-LOO, DIC), posterior predictive
# checks (e.g. overlaid KDEs of observed vs replicated data) and residual /
# QQ-plot diagnostics. A possible entry point:
#
#     model_fit_diagnostic(fo; plots = [:waic, :ppc, :residuals])
#
# For now, `prior_sensitivity` is the only diagnostic provided.

"""
    prior_sensitivity(fo::FitObject) -> Dict{Symbol,FitObject}

Refit `fo`'s model under a range of alternative prior scenarios and return a
dictionary of fitted [`FitObject`](@ref)s keyed by scenario.

`fo` must already be fitted; its results are copied under `:default`. The model
is then refitted (reusing all of `fo`'s data and sampler settings) under the
`:wide`, `:optimistic`, `:pessimistic` and `:high_scatter` priors. Comparing the
resulting posteriors shows how much the conclusions depend on the prior.

Note: this runs one full MCMC fit per alternative scenario, so it is as expensive
as several [`fit!`](@ref) calls.
"""
function prior_sensitivity(fo::FitObject)

    isnothing(fo.chains) && error("must be fitted before calling prior_sensitivity")

    priors = (
        wide = prior_wide,
        optimistic = prior_optimistic,
        pessimistic = prior_pessimistic,
        high_scatter = prior_high_scatter,
    )

    fos = Dict{Symbol,FitObject}()
    fos[:default] = FitObject(fo, copy_results = true)

    for (name, priorfun) in pairs(priors)
        foi = FitObject(fo)
        fit!(foi; prior = priorfun(fo.model))
        fos[name] = foi
    end

    return fos
end
