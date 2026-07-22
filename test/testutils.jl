# Test helpers: build a `FitObject` whose posterior "chains" are set to known
# constant values, so the numeric machinery (exceedance_probability, quantile,
# tolerance_limit, plotting) can be exercised WITHOUT running MCMC.

using FlexiChains
using Turing: @varname
using RETCore

"""
    constant_chains(; ni = 5, nc = 2, params...)

Build a `VNChain` with `ni` iterations and `nc` chains where every requested
parameter is held at a constant value across all draws. Accepts `b`, `m`, `s`
and/or `l` as keyword arguments, e.g. `constant_chains(b = 25.0, m = -5.0, s = 1.0)`.
Constant chains make the posterior mean equal to the value, so results can be
checked against closed-form expressions.
"""
function constant_chains(; ni = 5, nc = 2, params...)
    P = FlexiChains.Parameter
    d = Dict{Any,Matrix{Float64}}()
    for (k, v) in params
        vn = if k === :a
            @varname(a)
        elseif k === :b
            @varname(b)
        elseif k === :m
            @varname(m)
        elseif k === :s
            @varname(s)
        elseif k === :l
            @varname(l)
        else
            error("unknown parameter $k")
        end
        d[P(vn)] = fill(float(v), ni, nc)
    end
    return VNChain(ni, nc, d)
end

"""
    fit_with_chains(model; data, ni, nc, params...)

Return a `FitObject` for `model` whose `chains` field is populated with
[`constant_chains`](@ref). No sampling is performed.
"""
function fit_with_chains(model; data = DataObject([1.0, 2.0], [10.0, 20.0]), ni = 5, nc = 2, params...)
    fo = FitObject(model, data)
    fo.chains = constant_chains(; ni = ni, nc = nc, params...)
    return fo
end

# Constant-parameter fixtures for each model, used across several test files.
normal_fixture(; kwargs...) =
    fit_with_chains(normal_linear_model; b = 25.0, m = -5.0, s = 1.0, kwargs...)

lognormal_fixture(; kwargs...) =
    fit_with_chains(lognormal_linear_model; b = 25.0, m = -5.0, s = 1.0, l = 0.0, kwargs...)

exponential_fixture(; kwargs...) =
    fit_with_chains(exponential_linear_model; b = 25.0, m = -5.0, l = 1.0, kwargs...)