"""
    lognormal_linear_model(v, n, v_censored, n_censored; prior = prior_default(lognormal_linear_model))

Turing model for a linear RET curve with **log-normal, mode-anchored scatter** on
the log10 scale.

The linear prediction `pred = a + m·v` marks the mode of the cycle count: with
`z_mode = exp(l - s²)` the residual `z = pred + z_mode - n` follows
`z ~ LogNormal(l, s)`, so the most likely value of `n` is `pred`. Residuals that
would make `z ≤ 0` are rejected (`-Inf`). Censored points contribute
`logcdf(LogNormal(l, s), z)`.

Parameters: intercept `a`, slope `m` (`≤ 0`), log-scale scatter `s > 0` and the
log-normal location `l`. Priors come from `prior` (fields `a`, `m`, `s`, `l`);
see [`prior_default`](@ref).
"""
@model function lognormal_linear_model(v, n, v_censored, n_censored ; prior = prior_default(lognormal_linear_model))

    a ~ prior.a
    m ~ prior.m
    s ~ prior.s
    l ~ prior.l

    z_mode = exp(l - s^2)

    @inbounds for i in eachindex(v)

        pred = a + m*v[i]
        z = pred + z_mode - n[i]

        if z <= 0
            Turing.@addlogprob!(-Inf)
        else
            Turing.@addlogprob!(logpdf(LogNormal(l, s), z))
        end

    end

    @inbounds for i in eachindex(v_censored)
        pred = a + m*v_censored[i]
        z = pred + z_mode - n_censored[i]

        if z <= 0
            Turing.@addlogprob!(-Inf)
        else
            Turing.@addlogprob! logcdf(LogNormal(l, s), z)
        end
    end

end

prior_default(::typeof(lognormal_linear_model)) =
    merge(_base_prior(:default), (s = Exponential(1.0), l = Normal(0, 3.0)))

prior_wide(::typeof(lognormal_linear_model)) =
    merge(_base_prior(:wide), (s = Exponential(1.0), l = Normal(0, 6.0)))

prior_optimistic(::typeof(lognormal_linear_model)) =
    merge(_base_prior(:optimistic), (s = Exponential(1.0), l = Normal(0, 3.0)))

prior_pessimistic(::typeof(lognormal_linear_model)) =
    merge(_base_prior(:pessimistic), (s = Exponential(1.0), l = Normal(0, 3.0)))

prior_high_scatter(::typeof(lognormal_linear_model)) =
    merge(_base_prior(:high_scatter), (s = Exponential(5.0), l = Normal(0, 3.0)))


function log_log_quantile(fo::FitObject{typeof(lognormal_linear_model)}, prob::Real, logv::Real)

    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    s = fo.chains[@varname(s)]
    l = fo.chains[@varname(l)]

    mode = exp.(l - s.^2)
    pred = b + m* logv

    q = pred + mode - quantile.(LogNormal.(l, s), 1.0 - prob)

    return q
end


# Per-draw exceedance kernel P(N_obs > N | draw) for the log-normal model; the
# generic `exceedance_probability` in prediction.jl averages it over the chains.
function _exceedance_closure(fo::FitObject{typeof(lognormal_linear_model)})
    s = fo.chains[@varname(s)]
    l = fo.chains[@varname(l)]
    return function (i, j, pred, N_log)
        z_mode = exp(l[i, j] - s[i, j]^2)
        z = pred .+ z_mode .- N_log
        return cdf.(LogNormal(l[i, j], s[i, j]), z)
    end
end


function plot_means!(
    p::Plots.Plot,
    fo::FitObject{typeof(lognormal_linear_model)};
    n_lines::Int = 5,
    v_range = (100.0, 160.0),
    curve_res = 100,
    kwargs_mode = (color = :red,),
    kwargs_curve = (color = :green,),
)

    v_array = collect(logrange(v_range..., n_lines))
    inc = v_array[2]/v_array[1]

    b_mean = mean(fo.chains[@varname(b)])
    m_mean = mean(fo.chains[@varname(m)])
    s_mean = mean(fo.chains[@varname(s)])
    l_mean = mean(fo.chains[@varname(l)])

    # línea modal
    plot!(
        p,
        exp10.(b_mean .+ m_mean .* log10.(v_array)),
        v_array;
        label = false,
        kwargs_mode...,
    )

    zmode = exp(l_mean - s_mean^2)
    q = range(0.001, 0.999, curve_res)

    for vi in v_array[1:(end-1)]

        pred = b_mean + m_mean*log10(vi)
        z = quantile.(LogNormal(l_mean, s_mean), q)
        n = 10 .^ (pred .+ zmode .- z)
        dens = pdf.(LogNormal(l_mean, s_mean), z)
        dens ./= maximum(dens)
        v = vi .* (1 .+ dens .* (inc-1)*0.3)

        plot!(p, n, v; label = false, kwargs_curve...)
    end

    return p
end
