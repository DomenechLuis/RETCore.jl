"""
    exponential_linear_model(v, n, v_censored, n_censored; prior = prior_default(exponential_linear_model))

Turing model for a linear RET curve with **one-sided exponential scatter** on the
log10 scale.

The linear prediction `pred = a + m·v` acts as an upper edge: the shortfall
`d = pred - n` follows `d ~ Exponential(l)`, so observations fall on or below the
line and cluster just under it. Values with `d ≤ 0` are rejected (`-Inf`), and
censored points contribute `logcdf(Exponential(l), d)`.

Parameters: intercept `a`, slope `m` (`≤ 0`) and exponential scale `l > 0`.
Priors come from `prior` (fields `a`, `m`, `l`); see [`prior_default`](@ref).
"""
@model function exponential_linear_model(v, n, v_censored, n_censored; prior = prior_default(exponential_linear_model))

    a ~ prior.a
    m ~ prior.m
    l ~ prior.l

    @inbounds for i in eachindex(v)
        pred = a + m*v[i]
        d = pred - n[i]

        if d <= 0
            Turing.@addlogprob!(-Inf)
        else
            Turing.@addlogprob!(logpdf(Exponential(l), d))
        end
    end

    @inbounds for i in eachindex(v_censored)
        pred = a + m*v_censored[i]
        d = pred - n_censored[i]

        if d <= 0
            Turing.@addlogprob!(-Inf)
        else
            Turing.@addlogprob!(logcdf(Exponential(l), d))
        end
    end
end

prior_default(::typeof(exponential_linear_model)) =
    merge(_base_prior(:default), (l = Exponential(1.0),))

prior_wide(::typeof(exponential_linear_model)) =
    merge(_base_prior(:wide), (l = Exponential(1.0),))

prior_optimistic(::typeof(exponential_linear_model)) =
    merge(_base_prior(:optimistic), (l = Exponential(1.0),))

prior_pessimistic(::typeof(exponential_linear_model)) =
    merge(_base_prior(:pessimistic), (l = Exponential(1.0),))

prior_high_scatter(::typeof(exponential_linear_model)) =
    merge(_base_prior(:high_scatter), (l = Exponential(5.0),))


function log_log_quantile(fo::FitObject{typeof(exponential_linear_model)}, prob::Real, logv::Real)

    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    l = fo.chains[@varname(l)]

    q = b .+ m .* logv .- quantile.(Exponential.(l), 1.0 - prob)

    return q
end


# Per-draw exceedance kernel P(N_obs > N | draw) for the exponential model; the
# generic `exceedance_probability` in prediction.jl averages it over the chains.
function _exceedance_closure(fo::FitObject{typeof(exponential_linear_model)})
    l = fo.chains[@varname(l)]
    return (i, j, pred, N_log) -> cdf.(Exponential(l[i, j]), pred .- N_log)
end


function plot_means!(
    p::Plots.Plot,
    fo::FitObject{typeof(exponential_linear_model)};
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
    l_mean = mean(fo.chains[@varname(l)])

    plot!(
        p,
        exp10.(b_mean .+ m_mean .* log10.(v_array)),
        v_array;
        label = false,
        kwargs_mode...,
    )

    q = range(0.001, 0.999, curve_res)

    for vi in v_array[1:(end-1)]

        pred = b_mean + m_mean*log10(vi)

        d = quantile.(Exponential(l_mean), q)
        n = exp10.(pred .- d)

        dens = pdf.(Exponential(l_mean), d)
        dens ./= maximum(dens)

        v = vi .* (1 .+ dens .* (inc-1) * 0.3)
        plot!(p, n, v; label = false, kwargs_curve...)
    end

    return p
end
