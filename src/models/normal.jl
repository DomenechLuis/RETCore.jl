"""
    normal_linear_model(v, N, v_censored, n_censored; prior = prior_default(normal_linear_model))

Turing model for a linear RET curve with **Gaussian scatter** on the log10 scale.

For each observation the mean cycle count is `μ = a + m·v` (with `v`, `N` already
in log10 units) and `N ~ Normal(μ, s)`. Censored points (run-outs) contribute
`logccdf(Normal(μ, s), n_censored)`, i.e. the probability of surviving beyond the
observed run-out.

Parameters: intercept `a`, slope `m` (constrained `≤ 0`) and scatter `s > 0`.
Their priors are supplied through `prior` (a `NamedTuple` with fields `a`, `m`,
`s`); see [`prior_default`](@ref).
"""
@model function normal_linear_model(v, N, v_censored, n_censored; prior = prior_default(normal_linear_model))

    a ~ prior.a
    m ~ prior.m
    s ~ prior.s

    @inbounds for i in eachindex(v_censored)
        μ = a + m * v_censored[i]
        Turing.@addlogprob! logccdf(Normal(μ, s), n_censored[i])
    end

    @inbounds for i in eachindex(v)
        μ = a + m * v[i]
        Turing.@addlogprob! logpdf(Normal(μ, s), N[i])
    end

end

prior_default(::typeof(normal_linear_model)) =
    merge(_base_prior(:default), (s = Exponential(1.0),))

prior_wide(::typeof(normal_linear_model)) =
    merge(_base_prior(:wide), (s = Exponential(1.0),))

prior_optimistic(::typeof(normal_linear_model)) =
    merge(_base_prior(:optimistic), (s = Exponential(1.0),))

prior_pessimistic(::typeof(normal_linear_model)) =
    merge(_base_prior(:pessimistic), (s = Exponential(1.0),))

prior_high_scatter(::typeof(normal_linear_model)) =
    merge(_base_prior(:high_scatter), (s = Exponential(5.0),))


function log_log_quantile(fo::FitObject{typeof(normal_linear_model)}, prob::Real, logv::Real)

    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    s = fo.chains[@varname(s)]

    q = quantile.(Normal.(b + m * logv, s), prob)

    return q
end


# Per-draw exceedance kernel P(N_obs > N | draw) for the Gaussian model; the
# generic `exceedance_probability` in prediction.jl averages it over the chains.
function _exceedance_closure(fo::FitObject{typeof(normal_linear_model)})
    s = fo.chains[@varname(s)]
    return (i, j, pred, N_log) -> ccdf.(Normal(pred, s[i, j]), N_log)
end


function plot_means!(
    p::Plots.Plot,
    fo::FitObject{typeof(normal_linear_model)};
    n_lines::Int = 5,
    v_range = (100.0, 160.0),
    curve_res = 20,
    kwargs_mean = (color = :red,),
    kwargs_curve = (color = :green,),
)

    v_array = collect(logrange(v_range..., n_lines))
    inc = v_array[2]/v_array[1]
    b_mean = mean(fo.chains[@varname(b)])
    m_mean = mean(fo.chains[@varname(m)])
    s_mean = mean(fo.chains[@varname(s)])

    plot!(
        p,
        exp10.(b_mean .+ m_mean .* log10.(v_array)),
        v_array;
        label = false,
        kwargs_mean...,
    )

    for vi in v_array[1:(end-1)]
        N = exp10.(collect(range(-3, 3, curve_res))*s_mean .+ (b_mean + m_mean*log10(vi)))
        v =
            vi*(
                1 .+
                pdf.(Normal(0, 1), collect(range(-3, 3, curve_res))) / pdf(Normal(), 0) *
                (inc-1) *
                0.3
            )

        plot!(p, N, v; label = false, kwargs_curve...)
    end
    return p
end
