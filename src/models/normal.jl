
#@model function normal_linear_model(x, y)
#    a ~ Normal(10, 10)# Intercept
#    m ~ truncated(Normal(0, 12), -Inf, 0.0)# β ≤ 0
#    s ~ Exponential(3)#  sd
#
#    @inbounds for i in eachindex(x)
#        μ = a + m * x[i]
#        Turing.@addlogprob! logpdf(Normal(μ, s), y[i])
#    end
#end

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

function prior_default(::typeof(normal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5.0, 2.0), -Inf, 0.0),
        s = Exponential(1.0),
    )
end

function prior_wide(::typeof(normal_linear_model))
    return (
        a = Normal(25, 20),
        m = truncated(Normal(-5.0, 5.0), -Inf, 0.0),
        s = Exponential(1.0),
    )
end

function prior_optimistic(::typeof(normal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-1.5, 1.0), -Inf, 0.0),
        s = Exponential(1.0),
    )
end

function prior_pessimistic(::typeof(normal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-12.0, 2.0), -Inf, 0.0),
        s = Exponential(1.0),
    )
end

function prior_high_scatter(::typeof(normal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5.0, 2.0), -Inf, 0.0),
        s = Exponential(5),
    )
end

function prepare_data!(fo::FitObject{typeof(normal_linear_model)})
    return preparedata_standard!(fo)
end


function quantile(fo::FitObject{typeof(normal_linear_model)}, prob::Real, v::Real)

    0.0 < prob < 1.0 || throw(ArgumentError("v must be finite and strictly positive"))

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    s = fo.chains[@varname(s)]

    q = quantile.(Normal.(b + m * v, s), prob)

    return q
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
        10.0 .^ (b_mean .+ m_mean .* log10.(v_array)),
        v_array;
        label = false,
        kwargs_mean...,
    )

    for vi in v_array[1:(end-1)]
        N = 10.0 .^ (collect(range(-3, 3, curve_res))*s_mean .+ (b_mean + m_mean*log10(vi)))
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


function exceedance_probability(
    fo::FitObject{typeof(normal_linear_model)},
    v::Real,
    N::AbstractVector{<:Real};
    chains = :all,
)
    isnothing(fo.chains) && throw(ArgumentError("the model has not been fitted"))

    isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

    isempty(N) && return Float64[]

    all(ni -> isfinite(ni) && ni > 0, N) ||
        throw(ArgumentError("all N values must be finite and strictly positive"))

    v_log = log10(v)
    N_log = log10.(N)

    chain_indices = _chain_indices(fo, chains)

    probability_sum = zeros(Float64, length(N_log))
    nsamples_total = 0

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    s = fo.chains[@varname(s)]

    for chain_index in chain_indices
        @inbounds for sample_index in axes(b, 1)
            μ = b[sample_index, chain_index] + m[sample_index, chain_index] * v_log

            distribution = Normal(μ, s[sample_index, chain_index])

            probability_sum .+= ccdf.(Ref(distribution), N_log)
        end

        nsamples_total += size(b, 1)
    end

    nsamples_total > 0 || error("the selected chains contain no posterior samples")

    return probability_sum ./ nsamples_total
end
