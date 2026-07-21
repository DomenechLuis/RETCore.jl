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

function prepare_data!(fo::FitObject{typeof(exponential_linear_model)})
    return preparedata_standard!(fo)
end


function prior_default(::typeof(exponential_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5, 2.0), -Inf, 0.0),
        l = Exponential(1.0),
    )
end

function prior_wide(::typeof(exponential_linear_model))
    return (
        a = Normal(25, 20),
        m = truncated(Normal(-5.0, 5.0), -Inf, 0.0),
        l = Exponential(1.0),
    )
end

function prior_optimistic(::typeof(exponential_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-1.5, 1.0), -Inf, 0.0),
        l = Exponential(1.0),
    )
end

function prior_pessimistic(::typeof(exponential_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-12.0, 2.0), -Inf, 0.0),
        l = Exponential(1.0),
    )
end

function prior_high_scatter(::typeof(exponential_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5.0, 2.0), -Inf, 0.0),
        l = Exponential(5),
    )
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
        10.0 .^ (b_mean .+ m_mean .* log10.(v_array)),
        v_array;
        label = false,
        kwargs_mode...,
    )

    q = range(0.001, 0.999, curve_res)

    for vi in v_array[1:(end-1)]

        pred = b_mean + m_mean*log10(vi)

        d = quantile.(Exponential(l_mean), q)
        n = 10.0 .^ (pred .- d)

        dens = pdf.(Exponential(l_mean), d)
        dens ./= maximum(dens)

        v = vi .* (1 .+ dens .* (inc-1) * 0.3)
        plot!(p, n, v; label = false, kwargs_curve...)
    end

    return p
end



function exceedance_probability(
    fo::FitObject{typeof(exponential_linear_model)},
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
	l = fo.chains[@varname(l)]


    for chain_index in chain_indices
        @inbounds for sample_index in axes(b, 1)
            
			pred = b[sample_index, chain_index] + m[sample_index, chain_index]*v_log

			d = pred .- N_log

            distribution = Exponential(l[sample_index, chain_index])

            probability_sum .+= cdf.(Ref(distribution), d)
        end

        nsamples_total += size(b, 1)
    end

    nsamples_total > 0 || error("the selected chains contain no posterior samples")

    return probability_sum ./ nsamples_total
end