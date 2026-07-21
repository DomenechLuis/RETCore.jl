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

function prior_default(::typeof(lognormal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5, 2.0), -Inf, 0.0),
        s = Exponential(1.0),
        l = Normal(0, 3.0)
    )
end

function prior_wide(::typeof(lognormal_linear_model))
	return (
		a = Normal(25, 20),
		m = truncated(Normal(-5, 5.0), -Inf, 0.0),
		s = Exponential(1.0),
		l = Normal(0, 6.0)
	)
end

function prior_optimistic(::typeof(lognormal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-1.5, 1.0), -Inf, 0.0),
        s = Exponential(1.0),
		l = Normal(0, 3.0)
    )
end

function prior_pessimistic(::typeof(lognormal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-12.0, 2.0), -Inf, 0.0),
        s = Exponential(1.0),
		l = Normal(0, 3.0)
    )
end

function prior_high_scatter(::typeof(lognormal_linear_model))
    return (
        a = Normal(25, 10),
        m = truncated(Normal(-5.0, 2.0), -Inf, 0.0),
        s = Exponential(5),
		l = Normal(0, 3.0)
    )
end

function prepare_data!(fo::FitObject{typeof(lognormal_linear_model)})
    return preparedata_standard!(fo)
end


function log_log_quantile(fo::FitObject{typeof(lognormal_linear_model)}, prob::Real, logv::Real)

    0.0 < prob < 1.0 || throw(ArgumentError("v must be finite and strictly positive"))

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]
    s = fo.chains[@varname(s)]
    l = fo.chains[@varname(l)]

    mode = exp.(l - s.^2)
    pred = b + m* logv

    q = pred + mode - quantile.(LogNormal.(l, s), 1.0 - prob)

    return q
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
        10.0 .^ (b_mean .+ m_mean .* log10.(v_array)),
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


function exceedance_probability(
    fo::FitObject{typeof(lognormal_linear_model)},
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
	l = fo.chains[@varname(l)]

	z_mode = exp.(l - s.^2)

    for chain_index in chain_indices
        @inbounds for sample_index in axes(b, 1)
            
			pred = b[sample_index, chain_index] + m[sample_index, chain_index]*v_log

			z = pred + z_mode[sample_index, chain_index] .- N_log

            distribution = LogNormal(l[sample_index, chain_index], s[sample_index, chain_index])

            probability_sum .+= cdf.(Ref(distribution), z)
        end

        nsamples_total += size(b, 1)
    end

    nsamples_total > 0 || error("the selected chains contain no posterior samples")

    return probability_sum ./ nsamples_total
end