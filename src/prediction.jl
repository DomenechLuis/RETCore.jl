"""
    quantile(fo::FitObject, prob::Real, v::Real) -> Vector

Posterior distribution of the `prob`-quantile of the cycle count at velocity `v`
(physical units). One value is returned per posterior draw, so the result is a
`Vector` summarising uncertainty in the quantile itself (take `mean`/`quantile`
of it for a point estimate or credible interval).

`prob` must lie in `(0, 1)` and `v` must be finite and positive.
"""
function quantile(fo::FitObject, prob::Real, v::Real)
    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))
    isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

    return collect(vec(exp10.(log_log_quantile(fo, prob, log10(v)))))
end

"""
    quantile(fo::FitObject, prob::Real, v::AbstractVector{<:Real}) -> Vector{Vector}

Posterior `prob`-quantile evaluated at each velocity in `v`. Returns one entry
per velocity, each being the per-draw vector described above (so
`result[i]` corresponds to `v[i]`).
"""
function quantile(fo::FitObject, prob::Real, v::AbstractVector{<:Real})
    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))
    all(value -> isfinite(value) && value > 0, v) ||
        throw(ArgumentError("all velocities must be finite and strictly positive"))

    return [quantile(fo, prob, vi) for vi in v]
end

"""
    tolerance_limit(fo::FitObject, reliability, credibility, v; side = :lower)

Bayesian one-sided tolerance limit at velocity `v`.

A lower tolerance limit `L` satisfies

    P(P(X ≥ L) ≥ reliability | data) = credibility

An upper tolerance limit `U` satisfies

    P(P(X ≤ U) ≥ reliability | data) = credibility

`reliability` and `credibility` must lie in `(0, 1)` and `v` must be finite and
positive. `side` is `:lower` or `:upper`.
"""
function tolerance_limit(fo::FitObject, reliability::Real, credibility::Real, v::Real; side::Symbol = :lower)
	0.0 < reliability < 1.0 || throw(ArgumentError("reliability must be between 0 and 1"))
	0.0 < credibility < 1.0 || throw(ArgumentError("credibility must be between 0 and 1"))
	isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

	if side == :lower
		limit = quantile( quantile(fo, 1 - reliability, v ), 1 - credibility)
	elseif side == :upper
		limit = quantile( quantile(fo, reliability, v ), credibility)
	else
		throw(ArgumentError("side must be either :upper or :lower"))
	end

	return limit
end


"""
    exceedance_probability(fo::FitObject, v, N; chains = :all)

Posterior probability that the cycle count at velocity `v` exceeds `N`, i.e.
`P(N_obs > N | v, data)`, averaged over the selected posterior draws.

`v` and `N` may be:
- scalar `v`, scalar `N` → a single probability;
- scalar `v`, vector `N` → a probability per `N` (the model-specialised kernel);
- vector `v`, vector `N` (same length) → element-wise `(v[i], N[i])` pairs.

`chains` selects which chains to average (`:all`, an integer, or a collection of
integers). All velocities and cycle counts must be finite and strictly positive.
"""
function exceedance_probability(fo::FitObject, v::Real, N::Real; kwargs...)
    return only(exceedance_probability(fo, v, [N]; kwargs...))
end

function exceedance_probability(
    fo::FitObject,
    v::AbstractVector{<:Real},
    N::AbstractVector{<:Real};
    kwargs...,
)
    length(v) == length(N) ||
        throw(ArgumentError("v and N must have the same length"))

    return exceedance_probability.(Ref(fo), v, N; kwargs...)
end

"""
    exceedance_probability(fo::FitObject, v::Real, N::AbstractVector{<:Real}; chains = :all)

Core (model-generic) implementation: averages the per-draw exceedance kernel,
provided by each model through `_exceedance_closure(fo)`, over the selected
posterior draws. All the validation and chain-averaging logic lives here once,
so a new model only needs to supply its kernel.
"""
function exceedance_probability(
    fo::FitObject,
    v::Real,
    N::AbstractVector{<:Real};
    chains = :all,
)
    isnothing(fo.chains) && throw(ArgumentError("the model has not been fitted"))

    _validate_prediction_coordinates(v, N)

    isempty(N) && return Float64[]

    v_log = log10(v)
    N_log = log10.(N)

    chain_indices = _chain_indices(fo, chains)

    b = fo.chains[@varname(b)]
    m = fo.chains[@varname(m)]

    per_draw = _exceedance_closure(fo)

    probability_sum = zeros(Float64, length(N_log))
    nsamples_total = 0

    for chain_index in chain_indices
        @inbounds for sample_index in axes(b, 1)
            pred = b[sample_index, chain_index] + m[sample_index, chain_index] * v_log
            probability_sum .+= per_draw(sample_index, chain_index, pred, N_log)
        end
        nsamples_total += size(b, 1)
    end

    nsamples_total > 0 || error("the selected chains contain no posterior samples")

    return probability_sum ./ nsamples_total
end

"""
    _exceedance_closure(fo::FitObject) -> (i, j, pred, N_log) -> probs

Model hook returning a callable that, for posterior draw `(i, j)` with linear
prediction `pred = b + m·log10(v)`, yields the vector of exceedance
probabilities `P(N_obs > N | draw)` for the log10 cycle counts `N_log`. Each
model defines its own method. Internal.
"""
function _exceedance_closure end


function _chain_indices(fo::FitObject, chains)
    isnothing(fo.chains) && throw(ArgumentError("the model has not been fitted"))

    nchains = size(fo.chains, 2)

    indices = if chains === :all
        collect(1:nchains)
    elseif chains isa Integer
        [Int(chains)]
    elseif chains isa AbstractVector{<:Integer} || chains isa AbstractRange{<:Integer}
        collect(Int, chains)
    else
        throw(ArgumentError("chains must be :all, an integer, or a collection of integers"))
    end

    isempty(indices) && throw(ArgumentError("at least one chain must be selected"))

    all(index -> 1 <= index <= nchains, indices) ||
        throw(ArgumentError("chain indices must be between 1 and $nchains"))

    return indices
end

function _validate_prediction_coordinates(v::Real, N::AbstractVector{<:Real})
    isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

    all(value -> isfinite(value) && value > 0, N) ||
        throw(ArgumentError("all N values must be finite and strictly positive"))

    return nothing
end

"""
    exceedance_grid(fo::FitObject; v_range, N_range, v_res, N_res, chains = :all)

Evaluate [`exceedance_probability`](@ref) over a logarithmic grid of velocities
and cycle counts, returning `(v = v_values, N = N_values, probability = matrix)`
where `probability[i, j] = P(N_obs > N_values[j] | v_values[i])`.

A log-spaced grid is used because the process is modelled on the log-log scale.
`v_res` and `N_res` (grid sizes) must be at least `2`, and each range must be a
pair of positive, strictly increasing values.
"""
function exceedance_grid(
    fo::FitObject;
    v_range = (100.0, 160.0),
    N_range = (1e8, 1e10),
    v_res::Int = 100,
    N_res::Int = 100,
    chains = :all,
)
    v_res >= 2 || throw(ArgumentError("v_res must be at least 2"))

    N_res >= 2 || throw(ArgumentError("N_res must be at least 2"))

    v_range[1] > 0 && v_range[2] > v_range[1] ||
        throw(ArgumentError("v_range must contain two positive increasing values"))

    N_range[1] > 0 && N_range[2] > N_range[1] ||
        throw(ArgumentError("N_range must contain two positive increasing values"))

    # Dado que el proceso se modeliza en escala log-log, una malla
    # logarítmica representa mejor el dominio que una malla lineal.
    v_values = collect(logrange(v_range..., v_res))
    N_values = collect(logrange(N_range..., N_res))

    probabilities = Matrix{Float64}(undef, length(v_values), length(N_values))

    for x_index in eachindex(v_values)
        probabilities[x_index, :] .=
            exceedance_probability(fo, v_values[x_index], N_values; chains = chains)
    end

    return (v = v_values, N = N_values, probability = probabilities)
end
