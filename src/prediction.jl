

function quantile(fo::FitObject, prob::Real, v::Real; chains = :all)
    0.0 < prob < 1.0 || throw(ArgumentError("prob must be between 0 and 1"))
    isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

    return exp10.(log_log_quantile(fo, prob, log10(v), chains = chains))
end


"""
tolerance_limit(fo::FitObject, reliability::Real, credibility::Real, v::Real; side::Symbol = :lower)

Bayesian one-sided tolerance limit.

A lower tolerance limit L satisfies

    P(P(X ≥ L) ≥ reliability | data) = credibility

An upper tolerance limit U satisfies

    P(P(X ≤ U) ≥ reliability | data) = credibility
"""
function tolerance_limit(
    fo::FitObject,
    reliability::Real,
    credibility::Real,
    v::Real;
    side::Symbol = :lower,
    chains = :all 
)
    0.0 < reliability < 1.0 || throw(ArgumentError("reliability must be between 0 and 1"))
    0.0 < credibility < 1.0 || throw(ArgumentError("credibility must be between 0 and 1"))
    isfinite(v) && v > 0 || throw(ArgumentError("v must be finite and strictly positive"))

    if side == :lower
        limit = quantile(quantile(fo, 1 - reliability, v, chains = chains)[:], 1 - credibility)
    elseif side == :upper
        limit = quantile(quantile(fo, reliability, v, chains = chains)[:], credibility)
    else
        throw(ArgumentError("side must be either :upper or :lower"))
    end

    return limit
end


function exceedance_probability(fo::FitObject, v::Real, N::Real; kwargs...)
    return only(exceedance_probability(fo, v, [N]; kwargs...))
end


function exceedance_probability(
    fo::FitObject,
    v::AbstractVector{<:Real},
    N::AbstractVector{<:Real};
    kwargs...,
)
    length(v) == length(N) || throw(ArgumentError("v and N must have the same length"))

    exceedance_probability.(Ref(fo), v, N; kwargs...)
end


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
        throw(ArgumentError("all y values must be finite and strictly positive"))

    return nothing
end

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
