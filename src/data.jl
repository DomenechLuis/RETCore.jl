"""
struct DataObject{TX,TY}
    v::TX
    N::TY

    v_censored::TX
    N_censored::TY
end

This structure represents a dataset with two sets of values: `v` and `v_censored`, along with their respective counts `N` and `N_censored`. The type parameters `TX` and `TY` allow for flexibility in the data types used for these fields.
"""
struct DataObject{TX,TY}
    v::TX
    N::TY

    v_censored::TX
    N_censored::TY
end

function DataObject(v, N)
    length(v) == length(N) || throw(ArgumentError("v and N must have the same length"))
    DataObject(v, N, similar(v, 0), similar(N, 0))
end

function DataObject(v, N, v_censored, N_censored)

    length(v) == length(N) || throw(ArgumentError("v and N must have the same length"))
    length(v_censored) == length(N_censored) ||
        throw(ArgumentError("v_censored and N_censored must have the same length"))

    return DataObject{typeof(v),typeof(N)}(v, N, v_censored, N_censored)
end

function Base.vcat(d::DataObject...)
    DataObject(
        vcat((x.v for x in d)...),
        vcat((x.N for x in d)...),
        vcat((x.v_censored for x in d)...),
        vcat((x.N_censored for x in d)...),
    )
end


function plot!(
    p::Plots.Plot,
    dataobject::DataObject;
    data_kwargs = NamedTuple(),
    censored_kwargs = (markershape = :x, label = ""),
)

    if !isempty(dataobject.v)
        scatter!(p, dataobject.N, dataobject.v; data_kwargs...)
    end

    if !isempty(dataobject.v_censored)
        scatter!(p, dataobject.N_censored, dataobject.v_censored; censored_kwargs...)
    end

    return p
end


function plot(
    dataobject::DataObject;
    data_kwargs = NamedTuple(),
    censored_kwargs = (markershape = :x, label = ""),
    kwargs...,
)

    p = Plots.plot(; kwargs...)
    plot!(p, dataobject; data_kwargs = data_kwargs, censored_kwargs = censored_kwargs)

    return p
end


function plot!(
    p::Plots.Plot,
    dataobjects::AbstractVector{<:DataObject};
    data_kwargs = nothing,
    censored_kwargs = nothing,
)

    n = length(dataobjects)

    # Normaliza data_kwargs
    if isnothing(data_kwargs)
        data_kwargs = fill(NamedTuple(), n)
    elseif data_kwargs isa NamedTuple
        data_kwargs = fill(data_kwargs, n)
    else
        length(data_kwargs) == n || throw(ArgumentError("data_kwargs must have length $n"))
    end

    # Normaliza censored_kwargs
    if isnothing(censored_kwargs)
        censored_kwargs = fill((markershape = :x, label = ""), n)
    elseif censored_kwargs isa NamedTuple
        censored_kwargs = fill(censored_kwargs, n)
    else
        length(censored_kwargs) == n ||
            throw(ArgumentError("censored_kwargs must have length $n"))
    end

    # Dibuja cada DataObject
    for i in eachindex(dataobjects)
        plot!(
            p,
            dataobjects[i];
            data_kwargs = data_kwargs[i],
            censored_kwargs = censored_kwargs[i],
        )
    end

    return p
end


function plot(
    dataobjects::AbstractVector{<:DataObject};
    data_kwargs = nothing,
    censored_kwargs = nothing,
    kwargs...,
)

    p = Plots.plot(; kwargs...)

    plot!(p, dataobjects; data_kwargs = data_kwargs, censored_kwargs = censored_kwargs)

    return p
end


v_aeronordic = [
    107.81,
    108.7,
    109.58,
    110.46,
    111.35,
    112.23,
    113.11,
    114.0,
    114.88,
    115.76,
    116.65,
    117.53,
    118.41,
    119.3,
    120.18,
    121.06,
    121.95,
    122.83,
    123.71,
    124.6,
    125.48,
    126.36,
    127.25,
    128.13,
    129.01,
    129.9,
    130.78,
    131.66,
    132.55,
    133.43,
    134.31,
    135.2,
    136.08,
    136.96,
    137.85,
    138.73,
    139.61,
    140.5,
    141.38,
    142.26,
    143.15,
    144.03,
    144.91,
    145.8,
    146.68,
    147.56,
    148.45,
    149.33,
    150.21,
    151.1,
    151.98,
    152.86,
    153.75,
    154.63,
    155.51,
    156.4,
    157.28,
    158.16,
    159.05,
    159.93,
]


function capture_data(v_in, n_in, N_runout; v_array = v_aeronordic) #"v_array must be sorted"

    N_array = zeros(size(v_array)) .+ Inf
    for (vi, ni) in zip(v_in, n_in)
        ind = min(searchsortedfirst(v_array, vi), length(v_array))

        if N_array[ind] > ni
            N_array[ind] = ni
        end
    end

    mask = isfinite.(N_array)

    return DataObject(
        v_array[mask],
        N_array[mask],
        v_array[.!mask],
        fill(N_runout, count(.!mask)),
    )
end
