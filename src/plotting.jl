
function plot!(
    p::Plots.Plot,
    fo::FitObject;
    n_lines::Int = 100,
    v_range = (100.0, 160.0),
    color = :red,
    kwargs...,
)

    l_chain, n_chain = size(fo.chains)

    idx = randperm(l_chain)[1:min(n_lines, l_chain)]

    v_lines = collect(logrange(v_range..., 5))

    for j = 1:n_chain
        for i in idx
            plot!(
                p,
                exp10.(
                    fo.chains[@varname(b)][i, j] .+
                    fo.chains[@varname(m)][i, j] .* log10.(v_lines),
                ),
                v_lines;
                alpha = 0.05,
                color = color,
                label = false,
                kwargs...,
            )
        end
    end

    return p
end

function plot(fo::FitObject; n_lines::Int = 100, kwargs...)

    p = Plots.plot()

    plot!(p, fo; n_lines = n_lines, kwargs...)

    return p
end


function _histogram2d_scale(fo::FitObject, vars; nbins = 40)

    x = fo.chains[vars[1]]
    y = fo.chains[vars[2]]

    xedges = range(extrema(vec(x))...; length = nbins+1)
    yedges = range(extrema(vec(y))...; length = nbins+1)

    _, nchains = size(x)

    maxcount = 0

    for j = 1:nchains

        h = fit(Histogram, (x[:, j], y[:, j]), (xedges, yedges))

        maxcount = max(maxcount, maximum(h.weights))
    end

    return maxcount, xedges, yedges

end



function histogram2d(
    fo::FitObject;
    vars = (@varname(b), @varname(m)),
    chains = :all,
    nbins = 40,
    kwargs...,
)

    if chains === :split

        _, nchains = size(fo.chains)

        maxcount, xedges, yedges = _histogram2d_scale(fo, vars; nbins = nbins)

        pp = [
            Plots.histogram2d(
                fo.chains[vars[1]][:, j],
                fo.chains[vars[2]][:, j];
                bins = (xedges, yedges),
                clims = (0, maxcount),
                xlabel = string(vars[1]),
                ylabel = string(vars[2]),
                title = "Chain $j",
                kwargs...,
            ) for j = 1:nchains
        ]

        return plot(pp...; layout = nchains)
    end

    if chains === :all
        chains = collect(1:size(fo.chains, 2))
    elseif chains isa Integer
        chains = [chains]
    end

    return Plots.histogram2d(
        vec(fo.chains[vars[1]][:, chains]),
        vec(fo.chains[vars[2]][:, chains]);
        xlabel = string(vars[1]),
        ylabel = string(vars[2]),
        nbins = nbins,
        kwargs...,
    )

end

function plot_means(
    fo::FitObject;
    n_lines::Int = 5,
    v_range = (100.0, 160.0),
    curve_res = 20,
    kwargs_mean = (color = :red,),
    kwargs_curve = (color = :green,),
)

    p = Plots.plot()

    plot_means!(
        p,
        fo;
        n_lines = n_lines,
        v_range = v_range,
        curve_res = curve_res,
        kwargs_mean,
        kwargs_curve,
    )

    return p
end


function plot_exceedance!(
    p::Plots.Plot,
    grid::NamedTuple;
    heatmap::Bool = true,
    iso_levels = [0.80, 0.5, 0.95],
    heatmap_kwargs = NamedTuple(),
    contour_kwargs = NamedTuple(),
)
    hasproperty(grid, :v) || throw(ArgumentError("grid must contain the field :v"))

    hasproperty(grid, :N) || throw(ArgumentError("grid must contain the field :N"))

    hasproperty(grid, :probability) ||
        throw(ArgumentError("grid must contain the field :probability"))

    size(grid.probability) == (length(grid.v), length(grid.N)) || throw(
        ArgumentError(
            "grid.probability must have dimensions " * "(length(grid.v), length(grid.N))",
        ),
    )

    if heatmap
        heatmap_options = merge(
            (
                clims = (0.0, 1.0),
                colorbar_title = L"P(N > N_0 | v = v_0)",
                xlabel = "N",
                ylabel = "v",
                label = false,
            ),
            heatmap_kwargs,
        )

        Plots.heatmap!(p, grid.N, grid.v, grid.probability; heatmap_options...)
    end

    if !isnothing(iso_levels) && !isempty(iso_levels)
        all(level -> 0.0 <= level <= 1.0, iso_levels) ||
            throw(ArgumentError("all iso_levels must be between 0 and 1"))

        contour_options = merge(
            (
                levels = iso_levels,
                color = :black,
                linewidth = 1.0,
                clabels = true,
                label = true,
            ),
            contour_kwargs,
        )

        Plots.contour!(p, grid.N, grid.v, grid.probability; contour_options...)
    end

    return p, grid
end


function plot_exceedance(grid::NamedTuple; plot_kwargs = NamedTuple(), kwargs...)
    p = Plots.plot(; plot_kwargs...)

    plot_exceedance!(p, grid; kwargs...)

    return p, grid
end


function plot_exceedance!(
    p::Plots.Plot,
    fo::FitObject;
    v_range = (100.0, 160.0),
    N_range = (1e8, 1e10),
    v_res::Int = 100,
    N_res::Int = 100,
    chains = :all,
    heatmap::Bool = true,
    iso_levels = [0.05, 0.5, 0.95],
    heatmap_kwargs = NamedTuple(),
    contour_kwargs = NamedTuple(),
)
    grid = exceedance_grid(
        fo;
        v_range = v_range,
        N_range = N_range,
        v_res = v_res,
        N_res = N_res,
        chains = chains,
    )

    plot_exceedance!(
        p,
        grid;
        heatmap = heatmap,
        iso_levels = iso_levels,
        heatmap_kwargs = heatmap_kwargs,
        contour_kwargs = contour_kwargs,
    )

    return p, grid
end
function plot_exceedance(
    fo::FitObject;
    v_range = (100.0, 160.0),
    N_range = (1e8, 1e10),
    v_res::Int = 100,
    N_res::Int = 100,
    chains = :all,
    heatmap::Bool = true,
    iso_levels = [0.05, 0.5, 0.95],
    plot_kwargs = NamedTuple(),
    heatmap_kwargs = NamedTuple(),
    contour_kwargs = NamedTuple(),
)
    p = Plots.plot(; plot_kwargs...)

    return plot_exceedance!(
        p,
        fo;
        v_range = v_range,
        N_range = N_range,
        v_res = v_res,
        N_res = N_res,
        chains = chains,
        heatmap = heatmap,
        iso_levels = iso_levels,
        heatmap_kwargs = heatmap_kwargs,
        contour_kwargs = contour_kwargs,
    )
end


function plot_values!(
    p::Plots.Plot,
    fo::FitObject,
    v::AbstractVector{<:Real},
    N::AbstractVector{<:Real};
    kwargs...,
)

    probs = exceedance_probability(fo, v, N)

    plot!(
        p,
        N,
        v;
        series_annotations = [(pi, :green, :bottom) for pi in round.(probs, digits = 3)],
        kwargs...,
    )
end

function plot_values(
    fo::FitObject,
    v::AbstractVector{<:Real},
    N::AbstractVector{<:Real};
    kwargs...,
)
    p = plot()
    return plot_values!(p, fo, v, N; kwargs...)
end
