using Test
using RETCore
using Plots
using Turing: @varname

# headless GR: never try to open a window
ENV["GKSwstype"] = "100"

@testset "DataObject plotting" begin
    d = DataObject([120.0, 130.0], [1e8, 2e8], [140.0], [1e9])
    @test plot(d) isa Plots.Plot
    @test plot([d, d]) isa Plots.Plot

    # per-element kwargs of the wrong length are rejected
    p = Plots.plot()
    @test_throws ArgumentError plot!(p, [d, d]; data_kwargs = [NamedTuple()])
end

@testset "FitObject plotting (smoke)" begin
    fo = normal_fixture()

    @test plot(fo) isa Plots.Plot
    @test plot_means(fo) isa Plots.Plot
    @test histogram2d(fo) isa Plots.Plot
    @test histogram2d(fo; chains = :split) isa Plots.Plot

    p, grid = plot_exceedance(fo; v_res = 5, N_res = 5)
    @test p isa Plots.Plot
    @test haskey(grid, :probability)

    p2, _ = plot_exceedance(grid)
    @test p2 isa Plots.Plot

    # iso_levels out of range are rejected
    @test_throws ArgumentError plot_exceedance(grid; iso_levels = [1.5])
end

@testset "plot_values passes kwargs (regression)" begin
    fo = normal_fixture()
    # must not throw when a keyword is forwarded
    @test plot_values(fo, [120.0, 130.0], [1e8, 2e8]; label = "obs") isa Plots.Plot
end

@testset "plot_means for every model" begin
    for fo in (normal_fixture(), lognormal_fixture(), exponential_fixture())
        @test plot_means(fo) isa Plots.Plot
    end
end