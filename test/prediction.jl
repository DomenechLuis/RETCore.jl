using Test
using RETCore
using Distributions
using Statistics
using Turing: @varname

@testset "prediction validation helpers" begin
    fo = FitObject(normal_linear_model, DataObject([1.0, 2.0], [10.0, 20.0]))

    # not fitted yet
    @test_throws ArgumentError RETCore._chain_indices(fo, :all)

    @test_throws ArgumentError RETCore._validate_prediction_coordinates(0.0, [1.0])
    @test_throws ArgumentError RETCore._validate_prediction_coordinates(1.0, [0.0])
    @test isnothing(RETCore._validate_prediction_coordinates(1.0, [1.0, 2.0]))
end

@testset "_chain_indices selection" begin
    fo = normal_fixture(nc = 3)   # 3 chains

    @test RETCore._chain_indices(fo, :all) == [1, 2, 3]
    @test RETCore._chain_indices(fo, 2) == [2]
    @test RETCore._chain_indices(fo, [1, 3]) == [1, 3]
    @test RETCore._chain_indices(fo, 1:2) == [1, 2]

    @test_throws ArgumentError RETCore._chain_indices(fo, :bad)
    @test_throws ArgumentError RETCore._chain_indices(fo, 0)
    @test_throws ArgumentError RETCore._chain_indices(fo, 4)
    @test_throws ArgumentError RETCore._chain_indices(fo, Int[])
end

@testset "exceedance_probability numerics" begin
    b, m, s = 25.0, -5.0, 1.0
    fo = normal_fixture()   # b=25, m=-5, s=1
    v = 130.0

    # matches the closed-form Gaussian ccdf for constant chains
    N = 1e8
    got = exceedance_probability(fo, v, [N])[1]
    want = ccdf(Normal(b + m * log10(v), s), log10(N))
    @test got ≈ want

    # scalar / vector consistency
    @test exceedance_probability(fo, v, N) ≈ got
    @test exceedance_probability(fo, [v, v], [N, N]) ≈ [got, got]

    # bounds and monotonicity in N
    ps = exceedance_probability(fo, v, [1e6, 1e7, 1e8, 1e9, 1e10])
    @test all(0.0 .<= ps .<= 1.0)
    @test issorted(ps; rev = true)

    # empty N and error paths
    @test exceedance_probability(fo, v, Float64[]) == Float64[]
    @test_throws ArgumentError exceedance_probability(fo, -1.0, [N])
    @test_throws ArgumentError exceedance_probability(fo, [1.0, 2.0], [1.0])          # length mismatch
    @test_throws ArgumentError exceedance_probability(fo, v, [N]; chains = :bad)

    # not fitted
    foraw = FitObject(normal_linear_model, DataObject([1.0], [10.0]))
    @test_throws ArgumentError exceedance_probability(foraw, v, [N])
end

@testset "exceedance kernels match each model" begin
    v, N = 130.0, 1e8
    vlog, nlog = log10(v), log10(N)

    fo_n = normal_fixture()               # b=25,m=-5,s=1
    @test exceedance_probability(fo_n, v, [N])[1] ≈ ccdf(Normal(25 - 5vlog, 1.0), nlog)

    fo_e = exponential_fixture()          # b=25,m=-5,l=1
    @test exceedance_probability(fo_e, v, [N])[1] ≈ cdf(Exponential(1.0), (25 - 5vlog) - nlog)

    fo_l = lognormal_fixture()            # b=25,m=-5,s=1,l=0
    zmode = exp(0.0 - 1.0^2)
    @test exceedance_probability(fo_l, v, [N])[1] ≈ cdf(LogNormal(0.0, 1.0), (25 - 5vlog) + zmode - nlog)
end

@testset "quantile and tolerance_limit" begin
    fo = normal_fixture(ni = 5, nc = 2)
    v = 130.0

    # scalar v -> a Vector of per-draw quantiles (one per sample)
    q = quantile(fo, 0.5, v)
    @test q[:] isa Vector
    @test length(q) == 10                 # 5 iters * 2 chains
    # constant chains => the 0.5 quantile equals exp10(b + m*log10(v))
    @test all(≈(exp10(25 - 5 * log10(v))), q)


    @test_throws ArgumentError quantile(fo, 0.0, v)
    @test_throws ArgumentError quantile(fo, 0.5, -1.0)
    @test_throws MethodError quantile(fo, 1.5, [1.0])

    # tolerance limits
    lo = tolerance_limit(fo, 0.9, 0.95, v; side = :lower)
    hi = tolerance_limit(fo, 0.9, 0.95, v; side = :upper)
    @test isfinite(lo) && isfinite(hi)
    @test lo <= hi
    @test_throws ArgumentError tolerance_limit(fo, 1.1, 0.95, v)
    @test_throws ArgumentError tolerance_limit(fo, 0.9, 0.95, v; side = :sideways)

end

@testset "exceedance_grid" begin
    fo = normal_fixture()

    # default v_range must work (regression against the (100.0,160,0) typo)
    grid = exceedance_grid(fo; v_res = 5, N_res = 4)
    @test keys(grid) == (:v, :N, :probability)
    @test size(grid.probability) == (5, 4)
    @test all(0.0 .<= grid.probability .<= 1.0)

    @test_throws ArgumentError exceedance_grid(fo; v_res = 1)
    @test_throws ArgumentError exceedance_grid(fo; N_res = 1)
    @test_throws ArgumentError exceedance_grid(fo; v_range = (160.0, 100.0))
    @test_throws ArgumentError exceedance_grid(fo; N_range = (-1.0, 10.0))
end