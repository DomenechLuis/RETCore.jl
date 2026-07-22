using Test
using RETCore
using Distributions
using Statistics
using Turing: @varname

@testset "FitObject construction" begin
    data = DataObject([1.0, 2.0], [10.0, 20.0])
    fo = FitObject(normal_linear_model, data)

    @test fo.model === normal_linear_model
    @test fo.raw_data === data
    @test isnothing(fo.pre_data)
    @test fo.center_v == false
    @test fo.center_N == false
    @test fo.n_warmup == 20_000
    @test fo.n_samples == 5_000
    @test fo.n_chains == 4
    @test fo.target_accept == 0.85

    # keyword overrides
    fo2 = FitObject(normal_linear_model, data; center_v = true, n_chains = 2, target_accept = 0.9)
    @test fo2.center_v == true
    @test fo2.n_chains == 2
    @test fo2.target_accept == 0.9

    # copy constructor drops results unless asked
    fo2.chains = :dummy
    copy_plain = FitObject(fo2)
    @test isnothing(copy_plain.chains)
    copy_full = FitObject(fo2; copy_results = true)
    @test copy_full.chains == :dummy
end

@testset "prepare_data! (generic)" begin
    data = DataObject([100.0, 200.0], [10.0, 20.0])

    # no centring
    fo = FitObject(normal_linear_model, data)
    prepare_data!(fo)
    @test !isnothing(fo.pre_data)
    @test fo.pre_data.v == log10.(data.v)
    @test fo.pre_data.N == log10.(data.N)
    @test fo.vmean == 0.0
    @test fo.Nmean == 0.0

    # centring records the means and shifts the data
    foc = FitObject(normal_linear_model, data; center_v = true, center_N = true)
    prepare_data!(foc)
    @test foc.vmean ≈ mean(log10.(data.v))
    @test foc.Nmean ≈ mean(log10.(data.N))
    @test foc.pre_data.v ≈ log10.(data.v) .- foc.vmean
    @test foc.pre_data.N ≈ log10.(data.N) .- foc.Nmean
end

@testset "priors" begin
    models = (normal_linear_model, lognormal_linear_model, exponential_linear_model)
    scenarios = (
        RETCore.prior_default,
        RETCore.prior_wide,
        RETCore.prior_optimistic,
        RETCore.prior_pessimistic,
        RETCore.prior_high_scatter,
    )

    for model in models, scenario in scenarios
        p = scenario(model)
        @test haskey(p, :a) && haskey(p, :m)
        @test p.a isa Distribution
        @test p.m isa Distribution
        # slope is truncated at 0 from above
        @test maximum(p.m) <= 0
    end

    # model-specific dispersion parameters are present
    @test haskey(RETCore.prior_default(normal_linear_model), :s)
    @test haskey(RETCore.prior_default(lognormal_linear_model), :s)
    @test haskey(RETCore.prior_default(lognormal_linear_model), :l)
    @test haskey(RETCore.prior_default(exponential_linear_model), :l)

    # high_scatter widens the dispersion vs default
    @test RETCore.prior_high_scatter(normal_linear_model).s.θ >
          RETCore.prior_default(normal_linear_model).s.θ
end

@testset "build_chain! centring formula" begin
    # b = a - m*vmean + Nmean, for every centring combination
    for (vmean, Nmean) in ((0.0, 0.0), (2.0, 0.0), (0.0, 3.0), (2.0, 3.0))
        fo = fit_with_chains(normal_linear_model; a = 25.0, m = -5.0, s = 1.0)
        fo.vmean = vmean
        fo.Nmean = Nmean
        RETCore.build_chain!(fo)
        b = fo.chains[@varname(b)]
        @test all(≈(25.0 - (-5.0) * vmean + Nmean), b)
    end
end