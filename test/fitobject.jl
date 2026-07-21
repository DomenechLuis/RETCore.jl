using Test
using RETCore

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
end

@testset "prepare_data!" begin
    data = DataObject([100.0, 200.0], [10.0, 20.0])
    fo = FitObject(normal_linear_model, data; center_v = true, center_N = true)

    prepare_data!(fo)

    @test !isnothing(fo.pre_data)
    @test fo.pre_data.v == log10.(data.v) .- fo.vmean
    @test fo.pre_data.N == log10.(data.N) .- fo.Nmean
end
