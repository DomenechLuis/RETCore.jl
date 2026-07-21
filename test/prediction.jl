using Test
using RETCore

@testset "prediction helpers" begin
    data = DataObject([1.0, 2.0], [10.0, 20.0])
    fo = FitObject(normal_linear_model, data)

    @test_throws ArgumentError _chain_indices(fo, :all)
    @test_throws ArgumentError _validate_prediction_coordinates(0.0, [1.0])
    @test_throws ArgumentError _validate_prediction_coordinates(1.0, [0.0])
end

@testset "exceedance probability vectorization" begin
    data = DataObject([1.0, 2.0], [10.0, 20.0])
    fo = FitObject(normal_linear_model, data)

    @test_throws ArgumentError exceedance_probability(fo, [100.0, 200.0], [10.0])
    @test_throws ArgumentError exceedance_probability(fo, [100.0, 200.0], [10.0, 20.0]; chains = :bad)
end
