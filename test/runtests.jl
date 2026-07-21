using Test
using RETCore

@testset "DataObject" begin
    d = DataObject([1.0, 2.0], [10.0, 20.0])

    @test d.v == [1.0, 2.0]
    @test d.N == [10.0, 20.0]
    @test isempty(d.v_censored)
    @test isempty(d.N_censored)

    @test_throws ArgumentError DataObject([1.0, 2.0], [10.0])
end

@testset "capture_data" begin
    v_in = [108.7, 110.46]
    n_in = [1.0, 2.0]

    d = capture_data(v_in, n_in, 100.0)

    @test length(d.v) > 0
    @test length(d.N) > 0
    @test length(d.v_censored) >= 0
end