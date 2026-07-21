using Test
using RETCore

@testset "DataObject" begin
    v = [1.0, 2.0]
    N = [10.0, 20.0]

    d = DataObject(v, N)

    @test d.v == v
    @test d.N == N
    @test isempty(d.v_censored)
    @test isempty(d.N_censored)

    @test_throws ArgumentError DataObject([1.0, 2.0], [10.0])
    @test_throws ArgumentError DataObject([1.0, 2.0], [10.0, 20.0], [1.0], [10.0, 20.0, 30.0])
end

@testset "vcat and capture_data" begin
    d1 = DataObject([1.0], [10.0])
    d2 = DataObject([2.0], [20.0])

    d = vcat(d1, d2)

    @test d.v == [1.0, 2.0]
    @test d.N == [10.0, 20.0]
    @test isempty(d.v_censored)
    @test isempty(d.N_censored)

    v_in = [108.7, 110.46]
    n_in = [1.0, 2.0]

    dcap = capture_data(v_in, n_in, 100.0)

    @test length(dcap.v) == length(dcap.N)
    @test length(dcap.v_censored) == length(dcap.N_censored)
    @test all(x -> x > 0, dcap.N)
    @test all(x -> x > 0, dcap.N_censored)
end
