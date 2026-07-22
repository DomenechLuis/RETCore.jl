using Test
using RETCore

@testset "DataObject construction" begin
    v = [1.0, 2.0]
    N = [10.0, 20.0]

    d = DataObject(v, N)

    @test d.v == v
    @test d.N == N
    @test isempty(d.v_censored)
    @test isempty(d.N_censored)

    # length validation
    @test_throws ArgumentError DataObject([1.0, 2.0], [10.0])
    @test_throws ArgumentError DataObject([1.0, 2.0], [10.0, 20.0], [1.0], [10.0, 20.0, 30.0])

    # four-argument form keeps censored data
    dc = DataObject([1.0], [10.0], [2.0, 3.0], [100.0, 200.0])
    @test dc.v_censored == [2.0, 3.0]
    @test dc.N_censored == [100.0, 200.0]
end

@testset "vcat" begin
    d1 = DataObject([1.0], [10.0], [5.0], [500.0])
    d2 = DataObject([2.0], [20.0], [6.0], [600.0])

    d = vcat(d1, d2)

    @test d.v == [1.0, 2.0]
    @test d.N == [10.0, 20.0]
    @test d.v_censored == [5.0, 6.0]
    @test d.N_censored == [500.0, 600.0]

    # single argument is a no-op on contents
    d3 = vcat(d1)
    @test d3.v == d1.v && d3.N_censored == d1.N_censored
end

@testset "capture_data" begin
    grid = [100.0, 110.0, 120.0, 130.0]

    # nearest-bin assignment (searchsortedfirst) + run-out fill
    d = capture_data([111.0, 121.0], [5.0, 7.0], 999.0; v_array = grid)
    @test length(d.v) + length(d.v_censored) == length(grid)
    @test length(d.v) == length(d.N)
    @test length(d.v_censored) == length(d.N_censored)
    @test all(==(999.0), d.N_censored)
    @test all(x -> x > 0, d.N)

    # when two inputs share a bin, the smaller N is kept
    d2 = capture_data([111.0, 111.5], [8.0, 3.0], 999.0; v_array = grid)
    idx = findfirst(==(120.0), d2.v)   # 111.x snaps to the 120.0 bin edge
    @test d2.N[idx] == 3.0

    # values above the grid clamp to the last bin
    d3 = capture_data([10_000.0], [1.0], 999.0; v_array = grid)
    @test 130.0 in d3.v

    # unsorted grid is rejected
    @test_throws ArgumentError capture_data([111.0], [5.0], 999.0; v_array = [120.0, 100.0])
end