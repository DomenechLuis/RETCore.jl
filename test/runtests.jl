using Test
using RETCore

#= 
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using RETCore
Pkg.test("RETCore")
=#

# Shared helpers (build FitObjects with injected constant chains, no MCMC).
include("testutils.jl")

@testset "RETCore.jl" begin
    include("data.jl")
    include("fitobject.jl")
    include("prediction.jl")
    include("plotting.jl")
    include("integration.jl")
end