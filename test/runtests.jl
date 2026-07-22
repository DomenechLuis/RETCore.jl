using Test
using RETCore

# Shared helpers (build FitObjects with injected constant chains, no MCMC).
include("testutils.jl")

@testset "RETCore.jl" begin
    include("data.jl")
    include("fitobject.jl")
    include("prediction.jl")
    include("plotting.jl")
    include("integration.jl")
end
