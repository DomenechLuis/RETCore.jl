using Test
using RETCore
using Random
using Statistics
using Turing: @varname

# Full pipeline test with a REAL (tiny) MCMC run. It is slow because it compiles
# and runs Turing, so it only runs when RETCORE_RUN_MCMC=true is set in the
# environment (e.g. locally or in a dedicated CI job).
if get(ENV, "RETCORE_RUN_MCMC", "false") == "true"

    @testset "end-to-end fit (MCMC)" begin
        Random.seed!(1234)

        # synthetic data scattered around a decreasing log-log line
        # log10(N) = a + m*log10(v) + noise
        a_true, m_true = 25.0, -8.0
        v = collect(range(110.0, 150.0; length = 12))
        logN = a_true .+ m_true .* log10.(v) .+ 0.1 .* randn(length(v))
        N = 10.0 .^ logN

        fo = FitObject(normal_linear_model, DataObject(v, N);
                       n_warmup = 200, n_samples = 200, n_chains = 1)
        prepare_data!(fo)
        fit!(fo)

        @test !isnothing(fo.chains)

        # recovered slope should be clearly negative
        @test mean(fo.chains[@varname(m)]) < 0

        # predictions behave sensibly
        ps = exceedance_probability(fo, 130.0, [1e6, 1e12])
        @test all(0.0 .<= ps .<= 1.0)
        @test ps[1] >= ps[2]

        q = quantile(fo, 0.5, 130.0)
        @test all(isfinite, q)
    end
else
    @info "Skipping MCMC integration test (set RETCORE_RUN_MCMC=true to enable)"
end
