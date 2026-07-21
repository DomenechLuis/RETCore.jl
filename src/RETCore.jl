module RETCore

using Statistics
using StatsBase
using Turing
using Distributions
using FlexiChains
using LaTeXStrings
using Random
using Plots

import Plots: plot, plot!, histogram2d
import Base: vcat
import Statistics: quantile

export DataObject,
    capture_data,
    FitObject,
    fit!,
    prepare_data!,
    normal_linear_model,
    lognormal_linear_model,
    exponential_linear_model,
    exceedance_probability,
    exceedance_grid,
    plot_means!,
    plot_means,
	plot_exceedance,
	plot_exceedance!,
	plot_values!,
	plot_values,
    prior_sensitivity

include("data.jl")
include("fitobject.jl")
include("models/normal.jl")
include("models/lognormal.jl")
include("models/exponential.jl")

include("prediction.jl")
include("plotting.jl")
include("diagnostics.jl")

end
