# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

module HMMGradients

using LinearAlgebra, SparseArrays
using ChainRulesCore

include("forward.jl")
include("logforward.jl")
include("backward.jl")
include("logbackward.jl")
include("posteriors.jl")
include("losses.jl")
include("grad.jl")
include("loggrad.jl")
include("chain_rules.jl")
include("utils.jl")

end
