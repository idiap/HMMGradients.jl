# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using HMMGradients, ChainRulesCore
using Test, Documenter, LinearAlgebra, SparseArrays
using Random
Random.seed!(123)
include("utils.jl")

@testset "HMMGradients.jl" begin
  @testset "Basic tests" begin
    include("test_forwardbackward.jl")
  end
  @testset "prob space" begin
    include("test_grad.jl")
  end
  @testset "logprob space" begin
    include("test_loggrad.jl")
  end
  if VERSION >= v"1.5.0"
    # printing number is different in 1.0 
    doctest(HMMGradients)
  end
end
