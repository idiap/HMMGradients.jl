# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export posterior

"""
`posterior(Nt,a,A,y)`

Computes the state posterior probabilities (``\\gamma``) for an HMM model with initial probability `a`, transition probability matrix `A`  and observation likelihoods `y`.
"""
function posterior(Nt,a::AbstractArray{T},A,y) where {T<:AbstractFloat}
  alpha, c = forward(Nt,a,A,y)
  beta = backward(Nt,A,c,y)
  alpha .*= beta
  return alpha
end

"""
`posterior(Nt,t2tr,A,y)`

Computes the state posterior probabilities (``\\gamma``) for an HMM model with constrained path `t2tr` (see [`forward`](@ref)), transition probability matrix `A`  and observation likelihoods `y`.
"""
function posterior(Nt,t2tr,A,y)
  alpha, c = forward(Nt,t2tr,A,y)
  beta = backward(Nt,A,c,t2tr,y)
  alpha .*= beta
  return alpha
end

export logposterior

"""
`logposterior(Nt,a,A,y)`

Computes the state posterior log-probabilities (``\\hat{\\gamma}``) for an HMM model with initial probability `a`, transition probability matrix `A`  and observation likelihoods `y`.
"""
function logposterior(Nt,a::AbstractArray{T},A,y) where {T<:AbstractFloat}
  alpha, logML = logforward(Nt,a,A,y)
  beta = logbackward(Nt,A,y)
  alpha .+= beta .- logML .* Nt
  return alpha
end

"""
`logposterior(Nt,t2tr,A,y)`

Computes the state posterior log-probabilities (``\\hat{\\gamma}``) for an HMM model with constrained path `t2tr` (see [`forward`](@ref)), transition probability matrix `A`  and observation likelihoods `y`.
"""
function logposterior(Nt,t2tr,A,y)
  alpha, logML = logforward(Nt,t2tr,A,y)
  beta = logbackward(Nt,A,t2tr,y)
  alpha .+= beta .- logML .* Nt
  return alpha
end
