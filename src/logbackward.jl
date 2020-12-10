# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export logbackward!

"""
`logbackward!(logbeta,Nt,A,y)`

In-place version of [`logbackward`](@ref).
"""
function logbackward!(beta::AbstractMatrix{T},
                      Nt::Integer,
                      A::AbstractMatrix{T},
                      y::AbstractMatrix{T}) where {T<:AbstractFloat}
  Ns = size(A,1)
  fill!(beta, -T(Inf))
  for i = 1:Ns
    beta[Nt,i] = zero(T)
  end
  for t = Nt-1:-1:1
    for j = 1:Ns
      for k = 1:Ns
        ajk = A[j,k]
        if ajk != -T(Inf)
          beta[t,j] = logadd(beta[t,j], beta[t+1,k]+ajk+y[t+1,k])
        end
      end
    end
  end
  return beta
end

# backward with constrained path
function logbackward!(beta::Matrix{T},
                      Nt::Integer,
                      A::AbstractMatrix{T},
                      t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                      y::AbstractMatrix{T}) where {D<:Integer,T<:AbstractFloat}
  Ns = size(A,1)
  fill!(beta, -T(Inf))
  for i = 1:Ns
    beta[Nt,i] = zero(T)
  end

  for t = Nt-1:-1:1
    I,J = t2IJ[t]
    for k in eachindex(I)
      i,j = I[k],J[k]
      beta[t,i] = logadd(beta[t,i], beta[t+1,j]+A[i,j]+y[t+1,j])
    end
  end
  return beta
end

export logbackward

"""
`logbackward(Nt,A,y)`

Computes the backward log-probabilities (``\\hat{\\beta}``) of an Hidden Markov Model (HMM) of `Ns` states with log-transition matrix `A` (must be a `Ns` × `Ns` matrix), scale coefficients and observation log-likelihoods `y` (must be of size `Nt2` × `Ns` with `Nt2` ≥ `Nt`).
"""
function logbackward(Nt::Integer,A::AbstractMatrix{T},y::AbstractMatrix{T}) where {T <: AbstractFloat}
  Ns = size(A,1)
  beta = zeros(T,Nt,Ns)
  logbackward!(beta,Nt,A,y)
end

"""
`logbackward(Nt,A,t2tr,y)`

Computes the backward log-probabilities with constrained path indicated by the `t2tr` vector. See [`forward`](@ref) for a description of the structure and meaning of `t2tr`.
"""
function logbackward(Nt::Integer,
                     A::AbstractMatrix{T},
                     t2tr,
                     y::AbstractMatrix{T}) where {T <: AbstractFloat, D <: Integer}
  Ns = size(A,1)
  beta = zeros(T,Nt,Ns)
  logbackward!(beta,Nt,A,t2tr,y)
end
