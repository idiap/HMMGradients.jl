# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export backward!

"""
`backward!(beta,Nt,A,c,y)`

In-place version of [`backward`](@ref).
"""
function backward!(beta::AbstractMatrix{T},Nt::Integer,
                   A::AbstractMatrix{T},c::Vector{T},y::AbstractMatrix{T}) where {T<:AbstractFloat}
  Ns = size(A,1)
  fill!(beta,zero(T))
  beta[end,:] .= 1
  for t = Nt-1:-1:1
    for j = 1:Ns
      for k = 1:Ns
        aij = A[j,k]
        if  aij != T(0)
          beta[t,j] += beta[t+1,k]*aij*y[t+1,k]
        end
      end
    end
    for j = 1:Ns
      beta[t,j] /= c[t+1]
    end
  end
  return beta
end


"""
`backward!(beta,Nt,A,c,t2tr,y)`

In-place version [`backward`](@ref) with constrined path.
"""
function backward!(beta::AbstractMatrix{T},
                   Nt::Integer,
                   A::AbstractMatrix{T},
                   c::Vector{T},
                   t2tr::Vector{Dict{D,Vector{D}}},
                   y::AbstractMatrix{T}) where {D<:Integer,T<:AbstractFloat}
  Ns = size(A,1)
  fill!(beta,zero(T))
  beta[end,:] .= 1
  it2tr = invert_time2transitions(t2tr)

  for t = Nt-1:-1:1
    for k in keys(it2tr[t])
      for j in it2tr[t][k]
        beta[t,j] += beta[t+1,k]*A[j,k]*y[t+1,k]
      end
    end
    for j = 1:Ns
      beta[t,j] /= c[t+1]
    end
  end
  return beta
end

# backward with constrained path
function backward!(beta::AbstractMatrix{T},
                   Nt::Integer,
                   A::AbstractMatrix{T},
                   c::Vector{T},
                   t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                   y::AbstractMatrix{T}) where {D<:Integer,T<:AbstractFloat}
  Ns = size(A,1)
  fill!(beta,zero(T))
  @views beta[end,:] .= 1

  for t = Nt-1:-1:1
    I,J = t2IJ[t]
    for k in eachindex(I)
      i,j = I[k],J[k]
      beta[t,i] += beta[t+1,j]*A[i,j]*y[t+1,j]
    end
    for j = 1:Ns
      beta[t,j] /= c[t+1]
    end
  end
  return beta
end

export backward

"""
`backward(Nt,A,c,y)`

Computes the scaled backward probabilities (``\\bar{\\beta}``) of an Hidden Markov Model (HMM) of `Ns` states with transition matrix `A` (must be a `Ns` × `Ns` matrix), scale coefficients `c` (must be a `Nt`-long vector which is produced by [`forward`](@ref)) and observation likelihoods `y` (must be of size `Nt2` × `Ns` with `Nt2` ≥ `Nt`).
"""
function backward(Nt::Integer,A::AbstractArray{T},c::Vector{T},y::AbstractMatrix{T}) where {T <: AbstractFloat}
  Ns = size(y,2)
  beta = zeros(T,Nt,Ns)
  backward!(beta,Nt,A,c,y)
end

"""
`backward(Nt,A,c,t2tr,y)`

Computes the scaled backward probabilities with constrained path indicated by the `t2tr` vector. See [`forward`](@ref) for a description of the structure and meaning of `t2tr`.
"""
function backward(Nt::Integer,
                  A::AbstractMatrix{T},
                  c::Vector{T},
                  t2tr,
                  y::AbstractMatrix{T}) where {D <: Integer,
                                               T <: AbstractFloat}
  Ns = size(A,1)
  beta = zeros(T,Nt,Ns)
  backward!(beta,Nt,A,c,t2tr,y)
end

