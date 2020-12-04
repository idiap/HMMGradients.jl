# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export logforward!

"""
`logforward!(logalpha,Nt,a,A,y)`

In-place version of [`logforward`](@ref).
"""
function logforward!(alpha::AbstractArray{T},Nt::Integer,
                     a::AbstractVector{T},
                     A::AbstractMatrix{T},
                     y::AbstractArray{T}) where {T <: AbstractFloat}
  Ns = size(A,1)
  fill!(alpha,-T(Inf))
  for j = 1:Ns
    alpha[1,j] = a[j] + y[1,j]
  end

  for t = 2:Nt
    for j = 1:Ns
      for i = 1:Ns
        aij = A[i,j]
        if aij != -T(Inf)
          alpha[t,j] = logadd(alpha[t,j],alpha[t-1,i]+aij+y[t,j])
        end
      end
    end
  end

  logML = -T(Inf)
  for i = 1:Ns
    logML = logadd(logML,alpha[Nt,i]) 
  end
  return alpha, logML/Nt
end

# forward with constrained path
function logforward!(alpha::AbstractArray{T},
                     Nt::Integer,
                     t2tr::Vector{Dict{D,Vector{D}}},
                     A::AbstractMatrix{T},
                     y::AbstractArray{T}) where {D<:Integer, T <: AbstractFloat}
  Ns = size(A,1)

  fill!(alpha,-T(Inf))
  for j in keys(t2tr[1])
    alpha[1,j] = y[1,j] - log(length(t2tr[1]))
  end

  for t = 2:Nt
    for i in keys(t2tr[t-1])
      for j in t2tr[t-1][i]
        aij = A[i,j]
        if  aij == -T(Inf) error("zero hit in transition matrix at ($i,$j), something wrong in time2transition or A") end
        alpha[t,j] = logadd(alpha[t,j],alpha[t-1,i]+aij+y[t,j])
      end
    end
  end

  logML = -T(Inf)
  for i = 1:Ns
    logML = logadd(logML,alpha[Nt,i]) 
  end
  return alpha, logML/Nt
end

# forward with constrained path
function logforward!(alpha::AbstractArray{T},
                     Nt::Integer,
                     t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                     A::AbstractMatrix{T},
                     y::AbstractArray{T}) where {D<:Integer, T <: AbstractFloat}
  Ns = size(A,1)

  fill!(alpha,-T(Inf))
  idx0 = unique(t2IJ[1][1])
  for j in idx0 
    alpha[1,j] = y[1,j] - log(length(idx0))
  end

  for t = 2:Nt
    I,J = t2IJ[t-1]
    for k in eachindex(I)
      i,j = I[k],J[k]
      alpha[t,j] = logadd(alpha[t,j],alpha[t-1,i]+A[i,j]+y[t,j])
    end
  end

  logML = -T(Inf)
  for i = 1:Ns
    logML = logadd(logML,alpha[Nt,i]) 
  end
  return alpha, logML/Nt
end

export logforward

"""
`logalpha, logML = logforward(Nt,a,A,y)`

Computes the scaled forward log-probabilities (``\\hat{\\alpha}``) of an Hidden Markov Model (HMM) of `Ns` states with log-transition matrix `A` (must be a `Ns` × `Ns` matrix), initial log-probabilities `a` (must be a `Ns`-long vector) and observation log-likelihoods `y` (must be of size `Nt2` × `Ns` with `Nt2` ≥ `Nt`).

Returns:
* `logalpha` a matrix of size `(Nt,Ns)` containing the forward log-probabilities
* `logML` the log likelihood of the observation normalized by the sequence length, i.e. ``\\log(P(\\mathbf{X}))/N_t``. 

`logalpha, logML = logforward(Nt,t2tr,A,y)`

Returns the forward log-probabilities with a constrained path (see [`forward`](@ref)).
"""
function logforward(Nt::Integer,a,A::AbstractMatrix{T},y) where {T <: AbstractFloat}
  Ns = size(A,1)
  alpha = -T(Inf)*ones(T,Nt,Ns)
  logforward!(alpha,Nt,a,A,y)
end

# y is a tensor Nt2 × Ns × Nb
function logforward(Nts::Array{D},
                    as::AbstractMatrix{T},
                    As::AbstractArray{T,3},
                    y::AbstractArray{T,3}) where {D <: Integer,
                                          T <: AbstractFloat}
  Ns = size(As,1)
  Nb = size(y,3) # batch number
  @assert Nb == length(Nts) == size(as,2) == size(As,3)
  alphas = [-T(Inf)*ones(T,Nt,Ns) for Nt in Nts]
  logMLs = zeros(T,Nb)
  for i = 1:Nb
    _, logML = logforward!(alphas[i],Nts[i],view(as,:,i),view(As,:,:,i),view(y,:,:,i))
    logMLs[i] = logML
  end
  return alphas, logMLs
end

# y is a tensor Nt2 × Ns × Nb
function logforward(Nts::Array{D},
                    as::AbstractVector,
                    A::AbstractMatrix{T},
                    y::Array{T,3}) where {D <: Integer,
                                          T <: AbstractFloat}
  Ns = size(A,1)
  Nb = size(y,3) # batch number
  @assert Nb == length(Nts) == length(as)
  alphas = [-T(Inf)*ones(T,Nt,Ns) for Nt in Nts]
  logMLs = zeros(T,Nb)
  for i = 1:Nb
    _, logML = logforward!(alphas[i],Nts[i],as[i],A,view(y,:,:,i))
    logMLs[i] = logML
  end
  return alphas, logMLs
end
