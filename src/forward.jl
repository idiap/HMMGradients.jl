# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export forward!

"""
`forward!(alpha,c,Nt,a,A,y)`

In-place version of [`forward`](@ref).
"""
function forward!(alpha::AbstractArray{T},c::Vector{T},
                  Nt::Integer,
                  a::AbstractVector{T},
                  A::AbstractMatrix{T},
                  y::AbstractArray{T}) where {T <: AbstractFloat}
  Ns = size(A,1)
  fill!(alpha,zero(T))
  fill!(c,zero(T))
  for j = 1:Ns
    alpha[1,j] = a[j]*y[1,j]
    c[1] += alpha[1,j]
  end

  # normalize
  for j = 1:Ns
    alpha[1,j] /= c[1]
  end

  for t = 2:Nt
    for j = 1:Ns
      for i = 1:Ns
        aij = A[i,j]
        if aij != T(0)
          alpha[t,j] += alpha[t-1,i]*aij*y[t,j]
        end
      end
      c[t] += alpha[t,j]
    end
    for j = 1:Ns
      alpha[t,j] /= c[t] 
    end
  end
  return alpha, c
end

function forward!(alpha::AbstractArray{T},c::Vector{T},
                  Nt::Integer,
                  t2tr::Vector{Dict{D,Vector{D}}},
                  A::AbstractMatrix{T},
                  y::AbstractArray{T}) where {D<:Integer, T <: AbstractFloat}
  Ns = size(A,1)

  fill!(alpha,zero(T))
  fill!(c,zero(T))
  for j in keys(t2tr[1])
    alpha[1,j] = y[1,j]/length(t2tr[1])
    c[1] += alpha[1,j]
  end

  # normalize
  for j = 1:Ns
    alpha[1,j] /= c[1]
  end

  for t = 2:Nt
    for i in keys(t2tr[t-1])
      for j in t2tr[t-1][i]
        aij = A[i,j]
        if  aij == zero(T) error("zero hit in transition matrix at ($i,$j), something wrong in time2transition or A") end
        alpha[t,j] += alpha[t-1,i]*aij*y[t,j]
      end
    end
    for j = 1:Ns
      c[t] += alpha[t,j]
    end
    for j = 1:Ns
      alpha[t,j] /= c[t] 
    end
  end
  return alpha, c
end

# forward with constrained path
function forward!(alpha::AbstractArray{T},c::Vector{T},
                  Nt::Integer,
                  t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                  A::AbstractMatrix{T},
                  y::AbstractArray{T}) where {D<:Integer, T <: AbstractFloat}
  Ns = size(A,1)

  fill!(alpha,zero(T))
  fill!(c,zero(T))
  idx0 = unique(t2IJ[1][1])
  for i in idx0
    alpha[1,i] = y[1,i]/length(idx0)
    c[1] += alpha[1,i]
  end

  # normalize
  for j = 1:Ns
    alpha[1,j] /= c[1]
  end

  for t = 2:Nt
    I,J = t2IJ[t-1]
    for k in eachindex(I)
      i,j = I[k],J[k]
      alpha[t,j] += alpha[t-1,i]*A[i,j]*y[t,j]
    end
    for j = 1:Ns
      c[t] += alpha[t,j]
    end
    for j = 1:Ns
      alpha[t,j] /= c[t] 
    end
  end
  return alpha, c
end

export forward

"""
  `alpha, c = forward(Nt,a,A,y)`

Computes the scaled forward probabilities (``\\bar{\\alpha}``) of an Hidden Markov Model (HMM) of `Ns` states with transition matrix `A` (must be a `Ns` × `Ns` matrix), initial probabilities `a` (must be a `Ns`-long vector) and observation likelihoods `y` (must be of size `Nt2` × `Ns` with `Nt2` ≥ `Nt`).

Returns:
* `alpha` a matrix of size `(Nt,Ns)` containing the forward probabilities
* `c` a `Nt`-long vector containing the normalization coefficients which can be used to compute the backward probabilities (see `backward`) and the log maximum likelihood (`sum(log.(c))`).

`alpha, c = forward(Nt,t2tr,A,y)`

Here `t2tr` can be an `Nt-1` vector of `Pair{Vector{D},Vector{D}}`. For example given `Nt=3`:
```julia
t2tr = [[1,1]=>[1,2],[1,2]=>[3,3],[3]=>[4]]
```
indicates that at time `t=1` transitions from state `1` to `1` and from `1` to `2` are allowed. 
At time `t=2` transitions from state `1` to `3` and from `2` to `3` are allowed.
At time `t=3` only the transition from state `3` to `4` is allowed.
In practice these indices can be used to construct the (possibly sparse) time-dependent transition matrices described in the [theory section](@ref theory).
"""
function forward(Nt::Integer,a,A::AbstractArray{T},y) where {T <: AbstractFloat}
  Ns = size(y,2)
  alpha = zeros(T,Nt,Ns)
  c = zeros(T,Nt)
  forward!(alpha,c,Nt,a,A,y)
end

# y is a tensor Nt2 × Ns × Nb
function forward(Nts::Array{D},
                 as::AbstractMatrix{T},
                 As::AbstractArray{T,3},
                 ys::AbstractArray{T,3}) where {D <: Integer,
                                       T <: AbstractFloat}
  Ns = size(As,1)
  Nb = size(ys,3) # batch number
  @assert Nb == length(Nts) == size(as,2) == size(As,3)
  alphas = [zeros(T,Nt,Ns) for Nt in Nts]
  cs = [zeros(T,Nt) for Nt in Nts]
  for i = 1:Nb
    forward!(alphas[i],cs[i],Nts[i],view(as,:,i),view(As,:,:,i),view(ys,:,:,i))
  end
  return alphas, cs
end

# y is a tensor Nt2 × Ns × Nb, constraint path
function forward(Nts::Array{D},
                 as::AbstractVector,
                 A::AbstractMatrix{T},
                 y::Array{T,3}) where {D <: Integer,
                                       T <: AbstractFloat}
  Ns = size(A,1)
  Nb = size(y,3) # batch number
  @assert Nb == length(Nts) == length(as)
  alphas = [zeros(T,Nt,Ns) for Nt in Nts]
  cs = [zeros(T,Nt) for Nt in Nts]
  for i = 1:Nb
    forward!(alphas[i],cs[i],Nts[i],as[i],A,view(y,:,:,i))
  end
  return alphas, cs
end
