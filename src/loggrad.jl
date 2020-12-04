# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

function get_loggrad!(grada::AbstractVector{T},
                      gradA::AbstractArray{T}, 
                      grady::AbstractArray{T}, 
                      betat0::Vector{T}, betat1::Vector{T}, 
                      Nt::Integer,
                      a::AbstractVector{T}, 
                      A::AbstractMatrix{T},
                      y::AbstractArray{T}, 
                      alpha::Matrix{T},
                      logML::T
                     ) where {T <: AbstractFloat}
  Ns = size(A,1)
  C = logML*Nt + log(T(Nt))
  for t = Nt-1:-1:1
    for j = 1:Ns
      for k = 1:Ns
        ajk = A[j,k]
        if ajk != -T(Inf)
          betaA = ajk + betat1[k] + y[t+1,k]
          betat0[j] = logadd(betat0[j], betaA)  
          betaA += alpha[t,j]
          grady[t+1,k] = logadd(grady[t+1,k], betaA)
          gradA[j,k]   = logadd(gradA[j,k]  , betaA)
        end
      end
    end
    betat0, betat1 = betat1, betat0
    fill!(betat0,-T(Inf))
  end
  @views grady[1,:] .=  betat1 .+ alpha[1,:]
  grady .= -exp.(grady.-C)
  gradA .= -exp.(gradA.-C)
  @views grada .= grady[1,:]
  return grada, gradA, grady
end

# with constrained path
function get_loggrad!(grada,
                      gradA,
                      grady::AbstractArray{T}, 
                      betat0::Vector{T}, betat1::Vector{T}, 
                      Nt::Integer,
                      t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                      A::AbstractMatrix{T},
                      y::AbstractArray{T}, 
                      alpha::Matrix{T},
                      logML::T) where {D<:Integer,
                                       T <: AbstractFloat}
  C = logML*Nt+log(T(Nt))
  Ns = size(A,1)
  for t = Nt-1:-1:1
    I,J = t2IJ[t]
    for k in eachindex(I)
      i,j = I[k],J[k]
      betaA = A[i,j] + betat1[j] + y[t+1,j]
      betat0[i] = logadd(betat0[i], betaA)  
      betaA += alpha[t,i]
      grady[t+1,j] = logadd(grady[t+1,j], betaA)
      #gradA[i,j]   = logadd(gradA[i,j]  , betaA)
    end
    betat0, betat1 = betat1, betat0
    fill!(betat0,-T(Inf))
  end
  idx0 = unique(t2IJ[1][1])
  for i in idx0
    grady[1,i] = betat1[i] + alpha[1,i]
  end
  grady .= -exp.(grady.-C)
  #gradA .= -exp.(gradA.-C)
  return grada, gradA, grady
end

function get_loggrad(Nt::Integer,
                     a, 
                     A::AbstractMatrix{T},
                     y::AbstractArray{T}, 
                     alpha::Matrix{T},
                     logML::T
                    ) where {T <: AbstractFloat}
  Ns = size(A,1)
  grada = typeof(a) <: Array{T} ? -T(Inf) .* ones(T,Ns) : DoesNotExist() 
  gradA = typeof(a) <: Array{T} ? -T(Inf) .* ones(T,Ns,Ns) : DoesNotExist() 
  grady = -T(Inf) .* ones(T,size(y,1),Ns)
  betat1 = zeros(T,Ns)
  betat0 = -T(Inf).*ones(T,Ns)
  get_loggrad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,alpha,logML)
end

# y is a tensor Nt2 × Ns × Nb
function get_loggrad(
                     Nts::Vector{D},
                     as::AbstractMatrix{T}, 
                     As::AbstractArray{T,3},
                     ys::AbstractArray{T,3}, 
                     alphas::Vector{Matrix{T}},
                     logMLs::Vector{T}
                    ) where {T <: AbstractFloat, D <: Integer}
  Ns = size(As,1)
  Nb = size(ys,3)
  grada = -T(Inf) .* ones(T,Ns,Nb)    
  gradA = -T(Inf) .* ones(T,Ns,Ns,Nb) 
  grady = -T(Inf) .* ones(T,size(ys,1),Ns,Nb)
  betat1 = zeros(T,Ns)
  betat0 = -T(Inf).*ones(T,Ns)
  for i = 1:Nb
    get_loggrad!(
                 view(grada,:,i),
                 view(gradA,:,:,i),
                 view(grady,:,:,i),
                 betat0,betat1,
                 Nts[i],
                 view(as,:,i),
                 view(As,:,:,i),
                 view(ys,:,:,i),
                 alphas[i],logMLs[i])
    fill!(betat1, zero(T))
    fill!(betat0, -T(Inf))
  end
  return grada,gradA,grady
end

# y is a tensor Nt2 × Ns × Nb, constrained path
function get_loggrad(
                     Nts::Vector{D},
                     as::Vector{K}, 
                     A::AbstractMatrix{T},
                     y::AbstractArray{T,3}, 
                     alphas::Vector{Matrix{T}},
                     logMLs::Vector{T}
                    ) where {T <: AbstractFloat, Z, K<:AbstractVector{Z}, D <: Integer}
  Ns = size(A,1)
  Nb = size(y,3)
  grada = DoesNotExist()    
  gradA = DoesNotExist()    
  grady = -T(Inf) .* ones(T,size(y,1),Ns,Nb)
  betat1 = zeros(T,Ns)
  betat0 = -T(Inf).*ones(T,Ns)
  for i = 1:Nb
    get_loggrad!(
                 grada,
                 gradA,
                 view(grady,:,:,i),
                 betat0,betat1,
                 Nts[i],as[i],
                 A,
                 view(y,:,:,i),
                 alphas[i],logMLs[i])
    fill!(betat1, zero(T))
    fill!(betat0, -T(Inf))
  end
  return grada,gradA,grady
end
