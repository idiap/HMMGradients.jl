# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

# joint computation of gradient and backwards
function get_grad!(grada::AbstractVector{T},
                   gradA::AbstractMatrix{T},
                   grady::AbstractArray{T},
                   betat0::Vector{T}, betat1::Vector{T}, 
                   Nt::Integer,
                   a::AbstractVector{T}, 
                   A::AbstractMatrix{T},
                   y::AbstractArray{T}, 
                   alpha::Matrix{T},
                   c::Vector{T}) where {T <: AbstractFloat}
  Ns = size(A,1)
  for t = Nt-1:-1:1
    betat1 ./= c[t+1]
    for j = 1:Ns
      for k = 1:Ns
        ajk = A[j,k]
        if ajk != zero(T)
          betaA         = ajk * betat1[k]
          betat0[j]    += betaA * y[t+1,k]  
          grady[t+1,k] += betaA * alpha[t,j] 
          gradA[j,k]   +=  alpha[t,j] * betat1[k] * y[t+1,k]
        end
      end
    end
    betat0, betat1 = betat1, betat0
    fill!(betat0,zero(T))
  end
  for k in 1:Ns
    betat1n    = betat1[k]/c[1]
    grada[k]   = betat1n * y[1,k]  
    grady[1,k] = betat1n * a[k] 
  end
  grada ./= -Nt
  gradA ./= -Nt
  grady ./= -Nt
  return grada, gradA, grady
end

# with constrained path
function get_grad!(
                   grada,
                   gradA,
                   grady::AbstractArray{T}, 
                   betat0::Vector{T}, betat1::Vector{T}, 
                   Nt::Integer,
                   t2IJ::Vector{Pair{Vector{D},Vector{D}}},
                   A::AbstractMatrix{T},
                   y::AbstractArray{T}, 
                   alpha::Matrix{T},
                   c::Vector{T}) where {D<:Integer,
                                        T <: AbstractFloat}
  Ns = size(A,1)
  for t = Nt-1:-1:1
    betat1 ./= c[t+1]
    I,J = t2IJ[t]
    for k in eachindex(I)
      i,j = I[k],J[k]
      betaA         = A[i,j] * betat1[j]
      betat0[i]    += betaA * y[t+1,j]  
      grady[t+1,j] += betaA * alpha[t,i] 
      # gradA[i,j]   +=  alpha[t,i] * betat1[j] * y[t+1,j]
    end
    betat0, betat1 = betat1, betat0
    fill!(betat0,zero(T))
  end
  idx0 = unique(t2IJ[1][1])
  for i in idx0
    grady[1,i] += 1/c[1] * betat1[i]
  end
  # gradA ./= -Nt
  grady ./= -Nt
  return grada, gradA, grady
end

function get_grad(Nt::Integer,
                  a,
                  A::AbstractArray{T},
                  y::AbstractArray{T}, 
                  alpha::Matrix{T},
                  c::Vector{T}) where {T <: AbstractFloat}
  Ns = length(a)
  grada = typeof(a) <: AbstractVector{T} ? zeros(T,Ns) : DoesNotExist() 
  gradA = typeof(a) <: AbstractVector{T} ? zeros(T,Ns,Ns) : DoesNotExist() 
  grady = zeros(T,size(y))
  betat1 = ones(T,Ns)
  betat0 = zeros(T,Ns)
  get_grad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,alpha,c)
end

# y is a tensor Nt2 × Ns × Nb
function get_grad(Nts::Vector{D},
                  as::AbstractMatrix{T}, 
                  As::AbstractArray{T,3},
                  ys::AbstractArray{T,3}, 
                  alphas::Vector{Matrix{T}},
                  cs::Vector{Vector{T}}
                 ) where {T <: AbstractFloat, D <: Integer}
  Ns = size(As,1)
  Nb = size(ys,3)
  grada = zeros(T,Ns,Nb)
  gradA = zeros(T,Ns,Ns,Nb)
  grady = zeros(T,size(ys,1),Ns,Nb)
  betat1 = ones(T,Ns)
  betat0 = zeros(T,Ns)
  for i = 1:Nb
    get_grad!(view(grada,:,i),
              view(gradA,:,:,i),
              view(grady,:,:,i),
              betat0,betat1,
              Nts[i],
              view(as,:,i),
              view(As,:,:,i),
              view(ys,:,:,i),
              alphas[i],cs[i])
    fill!(betat1, one(T))
    fill!(betat0, zero(T))
  end
  return grada,gradA,grady
end

# y is a tensor Nt2 × Ns × Nb constrained path
function get_grad(Nts::Vector{D},
                  as::Vector{K}, 
                  A::AbstractMatrix{T},
                  y::AbstractArray{T,3}, 
                  alphas::Vector{Matrix{T}},
                  cs::Vector{Vector{T}}
                 ) where {T <: AbstractFloat, Z,
                          K<:AbstractVector{Z}, D <: Integer}
  Ns = size(A,1)
  Nb = size(y,3)
  grada = DoesNotExist() 
  gradA = DoesNotExist() 
  grady = zeros(T,size(y,1),Ns,Nb)
  betat1 = ones(T,Ns)
  betat0 = zeros(T,Ns)
  for i = 1:Nb
    get_grad!(grada,
              gradA,
              view(grady,:,:,i),
              betat0,betat1,
              Nts[i],as[i],
              A,
              view(y,:,:,i),
              alphas[i],cs[i])
    fill!(betat1, one(T))
    fill!(betat0, zero(T))
  end
  return grada, gradA, grady
end
