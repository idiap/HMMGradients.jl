# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

is_transition_matrix(A::AbstractMatrix) = 
(size(A,1) == size(A,2)) && all(abs.(sum(A,dims=2) .- 1) .< 1e-6)

function t2tr2t2IJ(t2tr)
  Z = [[k .* ones(Int,length(t2tr[t][k])) =>sort(t2tr[t][k]) for k in sort([keys(t2tr[t])...])] for t in eachindex(t2tr)]
  t2IJ=[vcat(getindex.(Z[t],1)...) => vcat(getindex.(Z[t],2)...) for t in eachindex(Z)]
  return t2IJ
end

function logadd(x::T, y::T) where {T<:AbstractFloat} 
  if isinf(x) return y end
  if isinf(y) return x end
  if x < y
    diff = x-y
    x = y 
  else
    diff = y-x
  end
  return x + log1p(exp(diff))
end
