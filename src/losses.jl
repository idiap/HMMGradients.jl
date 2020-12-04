# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export nlogML

"""
`nlogML(Nt,a,A,y)`

Computes the negative log Maximum Likelihood (ML) normalized by the sequence length (``\\log(P(\\mathbf{X}))/N_t``) of a Hidden Markov Model (HMM) with `Ns` states, transition matrix `A` (a `Ns` × `Ns` matrix), initial probabilities `a` (`Ns`-vector) and observation probabilities `y` (a `Ns`×`Nt` matrix) for `Nt` observations. All `Array`s must have the same element type.

`nlogML(Nt,t2tr,A,y)`

Returns the negative log Maximum Likelihood (ML) normalized by the sequence length (``\\log(P(\\mathbf{X}))/N_t``) with a constrained path (see [`forward`](@ref)).
"""
nlogML(Nt::Integer,c) = -sum(log.(c))/Nt
nlogML(Nt::Array,cs) = sum( [-sum(log.(c))/Nt[i] for (i,c) in enumerate(cs)])

function nlogML(Nt::Integer,a,A,y)
  alpha, c = forward(Nt,a,A,y)
  return nlogML(Nt,c)
end

function nlogML(Nt::Array,a,A,y)
  alphas, cs = forward(Nt,a,A,y)
  return nlogML(Nt,cs) 
end

export nlogMLlog
"""
`nlogMLlog(Nt,A,a,y)`

Same as [`nlogML`](@ref) but expects `A`, `a` and `y` to be in the log-domain.
"""
function nlogMLlog(Nt,a,A,y)
  alpha, logML = logforward(Nt,a,A,y)
  return -sum(logML) 
end
