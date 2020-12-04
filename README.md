# HMMGradients.jl

![CI](https://github.com/idiap/HMMGradients.jl/workflows/CI/badge.svg)

This package enables computing the gradient of the parameters of [Hidden Markov Models (HMMs)](https://en.wikipedia.org/wiki/Hidden_Markov_model). 
This makes it possible to perform HMM training using gradient based methods like stochastic gradient descent, which is necessary for example when neural networks are involved, e.g. in modern automatic speech recognition systems.

Formally, this package extends [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
making it possible to train HMM models using the automatic differentiation frameworks of Julia,
for example using [Zygote](https://github.com/FluxML/Zygote.jl) and machine learning libraries like [Flux](https://github.com/FluxML/Flux.jl). 
The package also provides numerical stable algorithms to compute forward, backward and posterior probabilities of HMMs.
