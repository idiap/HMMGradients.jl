# HMMGradients.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://idiap.github.io/HMMGradients.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://idiap.github.io/HMMGradients.jl/dev/)
![CI](https://github.com/idiap/HMMGradients.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/idiap/HMMGradients.jl/branch/main/graph/badge.svg?token=012MD4OIZY)](https://codecov.io/gh/idiap/HMMGradients.jl)
[![DOI](https://zenodo.org/badge/318543104.svg)](https://zenodo.org/badge/latestdoi/318543104)

This package enables computing the gradient of the parameters of [Hidden Markov Models (HMMs)](https://en.wikipedia.org/wiki/Hidden_Markov_model). 
This makes it possible to perform HMM training using gradient based methods like stochastic gradient descent, which is necessary for example when neural networks are involved, e.g. in modern automatic speech recognition systems. Check out this [TIDIGITS recipe](https://github.com/idiap/TIDIGITSRecipe.jl) for an example.

Formally, this package extends [ChainRulesCore](https://github.com/JuliaDiff/ChainRulesCore.jl)
making it possible to train HMM models using the automatic differentiation frameworks of Julia,
for example using [Zygote](https://github.com/FluxML/Zygote.jl) and machine learning libraries like [Flux](https://github.com/FluxML/Flux.jl). 
The package also provides numerical stable algorithms to compute forward, backward and posterior probabilities of HMMs.

## Installation

To install the package, simply issue the following command in the Julia REPL:

```julia
] add HMMGradients
```

For more information check the [documentation](https://idiap.github.io/HMMGradients.jl/stable/).
