# Demo & applications

A simple demo is provided [here](https://github.com/idiap/HMMGradients.jl/blob/main/demo/demo.jl). A set of one-dimensional signals generated from a HMM with Gaussian distributions is used to learn back the HMM parameters using a maximum likelihood training. 

In order to run the demo go to the folder `demo`, open a julia REPL and type the following:
```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()

julia> include("demo.jl")
```

Other applications:
* Automatic speech recognition: [TIDIGITS recipe](https://github.com/idiap/TIDIGITSRecipe.jl)
