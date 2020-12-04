# Computing gradients

This package uses [ChainRulesCore.jl](https://github.com/JuliaDiff/ChainRulesCore.jl)
API extending its `rrule` function. 
Specifically the following cost functions can be differentiated. 

```@docs
nlogML
nlogMLlog
```
For example if we want to get derivatives for the following model:
```jldoctest
julia> using HMMGradients, ChainRulesCore

julia> a = [1.0,0.0];

julia> A = [0.5 0.5; 0.5 0.5];

julia> y = [0.1 0.7; 0.5 0.8; 0.9 0.3];

julia> Ns,Nt = size(A,1), size(y,1);

julia> cost = nlogML(Nt,a,A,y)
1.0813978776174968
```
all we need to do is to use `rrule`, which will return cost and 
the [pullback function](https://www.juliadiff.org/ChainRulesCore.jl/stable/#frule-and-rrule):
```jldoctest; setup = :((using HMMGradients;using ChainRulesCore; Nt=3;a=[1.0, 0.0];A=[0.5 0.5; 0.5 0.5];y=[0.1 0.7; 0.5 0.8; 0.9 0.3]))
julia> cost, pullback_nlogML = ChainRulesCore.rrule(nlogML, Nt, a, A, y);

julia> _, _, grada, gradA, grady = pullback_nlogML(1.0);

julia> [println(grad) for grad in [grada,gradA,grady]];
[-0.3333333333333333, -2.3333333333333335]
[-0.4487179487179487 -0.4743589743589744; -0.30769230769230776 -0.10256410256410257]
[-3.3333333333333335 -0.0; -0.2564102564102564 -0.2564102564102564; -0.2777777777777778 -0.2777777777777778]

```
When we need to impose only specific paths to be allowed through the HMM
only the gradient with respect to ``y`` is computed.
```jldoctest; setup = :((using HMMGradients;using ChainRulesCore; Nt=3;a=[1.0, 0.0];A=[0.5 0.5; 0.5 0.5];y=[0.1 0.7; 0.5 0.8; 0.9 0.3]))
julia> t2tr = [[1,1]=>[1,2],[1,2]=>[2,2]];

julia> cost, pullback_nlogML = ChainRulesCore.rrule(nlogML, Nt, t2tr, A, y);

julia> _, _, grada, gradA, grady = pullback_nlogML(1.0);

julia> @assert grada == gradA == ChainRulesCore.DoesNotExist();

julia> grady
3×2 Array{Float64,2}:
 -3.33333  -0.0
 -0.25641  -0.25641
 -0.0      -1.11111

```
It is also possible to perform these operations in batches.
Say we have `Nb` sequences of different length then
`size(yb) == (Nt_max,Ns,Nb)` where `Nt_max` is the maximum of length sequence.
For example:
```jldoctest; setup = :((using HMMGradients;using ChainRulesCore;Ns=2;A=[0.5 0.5; 0.5 0.5];))
julia> Nb = 2; # number of sequences

julia> Nts = [3,4]; # sequences length

julia> ys = [[0.1 0.7; 0.5 0.8; 0.9 0.3], [0.5 0.3; 0.7 0.7; 0.1 0.1; 0.5 0.0]];

julia> Nt_max = maximum(size.(ys,1)); # calculate maximum seq length

julia> yb = zeros(Nt_max,Ns,Nb); 

julia> for b in 1:Nb yb[1:Nts[b],:,b] .= ys[b] end; # yb has zeropadded inputs

julia> t2trs = [ [[1,1]=>[1,2],[1,2]=>[2,2]], [[2]=>[2],[2]=>[2],[2]=>[1]] ];

julia> @assert all(length.(t2trs) .== Nts .-1);

julia> cost, pullback_nlogML = ChainRulesCore.rrule(nlogML, Nts, t2trs, A, yb);

julia> _, _, _, _, grady = pullback_nlogML(1.0);

julia> grady
4×2×2 Array{Float64,3}:
[:, :, 1] =
 -3.33333  -0.0
 -0.25641  -0.25641
 -0.0      -1.11111
 -0.0      -0.0

[:, :, 2] =
 -0.0  -0.833333
 -0.0  -0.357143
 -0.0  -2.5
 -0.5  -0.0
```
Batch computation is also available for the unconstrained case.
In this case `a` and `A` must be of the of size `(Ns,Nb)` and `(Ns,Ns,Nb)` respectively.
The same computations can be performed in the log domain using [`nlogMLlog`](@ref) 
after applying `log` to `a`, `A` and `y`.
