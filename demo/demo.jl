using HMMGradients, LinearAlgebra, Random, Statistics, Distributions, Zygote
use_log = true # do computation in log domain
plot_stuff = true

# define Gaussian probability distribution
logdist(μ::T,σ::T,x::T) where {T} = -( ((x-μ)/σ)^2/2 + log(σ*sqrt(T(2*π))))
dist(μ::T,σ::T,x::T) where {T} = exp(-((x-μ)/σ)^2 / 2 )/(σ*sqrt(T(2*π)))

# model parameters
mutable struct HMMGaussian{T<:AbstractFloat}
  a::Vector{T}
  A::Matrix{T}
  μs::Vector{T}
  σ::T
  use_log::Bool
  likelihood::Function
end

HMMGaussian(a,A,μs,σ;use_log=false) = use_log ? 
  HMMGaussian(log.(a),log.(A),μs,σ,use_log,nlogMLlog) : HMMGaussian(a,A,μs,σ,use_log,nlogML)

# calculate observation probability
function predict(model::HMMGaussian{T},x::AbstractArray{T}) where {T}
  #hcat([dist.(model.μs,model.σs,x[t]) for t in 1:length(x)]...)
  X = x*ones(T,length(model.μs))'
  M = ones(T,length(x))*model.μs'
  if model.use_log
    return logdist.(M,model.σ,X)
  else
    return dist.(M,model.σ,X)
  end
end

# obtain a random signal from HMM model
function Base.rand(model::HMMGaussian{T},Nt::Int) where {T}
  if model.use_log # go back to probability
    model.a .= exp.(model.a)
    model.A .= exp.(model.A)
  end
  x = zeros(T,Nt)
  states = zeros(Int,Nt)
  s = rand(Categorical(a))
  for i = 1:Nt
    states[i] = s
    x[i] = model.μs[s] + model.σ * randn(T)
    s = rand(Categorical(A[s,:])) # new state
  end
  if model.use_log # go back to log
    model.a .= log.(model.a)
    model.A .= log.(model.A)
  end
  return x,states
end
  
# define maximum likelihood function
loss(model::HMMGaussian,x) = model.likelihood(length(x),model.a,model.A,predict(model,x))

T=Float32
Random.seed!(123)
μs = T[0.0;0.1;0.7] 
σ = T(1e-1)         # standard deviation
Ns = 3              # number of states
a = T[1.,0.,0.] # initial probability distribution
A = T[0.5 0.9 0.0; 0.0 0.2 0.8; 0.5 0.0 0.5] # transition matrix
#a = rand(T,Ns)
#A = rand(T,Ns,Ns)
for s in 1:Ns A[s,:] ./= norm(A[s,:],1) end # normalize
a ./= norm(a,1) # normalize
@assert HMMGradients.is_transition_matrix(A)
@assert sum(a)-1.0 < 1e-5

model_gt = HMMGaussian(a,A,μs,σ;use_log=use_log)       # ground truth model
model = HMMGaussian(a,A,zeros(T,Ns),one(T);use_log=use_log) # initial model

Nt = 10 # number of samples
x, s = rand(model_gt,Nt)
fy_gt = loss(model_gt,x)
fy = loss(model,x)
println("cost ground truth model: $fy_gt")
println("cost initial model     : $fy   ")

# gradient descent
function update!(model::HMMGaussian{T},grads,γ=T(1e-3)) where {T}
  grads = grads[1][]
  model.μs .-= γ .* grads.μs
  model.σ -= γ * grads.σ
end

# generrate data
N = 200 # number od samples
X = [rand(model_gt,Nt)[1] for n=1:N]

epochs=10
for e in 1:epochs
  cost = 0
  for i = 1:N
    grads = gradient(model) do m
      f = loss(m,X[i])
      cost += f
      return f
    end
    update!(model,grads)
  end
  println("epoch: $e cost: $(cost/N)")
end

println("ground truth μs: $(model_gt.μs) ")
println("learned μs     : $(model.μs) ")
println("error μs       : $(norm(model.μs-model_gt.μs)) ")
println("ground truth σ : $(model_gt.σ) ")
println("learned σ      : $(model.σ) ")
println("error σ        : $(norm(model.σ-model_gt.σ)) ")

if plot_stuff
  # generate random samples from ground truth HMM 
  # plots a random signal x, the state sequence and observation probability y
  Nt = 50
  x_gt,s_gt = rand(model_gt,Nt)
  y_gt = predict(model_gt,x_gt)
  x,s = rand(model,Nt)
  y = predict(model,x)
  using Plots
  p1 = plot(x_gt, title="ground truth HMM", label="x")
  q1 = plot(s_gt, marker=(:star), title="states", label="s")
  r1 = heatmap(y_gt', title="y")
  p2 = plot(x, title="learned HMM", label="x")
  q2 = plot(s, marker=(:star), label="s")
  r2 = heatmap(y')
  plot(p1,q1,r1,p2,q2,r2)
end
