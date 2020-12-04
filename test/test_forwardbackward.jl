# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

# define Markov model's parameters (λ)
Ns = 3          # number of states
a = [0.1; 0.3; 0.6]  # initial state probability  
A = [0.1 0.9 0.0; 0.0 0.2 0.8; 0.5 0.0 0.5] # transition matrix
B = [0.8 0.1 0.1; 0.5 0.2 0.3; 0.7 0.1 0.2] # observation prob matrix
# Must be row stotastic
@test HMMGradients.is_transition_matrix(A)

Nt = 5                     # time window length
x = rand(1:Ns,5)           # observations

using Combinatorics
s_all = collect(multiset_permutations(1:Ns,Nt.*ones(Int,Ns),Nt))
@test length(unique(s_all)) == Ns^Nt
# this has all possible state sequences e.g. [1;1;1;1;1], [1;1;1;1;2] ...
# these are Ns^Nt permutations

# Probability of sequence of states
Ps = ones(Ns^Nt) 
for z = 1:length(s_all)
  for t = 1:Nt
    if t == 1
      Ps[z] *= a[ s_all[z][t] ] # initial prob
    else
      Ps[z] *= A[ s_all[z][t-1], s_all[z][t] ] # trans prob
    end
  end
end
@test sum(Ps) ≈ 1 # check it's a probability 

# Likelihood of x for a given sequence
Px_s = ones(length(s_all)) 
for z = 1:length(s_all)
  for t = 1:Nt
    Px_s[z] *= B[ x[t], s_all[z][t] ]
  end
end

P = sum(Ps .* Px_s)
y = [B[x[t],i] for t = 1:Nt, i = 1:Ns]

# testing Baum forward backward
alpha_b = Baum_forward(a,A,y)
alpha_m = Baum_forward_matrix(a,A,y)
beta_b  = Baum_backward(A,y)
beta_m = Baum_backward_matrix(A,y)
P2 = sum(alpha_m[end,:])
gamma_b = alpha_b .* beta_b ./ P2
@test norm(alpha_m-alpha_b) < 1e-7
@test norm(beta_m-beta_b) < 1e-7
@test P ≈ P2 
@test all(sum(alpha_b.*beta_b,dims=2) .≈ P)

## testing forward backward with normalization
alpha, c = forward(Nt,a,A,y)
logalpha, logML = logforward(Nt,log.(a),log.(A),log.(y))
logbeta = logbackward(Nt,log.(A),log.(y))
beta = backward(Nt,A,c,y)
gamma = alpha.*beta
gamma2 = posterior(Nt,a,A,y)
loggamma = logalpha .+ logbeta .- logML*Nt
loggamma2 = logposterior(Nt,log.(a),log.(A),log.(y))
@test P ≈ sum(exp.(logalpha[end,:])) 
@test nlogML(Nt,a,A,y) ≈ -1/Nt * log(P)
@test nlogML(Nt,a,A,y) ≈ nlogMLlog(Nt,log.(a),log.(A),log.(y))
@test all(sum(exp.(logalpha .+ logbeta),dims=2) .≈ P)
@test sum(log.(c)) ≈ log(sum(alpha_b[end,:]) )
@test all( sum(gamma,dims=2) .≈ 1.0 )
@test norm( gamma - gamma_b ) < 1e-7
@test norm( gamma2 - gamma_b ) < 1e-7
@test norm( gamma - exp.(loggamma) ) < 1e-7
@test norm( gamma - exp.(loggamma2) ) < 1e-7
# equivalence between baum alpha/beta and normalized ones
for t = 1:Nt
  @test norm( prod(1 ./ c[1:t]).*alpha_b[t,:] - alpha[t,:] ) < 1e-8 
  @test norm( exp.(logalpha[t,:]) - (prod(c[1:t])*alpha[t,:] ) ) < 1e-8 
end
for t = 1:Nt
  @test norm( prod(1 ./c[t+1:end]).*beta_b[t,:] - ( beta[t,:] ) )  < 1e-8
  @test norm( exp.(logbeta[t,:]) - (prod(c[t+1:end])*beta[t,:] ) )  < 1e-8
end

# testing with long sequence
T = Float64
Nt = 4000     
Ns = 100
A = rand(T,Ns,Ns)
a = rand(T,Ns)
a ./= norm(a,1)
for i in 1:Ns A[i,:] ./= norm(A[i,:],1) end
@assert HMMGradients.is_transition_matrix(A)
y = rand(T,Nt,Ns)

alpha, c = forward(Nt,a,A,y)
logalpha, logML = logforward(Nt,log.(a),log.(A),log.(y))
logbeta = logbackward(Nt,log.(A),log.(y))
beta = backward(Nt,A,c,y)
gamma = alpha.*beta
loggamma = logalpha .+ logbeta .- logML*Nt
@test norm( gamma - exp.(loggamma) ) < 1e-5
@test nlogML(Nt,a,A,y) ≈ nlogMLlog(Nt,log.(a),log.(A),log.(y))
