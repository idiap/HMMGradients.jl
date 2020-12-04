# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

T = Float32
Nt = 20
Ns = 5
a = rand(T,Ns)   # initial state probability  
a ./= norm(a,1)
A = rand(T,Ns,Ns)
for i in 1:Ns A[i,:] ./= norm(A[i,:],1) end
@assert HMMGradients.is_transition_matrix(A)
A = ones(T,Ns,Ns)
y = rand(T,Nt,Ns)

A, a, y = log.(A), log.(a), log.(y)
A, a, y = Float32.(A), Float32.(a), Float32.(y)

logalpha, logML = logforward(Nt,a,A,y)
logbeta = logbackward(Nt,A,y)
loggamma = logposterior(Nt,a,A,y)
gamma = posterior(Nt,exp.(a),exp.(A),exp.(y))
alpha,c = forward(Nt,exp.(a),exp.(A),exp.(y))
beta = backward(Nt,exp.(A),c,exp.(y))

grada = -T(Inf).*ones(T,Ns)
gradA = -T(Inf).*ones(T,Ns,Ns)
grady = -T(Inf).*ones(T,Nt,Ns)
betat1 = zeros(T,Ns)
betat0 = -T(Inf).*ones(T,Ns)
grada, gradA, grady = HMMGradients.get_loggrad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,logalpha,logML) 
grada2, gradA2, grady2 = HMMGradients.get_loggrad(Nt,a,A,y,logalpha,logML) 
A, a, y = Float64.(A), Float64.(a), Float64.(y) # needs double precision
grada_fd = gradient_fd(a -> nlogMLlog(Nt,a,A,y),a) 
gradA_fd = gradient_fd(A -> nlogMLlog(Nt,a,A,y),A) 
grady_fd = gradient_fd(y -> nlogMLlog(Nt,a,A,y),y) 
grada_an = -1/Nt .* ( gamma[1,:] )
gradA_an = -1/(Nt*exp(logML*Nt)).*sum([exp(logalpha[t,i] + A[i,j] + logbeta[t+1,i] + y[t+1,j]) for i=1:Ns, j=1:Ns] for t = 1:Nt-1)
grady_an = -1/Nt .* ( gamma )
fy,p = ChainRulesCore.rrule(nlogMLlog, Nt, a, A, y)
_,_,grada3,gradA3,grady3 = p(1.0)

if iseven(Nt)
  @test norm(logbeta[1,:]-betat0) < 1e-5
else
  @test norm(logbeta[1,:]-betat1) < 1e-5
end
@test norm(grada   - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA   - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady   - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada2  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA2  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady2  - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada3  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA3  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady3  - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada_an  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA_an  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady_an  - grady_fd, Inf)/norm(grady_fd) < 1e-4

# test with zeropadded input
Nz = 20
y2 = [y;zeros(eltype(y),Nz,size(y,2))]
logalpha, logML = logforward(Nt,a,A,y2)
grada,gradA,grady = HMMGradients.get_loggrad(Nt,a,A,y2,logalpha,logML) 
grady_fd = gradient_fd(y->nlogMLlog(Nt,a,A,y),Float64.(y2)) # needs double precision

@test size(grady) == size(y2)
@test norm(grada  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm( grady  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test sum(grady[end-(Nz-1):end,:]) == 0.0

## test with batch Nt × Ns × Nb
T = Float32
Nb = 3
Ns = 10
Nts = [10,15,20]
A = log.(rand(T,Ns,Ns))
As = repeat(A,1,1,Nb)
as = log.(rand(T,Ns,Nb))
ys = log.(rand(T,maximum(Nts),Ns,Nb))
logalphas,logMLs = logforward(Nts,as,As,ys)
logalphas_logMLs = [logforward(Nts[i],as[:,i],As[:,:,i],ys[:,:,i]) for i=1:Nb] 
@test all([norm(logalphas[i]-logalphas_logMLs[i][1])<1e-5 for i=1:Nb] )
@test all([norm(logMLs[i]-logalphas_logMLs[i][2])<1e-5 for i=1:Nb] )
@test all(length(logMLs) .== Nb)
@test all(size.(logalphas,2) .== Ns)
@test all(size.(logalphas,1) .== Nts)

grada, gradA, grady = HMMGradients.get_loggrad(Nts,as,As,ys,logalphas,logMLs) 
grada_gradA_grady = [HMMGradients.get_loggrad(Nts[i],as[:,i],As[:,:,i],ys[:,:,i],logalphas[i],logMLs[i]) for i=1:Nb] 
fy,p = ChainRulesCore.rrule(nlogMLlog, Nts, as, As, ys)
_,_,grada2,gradA2,grady2 = p(1.0)

@test -sum(logMLs) ≈ fy
@test all([norm(grada[:,i]-grada_gradA_grady[i][1]) for i = 1:Nb] .< 1e-5)
@test all([norm(gradA[:,:,i]-grada_gradA_grady[i][2]) for i = 1:Nb] .< 1e-5)
@test all([norm(grady[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)
@test all([norm(grada2[:,i]-grada_gradA_grady[i][1]) for i = 1:Nb] .< 1e-5)
@test all([norm(gradA2[:,:,i]-grada_gradA_grady[i][2]) for i = 1:Nb] .< 1e-5)
@test all([norm(grady2[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)

# constrained path
T = Float32
A = T[0.0 1.0 0.0; 0.0 0.5 0.5; 1.0 0.0 0.0]
t2tr = Dict{Int,Array{Int,1}}[Dict(1 => [2]), Dict(2 => [3, 2]), Dict(2 => [3, 2],3 => [1]), Dict(2 => [3, 2],3 => [1],1 => [2]), Dict(2 => [3, 2],3 => [1],1 => [2]), Dict(2 => [3, 2],3 => [1],1 => [2]), Dict(2 => [2],3 => [1],1 => [2]), Dict(2 => [2],1 => [2]), Dict(2 => [3])]
t2IJ= HMMGradients.t2tr2t2IJ(t2tr)

Nt = length(t2tr)+1
Ns = size(A,1)
y = rand(T,Nt,Ns)
A, y = log.(A), log.(y)

logalpha, logML = logforward(Nt,t2IJ,A,y)

logbeta = logbackward(Nt,A,t2IJ,y)

loggamma = logposterior(Nt,t2IJ,A,y)
@test all(sum( exp.(loggamma), dims=2) .≈ 1.0)

grada = zero(T)
gradA = zero(T)
grady = -T(Inf) .* ones(T,size(y,1),Ns)
betat1 = zeros(T,Ns)
betat0 = -T(Inf).*ones(T,Ns)
_,_,grady  = HMMGradients.get_loggrad!(grada,gradA,grady,betat0,betat1,Nt,t2IJ,A,y,logalpha,logML) 
_,_,grady2 = HMMGradients.get_loggrad(Nt,t2IJ,A,y,logalpha,logML) 

fy,p = ChainRulesCore.rrule(nlogMLlog, Nt, t2IJ, A, y)
_,_,grada3,gradA3,grady3 = p(1.0)

A,y = Float64.(A), Float64.(y) # needs double precision
grady_fd = gradient_fd(y->nlogMLlog(Nt,t2IJ,A,y),y) 

@test norm(exp.(logbeta[1,:])-exp.(betat0)) < 1e-5
@test norm(grady   - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test norm(grady2  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test norm(grady3  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test gradA3 == ChainRulesCore.DoesNotExist()
@test grada3 == ChainRulesCore.DoesNotExist()

# test with zeropadded input
Nz = 20
y2 = [y;zeros(eltype(y),Nz,size(y,2))]
logalpha, logML = logforward(Nt,t2IJ,A,y2)
_,_,grady = HMMGradients.get_loggrad(Nt,t2IJ,A,y2,logalpha,logML) 
grady_fd = gradient_fd(y->nlogMLlog(Nt,t2IJ,A,y),Float64.(y2)) # needs double precision

@test size(grady) == size(y2)
@test norm( grady  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test sum(grady[end-(Nz-1):end,:]) == 0.0

## test with batch Nt × Ns × Nb
T = Float32
Nb = 3
A = log.(T[0.0 1.0 0.0; 0.0 0.5 0.5; 1.0 0.0 0.0])
t2IJs = [t2IJ for i=1:Nb]
Nts = length.(t2IJs).+1
y = log.(rand(T,maximum(Nts),Ns,Nb))
logalphas,logMLs = logforward(Nts,t2IJs,A,y)
logalphas_logMLs = [logforward(Nts[i],t2IJs[i],A,y[:,:,i]) for i=1:Nb] 
@test all([norm(exp.(logalphas[i])-exp.(logalphas_logMLs[i][1]))<1e-5 for i=1:Nb] )
@test all([norm(logMLs[i]-logalphas_logMLs[i][2])<1e-5 for i=1:Nb] )
@test all(length(logMLs) .== Nb)
@test all(size.(logalphas,2) .== Ns)
@test all(size.(logalphas,1) .== Nts)
grada, gradA, grady = HMMGradients.get_loggrad(Nts,t2IJs,A,y,logalphas,logMLs) 
grada_gradA_grady = [HMMGradients.get_loggrad(Nts[i],t2IJs[i],A,y[:,:,i],logalphas[i],logMLs[i]) for i=1:Nb] 
fy,p = ChainRulesCore.rrule(nlogMLlog, Nts, t2IJs, A, y)
_,_,grada2,gradA2,grady2 = p(1.0)

@test all([norm(grady[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)
@test all([norm(grady2[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)
@test grada2 == ChainRulesCore.DoesNotExist()
@test gradA2 == ChainRulesCore.DoesNotExist()
