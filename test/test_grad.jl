# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

# testing underflow not occuring single precision
# full
T = Float32
Nt = 21
Ns = 10
a = rand(T,Ns)   # initial state probability  
a ./= norm(a,1)
A = rand(T,Ns,Ns)
for i in 1:Ns A[i,:] ./= norm(A[i,:],1) end
@assert HMMGradients.is_transition_matrix(A)

y = rand(T,Nt,Ns)
alpha, c = forward(Nt,a,A,y)
beta = backward(Nt,A,c,y)
gamma = posterior(Nt,a,A,y)
cost = nlogML(Nt,a,A,y)

grada = zeros(T,Ns)
gradA = zeros(T,Ns,Ns)
grady = zeros(T,Nt,Ns)
betat1 = ones(T,Ns)
betat0 = zeros(T,Ns)
grada, gradA, grady = HMMGradients.get_grad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,alpha,c) 
grada2, gradA2, grady2 = HMMGradients.get_grad(Nt,a,A,y,alpha,c) 
A, a, y = Float64.(A), Float64.(a), Float64.(y) # needs double precision
grada_fd = gradient_fd(a -> nlogML(Nt,a,A,y),a) 
gradA_fd = gradient_fd(A -> nlogML(Nt,a,A,y),A) 
grady_fd = gradient_fd(y -> nlogML(Nt,a,A,y),y) 
grada_an = -1/(Nt*c[1]) .* (Diagonal(y[1,:]) * beta[1,:])  # analyatic formula
grada_an2 = -1/(Nt) .* gamma[1,:] ./ a  # analyatic formula
gradA_an = -1/(Nt) * sum( alpha[t,:]*(Diagonal(1/c[t+1] .* y[t+1,:])*beta[t+1,:])' for t = 1:Nt-1)  # analyatic formula
gradA_an2 = -1/(Nt) * sum( alpha[t,:]*(A\beta[t,:])' for t = 1:Nt-1)  # analyatic formula
grady_an = -1/Nt .* gamma ./ y # analyatic formula
grady_an2 = similar(grady_an)
grady_an2[1,:] = -1/(c[1]*Nt) .* (Diagonal(a)*beta[1,:])
for t = 2:Nt
  grady_an2[t,:] .= -1/(c[t]*Nt) .* Diagonal(A'*alpha[t-1,:])*beta[t,:] # analyatic formula
end
fy,p = ChainRulesCore.rrule(nlogML, Nt, a, A, y)
_,_,grada3,gradA3,grady3 = p(1.0)

if iseven(Nt)
  @test norm(beta[1,:]-betat0) < 1e-5
else
  @test norm(beta[1,:]-betat1) < 1e-5
end
@test norm(grada  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady  - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada2  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA2  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady2  - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada3  - grada_fd, Inf)/norm(grada_fd) < 1e-4
@test norm(gradA3  - gradA_fd, Inf)/norm(gradA_fd) < 1e-4
@test norm(grady3  - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada_an   - grada_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grada_an2  - grada_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(gradA_an   - gradA_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(gradA_an2  - gradA_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grady_an   - grady_fd, Inf)/norm(grady_fd) < 1e-4
@test norm(grady_an2  - grady_fd, Inf)/norm(grady_fd) < 1e-4

# test with zeropadded
Nz = 20
y2 = [y;zeros(eltype(y),Nz,size(y,2))]
alpha, c = forward(Nt,a,A,y2)
grada,gradA,grady = HMMGradients.get_grad(Nt,a,A,y2,alpha,c) 
gradA_fd = gradient_fd(A->nlogML(Nt,a,A,y),Float64.(A)) 
grady_fd = gradient_fd(y->nlogML(Nt,a,A,y),Float64.(y2)) 

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
A  = rand(T,Ns,Ns)
As = zeros(T,Ns,Ns,Nb)
for i = 1:Nb As[:,:,i] .= A end
as = rand(T,Ns,Nb)
ys = rand(T,maximum(Nts),Ns,Nb)
alphas,cs = forward(Nts,as,As,ys)
alphas_cs = [forward(Nts[i],as[:,i],As[:,:,i],ys[:,:,i]) for i=1:Nb] 
@test all([norm(alphas[i]-alphas_cs[i][1])<1e-5 for i=1:Nb] )
@test all([norm(cs[i]-alphas_cs[i][2])<1e-5 for i=1:Nb] )
@test all(length(cs) .== Nb)
@test all(size.(alphas,2) .== Ns)
@test all(size.(alphas,1) .== Nts)
grada, gradA, grady = HMMGradients.get_grad(Nts,as,As,ys,alphas,cs) 
grada_gradA_grady = [HMMGradients.get_grad(Nts[i],as[:,i],As[:,:,i],ys[:,:,i],alphas[i],cs[i]) for i=1:Nb] 
fys = [nlogML(Nts[i],as[:,i],As[:,:,i],ys[:,:,i]) for i=1:Nb] 
fy2 = nlogML(Nts,as,As,ys)
fy,p = ChainRulesCore.rrule(nlogML, Nts, as, As, ys)
_,_,grada2,gradA2,grady2 = p(1.0)

@test sum(fys) ≈ fy
@test fy2 == fy
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

alpha, c = forward(Nt,t2IJ,A,y)
@test all( sum(alpha,dims=2) .≈ 1.0)

beta = backward(Nt,A,c,t2IJ,y)

gamma = posterior(Nt,t2IJ,A,y)
@test all( sum(gamma,dims=2) .≈ 1.0 )

grada = zero(T)
gradA = zero(T)
grady = zeros(T,Nt,Ns)
betat1 = ones(T,Ns)
betat0 = zeros(T,Ns)
_,_,grady  = HMMGradients.get_grad!(grada,gradA,grady,betat0,betat1,Nt,t2IJ,A,y,alpha,c) 
_,_,grady2 = HMMGradients.get_grad(Nt,t2IJ,A,y,alpha,c) 

fy,p = ChainRulesCore.rrule(nlogML, Nt, t2IJ, A, y)
_,_,grada3,gradA3,grady3 = p(1.0)

A,y = Float64.(A), Float64.(y) # needs double precision
grady_fd = gradient_fd(y->nlogML(Nt,t2IJ,A,y),y) 

@test norm(beta[1,:]-betat0) < 1e-5
@test norm(grady  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test norm(grady2  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test norm(grady3  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test gradA3 == ChainRulesCore.DoesNotExist()
@test grada3 == ChainRulesCore.DoesNotExist()

# test with zeropadded
Nz = 20
y2 = [y;zeros(eltype(y),Nz,size(y,2))]
alpha, c = forward(Nt,t2IJ,A,y2)
_,_,grady = HMMGradients.get_grad(Nt,t2IJ,A,y2,alpha,c) 
grady_fd = gradient_fd(y->nlogML(Nt,t2IJ,A,y),Float64.(y2)) # needs double precision

@test size(grady) == size(y2)
@test norm( grady  - grady_fd, Inf)/norm(grady_fd) < 1e-5
@test sum(grady[end-(Nz-1):end,:]) == 0.0

## test with batch Nt × Ns × Nb
T = Float32
Nb = 3
A = T[0.0 1.0 0.0; 0.0 0.5 0.5; 1.0 0.0 0.0]
t2IJs = [t2IJ for i=1:Nb]
Nts = length.(t2IJs).+1
y = rand(T,maximum(Nts),Ns,Nb)
alphas,cs = forward(Nts,t2IJs,A,y)
alphas_cs = [forward(Nts[i],t2IJs[i],A,y[:,:,i]) for i=1:Nb] 
@test all([norm(alphas[i]-alphas_cs[i][1])<1e-5 for i=1:Nb] )
@test all([norm(cs[i]-alphas_cs[i][2])<1e-5 for i=1:Nb] )
@test all(length(cs) .== Nb)
@test all(size.(alphas,2) .== Ns)
@test all(size.(alphas,1) .== Nts)
grada, gradA, grady = HMMGradients.get_grad(Nts,t2IJs,A,y,alphas,cs) 
grada_gradA_grady = [HMMGradients.get_grad(Nts[i],t2IJs[i],A,y[:,:,i],alphas[i],cs[i]) for i=1:Nb] 
fy,p = ChainRulesCore.rrule(nlogML, Nts, t2IJs, A, y)
fy2 = nlogML(Nts,t2IJs,A,y)
_,_,grada2,gradA2,grady2 = p(1.0)

@test fy == fy2
@test all([norm(grady[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)
@test all([norm(grady2[:,:,i]-grada_gradA_grady[i][3]) for i = 1:Nb] .< 1e-5)
@test gradA2 == ChainRulesCore.DoesNotExist()
@test grada2 == ChainRulesCore.DoesNotExist()
