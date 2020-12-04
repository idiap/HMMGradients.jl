using HMMGradients
using BenchmarkTools
using LinearAlgebra
using SparseArrays
using Random

const SUITE = BenchmarkGroup()

sizes =  Dict()
sizes["small"] = ( 100, 10) #Nt,Ns,Ny
sizes["big"]   = (1000,100)

k = "forward!"
SUITE[k] = BenchmarkGroup(["forward!"])
for T in [Float32]
  for s in keys(sizes)
        SUITE[k][T, s] = @benchmarkable forward!(alpha,c,Nt,a,A,y) setup=begin
            Random.seed!(0)
            Nt, Ns = sizes[$s]
            A = rand($T, Ns, Ns)
            a = rand($T, Ns)
            y = randn($T, Nt, Ns)
            alpha = rand($T, Nt, Ns)
            c = randn($T, Nt)
        end
    end
end

k = "get_grad!"
SUITE[k] = BenchmarkGroup(["get_grad!"])
for T in [Float32]
  for s in keys(sizes)
        SUITE[k][T, s] = @benchmarkable HMMGradients.get_grad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,alpha,c) setup=begin
            Random.seed!(0)
            Nt, Ns = sizes[$s]
            A,gradA = rand($T, Ns, Ns),zeros($T, Ns, Ns)
            a,grada = rand($T, Ns),zeros($T, Ns)
            y,grady = rand($T, Nt, Ns),zeros($T, Nt, Ns)
            betat1 = ones($T,Ns)
            betat0 = zeros($T,Ns)
            alpha,c = forward(Nt,a,A,y)

        end
    end
end

k = "logforward!"
SUITE[k] = BenchmarkGroup(["forward!"])
for T in [Float32]
  for s in keys(sizes)
        SUITE[k][T, s] = @benchmarkable logforward!(alpha,Nt,a,A,y) setup=begin
            Random.seed!(0)
            Nt, Ns = sizes[$s]
            A = rand($T, Ns, Ns)
            a = rand($T, Ns)
            y = rand($T, Nt, Ns)
            alpha = rand($T, Nt, Ns)
        end
    end
end

k = "get_loggrad!"
SUITE[k] = BenchmarkGroup(["get_loggrad!"])
for T in [Float32]
  for s in keys(sizes)
        SUITE[k][T, s] = @benchmarkable HMMGradients.get_loggrad!(grada,gradA,grady,betat0,betat1,Nt,a,A,y,logalpha,logML) setup=begin
            Random.seed!(0)
            Nt, Ns = sizes[$s]
            A,gradA = rand($T, Ns, Ns),-$T(Inf).*ones($T, Ns, Ns)
            a,grada = rand($T, Ns),-$T(Inf).*ones($T, Ns)
            y,grady = rand($T, Nt, Ns),-$T(Inf).*ones($T, Nt, Ns)
            betat1 = zeros($T,Ns)
            betat0 = -$T(Inf).*ones($T,Ns)
            logalpha,logML = logforward(Nt,a,A,y)
        end
    end
end
#run(SUITE)
