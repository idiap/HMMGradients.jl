# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

# "classic" forward - just for testing
function Baum_forward(a,A,y)
  Nt, Ns = size(y,1), size(A,1)
  alpha = zeros(Nt,Ns)
  for j = 1:Ns
    alpha[1,j] = a[j]*y[1,j]
  end

  for t = 2:Nt
    for j = 1:Ns
      for i = 1:Ns
        alpha[t,j] = alpha[t,j] + alpha[t-1,i]*A[i,j]*y[t,j]    
      end
    end
  end
  return alpha
end

function Baum_forward_matrix(a,A,y)
  Nt, Ns = size(y,1), size(A,1)
  alpha = zeros(Nt,Ns)
  alpha[1,:] .= Diagonal(y[1,:])*a
  for t = 2:Nt
    alpha[t,:] .= Diagonal(y[t,:])*A'*alpha[t-1,:]    
  end
  return alpha
end

# "classic" backwards - just for testing
function Baum_backward(A,y)
  Nt, Ns = size(y,1), size(A,1)
  beta = zeros(Nt,Ns)
  beta[end,:] .= 1
  for t = Nt-1:-1:1
    for j = 1:Ns
      for k = 1:Ns
        beta[t,j] += beta[t+1,k]*A[j,k]*y[t+1,k]
      end
    end
  end
  return beta
end

function Baum_backward_matrix(A,y)
  Nt, Ns = size(y,1), size(A,1)
  beta = zeros(Nt,Ns)
  beta[Nt,:] .= ones(Ns)
  for t = Nt-1:-1:1
    beta[t,:] .= A*Diagonal(y[t+1,:])*beta[t+1,:]    
  end
  return beta
end

# finite difference gradient
function gradient_fd(f, y) 
  fy = f(y)
	δ = sqrt(eps())
  grad = similar(y)
  for i in eachindex(y)
		yδ = copy(y)
		yδ[i] = yδ[i]+δ
    grad[i] = ( f(yδ) - fy )./δ
	end
	return grad
end
