# [Theory and Notation](@id theory)

## Hidden Markov models

A hidden Markov model (HMM) 
with ``N_s`` states is defined by the following parameters:

* **Initial state probability vector**:
  ``\mathbf{a} = [a_{1},\dots,a_{N_s}]^\intercal``
  where the ``j``-th element gives 
  the initial probability of being at state 
  ``s_t = j`` at time ``t=0``. 
  Summing the elements of the vector gives 1.

* **Transition matrix**:
  ``\mathbf{A} \in \mathbb{R}^{N_s \times N_s}``
  where the element ``a_{j,i}=p(s_t=j|s_{t-1}=i)`` 
  indicates the probability of going from state ``s_{t-1} = i`` to 
  state ``s_t = j``.
  Requires to satisfy
  ``\sum_{j=1}^{N_s} a_{j,i} = 1``
  for all 
  ``j= 1 \dots N_s``
  i.e. the elements of the rows of ``\mathbf{A}`` must sum to 1.

* **Observation probabilities**:
  each state ``j=1,\dots,N_s`` is associated 
  for example with a
  probability density function
  ``b_i(\mathbf{x}_t)``, 
  where ``\mathbf{x}_t \in \mathbb{R}^{N_x}`` is 
  a vector of observations at time ``t``.
  For a sequence of ``N_t`` observations, 
  ``\mathbf{X} = [\mathbf{x}_1 \dots \mathbf{x}_{N_t}]^\intercal``,
  the operator 
  ``B_{\mathcal{W}}: \mathbb{R}^{N_t \times N_x} \rightarrow \mathbb{R}^{N_t \times N_s}``
  where ``\mathcal{W}`` is a set of parameters,
  maps the observations to
  ``\mathbf{Y} = [\mathbf{y}_1 \dots \mathbf{y}_{N_t}]^\intercal``,
  where the element ``y_{t,j}=p(\mathbf{x}_t|s_t=j)`` 
  is the likelihood of ``\mathbf{x}_t`` 
  for a given state ``j``.

## Maximum likelihood (ML)

Given the observation ``\mathbf{X}``,
we would like to obtain a HMM 
capable of describing this
as sequences of states
that are directly measurable.
Typically a set of ``N_d`` observations 
``\mathcal{D} = \{ \mathbf{X}_1, \dots, \mathbf{X}_{N_d} \}``
is used but we consider only a single one 
to simplify the notation.
The optimal HMM parameters
``\mathcal{L}^\star = 
\{ \mathbf{a}^\star, \mathbf{A}^\star, \mathcal{W}^\star \}``
can be obtained by maximizing the likelihood of the observations.
This can be achieved by minimizing the 
negative logarithm of this likelihood:

```math
\text{minimize}_{\mathcal{L}} 
\{ 
-\log( p (\mathbf{X}) ) = 
-\log( \sum_{ \mathbf{s} \in \mathcal{N} } 
p ( \mathbf{X} | \mathbf{s}  ) p ( \mathbf{s} ) 
)
\},
```
where ``\mathbf{s} \in \mathbb{N}^{N_t}`` is sequence of states, 
and ``\mathcal{N}`` is the set of 
all possible state sequences.
Since ``\mathcal{N}`` has ``N_s^{N_t}`` elements,
in general it is not feasible to compute the cost function
by explicitly computing this summation.
Instead, it is convenient to "split" ``p(\mathbf{X})``
at a particular time ``t``:
```math
p(\mathbf{X}) = 
\sum_{i \in \mathcal{N}_{t,j}}
p(\mathbf{X},s_t=j),
```
where ``\mathcal{N}_{t,j}`` are the 
allowed transitions from state ``j`` at time ``t``,
```math
p(\mathbf{X},s_t=j) = 
p(\mathbf{x}_1,\dots,\mathbf{x}_t,s_t=j)
p(\mathbf{x}_{t+1},\dots,\mathbf{x}_{N_t} | \mathbf{x}_1,\dots,\mathbf{x}_t,s_t=j),
```
since we are assuming _observation independence_, 
we can drop the conditions of the second term:
```math
p(\mathbf{x}_1,\dots,\mathbf{x}_t,s_t=j)
p(\mathbf{x}_{t+1},\dots,\mathbf{x}_{N_t},s_t=j | s_t=j) =
\alpha_{t,j} \beta_{t,j},
```
where ``\alpha_{t,j}`` and ``\beta_{t,j}`` are the 
forward and backward probabilities.
The computation of these probabilities can be performed in 
a recursive way going either forward and backward in time.

The forward probabilities can be written as:
```math
\alpha_{1,j} = a_j y_{1,j}
\ \ \text{for} \ \ j =1,\dots,N_s
```
```math
\alpha_{t,j} = \sum_{i \in \mathcal{N}_{t,j}} \alpha_{t-1,i} 
a_{i,j} y_{t,j} 
\ \ \text{for} \ \ t = 2,\dots,N_t \ \ j=1,\dots,N_s,
```
and using matrix notation:
```math
\boldsymbol\alpha_1 = \text{diag}(\mathbf{y}_1) \mathbf{a}
```
```math
\boldsymbol\alpha_{t} =\text{diag}(\mathbf{y}_{t}) \mathbf{A}_t^\intercal  \boldsymbol\alpha_{t-1}
\ \ \text{for} \ \ t = 2,\dots,N_t \ \ j=1,\dots,N_s,
```
where ``\boldsymbol\alpha_t`` is a ``N_s``-long vector containing the 
forward probabilities at time ``t`` 
and ``\mathbf{A}_t = \mathbf{N}_t \mathbf{A}``,
where ``\mathbf{N}_t`` is a selection matrix which includes 
only the allowed indices at time ``t``.

Backward probabilities:
```math
\beta_{N_t,j} = 1
\ \ \text{for} \ \  j =1,\dots,N_s
```
```math
\beta_{t,j} = \sum_{k \in \bar{\mathcal{N}}_{t+1,j}} \beta_{t+1,k} 
a_{j,k} y_{t+1,k}
\ \ \text{for} \ \ t = N_t-1,\dots,1
```
where ``\bar{\mathcal{N}}_{t,j}`` are the 
allowed transitions arriving from state ``j`` at time ``t``.
Using matrix notation:
```math
\boldsymbol\beta_{N_t} = \mathbf{1}
```
```math
\boldsymbol\beta_{t} = \mathbf{A}_{t+1} \text{diag}(\mathbf{y}_{t+1})  \boldsymbol\beta_{t+1}
\ \ \text{for} \ \ t = N_t-1,\dots,1
```
where ``\boldsymbol\beta_t`` is a ``N_s``-long vector containing the 
backward probabilities at time ``t``. 
Finally the optimization problem can be written as:
```math
\text{minimize}_{\mathcal{L}} 
\{ 
-\log( p (\mathbf{X}) ) = -\log(\boldsymbol\alpha_{t}^\intercal \boldsymbol\beta_{t} )
\}
\ \text{s.t.} \ \mathbf{Y} = B_{\mathcal{W}}(\mathbf{X})
```
The classic way of solving this problem is Baum-Weltch,
however this approach is has the limitation that
``B_{\mathcal{W}}`` must satisfy certain properties [[1]](@ref references).
Another approach which enables the use of 
deep neural networks
is the gradient descent and all the other 
gradient based methods.
These methods requires the derivatives 
with respect to ``\mathbf{a}``, ``\mathbf{A}`` and ``\mathbf{Y}``
with the latter which is used to backpropagate through ``B_{\mathcal{W}}``.
As we will see the derivatives can be expressed 
in terms of posterior probabilities, 
i.e. the probability of being at state ``j`` given an observation at time ``t``:
```math
\gamma_{t,j} = p(s_t=j|\mathbf{x}_t) = \frac{\alpha_{t,j} \beta_{t,j}}{p(\mathbf{X})}.
```

Unfortunately forward probabilities suffer of underflow as ``N_t`` 
increases. To prevent this issue two techniques can be used: 
either ``\boldsymbol\alpha_{t}`` can be scaled at each time step ``t`` 
or the computation can be performed in the log domain. 

## Scaled probabilities

In order to keep ``\boldsymbol\alpha_{t}`` in a good numerical range,
forward probabilities can be normalized at every time step [[2-3]](@ref references):
```math
\bar{\alpha}_{t,j} = \prod_{\tau = 1}^t \frac{1}{c_\tau} \alpha_{t,j}, 
```
where ``c_t`` is a normalization coefficient 
that ensures that ``\sum_{j=1}^{N_s} \bar{\alpha}_{t,j} = 1``.
This normalization coefficients are then used in the backward computation as well
in order to satisfy:
```math
\bar{\beta}_{t,j} = \prod_{\tau = t+1}^{N_t} \frac{1}{c_\tau} \beta_{t,j}.
```
It turns out the normalization coefficients can be used to 
compute ``p(\mathbf{X})`` since at ``\boldsymbol\beta_{N_t} = \mathbf{1}``
then 
```math
p(\mathbf{X}) = \sum_{j=1}^{N_s} \alpha_{N_t,j} =  \prod_{\tau = 1}^t c_\tau \sum_{j=1}^{N_s} \bar{\alpha}_{t,j}
```
```math
p(\mathbf{X}) = \prod_{\tau = 1}^{N_t} c_\tau,
```
and hence the ML cost function can be written as:
```math
\log(p(\mathbf{X})) = \sum_{\tau = 1}^{N_t} \log c_\tau.
```
Moreover this implies that the posterior probabilities 
can be written as:
```math
\gamma_{t,j} = 
\frac{\alpha_{t,j} \beta_{t,j}}{p(\mathbf{X})} =
\bar{\alpha}_{t,j} \bar{\beta}_{t,j}
```
The following gradients can be derived:
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{a} } = 
 \frac{1}{c_1} \odot \text{diag}( \mathbf{y}_1 ) \bar{\boldsymbol\beta}_1,
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{A} } = 
\sum_{t=1}^{N_t-1}   
\bar{\boldsymbol\alpha}_t 
( 
\frac{1}{c_{t+1}} \odot
\text{diag}( \mathbf{y}_{t+1} ) 
\bar{\boldsymbol\beta}_{t+1} 
)^\intercal,
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{y}_1 } = 
\frac{1}{c_1} \odot \text{diag}(\mathbf{a}) \bar{\boldsymbol\beta}_1
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{y}_t } = 
\frac{1}{c_t} \odot 
\text{diag}(\mathbf{A}^\intercal\boldsymbol\alpha_{t-1}) \bar{\boldsymbol\beta}_t
\ \ \text{for} \ \ t = 2,\dots,N_t
```
where ``\odot`` is the element wise product (Hadamard product).
These can also be expressed more compactly as:
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{a} } = 
\text{diag}(\mathbf{a})^{-1} \boldsymbol\gamma_1 
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{A} } = 
\sum_{t=1}^{N_t-1}   
\bar{\boldsymbol\alpha}_t 
( 
\mathbf{A}_t^{-1}
\bar{\boldsymbol\beta}_{t} 
)^\intercal,
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \mathbf{y}_t } = 
\text{diag}(\mathbf{y}_t)^{-1} \boldsymbol\gamma_t 
\ \ \text{for} \ \ t = 1,\dots,N_t
```
Notice that in terms of implementation the 
former formulas are preferred as 
the latter formulas are however valid only for the case where 
all the elements of ``\mathbf{a}`` and ``\mathbf{Y}``
are not equal to zero and ``\mathbf{A}_t`` is invertible.

## Log probabilities

Another approach to prevent numerical overflow
is to compute the forward and backward probabilities in the log-domain.
```math
\hat{\alpha}_{1,j} = \hat{a}_j + \hat{y}_{1,j}
\ \ \text{for} \ \ j =1,\dots,N_s
```
```math
\hat{\alpha}_{t,j} = \bigoplus_{i \in \mathcal{N}_{t,j}} 
\hat{\alpha}_{t-1,i} +\hat{a}_{i,j} + \hat{y}_{t,j}
\ \ \text{for} \ \ t = 2,\dots,N_t \ \ j=1,\dots,N_s,
```
where the hat symbol indicates the variable is in the log-domain,
e.g. ``\hat{\alpha}_{i,j} = \log(\alpha_{t,j})``,
and ``\oplus(x,y) = \log(e^x+e^y)``.
Similarly the backward log-probabilities read:
```math
\hat{\beta}_{N_t,j} = 0
\ \ \text{for} \ \  j =1,\dots,N_s
```
```math
\hat{\beta}_{t,j} = \bigoplus_{k \in \bar{\mathcal{N}}_{t+1,j}} 
\hat{\beta}_{t+1,k} +
\hat{a}_{j,k} + \hat{y}_{t+1,k}
\ \ \text{for} \ \ t = N_t-1,\dots,1
```
The following gradients can be derived:
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \hat{\mathbf{a}} } = 
\boldsymbol\gamma_1
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \hat{\mathbf{A}} } = 
\frac{1}{p(\mathbf{X})}\sum_{t=1}^{N_t-1}
e^{
\hat{\alpha}_{t,i} + \hat{a}_{i,j} + \hat{\beta}_{t+1,i} + \hat{y}_{t+1,j} 
}
\ \ \text{for} \ \ i, j = 1,\dots,N_s 
```
```math
\frac{\partial \log (p(\mathbf{X}))}{\partial \hat{\mathbf{y}}_t } = 
\boldsymbol\gamma_t 
\ \ \text{for} \ \ t = 1,\dots,N_t
```

## [References](@id references)

- [1] L. A. Liporace, Maximum Likelihood Estimation for Multivariate Observations of Markov Sources, IEEE Trans. Inf. Theory, 1982. 
- [2] S. E. Levinson, L. R. Rabiner, M. M. Sondhi, An introduction to the application of the theory of probabilistic functions of a Markov process to automatic speech recognition, Bell System Technical Journal, 1983.
- [3] P. A. Devijver, Baum’s forward-backward algorithm revisited, Pattern Recognition Letters, 1985.

