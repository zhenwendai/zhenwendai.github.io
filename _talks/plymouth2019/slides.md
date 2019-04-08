---
title:  "Gaussian Process in Practice: Scalability and Uncertainty"
author: Zhenwen Dai
institute: Amazon
date:   2019-04-09
bibliography: ../GPSS2018/scalable_gp.bib
fontsize: 12pt
header-includes:
  \newcommand{\gaussianDist}[3]{\mathcal{N}\left(#1|#2,#3\right)}
  \newcommand{\expectation}[1]{\left\langle#1\right\rangle}
  \newcommand{\KL}[2]{\text{KL}\left(#1\,\|\,#2\right)}
  \newcommand{\argmax}{\operatorname{arg\,max}}
  \newcommand{\tr}[1]{\text{tr}\left(#1\right)}
---

# Gaussian process

$$
\yV = (y_1, \ldots, y_N), \quad \xM = (\xV_1, \ldots, \xV_N)^\top
$$
$$
p(\yV| \fV) = \gaussianDist{\yV}{\fV}{\sigma^2 \I}, \quad p (\fV| \xM) = \gaussianDist{\fV}{0}{\K(\xM, \xM)}
$$

![](../GPSS2018/diagrams/gp_first_example.pdf){ width=50% }

# The computational time of GP

- The time complexity of Gaussian process is $O(N^3)$.

- Take 1D regression problem as an example:

# GP meta-analysis

The computational cost of Gaussian process is $O(N^3)$.

![](../GPSS2018/diagrams/gp_scaling.pdf){ width=50% }

# What if we have 1 million data points?

# What about waiting for faster computers?

# What about parallel computing / GPU?

# Is this the end of the story?

- Apart from speeding up the exact computation, there have been a lot of works on approximation of GP inference.
- These methods often target at some specific scenario and provide good approximation for the targeted scenarios.
- Provide an overview about common approximations.

# Big data (?)

- lots of data $\neq$ complex function
- In real world problems, we often collect a lot of data for modeling relatively simple relations.

![](../GPSS2018/diagrams/gp_example_lots_data.pdf){ width=50% }

# Data subsampling?

- Real data often do not evenly distributed.
- We tend to get a lot of data on common cases and very few data on rare cases.

![](./diagrams/gp_example_lots_data_uneven.png){ width=45% }
![](./diagrams/X_histogram_lots_data_uneven.png){ width=45% }

# Covariance matrix of redundant data

- With redundant data, the covariance matrix becomes low rank.
- What about low rank approximation?

![](./diagrams/lots_data_covariance_matrix.png){ width=33% }
![](./diagrams/lots_data_eigen_values.png){ width=45% }

# Low-rank approximation

- Let's recall the log-likelihood of GP:
$$
\log p(\yV|\xM) =  \log \gaussianDist{\yV}{0}{\K+\sigma^2\I},
$$
where $\K$ is the covariance matrix computed from $\xM$ according to the kernel function $k(\cdot, \cdot)$ and $\sigma^2$ is the variance of the Gaussian noise distribution.
- Assume $\K$ to be low rank.
- This leads to Nyström approximation by Williams and Seeger [-@WilliamsSeeger2001].

# Nyström approximation [@WilliamsSeeger2001]

- Let's randomly pick a subset from the training data: $\zM \in \R^{M\times Q}$.
- Approximate the covariance matrix $\K$ by $\tilde{\K}$:
$$\tilde{\K} = \K_z \K_{zz}^{-1} \K_z^\top,$$
where $\K_{z} = \K(\xM, \zM)$ and $\K_{zz} = \K(\zM, \zM)$.

- Note that $\tilde{\K} \in \R^{N\times N}$, $\K_{z} \in \R^{N\times M}$ and $\K_{zz} \in \R^{M \times M}$.
- The log-likelihood is approximated by

$$
\log p(\yV|\xM, \theta) \approx  \log \gaussianDist{\yV}{0}{\K_z \K_{zz}^{-1} \K_z^\top+\sigma^2\I}.
$$

# Efficient computation using Woodbury formula

- The naive formulation does not bring any computational benefits.
$$
\tilde{\bound} = -\frac{1}{2}\log |2\pi (\tilde{\K}+\sigma^2\I)| - \frac{1}{2} \yV^\top (\tilde{\K}+\sigma^2\I)^{-1} \yV
$$
- Apply the Woodbury formula:
$$
(\K_z \K_{zz}^{-1} \K_z^\top+\sigma^2\I)^{-1} = \sigma^{-2}\I - \sigma^{-4} \K_z (\K_{zz} + \sigma^{-2}\K_z^\top\K_z)^{-1}\K_z^\top
$$
- Note that $(\K_{zz} + \sigma^{-2}\K_z^\top\K_z) \in \R^{M \times M}$.
- The computational complexity reduces to $O(NM^2)$.

<!--
# What is this different from data subsampling?

- The model parameters can be optimized
-->

# Gaussian process with pseudo data (1)

- Snelson and Ghahramani [-@SnelsonZoubin2006] propose the idea of having pseudo data. This approach is later referred to as   Fully Independent Training Conditional (FITC).
- Augment the training data ($\xM$, $\yV$) with pseudo data $\uV$ at location $\zM$.
$$
p\left(\begin{bmatrix}
\yV \\ \uV \end{bmatrix}|\begin{bmatrix}
\xM \\ \zM \end{bmatrix} \right) = \gaussianDist{\begin{bmatrix}
\yV \\ \uV \end{bmatrix}}{0}{\begin{bmatrix}
\K_{ff}+\sigma^2\I & \K_{fu}  \\ \K_{fu}^\top & \K_{uu} \end{bmatrix}}
$$
where $\K_{ff} = \K(\xM, \xM)$, $\K_{fu} = \K(\xM, \zM)$ and $\K_{uu} = \K(\zM, \zM)$.

# Gaussian process with pseudo data (2)

- Thanks to the marginalization property of Gaussian distribution,
$$
p(\yV| \xM) = \int_{\uV} p(\yV, \uV | \xM, \zM).
$$
- Further re-arrange the notation:
$$
p(\yV, \uV| \xM, \zM) = p(\yV| \uV, \xM, \zM) p(\uV| \zM)
$$
where $p(\uV| \zM) = \gaussianDist{\uV}{0}{\K_{uu}}$,
$p(\yV| \uV, \xM, \zM)=\gaussianDist{\yV}{\K_{fu} \K_{uu}^{-1} \uV}{\K_{ff} - \K_{fu} \K_{uu}^{-1} \K_{fu}^\top+\sigma^2\I}$.

# FITC approximation (1)

- So far, $p(\yV | \xM)$ has not been changed, but there is no speed-up, $\K_{ff} \in \R^{N\times N}$ in $\K_{ff} - \K_{fu} \K_{uu}^{-1} \K_{fu}^\top+\sigma^2\I$.

- The FITC approximation assumes
$$
\tilde{p}(\yV| \uV, \xM, \zM)=\gaussianDist{\yV}{\K_{fu} \K_{uu}^{-1} \uV}{\lambdaM +\sigma^2\I},
$$
where $\lambdaM = (\K_{ff} - \K_{fu} \K_{uu}^{-1} \K_{fu}^\top)\circ \I$.

# FITC approximation (2)

- Marginalize $\uV$ from the model definition:
$$
\tilde{p}(\yV| \xM, \zM) = \gaussianDist{\yV}{0}{\K_{fu} \K_{uu}^{-1} \K_{fu}^\top+\lambdaM +\sigma^2\I}
$$

- Woodbury formula can be applied in the sam way as in Nyström approximation:
$$
(\K_z \K_{zz}^{-1} \K_z^\top+\lambdaM+\sigma^2\I)^{-1} = \aM - \aM \K_z (\K_{zz} + \K_z^\top\aM\K_z)^{-1}\K_z^\top\aM,
$$
where $\aM = (\lambdaM+\sigma^2\I)^{-1}$.

# FITC approximation (3)

- FITC allows the pseudo data not being a subset of training data.
- The inducing inputs $\zM$ can be optimized via gradient optimization.
- Like Nyström approximation, when taking all the training data as inducing inputs, the FITC approximation is equivalent to the original GP:
$$
\tilde{p}(\yV| \xM, \zM=\xM) = \gaussianDist{\yV}{0}{\K_{ff} +\sigma^2\I}
$$
- FITC can be combined easily with expectation propagation (EP).
- [@BuiEtAl2017] provides an overview and a nice connection with variational sparse GP.

# Model Approximation vs. Approximate Inference

When the exact model/inference is intractable, typically there are two types of approaches:

- Approximate the original model with a simpler one such that inference becomes tractable, like Nyström approximation, FITC.
- Keep the original model but derive an approximate inference method which is often *not* able to return the true answer, like variational inference.

# Model Approximation vs. Approximate Inference

A problem with model approximation is that

- when an approximated model requires some tuning, e.g., for hyper-parameters, it is unclear how to improve it based on training data.

- In the case of FITC, we know the model is correct if $\zM = \xM$, however, optimizing $\zM$ will not necessarily lead to a better location.

- In fact, optimizing $\zM$ can lead to overfitting. [@QuinoneroRasmussen2005]

# Variational Sparse Gaussian Process (1)

- [@Titsias2009] introduces a variational approach for sparse GP.
- It follows the same concept of pseudo data:
$$
p(\yV| \xM) = \int_{\fV, \uV} p(\yV| \fV) p(\fV| \uV, \xM, \zM) p(\uV| \zM)
$$
where $p(\uV| \zM) = \gaussianDist{\uV}{0}{\K_{uu}}$,
$p(\yV| \uV, \xM, \zM)=\gaussianDist{\yV}{\K_{fu} \K_{uu}^{-1} \uV}{\K_{ff} - \K_{fu} \K_{uu}^{-1} \K_{fu}^\top+\sigma^2\I}$.

# Variational Sparse Gaussian Process (2)

- Instead of approximate the model, [@Titsias2009] derives a variational lower bound.

- Normally, a variational lower bound of a marginal likelihood, also known as evidence lower bound (ELBO), looks like
$$
\begin{aligned}
\log p(\yV | \xM) =& \log \int_{\fV, \uV} p(\yV | \fV) p(\fV| \uV, \xM, \zM) p(\uV | \zM) \\
\geq& \int_{\fV, \uV} q(\fV, \uV) \log \frac{p(\yV | \fV) p(\fV| \uV, \xM, \zM) p(\uV | \zM)}{q(\fV, \uV)}.
\end{aligned}
$$

# Special Variational Posterior

- [@Titsias2009] defines an unusual variational posterior:
$$
q(\fV, \uV) = p(\fV| \uV, \xM, \zM) q(\uV), \quad \text{where } q(\uV) = \gaussianDist{\uV}{\mu}{\Sigma}.
$$
- Plug it into the lower bound:
$$
\begin{aligned}
\bound =& \int_{\fV, \uV} p(\fV| \uV, \xM, \zM) q(\uV) \log \frac{p(\yV | \fV) p(\fV| \uV, \xM, \zM) p(\uV | \zM)}{p(\fV| \uV, \xM, \zM) q(\uV)} \\
=& \expectation{\log p(\yV | \fV)}_{p(\fV| \uV, \xM, \zM) q(\uV)} - \KL{q(\uV)}{p(\uV|\zM)}\\
=& \expectation{\log \gaussianDist{\yV}{\K_{fu}\K_{uu}^{-1}\uV}{\sigma^2\I}}_{q(\uV)} - \KL{q(\uV)}{p(\uV|\zM)}
\end{aligned}
$$

# Tighten the Bound

- Find the optimal parameters of $q(\uV)$:
$$
\mu^\ast, \Sigma^\ast = \argmax_{\mu, \Sigma} \bound(\mu, \Sigma).
$$
- Make the bound as tight as possible by plugging in $\mu^\ast$ and $\Sigma^\ast$:
$$
\bound = \log \gaussianDist{\yV}{0}{\K_{fu} \K_{uu}^{-1} \K_{fu}^\top+ \sigma^2\I} - \frac{1}{2\sigma^2}\tr{\K_{ff} - \K_{fu} \K_{uu}^{-1} \K_{fu}^\top}.
$$
- The overall complexity of the lower bound remains $O(NM^2)$.

# Variational sparse GP

- Note that $\bound$ is not a valid log-pdf, $\int_{\yV} \exp(\bound(\yV)) \leq 1$, due to the trace term.
- As inducing points are variational parameters, optimizing the inducing inputs $\zM$ always leads to a better bound.
- The model does not "overfit" with too many inducing points.

![](../GPSS2018/diagrams/sparsegp_example_lots_inducing_points.pdf){ width=50% }


# Recap approximation

- Inducing point approximation
- frequency approximation
- stochastic approximation
- distributed approximation
- approximation on matrix inversion (GPyTorch)
- approximation on banded precision matrix

# Why GP?

- Why do we spend all these energy on speeding up GP?
- Only for a non-parametric regressor?
- What about fitting a neural network?

![](./diagrams/lots_data_nn_fit.png){ width=50% }

# What is the difference?

- The error bar!

![](./diagrams/lots_data_nn_fit.png){ width=45% }
![](./diagrams/gp_example_lots_data_uneven.png){ width=45% }

# Can we learn an error bar with NN?

- Of course, we can. Let's add a likelihood to the neural network:
$$
p(y|x) = \gaussianDist{y}{f_\theta(x)}{\sigma^2}
$$

- Now, we have an error bar for our neural network. Are they the same?

# Two types of uncertainty

- In our GP regression model, we have two "layers" of distributions:
$$
p(\yV| \fV) = \gaussianDist{\yV}{\fV}{\sigma^2 \I}, \quad p (\fV| \xM) = \gaussianDist{\fV}{0}{\K(\xM, \xM)}
$$

# Sampling from the two types of uncertainty

- The left one is independent and the right one is correlated.

<!--
Too much technical details here

# A Bayesian linear model example

- It is easier to explain with a parametric model.

- A 1D quadratic regression example:
$$
p(\yV| \xV, \wV) = \gaussianDist{\yV}{\Psi\wV}{\sigma^2 \I}, \quad p (\wV) = \gaussianDist{\wV}{0}{\I}
$$
where $\Psi \in \R^{N \times 3}$ is the feature matrx:
$$
\Psi = \begin{bmatrix}
x_1^2& x_1& 1 \\
\vdots& \vdots& \vdots\\
x_N^2& x_N& 1
\end{bmatrix},
$$
and $\wV \in \R^3$ is the vector representing the coefficients of the quadratic equation.
-->

# Aleatoric and epistemic uncertainty

- Aleatoric uncertainty: the uncertainty about the noise in individual data points
- Epistemic uncertainty: the uncertainty in the model

$$
p(\theta|\mathcal{D}) = \frac{p(\mathcal{D}|\theta) p(\theta)}{p(\mathcal{D})}
$$

# Automated decision making

- Under the model assumption, the epistemic uncertainty allows us to know what the model *does not* know.

- This enables us to trade-off among different choices with limited information.

- Example: Bayesian optimization
$$
x^* = \arg \min_x f(x)
$$

# Bayesian optimization

A surrogate model guided search

![A 1D example](./diagrams/bo_surrogate_model.png){ width=50% }

# Possibility represented by uncertainty

![A 1D example](./diagrams/bo_surrogate_model_samples.png){ width=50% }

# Balance exploitation and exploration

# Acquisition function

- Formulate the policy of the exploitation and exploration tradeoff.

- The utility function about improvement:
$$
u(f) = \max(0, f' - f)
$$

- The expected improvement under our surrogate model:
$$
a_{\text{EI}}(x) = \int u(f) p(f|x, \mathcal{D}) \text{d}f
$$

# A BO algorithm

# Example

# Challenges in Bayesian Optimization

- dimensionality
- non-stationarity
- under safety constraints
- warm-starting

#
- Thank you!

# References {.allowframebreaks}
