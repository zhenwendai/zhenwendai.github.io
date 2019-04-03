---
title:  "Gaussian Process in Practice: Scalability and Uncertainty"
author: Zhenwen Dai
institute: Amazon
date:   2019-04-09
bibliography: ../GPSS2018/scalable_gp.bib
header-includes:
  \newcommand{\gaussianDist}[3]{\mathcal{N}\left(#1|#2,#3\right)}
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

# Is this the end of the talk?

- Apart from speeding up the exact computation, there have been a lot of works on approximation of GP inference.
- These methods often target at some specific scenario and provide good approximation for the targeted scenarios.
- Provide an overview about common approximations.

# Big data (?)

- lots of data $\neq$ complex function

In real world problems, we often collect a lot of data for relatively simple relations.

![](../GPSS2018/diagrams/gp_example_lots_data.pdf){ width=50% }

# Subsampling the data?

In real world problems, we often collect a lot of data for relatively simple relations.

![](../GPSS2018/diagrams/gp_example_lots_data.pdf){ width=50% }

# If the data are unevenly distributed?

- We often see a lot of "common" situations and a heavy rai of "rare" cases.

# Recap approximation

- Inducing point approximation
- frequency approximation
- stochastic approximation
- distributed approximation
- approximation on matrix inversion
- approximation on banded precision matrix

[@Titsias2009]

## Scalability is a big challenge for Gaussian process

Gaussian process is

GP computational time meta-analysis.

a gp fit on computational time. (what if we have a million data points?)

What about big computers?

What about parallelism?

So what else we can do?

cover different approaches:

1. Sparse GP
2. covariance matrix inversion
3. distributed GPs

## What does uncertainty in Gaussian process?



# References {.allowframebreaks}
