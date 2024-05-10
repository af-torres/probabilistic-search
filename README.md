# Binary Probabilistic Search

> Andres Torres
>
> University of Rhode Island, Kingston, United States

> code: [https://github.com/af-torres/probabilistic-search](https://github.com/af-torres/probabilistic-search)

### Abstract
This is a brief summary of your research paper.

## 1. Introduction

In recent years, the number of parameters used in statistical models has grown into the billions. Training these models is an expensive procedure that companies seek to optimize. During training, the most costly step is tuning the model's hyper-parameters. The function that describes the relation between hyper-parameters and the loss function of the model is non-differentiable, thus, researchers are constrain to the use black-box optimization methods that do not rely on derivation, like stochastic search and population algorithms.

Black-box optimization algorithms use samples from the objective function to find high value regions within the search space, and, when used for the purpose of hyper-parameter tunning, they require fitting the underlying model multiple times. When the underlying model is large, this cost adds up to millions of dollars in computing.

In this manuscript we develop a sample efficient algorithm, that can be used to optimize non-differentiable functions, called: "Binary probabilistic search". This procedure uses Bayesian statistics to efficiently reduce, by half, the search space.

## 2. Literature Review

The concept of halving the search space draws inspiration from the efficient strategy of binary search, a fundamental algorithmic technique used to locate a target value within a sorted array. Binary search operates by iteratively partitioning the search space into two halves, discarding the portion where the target value cannot reside, and focusing the search on the remaining portion [[1]](https://search.informit.org/doi/epdf/10.3316/informit.573360863402659). This iterative process leads to a logarithmic time complexity, making binary search exceptionally efficient for large datasets and search spaces.

In the realm of optimization, Bayesian optimization methods have gained prominence for their ability to efficiently optimize black-box functions with non-differentiable surfaces. These methods typically involve two primary components: a method for statistical inference, typically Gaussian process (GP) regression; and an acquisition function for deciding where to sample, which is often the expected improvement [[2]](https://arxiv.org/pdf/1807.02811). Bayesian optimization algorithms leverage surrogate models to estimate the quality of different regions within the search space. By dynamically updating their beliefs using Bayesian statistics as new samples are acquired, these algorithms intelligently direct the search towards promising areas, balancing exploration and exploitation.

## 3. Methodology

By combining the efficiency of binary search with the statistical robustness of Bayesian optimization, our proposed "Binary probabilistic search" algorithm offers a novel approach to optimizing non-differentiable functions. This methodology effectively reduces the search space by half using Bayesian statistics, enhancing sample efficiency and reducing computational costs in hyperparameter tuning tasks. By leveraging the complementary strengths of binary search and Bayesian optimization, our algorithm stands poised to address the challenges posed by high-dimensional and non-convex optimization problems in modern data science and machine learning applications.

Let's delve into optimizing a statistical model's hyperparameter, $\lambda$, by minimizing its underlying loss function $J$. To formalize this, we define our objective as:

$$
\begin{align}
    \underset{\lambda \in \mathbb{R} \: \mid \: 0 \leq \lambda \leq 1}{\mathop{\text{{argmin}}}} J(\lambda)
\end{align}
$$

Now, we introduce a pivotal concept: a random variable $Y$. This variable models the expected value of our objective function within a specified quantile of the domain, described by the set $\{(x_{1,i}, x_{2,i}) \in \mathbb{R}^2\}$. Formally:

$$
\begin{equation}
    E[Y(x_{1,i}, x_{2,i}) \mid x_{1,i}, x_{2,i}] = E[J(\lambda) \mid x_{1,i} \leq \lambda \leq x_{2,i}]
\end{equation}
$$

Here, $Y$ is defined as a normal random variable with semi-conjugate priors $\mu_i$ and $\sigma^2_i$, indicating the mean and variance respectively. Specifically:

$$
\begin{align}
    Y(x_{1,i}, x_{2,i}) \sim N(\mu_i, \sigma^2_i), \\
    \mu_i \sim N(\mu_{0, i}, \tau^2_{0, i}), \\
    \sigma^2_i \sim InverseGamma(a_{0, i}, b_{0, i})
\end{align}
$$

These probabilistic assumptions allow us to capture uncertainty in our model parameters. Utilizing this framework, we are able to reparametrize the objective function as the search of the quantile of the domain with the lowest expected loss:

$$
\begin{equation}
    \underset{x_{1,i}, x_{2,i}}{\mathop{\text{{argmin}}}} \; E[Y(x_{1,i}, x_{2,i}) \mid x_{1,i}, x_{2,i}]
\end{equation}
$$

The algorithm employed in the optimization process begins by initializing a uniformed prior distribution for $Y$, representing the expected value over the entire domain of $J$. The iterative process continues until convergence is achieved:

1. Randomly generate $N$ points within the specified range and utilize these points to compute an initial posterior distribution $Y_1$.
  
2. Partition the domain space into two halves, assigning each partition a prior distribution equal to the posterior found in the previous step $Y_1$.

3. For each partition, draw $M$ samples from the posterior predictive distribution and compute the probability that the predicted value ($\hat{Y_i}$) is the minimum among all partitions. This calculation involves determining the probability that $\hat{Y_i}$ is less than all other predicted values ($\hat{Y_1}, \ldots, \hat{Y_{i - 1}}, \hat{Y_{i + 1}}, \ldots, \hat{Y_{I}}$), where $I$ is the number of partitions.

4. Generate random uniform points from each partition, with the number of points proportional to the probability of the predictive value being the minimum.

5. Utilize these sampled points to evaluate $J$ and update the priors of the partitions based on the new information obtained from the evaluations.

## 6. Conclusion

In this manuscript, we have introduced a novel approach to optimizing non-differentiable functions, particularly focusing on hyperparameter tuning in statistical models. Motivated by the escalating complexity of modern statistical models with billions of parameters, we addressed the critical need for efficient optimization techniques to minimize computational costs.

Our proposed "Binary probabilistic search" algorithm combines the efficiency of binary search with the statistical robustness of Bayesian optimization, offering a powerful solution to navigate high-dimensional and non-convex search spaces. By leveraging Bayesian statistics, our algorithm effectively reduces the search space by half, enhancing sample efficiency and mitigating the computational burden associated with hyperparameter tuning tasks.

Moving forward, our focus will be on exploring innovative strategies to enhance the algorithm's ability to balance exploration and exploitation. While our current approach is a sample efficient approach, it fails to avoid local minima even over simple search spaces.

## References

[1] Lin, A. (2019). Binary search algorithm. WikiJournal of Science, 2(1), 1–13. https://search.informit.org/doi/10.3316/informit.573360863402659

[2] Peter I. Frazier. A tutorial on bayesian optimization, 2018.
