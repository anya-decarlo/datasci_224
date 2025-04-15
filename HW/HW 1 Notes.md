In logistic regression, each data point (x_i, y_i) has a label y_i that is either:
	•	y_i = 1 (“positive” class), or
	•	y_i = 0 (“negative” class).

    If y_i = 1, we want the model’s predicted probability of 1.
	•	If y_i = 0, we want the model’s predicted probability of 0, which is 1 - \text{probability of 1}.


----------------

### First Derivative

$$\frac{\partial \ell(\beta)}{\partial \beta}$$

- The gradient of the log-likelihood
- Direction of steepest ascent (how to change $\beta$ to make $\ell(\beta)$ increase fastest)

1. First derivative (gradient)
	•	Tells you the slope at your current \beta.
	•	If it’s zero, you’re at a peak or a flat spot.
	•	If it’s big, there’s a strong slope uphill.

In logistic regression:

\frac{\partial \ell(\beta)}{\partial \beta} = X^T (y - p)

This is the gradient vector — one entry for each parameter \beta_j.

⸻

### Gradient

Same as: $$\frac{\partial \ell(\beta)}{\partial \beta}$$

- Just another name for the first derivative when the parameter is a vector
- A vector of partial derivatives, one for each $\beta_j$
- Points "uphill" (in the direction of increasing function value)

### Second Derivative

$$\frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T}$$

- The Hessian matrix (matrix of second derivatives)
- How the slope (gradient) is changing
- It tells you the curvature ("sharpness" or "flatness") of $\ell(\beta)$

2. Second derivative (Hessian)
	•	Tells you how the gradient is changing.
	•	It describes the curvature of the surface.
	•	Are you on a steep narrow hill (sharp curvature)? A flat plateau (small curvature)?
	•	It’s a matrix because it measures how each \beta_j interacts with every other \beta_k.

In logistic regression:

\frac{\partial^2 \ell(\beta)}{\partial \beta \partial \beta^T} = - X^T W X


### Likelihood

Given data $(X_i, Y_i)$, the likelihood is:

$$L(\theta) = \prod_{i=1}^n \Pr(Y_i \mid X_i; \theta)$$

* $\theta$ are the parameters you are trying to estimate.
* You take the product over all data points.

Because:
	•	When \beta_{\text{new}} and \beta_{\text{old}} are almost the same,
	•	It means that one more optimization step barely moves \beta anymore,
	•	Which implies the gradient is close to 0 (flat),
	•	And you’re at or very close to a local maximum of the log likelihood.

✅ That’s the point where the model “feels” it has learned the best \beta.

\mathbb{R}
The set of real numbers
\mathbb{R}^p
The set of all real p-dimensional vectors
X_i \in \mathbb{R}^p
The i-th data point is a p-dimensional real vector

 \beta \in \mathbb{R}^p means:

“\beta is an element of \mathbb{R}^p.”

In normal words:

“\beta is a p-dimensional vector made of real numbers.”

1. Linear predictor
z_i = \beta_0 + \beta^T x_i
z = intercept + np.dot(X, beta)
2. Apply logistic function
p(x_i; \beta) = \frac{1}{1 + e^{-z_i}}
probabilities = 1 / (1 + np.exp(-z))
3. Generate Y_i
Draw 0 or 1 with probability p_i
np.random.binomial(1, probabilities)

