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