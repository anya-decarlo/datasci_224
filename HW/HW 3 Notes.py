# HW 3 Notes

# === Simple dataset for PCA and PPCA ===
# This dataset is designed for easy, intuitive understanding of PCA/PPCA.
# Points are aligned along the line x = y, centered around the origin.
# You can easily compute projections and visualize the transformation.
# Here, we use columns = observations and rows = variables
import numpy as np

X = np.array([
    [2, 2],
    [3, 3],
    [4, 4],
    [-2, -2],
    [-3, -3],
    [-4, -4]
])
# X is now a 6x2 matrix. Try running PCA or PPCA on X to see the principal components!

## Assumptions of PCA ONLY
## 1. We assume that each of the variables X has been centered to have a mean 0 
## 2. Stack observations vertically to form a data matrix X
## 3. We constrain the loadings so that their sum of squares is equal to one


## Assumptions of PPCA



# === PPCA NOTES ===

# Hidden vector z (size q X 1)
# A modeling assumption, not estimated from the data
z  ~  Normal(mean = 0,  covariance = I_q) #I_Q is the q by q identity matrix


# Observable vector x (size D X 1)
# Here, B is a D × q matrix, μ is a D × 1 mean vector, and I_D is the D × D identity matrix.
# σ² is a single positive number (the noise variance).

x  |  z  ~  Normal(mean = μ + B z ,  covariance = σ² I_D)

# === PPCA MODEL ===

# u = Data mean (D X 1 dimensions ) 
# D denotes the dimmensionality of the observed data vector X (e.g. 8x8 = 64)

# B = Loadings (D X q dimensions)
# Linear map from latent space to data space 

# Where q = number of latent dimensions, chose q < D (e.g. q = 2)

# === PPCA Implementation ===

def ppca(X, q):
    """
    Probabilistic PCA (closed-form ML solution).

    Args:
        X (np.ndarray): data matrix of shape (n_samples, D)
        q (int): number of latent dimensions (q < D)

    Returns:
        mu (np.ndarray): estimated data mean vector of shape (D,)
        B (np.ndarray): estimated loading matrix of shape (D, q)
        sigma2 (float): estimated isotropic noise variance (scalar)
    """
    # 1. Number of samples (n) and observed dimensions (D)
    n, D = X.shape

    # 2. Compute the mean vector mu (one mean per feature)
    mu = X.mean(axis=0)

    # 3. Center the data by subtracting mu from each sample
    X_centered = X - mu

    # 4. Compute the sample covariance matrix S = (1/n) * X_centered^T @ X_centered
    S = (1.0 / n) * (X_centered.T @ X_centered)

    # 5. Perform eigen-decomposition: S = eigvecs * diag(eigvals) * eigvecs^T
    eigvals, eigvecs = np.linalg.eigh(S)

    # 6. Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 7. Estimate noise variance sigma2 as the average of the eigenvalues beyond the first q
    if q < D:
        sigma2 = np.mean(eigvals[q:])
    else:
        sigma2 = 0.0

    # 8. Build the loading matrix B using the top-q eigenpairs
    #    For each of the top q components: scale eigenvector by sqrt(eigenvalue - sigma2)
    Lambda_q = np.diag(np.sqrt(np.maximum(eigvals[:q] - sigma2, 0)))
    U_q = eigvecs[:, :q]
    B = U_q @ Lambda_q

    return mu, B, sigma2


# Example usage on the toy dataset X defined above
mu_hat, B_hat, sigma2_hat = ppca(X, q=1)
print("Estimated mean (mu):", mu_hat)
print("Estimated loading matrix (B):", B_hat)
print("Estimated noise variance (sigma2):", sigma2_hat)
# === End PPCA Implementation ===




