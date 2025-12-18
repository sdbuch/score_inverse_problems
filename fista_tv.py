"""FISTA-TV solver for inverse problems.

Implementation of Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
with Total Variation (TV) regularization for solving linear inverse problems.

Reference:
    Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding
    algorithm for linear inverse problems. SIAM journal on imaging sciences.
"""

import jax
import jax.numpy as jnp

from transforms.fourier import fft, ifft


def tv_norm(x):
    """Compute the isotropic total variation norm."""
    # Compute gradients
    grad_x = jnp.diff(x, axis=-2, prepend=x[..., :1, :])
    grad_y = jnp.diff(x, axis=-1, prepend=x[..., :, :1])
    # Isotropic TV norm
    return jnp.sum(jnp.sqrt(grad_x**2 + grad_y**2 + 1e-8))


def tv_prox_gd(x, lambda_tv, num_steps=10, step_size=0.1):
    """TV proximal operator using gradient descent.

    Solves: argmin_z (1/2)||z - x||^2 + lambda_tv * TV(z)

    Args:
        x: Input image
        lambda_tv: TV regularization weight
        num_steps: Number of gradient descent steps
        step_size: Step size for gradient descent

    Returns:
        Denoised image
    """
    z = x

    for _ in range(num_steps):
        # Compute gradient of TV
        grad_x = jnp.diff(z, axis=-2, append=z[..., -1:, :])
        grad_y = jnp.diff(z, axis=-1, append=z[..., :, -1:])

        norm = jnp.sqrt(grad_x**2 + grad_y**2 + 1e-8)

        # Divergence of normalized gradient
        div_x = jnp.diff(
            grad_x / norm, axis=-2, prepend=jnp.zeros_like(grad_x[..., :1, :])
        )
        div_y = jnp.diff(
            grad_y / norm, axis=-1, prepend=jnp.zeros_like(grad_y[..., :, :1])
        )

        grad_tv = -(div_x + div_y)

        # Gradient descent step
        z = z - step_size * ((z - x) + lambda_tv * grad_tv)

    return z


def fista_tv_solve(measurements, mask, lambda_tv=0.001, max_iter=100, tol=1e-6):
    """FISTA-TV solver for undersampled MRI reconstruction.

    Solves: argmin_x (1/2)||A(x) - y||^2 + lambda_tv * TV(x)
    where A(x) = mask * FFT(x)

    Args:
        measurements: Undersampled k-space measurements (complex)
        mask: Undersampling mask
        lambda_tv: TV regularization parameter
        max_iter: Maximum number of FISTA iterations
        tol: Convergence tolerance

    Returns:
        Reconstructed image (real-valued)
    """

    # Forward operator: A(x) = mask * FFT(x)
    def forward(x):
        # Convert real image to complex for FFT
        x_complex = x.astype(jnp.complex64)
        # Use norm=None (default) for old JAX compatibility
        # Scale by sqrt(N) to match ortho normalization
        N = x.shape[-2] * x.shape[-1]
        return mask * fft(x_complex, center=True, norm=None) / jnp.sqrt(N)

    # Adjoint operator: A^T(y) = IFFT(mask * y)
    def adjoint(y):
        # Use norm=None and scale accordingly
        N = mask.shape[-2] * mask.shape[-1]
        return (ifft(mask * y, center=True, norm=None) / jnp.sqrt(N)).real

    # Initialize with adjoint of measurements
    x = adjoint(measurements)
    x = jnp.clip(x, 0, 1)  # Project to [0, 1]

    y = x
    t = 1.0

    # Lipschitz constant for masked FFT operator is 1 with ortho normalization
    L = 1.0

    for i in range(max_iter):
        # Gradient of data fidelity term: A^T(A(y) - measurements)
        grad = adjoint(forward(y) - measurements)

        # Gradient descent step
        x_new = y - (1.0 / L) * grad

        # TV proximal operator
        x_new = tv_prox_gd(x_new, lambda_tv / L, num_steps=5, step_size=0.05)

        # Ensure valid range
        x_new = jnp.clip(x_new, 0, 1)

        # FISTA acceleration
        t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        x = x_new
        t = t_new

    return x


def get_fista_tv_solver(config, shape, inverse_scaler):
    """Get FISTA-TV solver compatible with the evaluation pipeline.

    Args:
        config: Configuration object
        shape: Shape of images (batch_size, height, width, channels)
        inverse_scaler: Function to inverse the data scaling

    Returns:
        Solver function compatible with evaluation pipeline
    """
    from cs import get_masks

    # Lambda_tv will be passed as a hyperparameter
    # Default value can be overridden in config
    lambda_tv_default = getattr(config.sampling, "lambda_tv", 0.001)
    max_iter = getattr(config.sampling, "fista_max_iter", 100)

    def solver(rng, pstate, img, lambda_tv=None):
        """FISTA-TV solver function.

        Args:
            rng: Random number generator state array [num_devices] (unused for FISTA)
            pstate: Model state (unused for FISTA)
            img: Input measurements (scaled images) [num_devices, batch_per_device, H, W, C]
            lambda_tv: TV regularization parameter (optional)

        Returns:
            Reconstructed images [num_devices, batch_per_device, H, W, C]
        """
        if lambda_tv is None:
            lambda_tv = lambda_tv_default

        # Get undersampling mask - same for all images
        # img should have shape [num_devices, batch_per_device, H, W, C]
        # We need to get mask for a single image to use as template
        single_img = img[0:1, 0:1]  # Take one sample
        mask_template = get_masks(config, single_img)[
            0, 0
        ]  # Remove device and batch dims

        # Forward operator to get measurements
        def forward_single(x):
            """Apply forward operator to single image."""
            x_complex = x.astype(jnp.complex64)
            # Use norm=None for old JAX compatibility, scale manually
            N = x.shape[-2] * x.shape[-1]
            return mask_template * fft(x_complex, center=True, norm=None) / jnp.sqrt(N)

        # Run FISTA-TV on each image
        def solve_single_image(single_img):
            """Solve FISTA-TV for a single image."""
            # Create measurements
            measurement = forward_single(single_img[..., 0])  # Remove channel dim

            # Solve
            reconstructed = fista_tv_solve(
                measurement, mask_template, lambda_tv=lambda_tv, max_iter=max_iter
            )

            return reconstructed[..., None]  # Add channel dimension back

        # Process all images: vectorize over devices and batch
        # img shape: [num_devices, batch_per_device, H, W, C]
        def process_batch(batch_imgs):
            return jax.vmap(solve_single_image)(batch_imgs)

        # Vectorize over devices
        reconstructed = jax.vmap(process_batch)(img)

        return reconstructed

    return solver
