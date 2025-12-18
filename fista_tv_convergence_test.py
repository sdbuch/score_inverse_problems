"""Test FISTA-TV convergence on a single BraTS image.

This script runs FISTA-TV on one image and monitors the objective value
to verify convergence. Useful for tuning hyperparameters.

Usage:
    python fista_tv_convergence_test.py \
        --lambda_tv 0.001 \
        --max_iter 1000 \
        --tv_prox_steps 50 \
        --tv_prox_lr 0.01
"""

import jax
import jax.numpy as jnp
import numpy as np
from transforms.fourier import fft, ifft
from fista_tv import tv_norm, tv_prox_gd
import argparse


def get_cartesian_mask(shape, n_keep=30):
    """Create undersampling mask for MRI."""
    size = shape[0]
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = shape[0], shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))

    mask = np.zeros((num_rows, num_cols), dtype=np.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad: pad + num_low_freqs] = True

    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
        num_low_freqs * acceleration - num_cols
    )

    offset = round(adjusted_accel) // 2
    accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
    accel_samples = np.around(accel_samples).astype(np.uint32)
    mask[:, accel_samples] = True

    return jnp.array(mask)


def compute_objective(x, measurements, mask, lambda_tv):
    """Compute FISTA-TV objective value.
    
    Objective: (1/2)||A(x) - y||^2 + lambda_tv * TV(x)
    where A(x) = mask * FFT(x)
    """
    # Forward operator
    x_complex = x.astype(jnp.complex64)
    N = x.shape[-2] * x.shape[-1]
    Ax = mask * fft(x_complex, center=True, norm=None) / jnp.sqrt(N)
    
    # Data fidelity term
    data_fidelity = 0.5 * jnp.sum(jnp.abs(Ax - measurements) ** 2)
    
    # TV regularization term
    tv_value = tv_norm(x)
    
    # Total objective
    objective = data_fidelity + lambda_tv * tv_value
    
    return objective, data_fidelity, tv_value


def fista_tv_with_monitoring(measurements, mask, ground_truth, lambda_tv=0.001, max_iter=1000, 
                              tv_prox_steps=50, tv_prox_lr=0.01, print_every=10):
    """FISTA-TV with objective value and ground truth PSNR monitoring."""
    
    # Forward and adjoint operators
    def forward(x):
        x_complex = x.astype(jnp.complex64)
        N = x.shape[-2] * x.shape[-1]
        return mask * fft(x_complex, center=True, norm=None) / jnp.sqrt(N)
    
    def adjoint(y):
        N = mask.shape[-2] * mask.shape[-1]
        return (ifft(mask * y, center=True, norm=None) / jnp.sqrt(N)).real
    
    def compute_psnr(x, gt):
        """Compute PSNR against ground truth.
        
        PSNR = -10 * log10(MSE) where MSE = mean((x - gt)^2)
        Assumes both x and gt are in [0, 1] range.
        """
        mse = float(jnp.mean((x - gt) ** 2))
        if mse < 1e-10:
            return 100.0  # Perfect reconstruction
        psnr = -10 * np.log10(mse)
        return psnr
    
    # Initialize
    x = adjoint(measurements)
    x = jnp.clip(x, 0, 1)
    y = x
    t = 1.0
    L = 1.0
    
    objectives = []
    data_fidelities = []
    tv_values = []
    psnrs = []
    
    print(f"\n{'Iter':>6} {'Objective':>15} {'Data Fidelity':>15} {'TV':>12} {'GT PSNR':>10} {'Rel Change':>12}")
    print("-" * 90)
    
    for i in range(max_iter):
        # FISTA step
        grad = adjoint(forward(y) - measurements)
        x_new = y - (1.0 / L) * grad
        x_new = tv_prox_gd(x_new, lambda_tv / L, num_steps=tv_prox_steps, step_size=tv_prox_lr)
        x_new = jnp.clip(x_new, 0, 1)
        
        # Compute metrics
        if i % print_every == 0 or i == max_iter - 1:
            obj, df, tv = compute_objective(x_new, measurements, mask, lambda_tv)
            psnr = compute_psnr(x_new, ground_truth)
            
            objectives.append(float(obj))
            data_fidelities.append(float(df))
            tv_values.append(float(tv))
            psnrs.append(psnr)
            
            # Compute relative change
            if i > 0:
                rel_change = abs(objectives[-1] - objectives[-2]) / (abs(objectives[-2]) + 1e-8)
            else:
                rel_change = float('inf')
            
            print(f"{i:6d} {float(obj):15.6f} {float(df):15.6f} {float(tv):12.6f} {psnr:10.2f} {rel_change:12.6e}")
        
        # FISTA acceleration
        t_new = (1 + jnp.sqrt(1 + 4 * t**2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new
    
    return x, objectives, data_fidelities, tv_values, psnrs


def main():
    parser = argparse.ArgumentParser(description='Test FISTA-TV convergence')
    parser.add_argument('--lambda_tv', type=float, default=0.001, help='TV regularization weight')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum FISTA iterations')
    parser.add_argument('--tv_prox_steps', type=int, default=50, help='TV proximal operator steps')
    parser.add_argument('--tv_prox_lr', type=float, default=0.01, help='TV proximal operator learning rate')
    parser.add_argument('--n_projections', type=int, default=30, help='Number of k-space lines (acceleration)')
    parser.add_argument('--image_size', type=int, default=240, help='Image size')
    parser.add_argument('--print_every', type=int, default=10, help='Print frequency')
    args = parser.parse_args()
    
    print("=" * 70)
    print("FISTA-TV Convergence Test")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  lambda_tv:       {args.lambda_tv}")
    print(f"  max_iter:        {args.max_iter}")
    print(f"  tv_prox_steps:   {args.tv_prox_steps}")
    print(f"  tv_prox_lr:      {args.tv_prox_lr}")
    print(f"  n_projections:   {args.n_projections} ({args.image_size / args.n_projections:.1f}× acceleration)")
    print("=" * 70)
    
    # Load one BraTS test image
    print("\nLoading BraTS test data...")
    test_data_path = 'test_data/BraTS.npz'
    try:
        test_data = np.load(test_data_path)
        test_img = test_data['all_imgs'][0]  # Take first image
        test_img = test_img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        print(f"Loaded image shape: {test_img.shape}")
    except FileNotFoundError:
        print(f"Error: Could not find {test_data_path}")
        print("Creating synthetic test image instead...")
        # Create a simple test image
        test_img = np.zeros((args.image_size, args.image_size), dtype=np.float32)
        test_img[80:160, 80:160] = 0.8
        test_img[100:140, 100:140] = 0.3
    
    # Create undersampling mask
    print(f"\nCreating {args.n_projections}-line undersampling mask...")
    mask = get_cartesian_mask((args.image_size, args.image_size), n_keep=args.n_projections)
    print(f"Mask shape: {mask.shape}, sampling ratio: {mask.mean():.3f}")
    
    # Create measurements
    print("\nCreating undersampled measurements...")
    test_img_jax = jnp.array(test_img)
    img_complex = test_img_jax.astype(jnp.complex64)
    N = test_img.shape[-2] * test_img.shape[-1]
    measurements = mask * fft(img_complex, center=True, norm=None) / jnp.sqrt(N)
    
    # Test forward/adjoint operator consistency
    print("\nTesting forward/adjoint operator consistency...")
    reconstructed_direct = (ifft(measurements, center=True, norm=None) / jnp.sqrt(N)).real
    print(f"  Direct IFFT of measurements: min={float(reconstructed_direct.min()):.4f}, max={float(reconstructed_direct.max()):.4f}")
    
    # Apply adjoint (should be similar to direct IFFT)
    reconstructed_adjoint = (ifft(mask * measurements, center=True, norm=None) / jnp.sqrt(N)).real
    print(f"  Adjoint reconstruction: min={float(reconstructed_adjoint.min()):.4f}, max={float(reconstructed_adjoint.max()):.4f}")
    
    # Check if forward-adjoint gives back approximately the input
    test_forward = mask * fft(test_img_jax.astype(jnp.complex64), center=True, norm=None) / jnp.sqrt(N)
    test_back = (ifft(mask * test_forward, center=True, norm=None) / jnp.sqrt(N)).real
    forward_adjoint_error = float(jnp.mean((test_back - test_img_jax) ** 2))
    print(f"  Forward-adjoint error (MSE): {forward_adjoint_error:.6e}")
    print(f"  Forward-adjoint PSNR: {-10 * np.log10(forward_adjoint_error) if forward_adjoint_error > 0 else 100.0:.2f} dB")
    
    if forward_adjoint_error > 0.01:
        print("  ??  WARNING: Large forward-adjoint error! Operators may be inconsistent.")
    
    # Debugging: Check data ranges
    print(f"\nData range checks:")
    print(f"  Ground truth: min={float(test_img_jax.min()):.4f}, max={float(test_img_jax.max()):.4f}, mean={float(test_img_jax.mean()):.4f}")
    print(f"  Measurements: min={float(jnp.abs(measurements).min()):.4f}, max={float(jnp.abs(measurements).max()):.4f}")
    
    # Compute initial PSNR (zero-filled reconstruction)
    N = test_img.shape[-2] * test_img.shape[-1]
    zero_filled = (ifft(mask * measurements, center=True, norm=None) / jnp.sqrt(N)).real
    zero_filled = jnp.clip(zero_filled, 0, 1)
    
    print(f"  Zero-filled: min={float(zero_filled.min()):.4f}, max={float(zero_filled.max()):.4f}, mean={float(zero_filled.mean()):.4f}")
    
    zero_filled_mse = float(jnp.mean((zero_filled - test_img_jax) ** 2))
    zero_filled_psnr = -10 * np.log10(zero_filled_mse) if zero_filled_mse > 0 else 100.0
    print(f"\nInitial (zero-filled) PSNR: {zero_filled_psnr:.2f} dB")
    print(f"  MSE: {zero_filled_mse:.6f}")
    
    # Run FISTA-TV with monitoring
    print("\nRunning FISTA-TV...\n")
    reconstructed, objectives, data_fidelities, tv_values, psnrs = fista_tv_with_monitoring(
        measurements, mask, test_img_jax,
        lambda_tv=args.lambda_tv,
        max_iter=args.max_iter,
        tv_prox_steps=args.tv_prox_steps,
        tv_prox_lr=args.tv_prox_lr,
        print_every=args.print_every
    )
    
    # Convergence analysis
    print("\n" + "=" * 90)
    print("Convergence Analysis")
    print("=" * 90)
    
    if len(objectives) > 1:
        final_rel_change = abs(objectives[-1] - objectives[-2]) / (abs(objectives[-2]) + 1e-8)
        print(f"Final relative change: {final_rel_change:.6e}")
        
        # Check if converged
        if final_rel_change < 1e-6:
            print("? CONVERGED (rel_change < 1e-6)")
        elif final_rel_change < 1e-4:
            print("? Nearly converged (rel_change < 1e-4)")
        else:
            print("? NOT CONVERGED - Consider increasing max_iter")
    
    print(f"\nFinal objective value: {objectives[-1]:.6f}")
    print(f"  Data fidelity: {data_fidelities[-1]:.6f}")
    print(f"  TV term:       {tv_values[-1]:.6f}")
    
    # Compute reconstruction quality
    print(f"\nFinal reconstruction range: min={float(reconstructed.min()):.4f}, max={float(reconstructed.max()):.4f}, mean={float(reconstructed.mean()):.4f}")
    
    mse = float(jnp.mean((reconstructed - test_img_jax) ** 2))
    psnr = psnrs[-1]  # Use the last computed PSNR
    print(f"\nReconstruction quality vs Ground Truth:")
    print(f"  Initial (zero-filled): {zero_filled_psnr:.2f} dB (MSE: {zero_filled_mse:.6f})")
    print(f"  Final (FISTA-TV):      {psnr:.2f} dB (MSE: {mse:.6f})")
    print(f"  Improvement:           {psnr - zero_filled_psnr:.2f} dB")
    
    # Check if something went wrong
    if psnr < zero_filled_psnr:
        print("\n??  WARNING: FISTA-TV worse than zero-filled! Something may be wrong.")
        print("     Possible issues:")
        print("     - lambda_tv too high (oversmoothing)")
        print("     - TV prox not converging (try more tv_prox_steps)")
        print("     - Bug in forward/adjoint operators")
    
    # PSNR progression
    if len(psnrs) > 1:
        psnr_gain = psnrs[-1] - psnrs[0]
        print(f"\nPSNR progression during optimization:")
        print(f"  Iteration 0:    {psnrs[0]:.2f} dB")
        print(f"  Iteration {(len(psnrs)-1)*args.print_every}: {psnrs[-1]:.2f} dB")
        print(f"  Total gain:     {psnr_gain:.2f} dB")
    
    print("\n" + "=" * 90)
    print("Tip: If not converged, try:")
    print("  - Increase --max_iter (e.g., 2000)")
    print("  - Increase --tv_prox_steps (e.g., 100)")
    print("  - Adjust --lambda_tv if objective seems stuck")
    print("\nTip: Watch the GT PSNR column to see reconstruction quality improve!")
    print("=" * 90)


if __name__ == '__main__':
    main()
