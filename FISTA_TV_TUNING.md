# FISTA-TV Hyperparameter Tuning Guide

This guide explains how to tune FISTA-TV hyperparameters for optimal reconstruction quality.

## Hyperparameters

### 1. `lambda_tv` - TV Regularization Weight
**What it does**: Controls the trade-off between data fidelity and smoothness.
- **Higher values** (0.01): More smoothing, removes noise but may lose detail
- **Lower values** (0.0001): Less smoothing, preserves detail but may keep noise
- **Default**: 0.001

**Tuning strategy**:
- Start with 0.001
- If results are too noisy → increase to 0.005 or 0.01
- If results are too smooth/blurry → decrease to 0.0005 or 0.0001

### 2. `fista_max_iter` - Number of FISTA Iterations
**What it does**: How many optimization iterations to run.
- **More iterations**: Better convergence, higher quality, slower
- **Fewer iterations**: Faster, but may not converge fully
- **Default**: 100

**Tuning strategy**:
- Start with 100
- If results look under-reconstructed → increase to 150 or 200
- If speed is critical and quality is acceptable → reduce to 50

### 3. `tv_prox_steps` - TV Proximal Operator Steps
**What it does**: Number of gradient descent steps when solving TV subproblem.
- **More steps**: More accurate TV denoising per FISTA iteration
- **Fewer steps**: Faster but less accurate TV denoising
- **Default**: 5

**Tuning strategy**:
- Start with 5
- If TV regularization seems ineffective → increase to 10
- For faster evaluation → reduce to 3

### 4. `tv_prox_lr` - TV Proximal Operator Step Size
**What it does**: Learning rate for TV proximal operator gradient descent.
- **Higher values** (0.1): Faster TV denoising but may be unstable
- **Lower values** (0.01): More stable but slower convergence
- **Default**: 0.05

**Tuning strategy**:
- Start with 0.05
- If TV denoising is slow → increase to 0.1
- If results are unstable → decrease to 0.02

## Quick Tuning Workflow

### Step 1: Start with Default
```bash
python main.py \
  --config=configs/ve/brats_fista_tv.py \
  --workdir=./exp/fista_tv_default \
  --mode=eval \
  --eval_folder=eval
```

### Step 2: Tune Lambda (Most Important)

Try different `lambda_tv` values by editing `configs/ve/brats_fista_tv_tune.py`:

**Less smoothing** (preserve detail):
```python
sampling.lambda_tv = 0.0001
```

**Default**:
```python
sampling.lambda_tv = 0.001
```

**More smoothing** (remove noise):
```python
sampling.lambda_tv = 0.01
```

Run each:
```bash
python main.py \
  --config=configs/ve/brats_fista_tv_tune.py \
  --workdir=./exp/fista_tv_lambda_XXX \
  --mode=eval \
  --eval_folder=eval
```

### Step 3: Adjust Iterations if Needed

If results look under-converged, increase iterations:
```python
sampling.fista_max_iter = 200
```

### Step 4: Fine-tune TV Proximal Operator (Optional)

If TV regularization seems too weak:
```python
sampling.tv_prox_steps = 10  # More accurate TV
sampling.tv_prox_lr = 0.1    # Faster convergence
```

## Recommended Presets

### For Clean Data (Less Noise)
```python
sampling.lambda_tv = 0.0001
sampling.fista_max_iter = 100
sampling.tv_prox_steps = 5
sampling.tv_prox_lr = 0.05
```

### For Noisy Data (More Smoothing)
```python
sampling.lambda_tv = 0.01
sampling.fista_max_iter = 150
sampling.tv_prox_steps = 10
sampling.tv_prox_lr = 0.05
```

### For Fast Evaluation (Speed Priority)
```python
sampling.lambda_tv = 0.001
sampling.fista_max_iter = 50
sampling.tv_prox_steps = 3
sampling.tv_prox_lr = 0.1
```

### For High Quality (Quality Priority)
```python
sampling.lambda_tv = 0.001
sampling.fista_max_iter = 200
sampling.tv_prox_steps = 10
sampling.tv_prox_lr = 0.05
```

## Monitoring Results

After each run, check:
1. **PSNR/SSIM metrics**: Higher is better
2. **Visual quality**: Look at reconstructed images
3. **Runtime**: Balance quality vs speed

Results are saved in:
- Metrics: `{workdir}/{eval_folder}/host_0/metrics.npz`
- Images: `{workdir}/{eval_folder}/host_0/reconstructions.npz`

## Example Tuning Session

```bash
# 1. Baseline (default)
python main.py --config=configs/ve/brats_fista_tv.py \
  --workdir=./exp/fista_baseline --mode=eval --eval_folder=eval

# 2. Less regularization
# Edit config: lambda_tv = 0.0001
python main.py --config=configs/ve/brats_fista_tv_tune.py \
  --workdir=./exp/fista_low_reg --mode=eval --eval_folder=eval

# 3. More regularization  
# Edit config: lambda_tv = 0.01
python main.py --config=configs/ve/brats_fista_tv_tune.py \
  --workdir=./exp/fista_high_reg --mode=eval --eval_folder=eval

# 4. More iterations (if needed)
# Edit config: fista_max_iter = 200
python main.py --config=configs/ve/brats_fista_tv_tune.py \
  --workdir=./exp/fista_more_iter --mode=eval --eval_folder=eval

# Compare metrics from all runs
```

## Expected Performance Range

For BraTS MRI at 8× acceleration:
- **PSNR**: 28-33 dB (vs ~37-38 dB for learned methods)
- **SSIM**: 0.85-0.92 (vs ~0.95-0.96 for learned methods)
- **Runtime**: ~1-5 seconds per image on GPU (depending on iterations)

The gap between FISTA-TV and learned methods demonstrates the value of learning-based approaches!

