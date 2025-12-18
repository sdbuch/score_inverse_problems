# Solving Inverse Problems in Medical Imaging with Score-Based Generative Models

This repo contains the JAX code for experiments in the
paper [Solving Inverse Problems in Medical Imaging with Score-Based Generative Models](https://openreview.net/pdf?id=vaRCHVj0uGI)

by [Yang Song](https://yang-song.github.io)\*, [Liyue Shen](http://web.stanford.edu/~liyues/)\*, Lei Xing, and Stefano
Ermon. (*= joint first authors)

--------------------

We propose a general approach to solving linear inverse problems in medical imaging with score-based generative models.
Our method is purely generative, therefore does not require knowing the physical measurement process during training,
and can be quickly adapted to different imaging processes at test time without model re-training. We have demonstrated
superior performance on sparse-view computed tomography (CT), magnetic resonance imaging (MRI), and metal artifact
removal (MAR) in CT imaging.

## GPU Setup with Conda/Mamba

This repository requires old versions of JAX (0.2.18) and TensorFlow (2.5.0) that need CUDA 11 libraries. If your system has CUDA 12, follow these steps to set up a compatible environment:

### 1. Install Miniforge/Mamba

```bash
# Download and install Miniforge
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b

# Initialize conda
~/miniforge3/bin/conda init bash
source ~/.bashrc
```

### 2. Create Environment with CUDA 11

```bash
# Create environment with Python 3.9 and CUDA 11.8 libraries
mamba create -n score_gpu python=3.9 cudatoolkit=11.8 cudnn -c conda-forge -y
mamba activate score_gpu
```

### 3. Install Dependencies

```bash
# Install uv and project dependencies
pip install uv
uv sync --reinstall-package jaxlib

# Export library path so Python can find CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### 4. Run Evaluation

```bash
# Use specific GPU(s) - e.g., GPU 6 only
CUDA_VISIBLE_DEVICES=6 python main.py \
  --config=configs/ve/brats_ncsnpp_continuous.py \
  --workdir=./exp/brats_ncsnpp_ve \
  --mode=eval \
  --eval_folder=eval_test
```

**Note:** The code automatically uses all visible GPUs via JAX's `pmap`. Use `CUDA_VISIBLE_DEVICES` to control which GPUs are used.

### 5. Running FISTA-TV Classical Baseline

A classical optimization baseline (FISTA-TV) is also available for comparison with learning-based methods:

```bash
# Run FISTA-TV on BraTS MRI reconstruction (8× acceleration)
python main.py \
  --config=configs/ve/brats_fista_tv.py \
  --workdir=./exp/brats_fista_tv \
  --mode=eval \
  --eval_folder=eval_fista_tv
```

This baseline does not require a pretrained model checkpoint and uses classical total variation regularization.

--------------------

### Dependencies

See `requirements.txt`.

### Usage

Train and evaluate our models through `main.py`.

```sh
main.py:
  --config: Training configuration.
    (default: 'None')
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval|tune>: Running mode: train or eval or tune
  --workdir: Working directory
```

* `config` is the path to the config file. Our prescribed config files are provided in `configs/`. They are formatted
  according to [`ml_collections`](https://github.com/google/ml_collections) and should be mostly self-explanatory. `sampling.cs_solver` specifies which sampling method we use for solving the inverse problems. They have 5 possible values:
    * `baseline`: The "Score SDE" approach, as in the original paper [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
    * `langevin`: The "Langevin" approach, similar to the method in [Robust Compressed Sensing MRI with Deep Generative Priors](https://arxiv.org/abs/2108.01368) 
    * `langevin_projection`: The "ALD + Ours" approach, used to demonstrate advantages of the projection-based conditioning method compared to prior work.
    * `projection`: Our full method.
    * `fista_tv`: Classical FISTA-TV optimization baseline (learning-free).

* `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta
  checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is "train", "eval", or "tune". When set to "train", it starts the training of a new model, or resumes the
  training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist
  in `workdir/checkpoints-meta` . When set to "eval", it computes the PSNR/SSIM metrics on a test dataset. When set to "tune", it automatically tunes hyperparameters for the sampler with Bayesian optimization.

### Pretrained checkpoints

Checkpoints and test data are provided in
this [Google drive](https://drive.google.com/drive/folders/19G2zfKHX2ZCVh7H_B2BTPBNhMECZEE8H?usp=sharing). Please download the folder and move it to the same directory of this repo.

### References

If you find the code useful for your research, please consider citing

```bib
@inproceedings{
  song2022solving,
  title={Solving Inverse Problems in Medical Imaging with Score-Based Generative Models},
  author={Yang Song and Liyue Shen and Lei Xing and Stefano Ermon},
  booktitle={International Conference on Learning Representations},
  year={2022},
  url={https://openreview.net/forum?id=vaRCHVj0uGI}
}
```
and its prior work
```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```