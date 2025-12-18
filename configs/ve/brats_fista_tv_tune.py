# coding=utf-8
"""FISTA-TV with tunable hyperparameters for BraTS MRI reconstruction.

Hyperparameters to tune:
- lambda_tv: TV regularization weight (0.0001 to 0.01)
- fista_max_iter: Number of FISTA iterations (50 to 200)
- tv_prox_steps: Number of gradient descent steps in TV prox (3 to 10)
- tv_prox_lr: Step size for TV proximal operator (0.01 to 0.1)
"""

from configs.default_cs_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training - not used for FISTA-TV but required by config structure
  training = config.training
  training.batch_size = 64
  training.n_iters = 2400001
  training.snapshot_sampling = True
  training.sde = 'vesde'
  training.continuous = True
  
  # eval
  evaluate = config.eval
  evaluate.batch_size = 128  # Can process in parallel
  evaluate.num_samples = 50000
  evaluate.ckpt_id = 26  # Not used for FISTA-TV
  
  # sampling
  sampling = config.sampling
  sampling.n_projections = 30  # 8× acceleration (240/30 = 8)
  sampling.task = 'mri'
  sampling.cs_solver = 'fista_tv'
  
  # ========================================
  # TUNABLE HYPERPARAMETERS
  # ========================================
  
  # TV regularization weight (higher = smoother)
  # Recommended range: [0.0001, 0.001, 0.01]
  sampling.lambda_tv = 0.001
  
  # Number of FISTA iterations
  # Recommended range: [500, 1000, 2000] - Based on CCPi defaults
  # For quick testing: [100, 200]
  sampling.fista_max_iter = 1000
  
  # TV proximal operator: number of gradient descent steps
  # Recommended range: [20, 50, 100] - Based on CCPi FGP_TV without warm start
  # For quick testing: [10, 20]
  sampling.tv_prox_steps = 50
  
  # TV proximal operator: step size
  # Recommended range: [0.005, 0.01, 0.02]
  sampling.tv_prox_lr = 0.01
  
  # data
  data = config.data
  data.dataset = 'brats'
  data.image_size = 240
  data.num_channels = 1
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False
  
  # model - not used for FISTA-TV but required by config structure
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 128.
  model.num_scales = 1000
  model.ema_rate = 0.999
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 32
  model.ch_mult = (1, 1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (30,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optim - not used for FISTA-TV
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
