class Config(object):
  glimpse_size = 8
  win_size = 8
  bandwidth = win_size**2
  batch_size = 32
  eval_batch_size = 32
  loc_std = 0.17
  original_size = 28
  num_channels = 1
  depth = 1
  sensor_size = glimpse_size**2 * depth * num_channels
  minRadius = 4
  hg_size = hl_size = 128
  g_size = 256
  cell_output_size = 256
  loc_dim = 2
  cell_size = 256
  cell_out_size = cell_size
  num_glimpses = 6
  num_classes = 10
  max_grad_norm = 5.

  step = 5000000
  lr_start = 1e-3
  lr_min = 1e-4

  # Monte Carlo sampling
  M = 10
  M_TEST = 1
