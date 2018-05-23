# RAM

Implementation of "Recurrent Models of Visual Attention" V. Mnih et al.

Modified from https://github.com/zhongwen/RAM   https://github.com/jlindsey15/RAM   

NOTICE: tf.stop_gradient should be applied to location sampled from Gaussian distribution parametered by (mean, std), 
so actor-critic reinforcement learning back propagate mean with (reward-tf.stop_gradient(baselines)) * loglikelihood 
to update loc network. Futhermore, baseline network is updated by (R-baseline)^2.

Run by `python ram.py` and it can reproduce the result on Table 1 (a) 28x28 MNIST
