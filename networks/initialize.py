
import os
import torch
import torch.nn as nn
import functools
from .calculator import ssim_compute, psnr_compute, lpips_compute, norm11to01, norm01to11, histogram_percentile_width
from .losses import HistoSimLoss

from diffusers.models.autoencoders.vae import (
	Decoder,
	DiagonalGaussianDistribution,
	Encoder
)

from typing import Optional, Union, List

import lpips



	def _initialize_weights(self, net, init_type, init_gain):

		def init_func(m):  # define the initialization function
			classname = m.__class__.__name__
			if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
				match init_type:
					case 'normal':
						nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
					case 'xavier_uniform':
						nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
					case 'kaiming_uniform':
						nn.init.kaiming_uniform_(m.weight.data, mode="fan_in", nonlinearity="leaky_relu")
					case _:
						raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
					
				if hasattr(m, 'bias') and m.bias is not None:
					nn.init.constant_(m.bias.data, 0.0)

			elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
				nn.init.normal_(m.weight.data, 1.0, init_gain)
				nn.init.constant_(m.bias.data, 0.0)

		print('initialize network with %s' % init_type)
		net.apply(init_func)  # apply the initialization function <init_func>
	
	def initialize_networks(self, init_type='normal', init_gain=0.02, device='cuda'):
		if 'cuda' in str(device):
			assert(torch.cuda.is_available())
			self = self.to(device)
		self._initialize_weights(self.gene_a2b, init_type, init_gain)
		self._initialize_weights(self.gene_b2a, init_type, init_gain)
