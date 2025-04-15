
import os, sys
import torch
import torch.nn as nn
from torchvision.transforms.functional import crop
from .calculator import ssim_compute, psnr_compute, lpips_compute, hist_fwtm, des_image, mi_compute
from .losses import VGGPerceptualLoss
import lpips
from torchvision.transforms.functional import resize

class Twin_Generator(nn.Module):

	def __init__(self, model, model_args=dict(), is_train=True, optim_args=dict()):

		super().__init__()
		self.gene_a2b = model(**model_args)
		self.gene_b2a = model(**model_args)

		if is_train:
			# losses
			self.L1 = nn.L1Loss(reduction="mean")
			self.percept_loss = VGGPerceptualLoss(chans=3, reduction='mean')
			#optimizer
			params = list(self.gene_a2b.parameters()) + list(self.gene_b2a.parameters())
			self.optimizer = torch.optim.AdamW(params, **optim_args)

	def set_eval(self):
		self.gene_a2b.eval()
		self.gene_b2a.eval()
		self.set_requires_grad(False)

	def set_lpips(self, device):
		net_lpips = lpips.LPIPS(net='vgg').to(device)
		net_lpips.requires_grad_(False)
		self.net_lpips = net_lpips

	def set_train(self):
		self.gene_a2b.train()
		self.gene_b2a.train()
		self.set_requires_grad(True)

	def set_requires_grad(self, requires_grad=False):
		nets = [self.gene_a2b,self.gene_b2a]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def get_loss(self, input, target):
		generate_b	= self.gene_a2b.forward(input)
		generate_a	= self.gene_b2a.forward(target)
		l1_loss = (self.L1(generate_b, target) + self.L1(generate_a, input)) * 0.5
		perc_loss = (self.percept_loss(generate_b, target) + self.percept_loss(generate_a, input)) * 0.5
		return [l for l in [l1_loss,perc_loss]]

	def get_cycle_loss(self, input, target):
		recon_a		= self.gene_b2a.forward(self.gene_a2b.forward(input))
		recon_b		= self.gene_a2b.forward(self.gene_b2a.forward(target))
		cy = (self.L1(recon_b, target) + self.L1(recon_a, input)) * 0.5
		return [l for l in [cy]]

	def _get_eval(self, model, input, target, img_size=None, fwtm_roi_r=None, fwtm_roi_l=None): #model=set.gene_a2b
		generate = model.forward(input)#.sample
		if not img_size==None:
			input,target,generate = resize(input,img_size),resize(target,img_size),resize(generate,img_size)
		gene_des_s = des_image(input, generate, 0.5, 16, log_tr_for_image_a=True, hist=False)
		orig_des_s = des_image(input, target, 0.5, 16, log_tr_for_image_a=True, hist=False)

		ssim = ssim_compute(generate, target)
		psnr = psnr_compute(generate, target)
		lpips = lpips_compute(generate, target, self.net_lpips)
	#	hist_w = histogram_percentile_width(generate, input)
		hist_w_r = hist_fwtm(crop(gene_des_s, *fwtm_roi_r)) if not fwtm_roi_r==None else 0
		hist_w_l = hist_fwtm(crop(gene_des_s, *fwtm_roi_l)) if not fwtm_roi_l==None else 0
		hist_w_org_r = hist_fwtm(crop(orig_des_s, *fwtm_roi_r)) if not fwtm_roi_r==None else 0
		hist_w_org_l = hist_fwtm(crop(orig_des_s, *fwtm_roi_l)) if not fwtm_roi_l==None else 0
		return generate, {
							'ssim': ssim, 'psnr': psnr, 'lpips': lpips, 
							'hist_w_r': hist_w_r, 'hist_w_l': hist_w_l, 
							'hist_w_org_r': hist_w_org_r, 'hist_w_org_l': hist_w_org_l
							}
	
	def get_eval(self, model, input, target, img_size=None, fwtm_roi_r=None, fwtm_roi_l=None): #model=set.gene_a2b
		generate = model.forward(input)#.sample
		if not img_size==None:
			input,target,generate = resize(input,img_size),resize(target,img_size),resize(generate,img_size)
		gene_des_s = des_image(input, generate, 0.5, 16, log_tr_for_image_a=True, hist=False)
		orig_des_s = des_image(input, target, 0.5, 16, log_tr_for_image_a=True, hist=False)

		ssim = ssim_compute(generate, target)
		psnr = psnr_compute(generate, target)
		lpips = lpips_compute(generate, target, self.net_lpips)
	#	fwtm = histogram_percentile_width(generate, input)
		fwtm_gen_r = hist_fwtm(crop(gene_des_s, *fwtm_roi_r)) if not fwtm_roi_r==None else 0
		fwtm_gen_l = hist_fwtm(crop(gene_des_s, *fwtm_roi_l)) if not fwtm_roi_l==None else 0
		fwtm_org_r = hist_fwtm(crop(orig_des_s, *fwtm_roi_r)) if not fwtm_roi_r==None else 0
		fwtm_org_l = hist_fwtm(crop(orig_des_s, *fwtm_roi_l)) if not fwtm_roi_l==None else 0

		#mutal info
		input_r = crop(input, *fwtm_roi_r) if not fwtm_roi_r==None else input
		input_l = crop(input, *fwtm_roi_l) if not fwtm_roi_l==None else input
		mi_org_r = mi_compute(input_r, crop(target, *fwtm_roi_r) if not fwtm_roi_r==None else target)
		mi_org_l = mi_compute(input_l, crop(target, *fwtm_roi_l) if not fwtm_roi_l==None else target)
		mi_gen_r = mi_compute(input_r, crop(generate, *fwtm_roi_r) if not fwtm_roi_r==None else generate)
		mi_gen_l = mi_compute(input_l, crop(generate, *fwtm_roi_l) if not fwtm_roi_l==None else generate)

		return generate, {
							'ssim': ssim, 'psnr': psnr, 'lpips': lpips,
							'fwtm':{'gen_r': fwtm_gen_r, 'gen_l': fwtm_gen_l, 'org_r': fwtm_org_r, 'org_l': fwtm_org_l},
							'mi':{'gen_r': mi_gen_r, 'gen_l': mi_gen_l, 'org_r': mi_org_r, 'org_l': mi_org_l},
							}

	def print_networks(self, name = "net_generator", save_dir=None):
		net = self.gene_a2b
		print('-Python---------')
		print(sys.version)
		print('-PyTorch---------')
		print(torch.__version__)
		print('-CUDA---------')
		print(torch.version.cuda)
		print('-CUDNN---------')
		print(torch.backends.cudnn.version()) #cudnn
		print('---------- Networks initialized -------------')
		print(net)
		if not save_dir==None:
			with open(os.path.join(save_dir, name +".txt"),"w") as o:
				print(net, sep=",", file=o) 
				print('-Python---------')
				print(sys.version)
				print('-PyTorch---------')
				print(torch.__version__)
				print('-CUDA---------')
				print(torch.version.cuda)
				print('-CUDNN---------')
				print(torch.backends.cudnn.version()) #cudnn
		print('-----------------------------------------------')
	
	def load_networks(self, load_dir, load_optim=False):
		checkpoint = torch.load(load_dir)
		self.gene_a2b.load_state_dict(checkpoint['gene_a2b_state_dict'])
		self.gene_b2a.load_state_dict(checkpoint['gene_b2a_state_dict'])
		epoch = checkpoint['epoch']
		best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else epoch
		loss_gene = checkpoint['loss_gene']
		best_loss_gene = checkpoint['best_loss_gene'] if 'best_loss_gene' in checkpoint else loss_gene
		if load_optim:
			self.optimizer.load_state_dict(checkpoint['optim_gene_state_dict'])
		return epoch, best_epoch, loss_gene, best_loss_gene

	def save_networks(self, save_dir, epoch, best_epoch, loss_gene, best_loss_gene, net_disc, loss_disc):
		torch.save({'epoch': epoch,
					'best_epoch': best_epoch,
					'gene_a2b_state_dict': self.gene_a2b.state_dict(),
					'gene_b2a_state_dict': self.gene_b2a.state_dict(),
					'optim_gene_state_dict': self.optimizer.state_dict(),
					'loss_gene': loss_gene,
					'best_loss_gene': best_loss_gene,
					'disc_a_state_dict': net_disc.disc_a.state_dict(),
					'disc_b_state_dict': net_disc.disc_b.state_dict(),
					'optim_disc_state_dict': net_disc.optimizer.state_dict(),
					'loss_disc': loss_disc,
					},
			save_dir)

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

class Unet8(torch.nn.Module):
	def __init__(self, in_channels=1, mid_channels=[64,128,256,512,512,512,512], num_drop_layers=3, device='cuda'): #'color,translation,cutout'
		super().__init__(
		)
		self.channels=[in_channels]
		self.channels.extend(mid_channels)
		conv_args = dict(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
		self.encoder = []
		self.down_block = []
		self.up_block = []

		def init_block(in_channels, out_channels, conv_args):
			return nn.Sequential(nn.Conv2d(in_channels, out_channels, **conv_args))
		def down_block(in_channels, out_channels, conv_args):
			return nn.Sequential(
					nn.LeakyReLU(negative_slope=0.2, inplace=True),
					nn.Conv2d(in_channels, out_channels, **conv_args),
					nn.InstanceNorm2d(out_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
					)
		
		self.encoder.append(init_block(self.channels[0],self.channels[1],conv_args).to(device))
		for n in range(1, len(self.channels)-1-3): # [1,64,128,256,512] for Basic Discriminator (70x70 PatchGAN)
			self.encoder.append(down_block(self.channels[n],self.channels[n+1],conv_args).to(device))

		for n in range(len(self.channels)-1-3,len(self.channels)-1):
			self.down_block.append(down_block(self.channels[n],self.channels[n+1],conv_args).to(device))

		def bottleneck_block(io_channels, conv_args, add_drop):
			return nn.Sequential(
					nn.LeakyReLU(negative_slope=0.2, inplace=True),
					nn.Conv2d(io_channels, io_channels, **conv_args),
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(io_channels, io_channels, **conv_args),
					nn.InstanceNorm2d(io_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
#					nn.Dropout(p=0.5, inplace=False) if add_drop else nn.Identity(),
					MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
					)
		self.bottleneck_block = [bottleneck_block(self.channels[-1],conv_args,False).to(device)]

		def up_block(in_channels, out_channels, conv_args, add_drop): # skip-connection
			return nn.Sequential(
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(in_channels*2, out_channels, **conv_args),
					nn.InstanceNorm2d(out_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
#					nn.Dropout(p=0.5, inplace=False) if add_drop else nn.Identity(),
					MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
					)
		def last_block(in_channels, out_channels, conv_args): # skip-connection
			return nn.Sequential(
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(in_channels*2, out_channels, **conv_args),
					nn.Tanh(),
					)

		self.up_block.insert(0, last_block(self.channels[1],self.channels[0],conv_args).to(device))
		for n in range(1, len(self.channels)-1-num_drop_layers): # last 3-block: addition drop_out layer
			self.up_block.insert(0, up_block(self.channels[n+1],self.channels[n],conv_args,False).to(device))
		for n in range(len(self.channels)-1-num_drop_layers,len(self.channels)-1):
			self.up_block.insert(0, up_block(self.channels[n+1],self.channels[n],conv_args,True).to(device))

		self.model = nn.Sequential(*self.encoder, *self.down_block, *self.bottleneck_block, *self.up_block)

	def forward(self, images):
		x = [images]
		for enc in self.encoder:
			x.append(enc(x[-1]))
		for dw in self.down_block:
			x.append(dw(x[-1]))
		y = self.bottleneck_block[0](x[-1])
		x.reverse()
		for n, up in enumerate(self.up_block):
			y = up(torch.cat([x[n], y], dim=1))

		return y

class MyDropout(nn.Module):
	def __init__(self, p=0.5, enabled=True):
		super().__init__()
		self.p = p
		self.enabled = enabled

	def forward(self, x):
		return nn.functional.dropout(x, p=self.p, training=self.enabled)


class Unet8_drop(torch.nn.Module):
	def __init__(self, in_channels=1, mid_channels=[64,128,256,512,512,512,512], num_drop_layers=3, device='cuda'): #'color,translation,cutout'
		super().__init__(
		)
		self.channels=[in_channels]
		self.channels.extend(mid_channels)
		conv_args = dict(kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
		self.encoder = []
		self.down_block = []
		self.up_block = []

		def init_block(in_channels, out_channels, conv_args):
			return nn.Sequential(nn.Conv2d(in_channels, out_channels, **conv_args))
		def down_block(in_channels, out_channels, conv_args):
			return nn.Sequential(
					nn.LeakyReLU(negative_slope=0.2, inplace=True),
					nn.Conv2d(in_channels, out_channels, **conv_args),
					nn.InstanceNorm2d(out_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
					)
		
		self.encoder.append(init_block(self.channels[0],self.channels[1],conv_args).to(device))
		for n in range(1, len(self.channels)-1-3): # [1,64,128,256,512] for Basic Discriminator (70x70 PatchGAN)
			self.encoder.append(down_block(self.channels[n],self.channels[n+1],conv_args).to(device))

		for n in range(len(self.channels)-1-3,len(self.channels)-1):
			self.down_block.append(down_block(self.channels[n],self.channels[n+1],conv_args).to(device))

		def bottleneck_block(io_channels, conv_args, add_drop):
			return nn.Sequential(
					nn.LeakyReLU(negative_slope=0.2, inplace=True),
					nn.Conv2d(io_channels, io_channels, **conv_args),
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(io_channels, io_channels, **conv_args),
					nn.InstanceNorm2d(io_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
					nn.Dropout(p=0.5, inplace=False) if add_drop else nn.Identity(),
					)
		self.bottleneck_block = [bottleneck_block(self.channels[-1],conv_args,False).to(device)]

		def up_block(in_channels, out_channels, conv_args, add_drop): # skip-connection
			return nn.Sequential(
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(in_channels*2, out_channels, **conv_args),
					nn.InstanceNorm2d(out_channels, eps=1e-08, momentum=0.1, affine=False, track_running_stats=False),
#					nn.Dropout(p=0.5, inplace=False) if add_drop else nn.Identity(),
					MyDropout(p=0.5, enabled=True) if add_drop else nn.Identity(),
					)
		def last_block(in_channels, out_channels, conv_args): # skip-connection
			return nn.Sequential(
					nn.ReLU(inplace=True),
					nn.ConvTranspose2d(in_channels*2, out_channels, **conv_args),
					nn.Tanh(),
					)

		self.up_block.insert(0, last_block(self.channels[1],self.channels[0],conv_args).to(device))
		for n in range(1, len(self.channels)-1-num_drop_layers): # last 3-block: addition drop_out layer
			self.up_block.insert(0, up_block(self.channels[n+1],self.channels[n],conv_args,False).to(device))
		for n in range(len(self.channels)-1-num_drop_layers,len(self.channels)-1):
			self.up_block.insert(0, up_block(self.channels[n+1],self.channels[n],conv_args,True).to(device))

		self.model = nn.Sequential(*self.encoder, *self.down_block, *self.bottleneck_block, *self.up_block)

	def forward(self, images):
		x = [images]
		for enc in self.encoder:
			x.append(enc(x[-1]))
		for dw in self.down_block:
			x.append(dw(x[-1]))
		y = self.bottleneck_block[0](x[-1])
		x.reverse()
		for n, up in enumerate(self.up_block):
			y = up(torch.cat([x[n], y], dim=1))

		return y