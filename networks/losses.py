import torch
from torch import nn, tensor
from torch.nn import functional as F
from torchvision.models import vgg16
from torchist import histogram, normalize
import lpips


class GANLoss(nn.Module):
	def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.register_buffer('real_label', torch.tensor(target_real_label))
		self.register_buffer('fake_label', torch.tensor(target_fake_label))
		#self.gan_mode = gan_mode
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss() 
		elif gan_mode == 'srgan':
			self.loss = VGGContentLoss() # + vanilla gan loss

		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, pred, target_is_real):
		target_tensor = self.real_label if target_is_real else self.fake_label
		return target_tensor.expand_as(pred)

	def __call__(self, prediction, target_is_real):
		target_tensor = self.get_target_tensor(prediction, target_is_real)
		loss = self.loss(prediction, target_tensor)
		return loss.mean()

	
class mixGANLoss(nn.Module):
	def __init__(self, gan_mode):
		super(mixGANLoss, self).__init__()
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss() 
		else:
			raise NotImplementedError('gan mode %s not implemented' % gan_mode)

	def get_target_tensor(self, pred, label):
		if type(label)==torch.Tensor:
			return label.to(pred.device)
		elif type(label)==float:
			return torch.FloatTensor([label]).to(pred.device).expand_as(pred)
		else:
			raise NotImplementedError('[%s] is not use; available "float" or "torch.Tensor"' % type(label))

	def __call__(self, prediction, label):
		target_tensor = self.get_target_tensor(prediction, label)
		loss = self.loss(prediction, target_tensor)
		return loss.mean()


class VGGContentLoss(nn.Module):
	def __init__(self, chans=1, device="cuda", reduction='mean'):
		super(VGGContentLoss, self).__init__()
		self.reduction = reduction
		vgg = vgg16(pretrained=True)
		if chans==1:
			vgg_ = nn.Sequential(*list(vgg.children())[0]) #VGG16
			vgg_[0].weight = nn.Parameter(vgg_[0].weight.sum(dim=1).unsqueeze(1)) # single channel
		
		self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).to(device).eval()
		for param in self.contentLayers.parameters():
			param.requires_grad = False

	def forward(self, fake, real):
		content_loss = nn.functional.mse_loss(self.contentLayers(fake), self.contentLayers(real), reduction=self.reduction)
		return content_loss


class VGGPerceptualLoss(nn.Module):
	def __init__(self, chans=1, device="cuda", reduction='mean'):
		super(VGGPerceptualLoss, self).__init__()
		self.reduction=reduction
		vgg = vgg16(pretrained=True)
		if chans==1:
			vgg_ = nn.Sequential(*list(vgg.children())[0]) #VGG16
			vgg_[0].weight = nn.Parameter(vgg_[0].weight.sum(dim=1).unsqueeze(1)) # single channel

		blocks = []
		blocks.append(vgg.features[:4].eval())
		blocks.append(vgg.features[4:9].eval())
		blocks.append(vgg.features[9:16].eval())
		blocks.append(vgg.features[16:23].eval())
		blocks.append(vgg.features[23:30].eval())
		for bl in blocks:
			for p in bl:
				p.requires_grad = False
		self.blocks = nn.ModuleList(blocks).to(device)
		self.mean = nn.Parameter(tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), requires_grad=False).to(device) if chans==3 else\
					nn.Parameter(tensor([0.5,]).view(1, 1, 1, 1), requires_grad=False).to(device)
		self.std =	nn.Parameter(tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), requires_grad=False).to(device) if chans==3 else\
					nn.Parameter(tensor([0.5,]).view(1, 1, 1, 1), requires_grad=False).to(device)

	def forward(self, fake, real):
		x = (real - self.mean) / self.std
		y = (fake - self.mean) / self.std
		loss = 0.0
		for block in self.blocks:
			x = block(x)
			y = block(y)
			#nn.functional.mse_loss
			loss += nn.functional.l1_loss(x, y, reduction=self.reduction)
		return loss

