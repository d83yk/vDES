import os
import torch
from torch.amp.grad_scaler import GradScaler
from statistics import mean
import shutil
from utils.train_aides_utils import PairedDataset, parse_args_options, save_args, set_seed, save_image
from networks.calculator import des_image
import questionary
from glob import glob
from torchvision.transforms.functional import resize

import warnings
warnings.simplefilter('ignore')

def main(args):

	scaler_g = scaler_d = GradScaler(device=args.device, enabled=args.use_amp)
	#scaler_d = GradScaler(device=args.device, enabled=args.use_amp)
	os.makedirs(args.output_folder, exist_ok=True)
	set_seed(args.seed)

	print(args.output_folder)
	save_args(args, os.path.join(args.output_folder, "train_args.json"))

	os.makedirs(args.eval_folder, exist_ok=True)
	os.makedirs(args.ckpt_folder, exist_ok=True)
	os.makedirs(args.train_model_folder, exist_ok=True)

	currentmodel_cpt = os.path.join(args.train_model_folder, 'latest.pth')
	bestmodel_cpt = os.path.join(args.train_model_folder, 'best.pth')
	loss_txt = os.path.join(args.ckpt_folder, 'loss.txt')

	if not os.path.exists(args.num_training_epochs_txt[0]):
		with open(args.num_training_epochs_txt[0], 'w') as f:
			print(args.num_training_epochs, file=f)

	dataset_train = PairedDataset(dataset_folder=args.dataset_folder, json_folder=args.json_folder, image_prep=args.train_image_prep, split="train", 
							   		mask_range=args.mask_range, filename_A=args.filename_A, filename_B=args.filename_B)
	dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
	dataset_valid = PairedDataset(dataset_folder=args.dataset_folder, json_folder=args.json_folder, image_prep=args.val_image_prep, split="val", 
							   		mask_range=args.mask_range, filename_A=args.filename_A, filename_B=args.filename_B, args=args)
	dl_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, shuffle=False, num_workers=0)

	saveimage_train = dataset_train.img_ids[0]
	os.makedirs(os.path.join(args.ckpt_folder, saveimage_train, 'ORG'), exist_ok=True)
	os.makedirs(os.path.join(args.ckpt_folder, saveimage_train, 'DES'), exist_ok=True)

	net_gene = args.generator
	net_gene.print_networks(name='net_generator',save_dir=args.train_model_folder)
	net_gene.set_lpips(args.device)
	net_gene.initialize_networks(init_type='xavier_uniform', device=args.device)

	if args.allow_tf32:
		torch.backends.cuda.matmul.allow_tf32 = True

	net_disc = args.discriminator
	net_disc.print_networks(name="net_discriminator", save_dir=args.train_model_folder)
	net_disc.initialize_networks(init_type='xavier_uniform', device=args.device)

	# continue_train
	continue_train = False
	next_epoch = best_epoch = 0
	loss_gene_best = float('inf')
	pths = glob('*.pth', root_dir=args.train_model_folder)
	if os.path.exists(currentmodel_cpt):
		continue_train = bool(int(questionary.select('continue train ? [1:continue | 0:new training]',
														choices=["1","0"]
														).ask()))
	if continue_train:
		load_pth = questionary.select('loading the model from xxx.pth ?',
														choices=[os.path.splitext(pt)[0]+'.pth' for pt in pths]
														).ask()
			
		next_epoch, best_epoch, loss_gene_epochs, loss_gene_best = net_gene.load_networks(os.path.join(args.train_model_folder, load_pth), load_optim=True)
		_		  , loss_disc = net_disc.load_networks(os.path.join(args.train_model_folder, load_pth), load_optim=True)
		
		print('\r epoch: [%3d/%3d] LossG:%s%.3f (Best: %.3f@%3d) LossD:%s%.3f' %(	next_epoch, args.num_training_epochs, 
														 		" "*(4-len(str(loss_gene_epochs).split('.')[0])), loss_gene_epochs,
														 		loss_gene_best, best_epoch,
														 		" "*(5-len(str(loss_disc).split('.')[0])), loss_disc))
		next_epoch += 1
	else:
		with open(loss_txt, 'w') as f:
			print('epoch,iter,lossG,lossGfwd,lossGadv,lossD,lossDadv-org,lossDadv-gen,lossDadv-mix,SSIM,PSNR,LPIPS,HistW(R,org-R,L,org-L)', file=f)

	total_iters = next_epoch*len(dataset_train)

	def loss_x_lambda(_loss, _lambda):
		return [loss_*lambda_ for (loss_, lambda_) in zip(_loss, _lambda)]
	def a_x_b_div_2(loss_a, loss_b):
		return [(_a + _b)*0.5 for (_a, _b) in zip(loss_a, loss_b)]

	for epoch in range(next_epoch, args.num_training_epochs+1):

		loss_gene_fwd_epochs = loss_gene_adv_epochs =\
		loss_disc_adv_org_epochs = loss_disc_adv_gen_epochs = loss_disc_adv_mix_epochs = 0.

		#training step
		net_gene.set_train()
		net_disc.set_train()
		for iter, batch in enumerate(dl_train):
			img_ids, inputs, targets, masks, _, _ = batch

			inputs = inputs.to(args.device)
			targets = targets.to(args.device)
			masks = masks.to(args.device)
			total_iters += inputs.shape[0]

			#Discriminator loss
			net_disc.set_requires_grad(True)

#		parser.add_argument("--lambda_gan", default=[1.0,0.5,0.5], type=float, help=("lambda array of ['original','generate','cutmix']"))
#		parser.add_argument("--lambda_disc", default=0.1, type=float, help=("lambda facror for discriminator balanced between G and D]"))
#		parser.add_argument("--lambda_cr", default=10., type=float) # consistency regulation factor for discriminator training
#		parser.add_argument("--lambda_l1", default=10., type=float)
#		parser.add_argument("--lambda_cy", default=100., type=float)
#		parser.add_argument("--lambda_percept", default=1., type=float)



			with torch.autocast(device_type=str(args.device), enabled=args.use_amp, dtype=args.mixed_precision):
				# Discriminator for task a2b and b2a (generate inputs)
				generate_b	= net_gene.gene_a2b.forward(inputs)
				generate_a	= net_gene.gene_b2a.forward(targets)
				
				loss_disc_adv_gen_a = net_disc.get_loss_for_D_gen(net_disc.disc_a, generate_a)
				loss_disc_adv_gen_a = loss_x_lambda(loss_disc_adv_gen_a, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_gen_b = net_disc.get_loss_for_D_gen(net_disc.disc_b, generate_b)
				loss_disc_adv_gen_b = loss_x_lambda(loss_disc_adv_gen_b, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_gen, loss_disc_cr_gen = a_x_b_div_2(loss_disc_adv_gen_a, loss_disc_adv_gen_b)

				# Discriminator for task a2b and b2a (original inputs)
				loss_disc_adv_org_a = net_disc.get_loss_for_D_org(net_disc.disc_a, inputs)
				loss_disc_adv_org_a = loss_x_lambda(loss_disc_adv_org_a, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_org_b = net_disc.get_loss_for_D_org(net_disc.disc_b, targets)
				loss_disc_adv_org_b = loss_x_lambda(loss_disc_adv_org_b, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_org, loss_disc_cr_org = a_x_b_div_2(loss_disc_adv_org_a, loss_disc_adv_org_b)

				# Discriminator for task a2b and b2a (cutmix inputs)
				loss_disc_adv_mix_a = net_disc.get_loss_for_D_gen(net_disc.disc_a, net_disc.cutmix_image(inputs, generate_a, masks))
				loss_disc_adv_mix_a = loss_x_lambda(loss_disc_adv_mix_a, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_mix_b = net_disc.get_loss_for_D_gen(net_disc.disc_b, net_disc.cutmix_image(targets, generate_b, masks))
				loss_disc_adv_mix_b = loss_x_lambda(loss_disc_adv_mix_b, [args.lambda_gan[0],args.lambda_cr])
				loss_disc_adv_mix, loss_disc_cr_mix = a_x_b_div_2(loss_disc_adv_mix_a, loss_disc_adv_mix_b)

			loss_disc = sum(loss_x_lambda([loss_disc_adv_org,loss_disc_adv_gen,loss_disc_adv_mix], args.lambda_gan))\
						+ sum(loss_x_lambda([loss_disc_cr_org,loss_disc_cr_gen,loss_disc_cr_mix], args.lambda_gan))
			loss_disc *= args.lambda_disc

			net_disc.optimizer.zero_grad()
			scaler_d.scale(loss_disc).backward()
			scaler_d.step(net_disc.optimizer)
			scaler_d.update()			

			loss_disc_adv_org_item = loss_disc_adv_org.item()
			loss_disc_adv_gen_item = loss_disc_adv_gen.item()
			loss_disc_adv_mix_item = loss_disc_adv_mix.item()
			loss_disc_cr_org_item = loss_disc_cr_org.item()
			loss_disc_cr_gen_item = loss_disc_cr_gen.item()
			loss_disc_cr_mix_item = loss_disc_cr_mix.item()

			loss_disc_adv_item = loss_disc.item()
			
			# Generator loss
			net_disc.set_requires_grad(False)

			# Generator GAN loss
			with torch.autocast(device_type=str(args.device), enabled=args.use_amp, dtype=args.mixed_precision):
				generate_b	= net_gene.gene_a2b.forward(inputs)
				generate_a	= net_gene.gene_b2a.forward(targets)

				loss_gene_adv_b = net_disc.get_loss_for_G(net_disc.disc_b, generate_b)
				loss_gene_adv_a = net_disc.get_loss_for_G(net_disc.disc_a, generate_a)
				loss_gene_adv = (loss_gene_adv_a + loss_gene_adv_b)*args.lambda_gan[0]*0.5
				
				# Generator forward loss
				loss_gene_fwd_each = net_gene.get_loss(inputs, targets)
				loss_gene_fwd_each = loss_x_lambda(loss_gene_fwd_each, [args.lambda_l1,args.lambda_percept])
				loss_gene_fwd = sum(loss_gene_fwd_each)

				# Generator cycle consistency loss
				loss_gene_cycle_each = net_gene.get_cycle_loss(inputs, targets)
				loss_gene_cycle_each = loss_x_lambda(loss_gene_cycle_each, [args.lambda_cy])
				loss_gene_cycle	=  sum(loss_gene_cycle_each)#loss_gene_cycle_each[0]*args.lambda_cy

			net_gene.optimizer.zero_grad()
			scaler_g.scale(loss_gene_adv+loss_gene_fwd+loss_gene_cycle).backward()
			scaler_g.step(net_gene.optimizer)
			scaler_g.update()

			loss_gene_adv_item = loss_gene_adv.item()
			loss_gene_fwd_item = loss_gene_fwd.item() + loss_gene_cycle.item()

			print('\r epoch: [%3d/%3d] iter: [%3d/%3d] LossG: %.3f (L1:%.2f Pcp:%.2f Cy:%.2f) Adv: %.3f(a:%.2f b:%.2f) LossD: %.3f(O:%.2f|%.2f G:%.2f|%.2f M:%.2f|%.2f)       ' %(\
				epoch, args.num_training_epochs,\
				iter*inputs.shape[0], len(dataset_train),\
				loss_gene_fwd_item, loss_gene_fwd_each[0].item(), loss_gene_fwd_each[1].item(), loss_gene_cycle_each[0].item(),\
				loss_gene_adv_item, loss_gene_adv_a.item(), loss_gene_adv_b.item(), 
				loss_disc_adv_item, loss_disc_adv_org_item, loss_disc_cr_org_item,
									loss_disc_adv_gen_item, loss_disc_cr_gen_item,
									loss_disc_adv_mix_item, loss_disc_cr_mix_item,
				), end='')

			for _id, _i, _t, _a, _b, _m in zip(img_ids, inputs, targets, generate_a, generate_b, masks):
#				print(_id)
				if saveimage_train in _id:
#					print(os.path.exists(os.path.join(args.ckpt_folder, saveimage_train, 'ORG')))
					save_image(	torch.cat([	torch.cat([_i,_t],dim=2),
						   					torch.cat([_a,_b],dim=2)],
											dim=1),
								os.path.join(args.ckpt_folder, saveimage_train, 'ORG', str(epoch).zfill(4)+'.jpg'))
					_st = des_image(_i, _b, 0.5, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=True)
					_bo = des_image(_i, _b, 1.0, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=True)
					save_image(	torch.cat([_st,_bo,_m],dim=2),
								os.path.join(args.ckpt_folder, saveimage_train, 'DES', str(epoch).zfill(4)+'.jpg'))

				loss_gene_fwd_epochs += loss_gene_fwd_item
				loss_gene_adv_epochs += loss_gene_adv_item
				loss_disc_adv_org_epochs += loss_disc_adv_org_item + loss_disc_cr_org_item
				loss_disc_adv_gen_epochs += loss_disc_adv_gen_item + loss_disc_cr_gen_item
				loss_disc_adv_mix_epochs += loss_disc_adv_mix_item + loss_disc_cr_mix_item

		loss_gene_fwd_epochs = loss_gene_fwd_epochs / len(dl_train)
		loss_gene_adv_epochs = loss_gene_adv_epochs / len(dl_train)
		loss_gene_epochs = abs(loss_gene_adv_epochs)+abs(loss_gene_fwd_epochs)

		loss_disc_adv_org_epochs = loss_disc_adv_org_epochs / len(dl_train)
		loss_disc_adv_gen_epochs = loss_disc_adv_gen_epochs / len(dl_train)
		loss_disc_adv_mix_epochs = loss_disc_adv_mix_epochs / len(dl_train)
		loss_disc_epochs = sum(loss_x_lambda([loss_disc_adv_org_epochs,loss_disc_adv_gen_epochs,loss_disc_adv_mix_epochs], args.lambda_gan))

		# save BEST
		if loss_gene_best > loss_gene_epochs:
			best_epoch = epoch
			loss_gene_best = loss_gene_epochs
			net_gene.save_networks(bestmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc_epochs)

		print('\r epoch: [%3d/%3d] LossG:%s%.3f(Fwd:%s%.3f Adv:%s%.3f) LossD:%s%.3f (Org:%s%.3f Gen:%s%.3f Mix:%s%.3f %s)' %(
					epoch, args.num_training_epochs, 
					" "*(4-len(str(loss_gene_epochs).split('.')[0])), loss_gene_epochs, 
					" "*(4-len(str(loss_gene_fwd_epochs).split('.')[0])), loss_gene_fwd_epochs, 
					" "*(4-len(str(loss_gene_adv_epochs).split('.')[0])), loss_gene_adv_epochs, 
					" "*(5-len(str(loss_disc_epochs).split('.')[0])), loss_disc_epochs,
					" "*(5-len(str(loss_disc_adv_org_epochs).split('.')[0])),loss_disc_adv_org_epochs,
					" "*(5-len(str(loss_disc_adv_gen_epochs).split('.')[0])),loss_disc_adv_gen_epochs,
					" "*(5-len(str(loss_disc_adv_mix_epochs).split('.')[0])),loss_disc_adv_mix_epochs,
					"/BEST/" if best_epoch==epoch else "",
					),end='')

		if epoch%args.ckpt_by_epochs == 0:
			net_gene.save_networks(currentmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc_epochs)
		if epoch%args.save_by_epochs == 0:
			shutil.copy2(currentmodel_cpt, os.path.join(args.train_model_folder, str(epoch).zfill(4)+'.pth'))

		evals = [	'img_id',
					'psnr','ssim','lpips', 'fwtm', 'mi',]
		roi_rl = ['org_r','org_l','gen_r','gen_l',]

		if epoch%args.eval_by_epochs==0 or best_epoch==epoch:
			# validation step
			net_gene.set_eval()
			net_disc.set_eval()

			with torch.no_grad():
				eval_val = dict()
				for e in evals:
					if e=='fwtm' or e=='mi':
						eval_val[e] = dict()
						for r in roi_rl:
							eval_val[e][r] = []
					else:
						eval_val[e] = []

				for iter, batch in enumerate(dl_valid):
					#img_id, input, target, _, _, _ = batch
					img_id, input, target, img_size, org_des_b, org_des_s = batch

					input = input.to(args.device, non_blocking=True)
					target = target.to(args.device, non_blocking=True)
					output, eval_v = net_gene.get_eval(net_gene.gene_a2b, input, target, img_size)
					input,target = resize(input,img_size),resize(target,img_size)

					gene_des_s = des_image(input, output, 0.5, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=args.hist_eq_for_des)
					gene_des_b = des_image(input, output, 1.0, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=args.hist_eq_for_des)
					#org_des_s = des_image(input, target, 0.5, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=args.hist_eq_for_des)
					#org_des_b = des_image(input, target, 1.0, 16, log_tr_for_image_a=args.log_tr_for_image_a, hist=args.hist_eq_for_des)

					for k, v in eval_v.items():
						if isinstance(v, dict):
							for r in roi_rl:
								eval_val[k][r].append(v[r])
						else: #ssim, psnr, lpips
							eval_val[k].append(v)
					
					for _id, _i, _t, _o, _gb, _gs, _ob, _os in zip(img_id, input, target, output, gene_des_b, gene_des_s, org_des_b, org_des_s):
						os.makedirs(os.path.join(args.eval_folder, _id, 'ORG'), exist_ok=True)
						os.makedirs(os.path.join(args.eval_folder, _id, 'DES'), exist_ok=True)
#						save_image(torch.cat([_gs,resize(_os,img_size),_gb,resize(_ob,img_size)],dim=0), os.path.join(args.eval_folder, _id, 'DES', '%.4d.tif' %(epoch)))
#						save_image(torch.cat([_i,_t,_o],dim=2), os.path.join(args.eval_folder, _id,'%.4d.tif' %(epoch)))
						save_image(_o, os.path.join(args.eval_folder, _id, 'ORG', '%.4d.tif' %(epoch)))
						save_image(torch.cat([_gs,_gb],dim=0), os.path.join(args.eval_folder, _id, 'DES', '%.4d.tif' %(epoch)))
						if not os.path.exists(os.path.join(args.eval_folder, _id, 'ORG','original.tif')):
							save_image(torch.cat([_i,_t],dim=0), os.path.join(args.eval_folder, _id, 'ORG','original.tif'))
						if not os.path.exists(os.path.join(args.eval_folder, _id, 'DES','org_des.tif')):
							save_image(torch.cat([resize(_os,img_size),resize(_ob,img_size)],dim=0), os.path.join(args.eval_folder, _id, 'DES','org_des.tif'))

				print(" /VAL/",end='')
				print(f" SSIM: %.3f" %(mean(eval_val['ssim'])),end='')
				print(f" PSNR: %.3f" %(mean(eval_val['psnr'])),end='')
				print(f" LPIPS: %.3f" %(mean(eval_val['lpips'])),end='')

				print(f" FWTM_g: %.3f|%.3f" %(mean(eval_val['fwtm']['gen_r']),mean(eval_val['fwtm']['gen_l'])),end='')
				print(f" FWTM_o: %.3f|%.3f" %(mean(eval_val['fwtm']['org_r']),mean(eval_val['fwtm']['org_l'])),end='')
				print(f" MI_g: %.3f|%.3f" %(mean(eval_val['mi']['gen_r']),mean(eval_val['mi']['gen_l'])),end='')
				print(f" MI_o: %.3f|%.3f" %(mean(eval_val['mi']['org_r']),mean(eval_val['mi']['org_l'])))

		print(" //                      ")
		with open(loss_txt, 'a') as f:
			print('%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f' %(epoch, total_iters,
										loss_gene_epochs, loss_gene_fwd_epochs, loss_gene_adv_epochs,
										loss_disc_epochs, loss_disc_adv_org_epochs, loss_disc_adv_gen_epochs,loss_disc_adv_mix_epochs,
										mean(eval_val['ssim']) if epoch%args.eval_by_epochs==0 else 0, 
										mean(eval_val['psnr']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['lpips']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['fwtm']['gen_r']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['fwtm']['org_r']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['fwtm']['gen_l']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['fwtm']['org_l']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['mi']['gen_r']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['mi']['org_r']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['mi']['gen_l']) if epoch%args.eval_by_epochs==0 else 0,
										mean(eval_val['mi']['org_l']) if epoch%args.eval_by_epochs==0 else 0,
										), file=f)
		
		with open(args.num_training_epochs_txt[0]) as f:
			num_training_epochs = int(f.readlines()[args.num_training_epochs_txt[1]])
			if epoch >= num_training_epochs:
				net_gene.save_networks(currentmodel_cpt, epoch, best_epoch, loss_gene_epochs, loss_gene_best, net_disc, loss_disc_epochs)
				print("epoch=%d > %d @%s done (saved latest.pth)" %(epoch,num_training_epochs,args.num_training_epochs_txt[0]))
				break


if __name__ == "__main__":
	args = parse_args_options(is_train=True)
	main(args)


