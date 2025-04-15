import os
import numpy as np
import torch

from statistics import mean
from networks.calculator import des_image 
from utils.train_aides_utils import PairedDataset, save_image, parse_args_options, save_args

import json, csv

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms.functional import resize

import warnings
warnings.simplefilter('ignore')

def main(args):
	batch_size = 1
	os.makedirs(args.test_folder, exist_ok=True)
	args.testmodel_pth = os.path.join(args.train_model_folder, 'latest.pth')
	print(args.testmodel_pth)
	
	dataset_test = PairedDataset(dataset_folder=args.dataset_folder, json_folder=args.json_folder, image_prep=args.test_image_prep, split=os.path.basename(args.test_folder),#"eval", 
							  mask_range=[0,0,100,100], filename_A=args.filename_A, filename_B=args.filename_B, args=args)
	dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

	net_gene = args.generator
	epoch,*loss = net_gene.load_networks(args.testmodel_pth, load_optim=False)

	args.output_image_folder = os.path.join(args.test_folder, 'epoch=%.4d' %(epoch))
	os.makedirs(os.path.join(args.output_image_folder, 'ORG'), exist_ok=True)
	os.makedirs(os.path.join(args.output_image_folder, 'DES'), exist_ok=True)

	net_gene.print_networks(save_dir=args.output_image_folder) # test_folder
	net_gene.set_eval()
	net_gene.set_lpips(args.device)

	fid_st = FrechetInceptionDistance(normalize=True)
	fid_bo = FrechetInceptionDistance(normalize=True)

	net_gene = net_gene.to(args.device)
	save_args(args, os.path.join(args.output_folder, "epoch=%.4d_%s_args.json" %(epoch,os.path.basename(args.test_folder))))

	with open(args.fwtm_roi_json, "r") as f:
		_fwtm_roi = json.load(f)
		fwtm_roi_r = _fwtm_roi["R"]
		fwtm_roi_l = _fwtm_roi["L"]

	evals = [	'img_id',
				'psnr','ssim','lpips', 'fwtm', 'mi',
				'fid_st','fid_bo']
	roi_rl = ['org_r','org_l','gen_r','gen_l',]

	# test step
	with torch.no_grad():
		eval_val = dict()
		for e in evals:
			if e=='fwtm' or e=='mi':
				eval_val[e] = dict()
				for r in roi_rl:
					eval_val[e][r] = []
			else:
				eval_val[e] = []

		org_st_imgs,gen_st_imgs = torch.empty(0), torch.empty(0)
		org_bo_imgs,gen_bo_imgs = torch.empty(0), torch.empty(0)
		print(args.output_image_folder)

		for _, batch in enumerate(sorted(dl_test)):
			img_id, input, target, img_size, org_des_bo, org_des_st = batch

			input = input.to(args.device, non_blocking=True)
			target = target.to(args.device, non_blocking=True)
			org_des_bo = resize(org_des_bo,img_size).to(args.device, non_blocking=True)
			org_des_st = resize(org_des_st,img_size).to(args.device, non_blocking=True)
			output, eval_v = net_gene.get_eval(net_gene.gene_a2b, input, target, img_size, 
									  fwtm_roi_r[img_id[0]] if img_id[0] in fwtm_roi_r.keys() else None,
									  fwtm_roi_l[img_id[0]] if img_id[0] in fwtm_roi_l.keys() else None,
									  )
			input,target = resize(input,img_size),resize(target,img_size)

			gen_des_st = des_image(input, output, 0.5, 16, log_tr_for_image_a=args.log_tr_for_image_a)
			gen_des_bo = des_image(input, output, 1.0, 16, log_tr_for_image_a=args.log_tr_for_image_a)

			org_st_imgs = to_fid_tensors(org_st_imgs,dataset_test.T(org_des_st))
			gen_st_imgs = to_fid_tensors(gen_st_imgs,dataset_test.T(gen_des_st))
			org_bo_imgs = to_fid_tensors(org_bo_imgs,dataset_test.T(org_des_bo))
			gen_bo_imgs = to_fid_tensors(gen_bo_imgs,dataset_test.T(gen_des_bo))

			for k, v in eval_v.items():
				if isinstance(v, dict):
					for r in roi_rl:
						eval_val[k][r].append(v[r])
				else: #ssim, psnr, lpips
					eval_val[k].append(v)

			for _id, _i, _t, _o, _gb, _gs, _ob, _os in zip(img_id, input, target, output, gen_des_bo, gen_des_st, org_des_bo, org_des_st):
				save_image(torch.cat([_i,_t,_o],dim=0), os.path.join(args.output_image_folder, 'ORG', _id+".tif"), dtype=np.uint16, dyn_range=2**12-1)
				save_image(torch.cat([_gs,_os,_gb,_ob],dim=0), os.path.join(args.output_image_folder, 'DES', _id+".tif"), dtype=np.uint16, dyn_range=2**12-1)
				eval_val['img_id'].append(_id)

			#print(f"%s" %(_id),end='')
			print(f" ID: %s" %(eval_val['img_id'][-1]),end='')
			print(f" SSIM: %.3f" %(eval_val['ssim'][-1]),end='')
			print(f" PSNR: %.3f" %(eval_val['psnr'][-1]),end='')
			print(f" LPIPS: %.3f" %(eval_val['lpips'][-1]),end='')

			print(f" FWTM_g: %.3f|%.3f" %(eval_val['fwtm']['gen_r'][-1],eval_val['fwtm']['gen_l'][-1]),end='')
			print(f" FWTM_o: %.3f|%.3f" %(eval_val['fwtm']['org_r'][-1],eval_val['fwtm']['org_l'][-1]),end='')
			print(f" MI_g: %.3f|%.3f" %(eval_val['mi']['gen_r'][-1],eval_val['mi']['gen_l'][-1]),end='')
			print(f" MI_o: %.3f|%.3f" %(eval_val['mi']['org_r'][-1],eval_val['mi']['org_l'][-1]))

		fid_st.update(org_st_imgs, real=True)
		fid_st.update(gen_st_imgs, real=False)
		eval_val['fid_st'].append(float(fid_st.compute()))
		fid_bo.update(org_bo_imgs, real=True)
		fid_bo.update(gen_bo_imgs, real=False)
		eval_val['fid_bo'].append(float(fid_bo.compute()))

		print("/MEAN/",end='')
		print(f" PSNR: %.3f" %(mean(eval_val['psnr'])),end='')
		print(f" SSIM: %.3f" %(mean(eval_val['ssim'])),end='')
		print(f" LPIPS: %.3f" %(mean(eval_val['lpips'])),end='')
		print(f" FWTM_g: %.3f|%.3f" %(mean(eval_val['fwtm']['gen_r']),mean(eval_val['fwtm']['gen_l'])),end='')
		print(f" FWTM_o: %.3f|%.3f" %(mean(eval_val['fwtm']['org_r']),mean(eval_val['fwtm']['org_l'])),end='')
		print(f" MI_g: %.3f|%.3f" %(mean(eval_val['mi']['gen_r']),mean(eval_val['mi']['gen_l'])),end='')
		print(f" MI_o: %.3f|%.3f" %(mean(eval_val['mi']['org_r']),mean(eval_val['mi']['org_l'])))
		print(f" FID_St: %.3f" %(mean(eval_val['fid_st'])),end='')
		print(f" FID_Bo: %.3f" %(mean(eval_val['fid_bo'])),end='')
		print(" //")

		with open(os.path.join(args.output_image_folder, "scores.csv"), 'w', newline="") as file:
			writer = csv.writer(file)
			for k in eval_val.keys():
				if isinstance(eval_val[k], dict):
					for r in roi_rl:
						writer.writerow([k+'-'+r, *eval_val[k][r]])
				else:
					writer.writerow([k, *eval_val[k]])



		print(args.test_folder)

def to_fid_tensors(img_list, image):
	img_list = torch.cat([img_list.cpu(),torch.cat([image,image,image],dim=1).cpu()], dim=0).detach().cpu()
	return img_list if (len(img_list.shape)>=4) else img_list.unsqueeze_(0)

if __name__ == "__main__":
	args = parse_args_options(is_train=False)
	main(args)


