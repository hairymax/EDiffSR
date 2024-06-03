import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse
import logging
import os.path
import sys
import time
from collections import OrderedDict
import torchvision.utils as tvutils

import numpy as np
import torch
from IPython import embed
##import lpips

import options as option
from models import create_model

sys.path.insert(0, "../../")
import utils as util
from data import create_dataloader, create_dataset
from data.util import bgr2ycbcr
from data.data_sampler import DistIterSampler

import torch.distributed as dist
import torch.multiprocessing as mp

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

def init_dist(backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != "spawn":  
        # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    rank = int(os.environ["RANK"])  # system env process ranks
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend, **kwargs
    )  # Initializes the default distributed process group

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, default="options/test/aid.yml", help="Path to options YMAL file.")
    parser.add_argument(
            "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"  # none means disabled distributed training
        )
    args = parser.parse_args()
    opt = option.parse(parser.parse_args().opt, is_train=False)

    opt = option.dict_to_nonedict(opt)

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist()
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True

    #### mkdir and logger
    if rank <= 0:
        util.mkdirs(
            (
                path
                for key, path in opt["path"].items()
                if not key == "experiments_root"
                and "pretrain_model" not in key
                and "resume" not in key
            )
        )

        # os.system("rm ./result")
        # os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

        util.setup_logger(
            "base",
            opt["path"]["log"],
            "test_" + opt["name"],
            level=logging.INFO,
            screen=True,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt["datasets"].items()):
        test_set = create_dataset(dataset_opt)
        if opt["dist"]:
            test_sampler = DistIterSampler(test_set, world_size, rank, 1)
        else:
            test_sampler = None
        test_loader = create_dataloader(test_set, dataset_opt, opt, test_sampler)
        if rank <= 0:
            logger.info(
                "Number of test images in [{:s}]: {:d}".format(
                    dataset_opt["name"], len(test_set)
                )
        )
        test_loaders.append(test_loader)

    # load pretrained model by default
    model = create_model(opt)
    device = model.device

    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], 
                     schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)

    sde.set_model(model.model)

    scale = opt['degradation']['scale']

    psnr = PeakSignalNoiseRatio(data_range=(0,1), reduction='sum', dim=(1,2,3)).to(rank)
    ssim = StructuralSimilarityIndexMeasure(data_range=(0,1), reduction='sum').to(rank)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze', reduction='sum', normalize=True).to(rank)
    fid = FrechetInceptionDistance(feature=64, normalize=True).to(rank)
    def calculate_fid(real_img, fake_img):
        fid.update(real_img, real=True)
        fid.update(fake_img, real=False)
        value = fid.compute()
        fid.reset()
        return value

    
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt["name"]  # path opt['']
        test_start_time = time.time()
        dataset_dir = os.path.join(opt["path"]["results_root"], test_set_name)
        # print(dataset_dir)
        if rank <= 0:
            logger.info("\nTesting [{:s}]...".format(test_set_name))
            util.mkdir(dataset_dir)

        avg_psnr = torch.tensor(0.0).to(rank)
        avg_ssim = torch.tensor(0.0).to(rank)
        avg_lpips = torch.tensor(0.0).to(rank)
        avg_fid = torch.tensor(0.0).to(rank)
        cnt = torch.tensor(0).to(rank) #if opt["dist"] else 0
        batch_cnt = torch.tensor(0).to(rank)
        
        for i, test_data in enumerate(test_loader):
            need_GT = False if test_loader.dataset.opt["dataroot_GT"] is None else True
            # img_path = test_data["GT_path"][0] if need_GT else test_data["LQ_path"][0]
            # img_name = os.path.splitext(os.path.basename(img_path))[0]

            #### input dataset_LQ
            LQ, GT = test_data["LQ"], test_data["GT"]
            LQ = util.upscale(LQ, scale)
            noisy_state = sde.noise_state(LQ)

            model.feed_data(noisy_state, LQ, GT)
            tic = time.time()
            model.test(sde, save_states=True)
            toc = time.time()
            
            fake_img = model.output.clamp(min=0, max=1)
            avg_psnr += psnr(model.state_0, fake_img)
            avg_ssim += ssim(model.state_0, fake_img)
            avg_lpips += lpips(model.state_0, fake_img)
            avg_fid += calculate_fid(model.state_0, fake_img)
            cnt += LQ.size(0)
            batch_cnt += 1
            
            if rank <= 0:
                logger.info(f"{i}, {toc - tic:.4f}, psnr: {avg_psnr / cnt:.4f}, ssim: {avg_ssim / cnt:.4f}, " 
                            f"lpips: {avg_lpips / cnt:.4f}, fid: {avg_fid / batch_cnt:.4f}")
            
            for j in range(LQ.size(0)):
                visuals = model.get_current_visuals(j)
                SR_img = visuals["Output"]
                output = util.tensor2img(SR_img.squeeze())  # uint8
                # LQ_ = util.tensor2img(visuals["Input"].squeeze())  # uint8
                # GT_ = util.tensor2img(visuals["GT"].squeeze())  # uint8
            
                suffix = opt["suffix"]
                save_name = dataset_dir + '/'+test_data['GT_path'][0].split('/')[-1]
                util.save_img(output, save_name)
            
        if opt["dist"]:
            dist.all_reduce(cnt)
            dist.all_reduce(batch_cnt)
            dist.all_reduce(avg_psnr)
            dist.all_reduce(avg_ssim)
            dist.all_reduce(avg_lpips)
            dist.all_reduce(avg_fid)
        
        if rank <= 0:    
            avg_psnr /= cnt
            avg_ssim /= cnt
            avg_lpips /= cnt
            avg_fid /= batch_cnt

        
        if rank <= 0:
            logger.info(f"\n\nResults: \n" 
                        f"psnr: {avg_psnr:.4f},\nssim: {avg_ssim:.4f},\n" 
                        f"lpips: {avg_lpips:.4f},\nfid: {avg_fid:.4f},\n" 
            )   

if __name__ == "__main__":
    main()