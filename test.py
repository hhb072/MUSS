
import os
import argparse
import numpy as np
import random 
import time
from os.path import join
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataset import *

try:
    from ruamel import yaml
except:
    import yaml
from easydict import EasyDict as edict

from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

import torchvision.utils as vutils

import skimage
from skimage import io,transform
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from networks import MemoryAE

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def readlinesFromFile(path):
    print("Load from file %s" % path)        
    f=open(path)
    data = []
    for line in f:
      content = line.split()        
      data.append(content)      
          
    f.close()  
    return data

def enable_path(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def time2str(t):
    t = int(t)
    day = t // 86400
    hour = t % 86400 // 3600
    minute = t % 3600 // 60
    second = t % 60
    return "{:02d}/{:02d}/{:02d}/{:02d}".format(day, hour, minute, second) 

def get_config(parser):

    args = parser.parse_args()
    args = edict(vars(args))

    cfg_file_path = args.config_file

    with open(cfg_file_path, 'r') as stream:
        config = edict(yaml.safe_load(stream))

    config.update(args)
    return config
  
def compute_metrics(img, gt):
    img = img.numpy().transpose((0, 2, 3, 1))
    gt = gt.numpy().transpose((0, 2, 3, 1)) 
    img = img[0,:,:,:] * 255.
    gt = gt[0,:,:,:] * 255.
    img = np.array(img, dtype = 'uint8')
    gt = np.array(gt, dtype = 'uint8')
    # gt = skimage.color.rgb2ycbcr(gt)[:,:,0]
    # img = skimage.color.rgb2ycbcr(img)[:,:,0] 
    cur_psnr = compare_psnr(img, gt, data_range=255)
    cur_ssim = compare_ssim(img, gt, data_range=255, multichannel=True)
    return cur_psnr, cur_ssim       

def load_model(model, pretrained, strict=False):
    state = torch.load(pretrained)
    model.load_state_dict(state['model'], strict=strict)
    # print('\nBest: {}, {}, {}\n'.format(state['best'][0], state['best'][1]))
    return state['epoch'], state['best']
            
def save_checkpoint(model, best, epoch, iteration, prefix="", manualSeed=0):
    enable_path('model')
    if 'best' in prefix:
        model_out_path = "model/" + prefix +"_model_seed_{}.pth".format(manualSeed)
    else:    
        model_out_path = "model/" + prefix +"_model_epoch_{}_iter_{}_seed_{}.pth".format(epoch, iteration, manualSeed)
    state = {"epoch": epoch, "iter": iteration, "model": model.state_dict(), 'best':best}    
    torch.save(state, model_out_path)        
    print("Checkpoint saved to {}".format(model_out_path))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main(config):

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    cudnn.benchmark = True
    
    start_epoch = 0
    best_psnr  = 0
    best_ssim = 0
    
    model = MemoryAE(config).cuda()
    if config.pretrained:
        state = torch.load(config.pretrained)
        model.load_state_dict(state['model'])
    
    # torch.save(model.netG.state_dict(), 'tmp.pth')
    # exit()
    
    test_list = readlinesFromFile(config.testfiles)
      
    assert len(test_list) > 0
        
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    test_set = SingleImageDataset(test_list, config.testroot, crop_height=None, output_height=None, is_random_crop=False, is_mirror=False, normalize=normalize)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=int(config.workers))
    
    start_time = time.time()
    
    
    def test_real(epoch):
        psnrs, ssims = [], []
        end_time = time.time()
        save_image_path = join(config.save_image_path, 'real')
        enable_path(save_image_path)
    
        model.eval()
        lossesPixel = AverageMeter()        
        
        with torch.no_grad():
            for iteration, batch in enumerate(test_dataloader, 0):                
                data = batch
                if len(data.size()) == 3:
                    data = data.unsqueeze(0)               
                data = Variable(data).cuda()
                _, c, h, w = data.size()
                h1 = math.ceil(h / 8.) * 8
                w1 = math.ceil(w / 8.) * 8
                if h1 != h or w1 != w:
                    data = F.interpolate(data, (h1, w1), mode='bicubic')
                               
                fake = model.generate(data)
                if h1 != h or w1 != w:
                    fake = F.interpolate(fake, (h, w), mode='bicubic')
                               
                vutils.save_image(fake*0.5+0.5, '{}/{:03d}.png'.format(save_image_path, iteration+1))
               
      
   
    test_real(0)
                
            
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="configG.yaml", type=str, help='the path of config file')         

    main(get_config(parser))

