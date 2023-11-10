
import os
import argparse
import numpy as np
import random 
import time
from os.path import join

import copy

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

def readlinesFromFile(path, root=''):
    print("Load from file %s" % path)        
    f=open(path)
    data = []
    for line in f:
      content = line.split()
      content = [join(root, x) for x in content]
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
        config = edict(yaml.load(stream))

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
    enable_path('ckpt')       
    model_out_path = "ckpt/" + prefix +"_model_epoch_{}_iter_{}_seed_{}.pth".format(epoch, iteration, manualSeed)
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

def norm_range(x, margin):
    low = float(x.min())
    high = float(x.max())
    x = x.sub_(low).div_(max(high-low,1e-5))
    x = x.clamp_(min=margin)
    return x

def augment_data(data, label, real_data, pseudo_label, confidence_map):
    batchsize = data.shape[0]
    density = torch.rand(data.shape[0], 1, 1, 1).mul(0.5).add(0.5).cuda()    
    
    rainy0 = data.data - label.data
    rainy1 = real_data.data - pseudo_label.data
    rainy = torch.cat((rainy0, rainy1), dim=0)
    rainy = rainy[torch.randperm(rainy.shape[0])[:batchsize]]
    
    background = torch.cat((real_data.data, pseudo_label.data, data.data, label.data), dim=0)
    pseudo_label = torch.cat((pseudo_label, pseudo_label, label, label), dim=0)
    confidence_map = torch.cat((confidence_map, confidence_map, torch.ones_like(label), torch.ones_like(label)), dim=0)
    
    index = torch.randperm(background.shape[0])[:batchsize]
    background = background[index]
    pseudo_label = pseudo_label[index]
    confidence_map = confidence_map[index]
    
    noised_data = background + rainy * density
    noised_data = Variable(noised_data.clamp(-1, 1))
    
    return noised_data, pseudo_label, confidence_map
    
def augment_data2(data, label, real_data, pseudo_label, confidence_map):
    density = torch.rand(data.shape[0]*2+real_data.shape[0]*2, 1, 1, 1).mul(0.5).add(0.5).cuda()    
    rainy0 = data.data - label.data
    rainy1 = real_data.data - pseudo_label.data    
    rainy = torch.cat((rainy0, rainy0, rainy1, rainy1), dim=0)
    rainy = rainy[torch.randperm(rainy.shape[0])]
    
    background = torch.cat((real_data.data, pseudo_label.data, data.data, label.data), dim=0)
    pseudo_label = torch.cat((pseudo_label, pseudo_label, label, label), dim=0)
    confidence_map = torch.cat((confidence_map, confidence_map, torch.ones_like(label), torch.ones_like(label)), dim=0)
    
    noised_data = background + rainy * density
    noised_data = Variable(noised_data.clamp(-1, 1))
    
    return noised_data, pseudo_label, confidence_map    

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
    print(model)
    if config.pretrained:
        state = torch.load(config.pretrained)
        model.load_state_dict(state['model'], strict=False)
        start_epoch = state['epoch']
        best_psnr, best_ssim = state['best']
        print('\nBest: PSNR: {}, SSIM: {},\n'.format(state['best'][0], state['best'][1]))
        model.reset_moving_average()
        model.netT = model._get_target_network()
        
    
    optimizerG = optim.Adam(model.netG.parameters(), lr = config.learning_rate_g)
    optimizerD = optim.Adam(model.netD.parameters(), lr = config.learning_rate_d)
    

    syn_train_list_0 = readlinesFromFile(config.syn_files.split(',')[0], config.syn_trainroot.split(',')[0])
    syn_train_list_1 = readlinesFromFile(config.syn_files.split(',')[1], config.syn_trainroot.split(',')[1])
    max_len = max(len(syn_train_list_0), len(syn_train_list_1))
    syn_train_list = syn_train_list_0 * (max_len//len(syn_train_list_0)) + syn_train_list_1 * (max_len//len(syn_train_list_1))
    
    real_train_list = readlinesFromFile(config.real_files)
    real_train_list = real_train_list[:10000]
    test_list = readlinesFromFile(config.testfiles)
    assert len(syn_train_list) > 0 and len(real_train_list) > 0 and len(test_list) > 0
            
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    syn_train_set = DoubleImageDataset(syn_train_list, '', crop_height=config.output_size, output_height=config.output_size, is_random_crop=True, is_mirror=True, random_scale=config.random_scale, normalize=normalize, align_scale=config.align_scale, augment=config.data_augment)
    syn_train_dataloader = DataLoader(syn_train_set, batch_size=config.batchsize, shuffle=True, num_workers=int(config.workers))
            
    real_train_set = SingleImageDataset(real_train_list, config.real_trainroot, crop_height=config.output_size, output_height=config.output_size, is_random_crop=True, is_mirror=True, random_scale=config.random_scale, align_scale=config.align_scale,  normalize=normalize, augment=config.data_augment*int(len(syn_train_list)/len(real_train_list)))
    real_train_dataloader = DataLoader(real_train_set, batch_size=config.batchsize, shuffle=True, num_workers=int(config.workers))
    
    test_set = DoubleImageDataset(test_list, config.testroot, crop_height=config.test_output_size, output_height=config.test_output_size, is_random_crop=False, is_mirror=False, normalize=normalize)
    test_data_loader = DataLoader(test_set, batch_size=config.test_batchsize, shuffle=False, num_workers=int(config.workers))
           
    start_time = time.time()
    
    real_iter = iter(real_train_dataloader)
            
    def test(epoch):
        psnrs, ssims = [], []
        end_time = time.time()
        save_image_path = join(config.save_image_path, 'Epoch_{}'.format(epoch))
        enable_path(save_image_path)
    
        model.netG.eval()
        lossesPixel = AverageMeter()        
        
        with torch.no_grad():
            for iteration, batch in enumerate(test_data_loader, 0):                
                data, label = batch
                if len(data.size()) == 3:
                    data, label = data.unsqueeze(0), label.unsqueeze(0)                
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                
                fake = model.generate(data)
                                                
                loss_pixel = model.get_l1_loss(fake, label)
                                                
                lossesPixel.update(loss_pixel.data, fake.shape[0])
                
                data, label, fake = [x*0.5+0.5 for x in [data, label, fake]]
                
                fake, label = fake.data.cpu(), label.data.cpu()
                for i in range(fake.shape[0]):
                    psnr, ssim = compute_metrics(fake[i:i+1], label[i:i+1])
                    psnrs.append(psnr)
                    ssims.append(ssim)
                
                tensor = torch.cat((data.data.cpu(), label, fake), dim=0)
                vutils.save_image(tensor, '{}/{}.jpg'.format(save_image_path, iteration), nrow=4, padding=0)                                
                
                print('===> Test: Iteration: {}[{}], LossPixel: {:.4f}, '.format(iteration, len(test_data_loader), lossesPixel.val), flush=True)
               
        print('===> Test: Epoch: {}, LossPixel: {:.4f}, PSNR: {:.2f}, SSIM: {:.4f} '.format(epoch, lossesPixel.avg, np.mean(psnrs), np.mean(ssims)), flush=True)
        
        dense_psnrs, dense_ssims = np.mean(psnrs[:10]), np.mean(ssims[:10])
        sparse_psnrs, sparse_ssims = np.mean(psnrs[10:]), np.mean(ssims[10:])
                
        print('Dense:\tPSNR: {:.2f}, SSIM: {:.4f}'.format(dense_psnrs, dense_ssims), flush=True)
        print('Sparse:\tPSNR: {:.2f}, SSIM: {:.4f}'.format(sparse_psnrs, sparse_ssims), flush=True)
                
        model.netG.train()
        
        return np.mean(psnrs), np.mean(ssims) #dense_psnrs, dense_ssims
    
    def train_sl(epoch, best_psnr, best_ssim):
        model.netG.train()
        lossesPixel = AverageMeter()
        #lossesSSIM = AverageMeter()
        
        for iteration, batch in enumerate(syn_train_dataloader, 0):
            if iteration % config.test_iter == 0:
                psnr, ssim = test(epoch)
                if best_psnr < psnr:
                    best_psnr = psnr
                    save_checkpoint(model, [psnr, ssim], epoch, 0, 'bestPSNR', manualSeed)
                   
            
            # synthetic data
            data, label = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            batchsize = data.shape[0]                       
            
            preds = model.generate(data)
            
            lossL1 = model.get_l1_loss(preds, label)
                        
            loss = lossL1
           
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            
            lossesPixel.update(lossL1.data, batchsize)
            #lossesSSIM.update(lossSSIM.data, batchsize)
                                 
            if iteration % 10 == 0:
                info = "\n====> Epoch[{}]({}/{}): time: {}: ".format(epoch, iteration, len(syn_train_dataloader), time2str(time.time()-start_time))
                info += 'LossPixel: {:.4f}({:.4f}), '.format(lossesPixel.val, lossesPixel.avg)
                #info += 'LossSSIM: {:.4f}({:.4f}), '.format(lossesSSIM.val, lossesSSIM.avg)                
                print(info, flush=True)
                
            if iteration % 200 == 0:
                tensor = torch.cat((data[:8], label[:8], preds[:8]), dim=0)
                tensor = tensor * 0.5 + 0.5
                vutils.save_image(tensor.data.cpu(), '{}/training.jpg'.format(config.save_image_path), nrow=8, padding=0)
                
            
        return best_psnr, best_ssim 
    
    def train_semi(epoch, best_psnr, best_ssim):
        model.netG.train()
        model.netD.train()
        model.netT.eval()
        
        lossesPixel = AverageMeter()
        lossesPseudo = AverageMeter()
        lossesTV = AverageMeter()
        lossesDR = AverageMeter()
        lossesDF = AverageMeter()
        lossesG = AverageMeter()
                
        for iteration, batch in enumerate(syn_train_dataloader, 0):
            if iteration % config.test_iter == 0:
                psnr, ssim = test(epoch)
                if best_psnr < psnr:
                    best_psnr = psnr
                    save_checkpoint(model, [psnr, ssim], epoch, 0, 'bestPSNR', manualSeed)                
                if epoch > 0 and iteration % config.save_iter == 0:
                    save_checkpoint(model, [psnr, ssim], epoch, iteration, '', manualSeed)
            
            # synthetic data
            data, label = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            batchsize = data.shape[0]  

            preds = model.generate(data)            
            loss_pixel = model.get_l1_loss(preds, label)

            # real data
            try:
                real_data = real_iter.next()
            except:
                real_iter = iter(real_train_dataloader)
                real_data = real_iter.next()
            
            real_data = Variable(real_data).cuda()
            
            pseudo_label, confidence_map = model.produce_pseudo_label_with_uncertainty(real_data, config.uncertainty_times)
            confidence_map = norm_range(confidence_map, config.uncertainty_low)
                        
            if config.with_noise and batchsize == real_data.shape[0]:                            
                noised_data, pseudo_label, confidence_map = augment_data(data, label, real_data, pseudo_label, confidence_map)                
            else:
                noised_data = real_data
            
            real_preds = model.generate(noised_data)                        
            loss_pseudo = model.get_l1_loss_with_uncertainty(real_preds, pseudo_label, confidence_map)                
            loss_tv = model.get_tv_loss(real_preds)
            
            # update netD
            realD = model.discriminate(label)
            fakeD = model.discriminate(real_preds.detach())
            lossD_real = ((realD - 1)**2).mean()
            lossD_fake = (fakeD**2).mean()
            lossD = lossD_real + lossD_fake
            
            optimizerD.zero_grad()
            lossD.backward()
            optimizerD.step()
            
            # update netG
            fakeG = model.discriminate(real_preds)
            lossG = ((fakeG - 1)**2).mean()
            
            # total loss
            loss = loss_pixel * config.weight_pixel + (loss_pseudo * config.weight_pseudo + loss_tv * config.weight_tv + lossG * config.weight_adv) * config.weight_real
                       
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            
            model.update_moving_average()
            
            lossesPixel.update(loss_pixel.data, batchsize)
            batchsize = real_data.shape[0]
            lossesPseudo.update(loss_pseudo.data, batchsize)
            lossesTV.update(loss_tv.data, batchsize)
                        
            lossesDR.update(lossD_real.data, batchsize)
            lossesDF.update(lossD_fake.data, batchsize)
            lossesG.update(lossG.data, batchsize)
                                  
            if iteration % 10 == 0:
                info = "\n====> Epoch[{}]({}/{}): time: {}: ".format(epoch, iteration, len(syn_train_dataloader), time2str(time.time()-start_time))
                info += 'LossPixel: {:.4f}({:.4f}), '.format(lossesPixel.val, lossesPixel.avg)
                info += 'LossPseudo: {:.4f}({:.4f}), '.format(lossesPseudo.val, lossesPseudo.avg)
                info += 'LossTv: {:.4f}({:.4f}), '.format(lossesTV.val, lossesTV.avg)                   
                info += 'LossD: {:.4f}({:.4f}), {:.4f}({:.4f}), '.format(lossesDR.val, lossesDR.avg, lossesDF.val, lossesDF.avg) 
                info += 'LossG: {:.4f}({:.4f}), '.format(lossesG.val, lossesG.avg)   
                print(info, flush=True)
                
            if iteration % 200 == 0:
                tensor = torch.cat((data[:8], label[:8], preds[:8], noised_data[:8], pseudo_label[:8], real_preds[:8]), dim=0)
                tensor = tensor * 0.5 + 0.5
                vutils.save_image(tensor.data.cpu(), '{}/training.jpg'.format(config.save_image_path), nrow=8, padding=0)               
                vutils.save_image(confidence_map[:8].data.cpu(), '{}/confidence.jpg'.format(config.save_image_path), nrow=8, padding=0)
            
        return best_psnr, best_ssim 
    
    def train_st(epoch, best_psnr, best_ssim):
        model.netG.train()
        # model.netD.train()
        model.netT.eval()
        
        lossesPixel = AverageMeter()
        lossesPseudo = AverageMeter()
        lossesTV = AverageMeter()
                
        for iteration, batch in enumerate(syn_train_dataloader, 0):
                        
            if iteration % config.test_iter == 0:
                psnr, ssim = test(epoch)                
                if best_psnr < psnr:
                    best_psnr = psnr
                    save_checkpoint(model, [psnr, ssim], epoch, iteration, 'bestPSNR', manualSeed)
                    
            
            # synthetic data
            data, label = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
            batchsize = data.shape[0]  

            preds = model.generate(data)            
            loss_pixel = model.get_l1_loss(preds, label)

            # real data
            try:
                real_data = real_iter.next()
            except:
                real_iter = iter(real_train_dataloader)
                real_data = real_iter.next()
            
            real_data = Variable(real_data).cuda()
            
            # pseudo_label = model.produce_pseudo_label(real_data)
            pseudo_label, confidence_map = model.produce_pseudo_label_with_uncertainty(real_data, config.uncertainty_times)
            confidence_map = norm_range(confidence_map, config.uncertainty_low)
                        
            if config.with_noise and batchsize == real_data.shape[0]:                            
                noised_data, pseudo_label, confidence_map = augment_data(data, label, real_data, pseudo_label, confidence_map)
                real_preds = model.generate(noised_data)
            else:
                real_preds = model.generate(real_data)
            
                                    
            loss_pseudo = model.get_l1_loss_with_uncertainty(real_preds, pseudo_label, confidence_map)
                
            loss_tv = model.get_tv_loss(real_preds)
                        
            # total loss
            loss = loss_pixel * config.weight_pixel + (loss_pseudo * config.weight_pseudo + loss_tv * config.weight_tv ) * config.weight_real
                       
            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()
            
            model.update_moving_average()
            
            lossesPixel.update(loss_pixel.data, batchsize)
            batchsize = real_data.shape[0]
            lossesPseudo.update(loss_pseudo.data, batchsize)
            lossesTV.update(loss_tv.data, batchsize)
                                              
            if iteration % 10 == 0:
                info = "\n====> Epoch[{}]({}/{}): time: {}: ".format(epoch, iteration, len(syn_train_dataloader), time2str(time.time()-start_time))
                info += 'LossPixel: {:.4f}({:.4f}), '.format(lossesPixel.val, lossesPixel.avg)
                info += 'LossPseudo: {:.4f}({:.4f}), '.format(lossesPseudo.val, lossesPseudo.avg)
                info += 'LossTv: {:.4f}({:.4f}), '.format(lossesTV.val, lossesTV.avg)  
                print(info, flush=True)
                
            if iteration % 200 == 0:
                tensor = torch.cat((data[:8], label[:8], preds[:8], real_data[:8], real_preds[:8]), dim=0)
                if config.with_noise  and real_data.shape[0] == batchsize:
                    tensor = torch.cat((tensor, noised_data[:8]), dim=0)
                tensor = tensor * 0.5 + 0.5
                vutils.save_image(tensor.data.cpu(), '{}/training.jpg'.format(config.save_image_path), nrow=8, padding=0)
                vutils.save_image(confidence_map[:8].data.cpu(), '{}/confidence.jpg'.format(config.save_image_path), nrow=8, padding=0)
            
        return best_psnr, best_ssim 
    
      
    
    for epoch in range(start_epoch, config.total_epochs + 1): 
        if epoch < config.pretrain_epochs:
            best_psnr, best_ssim = train_sl(epoch, best_psnr, best_ssim)
        else:
            best_psnr, best_ssim = train_semi(epoch, best_psnr, best_ssim)  
            # best_psnr, best_ssim = train_st(epoch, best_psnr, best_ssim) 
            
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default="configG.yaml", type=str, help='the path of config file')         

    main(get_config(parser))

