import numbers
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from  torch.nn.parallel import data_parallel
from torch.nn.utils import spectral_norm
import torch.optim as optim

from timm.models.layers import trunc_normal_

from functools import wraps

from PIL import ImageFilter
import random

import torchvision

from torch.autograd import Variable

from thop import profile

import time

from SSIM import SSIM

str_to_list = lambda x: [int(xi) for xi in x.split(',')]

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn
 
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# use InfoNCE来约束特征的diversity

class _L1_Uncertainty_loss(nn.Module):
    def __init__(self, ):
        super().__init__()
                
    def forward(self, x, y, c):
        loss = F.l1_loss(x, y, reduction='none') 
        return (loss * c).mean()

class _TV_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        return (self.criterion(x[:,:,1:], x[:,:,:-1]) + self.criterion(x[:,:,:,1:], x[:,:,:,:-1]))/2

class _KL_Div_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.kl = nn.KLDivLoss(reduction='sum')
        
    def forward(self, x, y, dim):
        x = F.log_softmax(x, dim=dim)
        y = F.softmax(y, dim=dim)
        return self.kl(x, y)
        
class _Entropy_Loss(nn.Module):
    def __init__(self):
        super().__init__()               
        
    def forward(self, x, dim):
        c = x.shape[dim]
        # sm = F.softmax(x, dim=dim)
        lsm = F.log_softmax(x, dim=dim)
        sm = lsm.exp()
        return - (sm * lsm).mean().mul(c)

class _Diverse_Loss(nn.Module):
    def __init__(self):
        super().__init__()               
        
    def forward(self, x, dim):
        for i in range(len(x.shape)):
            if i != dim:
                x = x.mean(dim=i, keepdim=True)
        x = x.squeeze()
        sm = F.softmax(x, dim=0)
        lsm = F.log_softmax(x, dim=0)
        return (sm * lsm).sum()

class _Memory_Block(nn.Module):        
    def __init__(self, hdim, kdim, max_T=1., min_T=0.03, decay_alpha=0.99998, moving_average_rate=0.999):
        super().__init__()
        
        self.c = hdim
        self.k = kdim
        
        self.moving_average_rate = moving_average_rate
                
        self.units = nn.Embedding(kdim, hdim)
        
        
        self.max_T = max_T
        self.min_T = min_T
        self.decay_alpha = decay_alpha
        self.register_buffer('cur_T', torch.tensor(max_T))
                
    def update(self, x, score, m=None):
        '''
            x: (n, c)
            e: (k, c)
            score: (n, k)
        '''
        if m is None:
            m = self.units.weight.data
        x = x.detach()
        embed_ind = torch.max(score, dim=1)[1] # (n, )
        embed_onehot = F.one_hot(embed_ind, self.k).type(x.dtype) # (n, k)
        # embed_onehot = embed_onehot * score
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = x.transpose(0, 1) @ embed_onehot # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_average_rate + embed_mean.t() * (1 - self.moving_average_rate)
        if self.training:
            self.units.weight.data = new_data
        return new_data
                
    def forward(self, x):
        '''
          x: (b, c, h, w)
          embed: (k, c)
        '''
        
        if self.training and self.cur_T > self.min_T:
            _cur_T = self.cur_T
        else:
            _cur_T = self.min_T
        
        b, c, h, w = x.size() 
        # print([c, self.c])
        assert c == self.c        
        k, c = self.k, self.c
        
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, c) # (n, c)
        
        m = self.units.weight.data # (k, c)
                
        xn = F.normalize(x, dim=1) # (n, c)
        mn = F.normalize(m, dim=1) # (k, c)
        score = torch.matmul(xn, mn.t()) # (n, k)
        
        if self.training:
            # for i in range(1):
            m = self.update(x, score, m)
            mn = F.normalize(m, dim=1) # (k, c)
            score = torch.matmul(xn, mn.t()) # (n, k)
        
        # soft_label = F.softmax(score, dim=1)
        gumbel_label = F.gumbel_softmax(score, tau=_cur_T, hard=True, eps=1e-8)
        # gumbel_label = (gumbel_label - soft_label).detach() + soft_label
        # print(gumbel_label)
        out = torch.matmul(gumbel_label, m) # (n, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)
        
        if self.training:
            self.cur_T = self.cur_T * self.decay_alpha
                                   
        return out, score

class Discriminator(nn.Module): # 用biggan/stylegan2的判别器
    def __init__(self, ndf):
        super().__init__()
        
        layers = []
        
        layers += [nn.Sequential(
                    spectral_norm(nn.Conv2d(3, ndf, 5, 2, 2, bias=False)),
                    nn.BatchNorm2d(ndf),
                    nn.LeakyReLU(0.02, True),
                )]
                
        layers += [nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf, ndf, 5, 2, 2, bias=False)),
                    nn.BatchNorm2d(ndf),
                    nn.LeakyReLU(0.02, True),
                )]
                
        layers += [nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf, ndf*2, 5, 2, 2, bias=False)),
                    nn.BatchNorm2d(ndf*2),
                    nn.LeakyReLU(0.02, True),
                )]
        
        layers += [nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf*2, ndf*4, 5, 2, 2, bias=False)),
                    nn.BatchNorm2d(ndf*4),
                    nn.LeakyReLU(0.02, True),
                )]
        
        
        self.layers = nn.ModuleList(layers)
        
        self.predict = spectral_norm(nn.Conv2d(ndf*4, 1, 1, 1, 0, bias=False))
                              
    def forward(self, x):
        # feats = []
        for layer in self.layers:
            x = layer(x)
            # feats += [x]
            
        y = self.predict(x)
        return y

class _Split_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.mask = nn.Conv2d(dim, 1, 1, 1, 0, bias=False)
    
    def forward(self, x):
        mask = torch.sigmoid(self.mask(x))        
        x_m = x * mask
        x_s = x - x_m
        return x_m, x_s

class _Fuse_Block(nn.Module):
    def __init__(self, x_dim, skip_dim, mem_dim):
        super().__init__()
        
        self.norm = nn.InstanceNorm2d(x_dim, affine=False)
        
        self.alpha = nn.Conv2d(x_dim, 1, 1, 1, 0, bias=False)
        
        self.gamma_skip = nn.Conv2d(skip_dim, x_dim, 1, 1, 0, bias=False)
        self.beta_skip = nn.Conv2d(skip_dim, x_dim, 1, 1, 0, bias=False)
        
        self.gamma_mem = nn.Conv2d(mem_dim, x_dim, 1, 1, 0, bias=False)
        self.beta_mem = nn.Conv2d(mem_dim, x_dim, 1, 1, 0, bias=False)
        
        # self.refine = self.Conv2d(x_dim, x_dim, 3, 1, 1, bias=False)
        
    def forward(self, x, skip, mem):
                
        x = self.norm(x)
        
        alpha = torch.sigmoid(self.alpha(x))
        
        gs = self.gamma_skip(skip) + 1
        bs = self.beta_skip(skip)
        
        gm = self.gamma_mem(mem) + 1
        bm = self.beta_mem(mem)
        
        xs = x * gs + bs
        xm = x * gm + bm
        
        y = alpha * xs + (1 - alpha) * xm
        
        # y = self.refine(y)
        # y = torch.relu(y)
        
        return y

class _Encoder_layer(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, wide_width=True, downsample=False):
        super().__init__()
        
        self.downsample = downsample
                
        middle_planes = (in_planes if in_planes > out_planes else out_planes) if wide_width else out_planes
        
        self.conv_1 = nn.Conv2d(in_planes, middle_planes, 3, 1, 1, bias=False, groups=groups)        
        self.relu_1 = nn.LeakyReLU(0.02, inplace=True)        
        self.conv_2 = nn.Conv2d(middle_planes, out_planes, 3, 1, 1, bias=False, groups=groups)       
        
        if in_planes != out_planes:            
            self.translation = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False, groups=groups)
        else:
            self.translation = None
        
            
    def forward(self, x):
           
        identity = x

        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)
        
        if self.translation is not None:
            identity = self.translation(identity)
        
        out = out + identity
                
        if self.downsample:
            out = F.avg_pool2d(out, 2)
                    
        return out

def resize(src, tgt, mode='nearest'):
    if src.shape[-1] != tgt.shape[-1] or src.shape[-2] != tgt.shape[-2]:
        src = F.interpolate(src, (tgt.shape[-2], tgt.shape[-1]), mode=mode)
    return src
        
class _Decoder_layer(nn.Module):
    def __init__(self, in_planes, out_planes, groups=1, wide_width=True, upsample=False):
        super().__init__()
        
        self.upsample = upsample
        
        middle_planes = (in_planes if in_planes > out_planes else out_planes) if wide_width else out_planes
        
        
        self.conv_1 = nn.Conv2d(in_planes, middle_planes, 3, 1, 1, bias=False, groups=groups)        
        self.relu_1 = nn.LeakyReLU(0.02, inplace=True)
        
        self.conv_2 = nn.Conv2d(middle_planes, out_planes, 3, 1, 1, bias=False, groups=groups)       
        
        if in_planes != out_planes:            
            self.translation = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False, groups=groups)
        else:
            self.translation = None
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
        
            
        identity = x
               
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        
        if self.translation is not None:
            identity = self.translation(identity)
        
        x = x + identity
        return x

class _Mem_Res_layer(nn.Module):
    def __init__(self, in_planes, out_planes, kdim, max_T=1., min_T=0.03, decay_alpha=0.99998, moving_average_rate=0.999, wide_width=True, downsample=False, upsample=False, with_memory=False):
        super().__init__()
        
        self.downsample = downsample
        self.upsample = upsample
        self.with_memory = with_memory
                
        middle_planes = (in_planes if in_planes > out_planes else out_planes) if wide_width else out_planes
        
        self.conv_1 = nn.Conv2d(in_planes, middle_planes, 3, 1, 1, bias=False)        
        self.relu_1 = nn.LeakyReLU(0.02, inplace=True)        
        self.conv_2 = nn.Conv2d(middle_planes, out_planes, 3, 1, 1, bias=False)  
        
        if with_memory:
            self.mem = _Memory_Block(in_planes, kdim, max_T=max_T, min_T=min_T, decay_alpha=decay_alpha, moving_average_rate=moving_average_rate)
            self.alpha = nn.Parameter(torch.zeros(1))
        
        if in_planes != out_planes:            
            self.translation = nn.Conv2d(in_planes, out_planes, 1, 1, 0, bias=False)
        else:
            self.translation = None
        
            
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            
        identity = x
        
        if self.with_memory:
            mem, _ = self.mem(x)            
            x = x + mem * self.alpha
           
        
        out = self.conv_1(x)
        out = self.relu_1(out)
        out = self.conv_2(out)        
                
        if self.translation is not None:
            identity = self.translation(identity)
        
        out = out + identity
                
        if self.downsample:
            out = F.avg_pool2d(out, 2)
         
                    
        return out
      
class NetG(nn.Module):
    def __init__(self, ngf, delta, num_blocks, num_scales, num_layers, kdim, max_T, min_T, decay_alpha, moving_average_rate):
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        
        self.head = nn.Conv2d(3, ngf, 3, 1, 1, bias=False)
        
        cc = ngf
        self.enc = nn.ModuleDict()
        for i in range(num_blocks):
            self.enc['enc{}'.format(i)] = nn.ModuleList([_Mem_Res_layer(cc if j==0 else cc+delta, cc+delta, kdim, max_T=max_T, min_T=min_T, decay_alpha=decay_alpha, moving_average_rate=moving_average_rate, downsample=(j==0 and i < num_scales), with_memory=(j==0 or i >= num_scales)) for j in range(num_layers)])            
            cc += delta 
        
        self.dec = nn.ModuleDict()
        self.fuse = nn.ModuleDict()
        for i in range(num_blocks):              
            self.dec['dec{}'.format(i)] = nn.ModuleList([_Mem_Res_layer(cc if j ==0 else cc-delta, cc-delta, kdim, max_T=max_T, min_T=min_T, decay_alpha=decay_alpha, moving_average_rate=moving_average_rate, upsample=(j==0 and i > num_blocks - 1 - num_scales), with_memory=(j==0 or i <= num_blocks - 1 - num_scales)) for j in range(num_layers)]) 
            cc -= delta            
            if i < num_blocks-1:
                self.fuse['fuse{}'.format(i)] = nn.Conv2d(cc*2, cc, 1, 1, 0, bias=False)
                                      
        self.tail = nn.Conv2d(ngf, 3, 3, 1, 1)
            
    def forward(self, x):
        x = self.head(x)
        xi = x
        
        res = []
        for i in range(self.num_blocks):
            for layer in self.enc['enc{}'.format(i)]:
                x = layer(x)
            # print(x.shape)
            res.append(x)
        
        # print('================')
                
        res = res[::-1]
        for layer in self.dec['dec0']:
            x = layer(x)
        for i in range(self.num_blocks-1):            
            x = self.fuse['fuse{}'.format(i)](torch.cat((x, res[i+1]), dim=1))
            for layer in self.dec['dec{}'.format(i+1)]:
                x = layer(x)
        
        x = xi - x
        x = self.tail(x)
        x = torch.tanh(x)
        return x
      
class MemoryAE(nn.Module):
    def __init__(self, config):
        super().__init__() 

        self.netG = NetG(config.ngf, config.delta, config.num_blocks, config.num_scales, config.num_layers, config.kdim, config.max_T, config.min_T, config.decay_alpha, config.moving_average_rate)
        
        # print(self.netG)
                    
        self.netD = Discriminator(config.ndf)
        
        self.netT = None
        self.ema_updater = EMA(config.moving_average_decay)
        self.init_netT(torch.zeros(1,3,32,32))
                
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l1_un = _L1_Uncertainty_loss()
        self.criterion_l2 = nn.MSELoss()
        self.criterion_kl = _KL_Div_Loss()
        self.criterion_entropy = _Entropy_Loss()
        self.criterion_diverse = _Diverse_Loss() 
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_tv = _TV_Loss()
        self.SSIM = SSIM()
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        pass
        
    def generate(self, x):
        return data_parallel(self.netG, x)
        
    def discriminate(self, x):
        return data_parallel(self.netD, x)
            
    @singleton('netT')
    def _get_target_network(self):
        netT = copy.deepcopy(self.netG)
        return netT

    def reset_moving_average(self):
        del self.netT
        self.netT = None

    def update_moving_average(self):
        assert self.netT is not None, 'target network has not been created yet'
        update_moving_average(self.ema_updater, self.netT, self.netG)
    
    def init_netT(self, x):
        with torch.no_grad():
            self.netT = self._get_target_network()
            return self.netT(x)
    
    def produce_pseudo_label(self, x):
        with torch.no_grad():            
            return data_parallel(self.netT, x)
            
    def produce_pseudo_label_with_uncertainty(self, x, repeat=1):
        b, c, h, w = x.shape
        with torch.no_grad():             
            # x = x.repeat(repeat, 1, 1, 1)
            # y = data_parallel(self.netT, x)
            # y = y.view(repeat, b, c, h, w)
            yy = []
            for _ in range(repeat):
                y = data_parallel(self.netT, x)
                yy += [y.unsqueeze(0)]
            y = torch.cat(yy, dim=0)
                
            var, mean = torch.var_mean(y, dim=0, unbiased=False)
           
            return mean, var.mul_(-1).exp_()
                   
    def get_l1_loss(self, x, y):
        return data_parallel(self.criterion_l1, (x, y)).mean()
        
    def get_l1_loss_with_uncertainty(self, x, y, c):        
        return data_parallel(self.criterion_l1_un, (x, y, c)).mean()
     
    def get_l2_loss(self, x, y):
        return data_parallel(self.criterion_l2, (x, y)).mean()
        
    def get_kl_loss(self, clean, rainy, dim=1):
        '''
            clean: score from clean images, [b, k, h, w]
            rainy: score from rainy images, [b, k, h, w]
        '''
        return data_parallel(self.criterion_kl, (clean, rainy, dim)).mean()
        
    def get_entropy_loss(self, x, dim=1):
        return data_parallel(self.criterion_entropy, (x, dim)).mean()
        
    def get_tv_loss(self, x):
        return data_parallel(self.criterion_tv, x).mean()
        
    def get_div_loss(self, x, dim=1):
        return data_parallel(self.criterion_diverse, (x, dim)).mean()
    
    def get_ssim_loss(self, x, y):
        return 1 - data_parallel(self.SSIM, (x, y)).mean()
            
def test():
    # model = NetG(ngf=16, delta=16, num_layers=20, change_channel_layers='0,4,8,12,16', resize_layers='0,4,8', kdim=512, max_T=1., min_T=0.1, decay_alpha=0.99998, moving_average_rate=0.999)
    model = NetG(ngf=16, delta=16, num_blocks=5, num_scales=3, num_layers=4, kdim=512, max_T=1., min_T=0.1, decay_alpha=0.99998, moving_average_rate=0.999)
    print(model)         
    model.eval()
    
    total = sum([param.nelement() for param in model.parameters()])
    
    
    input = torch.randn(1, 3, 256, 256).cuda()
    model.cuda()
    
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            out = model(input)
    print((time.time()-t0)/100)
    
    # out = model(input)
    
    flops, params = profile(model, inputs = (input,))   
    
    # print(model)
    
    print("Number of parameter: %.4fM" % (total / 1e6))
    print("FLOPS: %.4G" % (flops/1e9))
    
            
if __name__ == '__main__':
    test()  


