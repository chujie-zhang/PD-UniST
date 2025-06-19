#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:43:06 2024

@author: zcj
"""

import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
#from .Dsmap_layers import *
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .trans_layers import Attention, CrossAttention, LayerNorm, Mlp, PreNorm
from transformers import AutoTokenizer, AutoModel
from einops import rearrange, repeat
from copy import deepcopy

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    


        

    

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size=1, temperature=1):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
 
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        #denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        a = torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool).float()
        b =torch.exp(similarity_matrix / self.temperature)
        a=a.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        b=b.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        denominator = torch.mul(a,b)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class Discriminator1(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        self.model_second = nn.Conv2d(512, 20, 4, padding=1)
        self.model_final = nn.Conv2d(512, 1, 4, padding=1)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.model_final(x1)
        #x3 = self.model_second(x1)
        # Average pooling and flatten
        #return F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1),x1,x3,x3
        return F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1),x1,x1,x1

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model1 = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True)]

        model2 = [nn.Conv2d(64, 128, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True)]

        model3 = [nn.Conv2d(128, 256, 4, stride=2, padding=1),
                  nn.InstanceNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True)]

        model4 = [nn.Conv2d(256, 512, 4, padding=1),
                  nn.InstanceNorm2d(512),
                  nn.LeakyReLU(0.2, inplace=True)]

        # FCN classification layer
        self.model_second = nn.Conv2d(960, 20, 4, padding=1)
        self.model_final = nn.Conv2d(512, 1, 4, padding=1)
        
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.k_model = nn.Flatten()
        
        n_residual_blocks=8
        alt_leak=False
        neg_slope=1e-2
        
        
    def forward(self, x):
        x1_1 = self.model1(x)
        x1_2 = self.model2(x1_1)
        x1_3 = self.model3(x1_2)
        x1_4 = self.model4(x1_3)
        
        x1_1 = F.interpolate(x1_1, scale_factor=0.125)
        diffY1 = x1_4.size()[2] - x1_1.size()[2]
        diffX1 = x1_4.size()[3] - x1_1.size()[3]
        x1_1 = F.pad(x1_1, [diffX1 // 2, diffX1 - diffX1 // 2,
                        diffY1 // 2, diffY1 - diffY1 // 2])
        #x1 = torch.cat([x1_1, x1_4], dim=1)

        x1_2 = F.interpolate(x1_2, scale_factor=0.25)
        diffY2 = x1_4.size()[2] - x1_2.size()[2]
        diffX2 = x1_4.size()[3] - x1_2.size()[3]
        x1_2 = F.pad(x1_2, [diffX2 // 2, diffX2 - diffX2 // 2,
                        diffY2 // 2, diffY2 - diffY2 // 2])
        #x1 = torch.cat([x1, x1_2], dim=1)
        
        x1_3 = F.interpolate(x1_3, scale_factor=0.25)
        diffY3 = x1_4.size()[2] - x1_3.size()[2]
        diffX3 = x1_4.size()[3] - x1_3.size()[3]
        x1_3 = F.pad(x1_3, [diffX3 // 2, diffX3 - diffX3 // 2,
                        diffY3 // 2, diffY3 - diffY3 // 2])

        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)

 
        x2 = self.model_final(x1_4)
        x3 = self.model_second(x1)

        return F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1),x1,x3,x3
        #return F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1),x1,x2

    

    def compute_contrastive_loss(self,input_fake,input_real):

        _,outs0,fakes = self.forward(input_fake)
        _,outs1,reals = self.forward(input_real)
        loss = 0
        self.compute = ContrastiveLoss(1,1)
        for i in range(len(fakes)):

            loss = loss + self.compute(fakes[i],reals[i])
        return loss
    
    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0,_,_ = self.forward(input_fake)
        outs1,_,_ = self.forward(input_real)
        loss = 0
        #SR_loss = 10 * self.L1_loss(SysRegist_AB, input_real)  ###SR
        #SM_loss = 5 * smooothing_loss(Trans)
        #loss = loss + SM_loss
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'ralsgan':
                # all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += (torch.mean((out0 - torch.mean(out1) + 1)**2) +
                        torch.mean((out1 - torch.mean(out0) - 1)**2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
    
    def calc_gen_loss(self, input_fake, input_real):
        # calculate the loss to train G
        outs0,_,_ = self.forward(input_fake)
        outs1,_,_ = self.forward(input_real)

        loss = 0
        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'ralsgan':
                # all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += (torch.mean((out0 - torch.mean(out1) - 1)**2) +
                        torch.mean((out1 - torch.mean(out0) + 1)**2))/2
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class Discriminator_umdst(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator_umdst, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)

        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))


        self.gap_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 5, bias=False))
        self.gmp_fc_2 = nn.utils.spectral_norm(nn.Linear(ndf * mult, 5, bias=False))
        self.leaky_relu_2 = nn.LeakyReLU(0.2, True)
        self.pad_2 = nn.ReflectionPad2d(1)
        for i in range(5):
            setattr(self, 'conv1*1_2_' + str(i), nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True))
        for i in range(5):
            setattr(self, 'conv_2_' + str(i), nn.utils.spectral_norm(nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False)))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x0 = self.model(input)
        heatmap_0 = torch.sum(x0, dim=1, keepdim=True)

        #Discrimination Module and Auxiliary Classifier 2
        x=x0
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap_1 = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)


        #Classification Module and Auxiliary Classifier 1
        out_2=torch.zeros((input.shape[0],5),dtype=float)
        x=x0

        gap_2 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit_2 = self.gap_fc_2(gap_2.view(x.shape[0], -1)).unsqueeze(2)

        gmp_2 = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit_2 = self.gmp_fc_2(gmp_2.view(x.shape[0], -1)).unsqueeze(2)

        cam_logit_2_0 = torch.cat([gap_logit_2, gmp_logit_2], 2)
        cam_logit_2 = torch.mean(cam_logit_2_0,dim=2,keepdim=False)
        Softmax = nn.Softmax(dim=1)
        cam_logit_2=Softmax(cam_logit_2)

        heatmap_2=torch.Tensor(x.shape[0],0,x.shape[2],x.shape[3]).cuda()

        for i in range(5):
            x = x0
            gap_weight_2 = list(self.gap_fc_2.parameters())[0][i].unsqueeze(0)
            gap_2 = x * gap_weight_2.unsqueeze(2).unsqueeze(3)

            gmp_weight_2 = list(self.gmp_fc_2.parameters())[0][i].unsqueeze(0)
            gmp_2 = x * gmp_weight_2.unsqueeze(2).unsqueeze(3)

            x = torch.cat([gap_2, gmp_2], 1)
            x = self.leaky_relu_2(getattr(self, 'conv1*1_2_' + str(i))(x))

            heatmap_2_0 = torch.sum(x, dim=1, keepdim=True)
            heatmap_2=torch.cat((heatmap_2,heatmap_2_0),1)#attention map

            x = self.pad_2(x)
            out_2_0 = getattr(self, 'conv_2_' + str(i))(x)
            if i==0:
                out_2 = torch.mean(out_2_0,axis=(2,3),keepdim=False)
            else:out_2=torch.cat((out_2,torch.mean(out_2_0,axis=(2,3),keepdim=False)),1)

        Softmax = nn.Softmax(dim=1)
        out_2 = Softmax(out_2)

        return out, cam_logit,out_2,cam_logit_2,heatmap_0,heatmap_1,heatmap_2[:,0],heatmap_2[:,1],heatmap_2[:,2],heatmap_2[:,3],heatmap_2[:,4]
    
    
    

    
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice0 = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1,padding=0))
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = self.slice0(X)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

#torch.cat([fake_B,fake_B,fake_B],dim=1)
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(torch.nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
    


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

   
class new_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(new_Generator, self).__init__()

        # Initial convolution block
        model_head = [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, 64, 7),
                      nn.InstanceNorm2d(64),
                      nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model_head += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        model_body = []
        for _ in range(n_residual_blocks):
            model_body += [ResidualBlock(in_features)]

        # Upsampling
        model_tail = []
        out_features = in_features // 2
        for _ in range(2):
            model_tail += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                           nn.InstanceNorm2d(out_features),
                           nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model_tail += [nn.ReflectionPad2d(3),
                       nn.Conv2d(64, output_nc, 7),
                       nn.Tanh()]

        self.model_head = nn.Sequential(*model_head)
        self.model_body = nn.Sequential(*model_body)
        self.model_tail = nn.Sequential(*model_tail)

    def forward(self, x):
        x = self.model_head(x)
        x = self.model_body(x)
        x = self.model_tail(x)

        return x


import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
        
class DualPreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x1, x2, **kwargs):
        return self.fn(self.norm1(x1), self.norm2(x2), **kwargs)


class PriorAttentionBlock(nn.Module):
    def __init__(self, feat_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head

        dim = feat_dim
        mlp_dim = dim * 4

        # update priors by aggregating from the feature map
        self.prior_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.prior_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))

        # update the feature map by injecting knowledge from the priors
        self.feat_aggregate_block = DualPreNorm(dim, CrossAttention(dim, heads, dim_head, attn_drop, proj_drop))
        self.feat_ffn = PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))


    def forward(self, x1, x2):
        # x1: image feature map, x2: priors

        x2 = self.prior_aggregate_block(x2, x1) + x2
        x2 = self.prior_ffn(x2) + x2

        x1 = self.feat_aggregate_block(x1, x2) + x1
        x1 = self.feat_ffn(x1) + x1

        return x1, x2


class PriorInitFusionLayer(nn.Module):
    def __init__(self, feat_dim, prior_dim, block_num=2, task_prior_num=5, l=10):
        super().__init__()
        
        # random initialize the priors
        self.task_prior = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(task_prior_num+1, prior_dim))) # +1 for null token
        

        self.attn_layers = nn.ModuleList([])
        for i in range(block_num):
            self.attn_layers.append(PriorAttentionBlock(feat_dim, heads=feat_dim//32, dim_head=32, attn_drop=0, proj_drop=0))

    def forward(self, x, tgt_idx):
        # x: image feature map, tgt_idx: target task index, mod_idx: modality index
        B, C, H, W = x.shape
        
        task_prior_list = []
  
        # prior selection
        task_prior_list.append(self.task_prior[tgt_idx, :])
        

        task_priors = torch.stack(task_prior_list)


        
        #x = rearrange(x, 'b c d h w -> b (d h w) c', d=D, h=H, w=W)
        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        x = x.permute(0, 2, 1).contiguous()

        
        for layer in self.attn_layers:
            x, priors = layer(x, task_priors)
        
        #x = rearrange(x, 'b (d h w) c -> b c d h w', d=D, h=H, w=W, c=C)
        x = x.permute(0, 2, 1)
        x = x.view(b, c, h, w).contiguous()

        return x, task_priors



    



class GeneratorPromptEnhanced(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super().__init__()
        self.n_residual_blocks = n_residual_blocks
        # Channel configuration
        self.channels = {
            'head': [64, 128, 256],
            'body': 256,
            'tail': [128, 64, output_nc]
        }
        
        # Initial convolution block
        self.init_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling layers with dual-prompt
        self.down_layers1 = self._build_downsample(64, 128)
        self.prior_init_fuse_1 = PriorInitFusionLayer(128, 128, block_num=2, task_prior_num=5)
        
        self.down_layers2 = self._build_downsample(128, 256) 
        self.prior_init_fuse_2 = PriorInitFusionLayer(256, 256, block_num=2, task_prior_num=5)

        # Residual blocks with task prompt
        self.res_blocks = nn.ModuleList(
            [ResidualBlockDualPrompt(256) for _ in range(n_residual_blocks)]
        )

        # Upsampling with progressive prompting
        self.up_layers1 = self._build_upsample(256, 128)
        self.prior_init_fuse_3 = PriorInitFusionLayer(128, 128, block_num=2, task_prior_num=5)
        
        self.up_layers2 = self._build_upsample(128, 64)
        self.prior_init_fuse_4 = PriorInitFusionLayer(64, 64, block_num=2, task_prior_num=5)

        # Output refinement
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        )


        self.task_prompt = nn.Parameter(torch.randn(1, 6, 256, 256))

        

        

        self.task_enhancer = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 6, 1)
        )

    def _build_downsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _build_upsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, 
                             stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, IHC_id, layers=[], encode_only=False):
        if encode_only:
            return self._encode_only(x,IHC_id)
            
        bs = x.size(0)
        intermediates = {}
    
        # Stage 1: 
        x = self.init_conv(x)
        

        # Stage 2: 
        x = self.down_layers1(x)
        x, t_prompt1 = self.prior_init_fuse_1(x, IHC_id)  # 

        
        # Stage 3: 
        x = self.down_layers2(x)
        x, t_prompt2 = self.prior_init_fuse_2(x, IHC_id)
        intermediates['encoder_output'] = x.clone()
 
  
        # Stage 4: 
        for i, block in enumerate(self.res_blocks):
            if i<self.n_residual_blocks-1:
                x = block(x, IHC_id, False)
            else:
                x = block(x, IHC_id, True)
  
        # Stage 5: 
        x = self.up_layers1(x)
        x, _ = self.prior_init_fuse_3(x, IHC_id)
        
        x = self.up_layers2(x)
        x, _ = self.prior_init_fuse_4(x, IHC_id)
        return self.output_layer(x)  

    def _encode_only(self, x,IHC_id):
        feats = []
        x = self.init_conv(x)
        feats.append(x)
        
        x = self.down_layers1(x)
        x, _ = self.prior_init_fuse_1(x, IHC_id)  # disable task prompt
        feats.append(x)
        
        x = self.down_layers2(x)
        x, _ = self.prior_init_fuse_2(x, IHC_id)
        feats.append(x)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        feats.append(x)
        
        for layer_id, layer in enumerate(self.res_blocks):
            feat = layer(x)
            if layer_id==0 or layer_id==4 or layer_id==8:
                feats.append(feat)
        return feats
        
        


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.url = "microsoft/BiomedVLP-CXR-BERT-specialized"
        #self.url = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(self.url, trust_remote_code=True)
        self.bert = AutoModel.from_pretrained(self.url, trust_remote_code=True)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
            
        #self.projection = nn.Linear(768, 256)  
        self.projection = nn.Sequential(
                            nn.Linear(768, 256),  
                            nn.ReLU(),            
                            nn.Linear(256, 256),
                            nn.Dropout(p=0.1)
                        )
        # 简化版描述
        self.simple_desc = {
            0: 'cell nucleus',  # HE
            1: 'cell nucleus',  # ER 
            2: 'cell membrane',  # HER2
            3: 'cell nucleus',  # Ki67
            4: 'cell nucleus' , # PR
            5: 'cell membrane'  # PDL1
        }
    def forward(self, ihc_id):
        if isinstance(ihc_id, torch.Tensor):
            ihc_id = ihc_id.item()  
        # 获取位置描述
        text = self.simple_desc[ihc_id] 
        
        tokens = self.tokenizer(text, return_tensors="pt", padding=True).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        with torch.no_grad():
            outputs = self.bert(**tokens)
            bert_features = outputs.last_hidden_state[:, 0, :]
            
        text_features = self.projection(bert_features)
        return text_features

class CrossAttentionBlock(nn.Module):
    def __init__(self, channels, text_dim):
        super().__init__()
        self.channels = channels
        self.text_dim = text_dim
        
        # Query from image features
        self.query = nn.Conv2d(channels, channels, 1)
        # Key and Value from text features
        self.key = nn.Linear(text_dim, channels)
        self.value = nn.Linear(text_dim, channels)
        
        self.scaling = channels ** -0.5
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x, text_feat):
        b, c, h, w = x.shape
        
        # Shape: [B, C, H*W]
        q = self.query(x).view(b, c, -1)
        
        # Shape: [B, C, 1]
        k = self.key(text_feat).unsqueeze(-1)
        v = self.value(text_feat).unsqueeze(-1)
        
        # Attention weights
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scaling
        attn = F.softmax(attn, dim=1)
        
        # Attended features
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        return self.to_out(out)

class ResidualBlockDualPrompt(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )
        self.textenc = TextEncoder()
        
        self.cross_attention = CrossAttentionBlock(channels, 256)
        
    def forward(self, x, IHC_id, text_prompt=False):
    
        residual = x
        if not text_prompt:
            out = self.conv_block(x)
            out = out + residual
            return out
        
        text_feat = self.textenc(IHC_id)
        out = self.conv_block(x)
        out = self.cross_attention(out, text_feat)
        out = out + residual
        
            
        return out











