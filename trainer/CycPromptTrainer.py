#!/usr/bin/python3

import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from .utils import LambdaLR,Logger,ReplayBuffer
from .utils import weights_init_normal,get_config,Normalize
from .datasets import ImageDataset_MIST_prompt,ValDataset_MIST_prompt
from Model.CycleGan import *
from .utils import Resize,ToTensor,smooothing_loss,seg_img
from .utils import Logger
from torchvision.transforms import RandomAffine
from torchvision.transforms import RandomAffine,ToPILImage
from .transformer import Transformer_2D
from skimage import measure

from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision
import torch
import imageio
from PIL import Image
from ptflops import get_model_complexity_info
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    
class CycPrompt_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        ## def networks
        self.netG_A2B = GeneratorPromptEnhanced(config['input_nc'], config['output_nc']).cuda()
        self.disGA = Discriminator_umdst(input_nc=3, ndf=64, n_layers=7).cuda()
        self.disLA = Discriminator_umdst(input_nc=3, ndf=64, n_layers=5).cuda()
        self.vgg_loss = VGGLoss(1)
        self.netF = PatchSampleF(use_mlp=True, init_type='normal', init_gain=0.02, gpu_ids=[1], nc=256).cuda()
        self.netD_A = Discriminator(self.config["input_nc"]).cuda()  # discriminator for domain a
        self.netD_B = Discriminator(self.config["input_nc"]).cuda()  # discriminator for domain b
	
        params = list(self.netF.parameters())
        if len(params) == 0:
            print("Warning: netF has no parameters!")
        else:
            print(f"Number of parameters in netF: {len(params)}")
            self.optimizer_F = torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))
    
    
        #self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=0.0002,betas=(0.5, 0.999))
        self.nce_layers = '0,1,2,3,4,5'
        self.nce_layers = [int(i) for i in self.nce_layers.split(',')]
        self.criterionNCE = []

        for nce_layer in self.nce_layers:
            self.criterionNCE.append(PatchNCELoss().cuda())
                
                
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disLA.parameters()), lr=config['lr'], betas=(0.5, 0.999), weight_decay=0.0001)
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['lr'], betas=(0.5, 0.999),weight_decay=0.0001)
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=config['lr'], betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=config['lr'], betas=(0.5, 0.999))
            

        # Lossess
        self.MSE_loss = torch.nn.MSELoss()
        self.L1_loss = torch.nn.L1Loss()
        self.BCE_loss = torch.nn.BCEWithLogitsLoss()
        
        self.new_L1_loss = torch.nn.L1Loss(reduction='sum')
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor
        self.input_A = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_A_label = Tensor([1]).long()
        self.input_B_label = Tensor([1]).long()
        
        self.input_A1 = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B1 = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_A2 = Tensor(config['batchSize'], config['input_nc'], config['size'], config['size'])
        self.input_B2 = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        
        self.input_C = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.input_D = Tensor(config['batchSize'], config['output_nc'], config['size'], config['size'])
        self.target_real = Variable(Tensor(1,1).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(1,1).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
        def __make_power_2(img, base, method=Image.BICUBIC):
            ow, oh = img.size
            h = int(round(oh / base) * base)
            w = int(round(ow / base) * base)
            if h == oh and w == ow:
                return img

            return img.resize((w, h), method)
        
        #Dataset loader
        transforms_1 = [
                        transforms.Resize([256,256], Image.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        #Resize(size_tuple = (config['size'], config['size']))
                        ]
    
        transforms_2 = [ToPILImage(),
                        
                        ToTensor(),
                        Resize(size_tuple = (config['size'], config['size']))]
        
        transforms_3 = [ToPILImage(),
                        RandomAffine(degrees=0, translate=[0.02 * 0, 0.02 * 0],
                                     scale=[1 - 0.02 * 0, 1 + 0.02 * 0], fill=-1),  #
                        ToTensor(),
                        Resize(size_tuple=(config['size'], config['size']))]
        
        train_transform = [ToPILImage(),           
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.ToTensor(),
                           Resize(size_tuple = (config['size'], config['size']))]
        
        transform222 = [transforms.RandomCrop(32, padding=4),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                      transforms.RandomGrayscale(p=0.2),
                      transforms.ToTensor(),
                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                      Resize(size_tuple = (config['size'], config['size']))]

        

        self.dataloader = DataLoader(ImageDataset_MIST_prompt(config['dataroot'], transforms_1=transforms_1, transforms_2=transforms_2,  transforms_3=transforms_2, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])

        val_transforms = [ToTensor(),
                          Resize(size_tuple = (config['size'], config['size']))]
        
        self.val_data = DataLoader(ValDataset_MIST_prompt(config['val_dataroot'],  transforms_1=transforms_1, transforms_2=transforms_2,  transforms_3=transforms_2, unaligned=False),
                                batch_size=config['batchSize'], shuffle=False, num_workers=config['n_cpu'])


       # Loss plot
        self.logger = Logger(config['name'],config['port'],config['n_epochs'], len(self.dataloader))       
        
    def train(self):
        ###### Training ######
        #self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + '24_netG_A2B.pth'))

            
        for epoch in range(self.config['epoch'], self.config['n_epochs']):
            
            for i, batch in enumerate(self.dataloader):
                
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                
                A_label = Variable(self.input_A_label.copy_(batch['A_label']))
                B_label = Variable(self.input_A_label.copy_(batch['B_label']))

                
                self.optimizer_G.zero_grad()
                # GAN loss

                fake_B2A = self.netG_A2B(real_B,A_label)
                fake_B2A2B = self.netG_A2B(fake_B2A,B_label)
                fake_B2B = self.netG_A2B(real_B,B_label) #identity

                pred_fake,_,_,_ = self.netD_B(fake_B2A)
                loss_GAN_A2B = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_real)

                G_recon_loss_A = self.L1_loss(fake_B2A2B, real_B)

                G_identity_loss_A = self.L1_loss(fake_B2B, real_B)
                
                # Total loss
                loss_Total = loss_GAN_A2B + G_recon_loss_A + G_identity_loss_A 
                loss_Total.backward()
                self.optimizer_G.step()
                        

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real,_,real_feature,_ = self.netD_B(real_A)
                loss_D_real = self.config['Adv_lamda'] * self.MSE_loss(pred_real, self.target_real)

                # Fake loss
                fake_B2A = self.fake_B_buffer.push_and_pop(fake_B2A)
                pred_fake,_,fake_feature,_ = self.netD_B(fake_B2A.detach())
                loss_D_fake = self.config['Adv_lamda'] * self.MSE_loss(pred_fake, self.target_fake)
                

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)
                loss_D_B.backward()

                self.optimizer_D_B.step()
                

                self.logger.log({
                                'loss_cycle_BAB':loss_Total,
                                'G_recon_loss_A':G_recon_loss_A,
                                'loss_D_A':loss_D_B,
                                 },
                                images={'rd_real_B': real_B, 'rd_real_A': real_A, 'fake_B2A':fake_B2A})
            if epoch>=50 and epoch%1==0:
            	torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + '%s_netG_A2B.pth'%epoch)

            if not os.path.exists(self.config["save_root"]):
                os.makedirs(self.config["save_root"])
            torch.save(self.netG_A2B.state_dict(), self.config['save_root'] + 'netG_A2B.pth')


                         
    def find_test(self,epoch):
        #self.netG_B2A.load_state_dict(torch.load(self.config['save_root'] + '%s_netG_B2A.pth'%epoch))
        #self.netG_B2A.load_state_dict(torch.load(self.config['save_root'] + '1_netG_B2A.pth'))
        self.netG_A2B.load_state_dict(torch.load(self.config['save_root'] + 'netG_A2B.pth'))
        with torch.no_grad():

                for i, batch in enumerate(self.val_data):
                    real_A = Variable(self.input_A.copy_(batch['A'])).detach().cpu().numpy().squeeze()
                    real_B = Variable(self.input_B.copy_(batch['B']))
                    
                    A_label = Variable(self.input_A_label.copy_(batch['A_label']))
                    B_label = Variable(self.input_A_label.copy_(batch['B_label']))
                
                
                    fake_A, = self.netG_A2B(real_B,A_label)
                    
 
                    num += 1
                    real_B = real_B.detach().cpu().numpy().squeeze() 
                    fake_A = fake_A.detach().cpu().numpy().squeeze() 


                    self.save_img(i,real_A,real_B,fake_A,fake_A)

    
    def test(self,):
        for i in range(1):
            self.find_test(i)
            
        
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                    
                    


    def save_img(self,i,real_A,real_B,fake_A,fake_B):

        real_B = np.transpose(real_B, (1, 2, 0))
        fake_A = np.transpose(fake_A, (1, 2, 0))
        real_A = np.transpose(real_A, (1, 2, 0))
        imageio.imwrite('./output/CYC_uniseg/save_img/real_B/'+'%s.jpg'%i,real_B)
        imageio.imwrite('./output/CYC_uniseg/save_img/real_A/'+'%s.jpg'%i,real_A)
        imageio.imwrite('./output/CYC_uniseg/save_img/fake_A/'+'%s.jpg'%i,fake_A)
    

    

