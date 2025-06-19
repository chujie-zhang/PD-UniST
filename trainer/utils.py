import random
import time
import datetime
import sys
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad

import time
from pykeops.torch import LazyTensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Resize():
    def __init__(self, size_tuple, use_cv = True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv
        #self.mode = mode

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
 
        tensor = F.interpolate(tensor, size = [self.size_tuple[0],self.size_tuple[1]])

        tensor = tensor.squeeze(0)
 
        return tensor#1, 64, 128, 128
class ToTensor():
    def __call__(self, tensor):
        tensor = np.expand_dims(tensor, 0)
        return torch.from_numpy(tensor)
    
def Normalize(img):
    ymax = 255
    ymin = 0
    xmax = np.max(img)
    xmin = np.min(img)
    norm_img = ((img-xmin)/(xmax-xmin)*(ymax-ymin))+ymin #-->[0,255]
    return norm_img

def tensor2image(tensor):
    image = Normalize(tensor.cpu().float().numpy())
    #image = (127.5*(tensor.cpu().float().numpy()))+127.5
    #image = tensor.cpu().float().numpy()
    image1 = image[0]
    for i in range(1,tensor.shape[0]):
        image1 = np.hstack((image1,image[i]))
    
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    #print ('image1.shape:',image1.shape)
    return image1.astype(np.uint8)


class Logger():
    def __init__(self, env_name ,ports, n_epochs, batches_epoch):
        self.viz = Visdom(port= ports,env = env_name)
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        # Draw images
        for image_name, tensor in images.items():
            if image_name not in self.image_windows:

                self.image_windows[image_name] = self.viz.image(tensor2image(tensor.data), opts={'title': image_name})
            else:
                self.viz.image(tensor2image(tensor.data), win=self.image_windows[image_name],
                               opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)
        
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)

def smooothing_loss(y_pred):
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx*dx
    dy = dy*dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d 
    return d



def seg_img(img,mask): 
    return torch.mul(img,mask)


def R1Penalty(real_img, f):
    # gradient penalty
    reals = Variable(real_img, requires_grad=True).to(real_img.device)
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
    undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

    real_logit = apply_loss_scaling(torch.sum(real_logit))
    real_grads = grad(real_logit, reals, grad_outputs=torch.ones(real_logit.size()).to(reals.device), create_graph=True)[0].view(reals.size(0), -1)
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = torch.sum(torch.mul(real_grads, real_grads))
    return r1_penalty

class SCCLoss(torch.nn.Module):
    def __init__(self):
        super(SCCLoss, self).__init__()

    def forward(self, fakeI, realI, patch_ver=False):
        def batch_ERSMI(I1, I2):
            batch_size = I1.shape[0]
            # channel_size =
            img_size = I1.shape[1] * I1.shape[2] * I1.shape[3]
            # if I2.shape[1] == 1 and I1.shape[1] != 1:
            #     I2 = I2.repeat(1,3, 1, 1)

            def kernel_F(y, mu_list, sigma):
                # tmp_mu = mu_list.view(-1,1).repeat(1, img_size).repeat(,1,1).cuda()  # [81, 784]
                tmp_mu = mu_list.view(-1, 1).repeat(1, img_size).cuda()
                tmp_y = y.view(batch_size,1, -1).repeat(1,81, 1)
                tmp_y = tmp_mu - tmp_y
                mat_L = torch.exp(tmp_y.pow(2) / (2 * sigma ** 2))
                return mat_L

            mu = torch.Tensor([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]).cuda()

            x_mu_list = mu.repeat(9).view(-1, 81)
            y_mu_list = mu.unsqueeze(0).t().repeat(1, 9).view(-1, 81)

            mat_K = kernel_F(I1, x_mu_list, 1)
            mat_L = kernel_F(I2, y_mu_list, 1)

            H1 = ((mat_K.matmul(mat_K.transpose(1,2))).mul(mat_L.matmul(mat_L.transpose(1,2))) / (img_size ** 2)).cuda()
            # h1 = (mat_K.mul(mat_L)).mm(torch.ones(img_size, 1)) / img_size

            H2 = ((mat_K.mul(mat_L)).matmul((mat_K.mul(mat_L)).transpose(1,2)) / img_size).cuda()
            h2 = ((mat_K.sum(2).view(batch_size,-1, 1)).mul(mat_L.sum(2).view(batch_size,-1, 1)) / (img_size ** 2)).cuda()
            # h2 = (((mat_K.sum(1).view(-1,1)).mul(mat_L.sum(1).view(-1,1)) / (img_size ** 2)).double()).cuda()

            H2 = 0.5 * H1 + 0.5 * H2
            tmp = H2 + 0.05 * torch.eye(H2.shape[1]).cuda()

            alpha = (tmp.inverse())

            alpha = alpha.matmul(h2)
            # end = time.clock()
            # print('2:', end - start)
            ersmi = (2 * (alpha.transpose(1,2)).matmul(h2) - ((alpha.transpose(1,2)).matmul(H2)).matmul(alpha) - 1).squeeze()
            ersmi = -ersmi.mean()
            return ersmi


        batch_loss = batch_ERSMI(fakeI, realI)

        return batch_loss
    
def KMeans(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

    if verbose:  # Fancy display -----------------------------------------------
        if True:
            torch.cuda.synchronize()
        end = time.time()
        '''
        print(
            f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
            
        )
        '''
    #return cl.to(device='cuda:0',dtype=torch.float64), c
    return torch.tensor(cl,dtype=torch.float32).to(DEVICE),c
    #return cl,c
