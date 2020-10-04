

#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

torch.manual_seed(2)

cfg = {
	'VGG5' : [64, 'A', 128, 'D', 128, 'A'],
	'VGG9': [64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'A'],
	'VGG11': [64, 'A', 128, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'A'],
	'VGG16': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D', 512, 'D']
}


from typing import Union


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k)[0]
    return result

class LinearSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass, we compute a step function of the input Tensor and
        return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use
        the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass, we receive a Tensor we need to compute
        the surrogate gradient of the loss with respect to the input.
        Here we use the piecewise-linear surrogate gradient as was
        done in Bellec et al. (2018).
        """
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = grad_input*LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad

from torch.autograd import Variable
class Sampled_DCT2ov(nn.Module):
    def __init__(self, block_size=8, stride=None, p=0, mode = 'random', mean = None, std=None, device = 'cpu'):

      super(Sampled_DCT2ov, self).__init__()
      ### forming the cosine transform matrix
      self.block_size = block_size
      self.device = device
      self.mean =mean
      self.std =std
      if stride==None:
        self.stride=block_size
      else:
        self.stride=stride
      self.Q = torch.zeros((self.block_size,self.block_size)).cuda()
      self.bases=torch.zeros(self.block_size,self.block_size,self.block_size,self.block_size).cuda()
      self.Q[0] = math.sqrt( 1.0/float(self.block_size) )
      for i in range (1,self.block_size,1):
        for j in range(self.block_size):
          self.Q[i,j] = math.sqrt( 2.0/float(self.block_size) ) * math.cos( float((2*j+1)*math.pi*i) /float(2.0*self.block_size) )

      
        ### forming the 2d DCT bases
      for i in range (self.block_size):
        for j in range(self.block_size):
          c = torch.zeros((self.block_size,self.block_size)).cuda()
          c[i,j]=1.0
          self.bases[i,j] = torch.matmul(torch.matmul(self.Q.permute(1,0).contiguous(), c), self.Q )

      self.tst=self.block_size*self.block_size
      self.loc=torch.zeros(self.tst, 2).cuda()
      if self.block_size==4:
          self.loc[0]=torch.tensor([0,0])
          self.loc[1]=torch.tensor([0,1])
          self.loc[2]=torch.tensor([1,0])
          self.loc[3]=torch.tensor([2,0])
          self.loc[4]=torch.tensor([1,1])
          self.loc[5]=torch.tensor([0,2])
          self.loc[6]=torch.tensor([0,3])
          self.loc[7]=torch.tensor([1,2])
          self.loc[8]=torch.tensor([2,1])
          self.loc[9]=torch.tensor([3,0])
          self.loc[10]=torch.tensor([3,1])
          self.loc[11]=torch.tensor([2,2])
          self.loc[12]=torch.tensor([1,3])
          self.loc[13]=torch.tensor([2,3])
          self.loc[14]=torch.tensor([3,2])
          self.loc[15]=torch.tensor([3,3])

      if self.block_size==8:
          self.loc[0]=torch.tensor([0,0])
          self.loc[1]=torch.tensor([0,1])
          self.loc[2]=torch.tensor([1,0])
          self.loc[3]=torch.tensor([2,0])
          self.loc[4]=torch.tensor([1,1])
          self.loc[5]=torch.tensor([0,2])
          self.loc[6]=torch.tensor([0,3])
          self.loc[7]=torch.tensor([1,2])
          self.loc[8]=torch.tensor([2,1])
          self.loc[9]=torch.tensor([3,0])
          self.loc[10]=torch.tensor([4,0])
          self.loc[11]=torch.tensor([3,1])
          self.loc[12]=torch.tensor([2,2])
          self.loc[13]=torch.tensor([1,3])
          self.loc[14]=torch.tensor([0,4])
          self.loc[15]=torch.tensor([0,5])
          self.loc[16]=torch.tensor([1,4])
          self.loc[17]=torch.tensor([2,3])
          self.loc[18]=torch.tensor([3,2])
          self.loc[19]=torch.tensor([4,1])
          self.loc[20]=torch.tensor([5,0])
          self.loc[21]=torch.tensor([6,0])
          self.loc[22]=torch.tensor([5,1])
          self.loc[23]=torch.tensor([4,2])
          self.loc[24]=torch.tensor([3,3])
          self.loc[25]=torch.tensor([2,4])
          self.loc[26]=torch.tensor([1,5])
          self.loc[27]=torch.tensor([0,6])
          self.loc[28]=torch.tensor([0,7])
          self.loc[29]=torch.tensor([1,6])
          self.loc[30]=torch.tensor([2,5])
          self.loc[31]=torch.tensor([3,4])
          self.loc[32]=torch.tensor([4,3])
          self.loc[33]=torch.tensor([5,2])
          self.loc[34]=torch.tensor([6,1])
          self.loc[35]=torch.tensor([7,0])
          self.loc[36]=torch.tensor([7,1])
          self.loc[37]=torch.tensor([6,2])
          self.loc[38]=torch.tensor([5,3])
          self.loc[39]=torch.tensor([4,4])
          self.loc[40]=torch.tensor([3,5])
          self.loc[41]=torch.tensor([2,6])
          self.loc[42]=torch.tensor([1,7])
          self.loc[43]=torch.tensor([2,7])
          self.loc[44]=torch.tensor([3,6])
          self.loc[45]=torch.tensor([4,5])
          self.loc[46]=torch.tensor([5,4])
          self.loc[47]=torch.tensor([6,3])
          self.loc[48]=torch.tensor([7,2])
          self.loc[49]=torch.tensor([7,3])
          self.loc[50]=torch.tensor([6,4])
          self.loc[51]=torch.tensor([5,5])
          self.loc[52]=torch.tensor([4,6])
          self.loc[53]=torch.tensor([3,7])
          self.loc[54]=torch.tensor([4,7])
          self.loc[55]=torch.tensor([5,6])
          self.loc[56]=torch.tensor([6,5])
          self.loc[57]=torch.tensor([7,4])
          self.loc[58]=torch.tensor([7,5])
          self.loc[59]=torch.tensor([6,6])
          self.loc[60]=torch.tensor([5,7])
          self.loc[61]=torch.tensor([6,7])
          self.loc[62]=torch.tensor([7,6])
          self.loc[63]=torch.tensor([7,7])

    def rgb_to_ycbcr(self,input):
        
        # input is mini-batch N x 3 x H x W of an RGB image
        #output = Variable(input.data.new(*input.size())).to(self.device)
        output = Variable(torch.zeros_like(input)).cuda()
        input = (input * 255.0)
        output[:, 0, :, :] = input[:, 0, :, :] * 0.299+ input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114 
        output[:, 1, :, :] = input[:, 0, :, :] * -0.168736 - input[:, 1, :, :] *0.331264+ input[:, 2, :, :] * 0.5 + 128
        output[:, 2, :, :] = input[:, 0, :, :] * 0.5 - input[:, 1, :, :] * 0.418688- input[:, 2, :, :] * 0.081312+ 128
        return output/255.0

    def ycbcr_to_freq(self,input): 
        
        x=int(((input.shape[2]-self.block_size)/self.stride)+1)*self.block_size
        y=int(((input.shape[3]-self.block_size)/self.stride)+1)*self.block_size
        output = Variable(torch.zeros(self.tst, input.shape[0],input.shape[1],x,y)).cuda()
        dctcoeff= torch.zeros(input.shape[0],input.shape[1],self.block_size,self.block_size).cuda()
        #a=int(input.shape[2]/self.block_size)
        #b=int(input.shape[3]/self.block_size)
        
#        print(input.device)
#        print(self.Q.device)
        self.Q=self.Q.to(input.device)
        self.bases=self.bases.to(dctcoeff.device)
        m1=-1
        # Compute DCT in block_size x block_size blocks 
        for i in range(0, input.shape[2] - self.block_size + 1, self.stride):
            m1=m1+1
            n1=-1
            for j in range(0, input.shape[3] - self.block_size + 1, self.stride):
                n1=n1+1
                dctcoeff = torch.matmul(torch.matmul(self.Q, input[:, :, i:i+self.block_size, j:j+self.block_size]), self.Q.permute(1,0).contiguous() )
                
                for k in range(self.tst):
                  m,n=self.loc[k]
                  output[k,:,:,m1*self.block_size  : (m1+1)*self.block_size ,n1*self.block_size  : (n1+1)*self.block_size ]=torch.einsum('ij,kl->ijkl', dctcoeff[:,:,int(m),int(n)], self.bases[int(m),int(n)])




        #return dctcoeff 
        return output
    def forward(self, x):
        #return self.ycbcr_to_freq( x )
        if (x.shape[1]==3):
          return self.ycbcr_to_freq( self.rgb_to_ycbcr(x) )
        else:
          return self.ycbcr_to_freq(x ) 


class VGG_SNN_STDB_lin(nn.Module):
    def __init__(self, vgg_name, activation='STDB', labels=1000, timesteps=75, leak_mem=0.99, drop=0.2):
        super().__init__()
        
        self.timesteps= timesteps
        self.vgg_name= vgg_name
        self.labels= labels
        self.leak_mem=leak_mem
        self.act_func	= LinearSpike.apply
        use_cuda =torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        
        self.process 	= Sampled_DCT2ov(block_size=4, stride=2,device = device)
        
        self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
        
    def threshold_init(self, scaling_threshold=1.0, reset_threshold=0.0, thresholds=[], default_threshold=1.0):
        self.scaling_threshold 	= scaling_threshold
        self.reset_threshold 	= reset_threshold
        self.threshold 			= {}
        print('\nThresholds:')
        
        for pos in range(len(self.features)):
            if isinstance(self.features[pos], nn.Conv2d):
                self.threshold[pos] = round(thresholds.pop(0) * self.scaling_threshold  + self.reset_threshold * default_threshold, 2)
                print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))
                
        prev = len(self.features)
        
        for pos in range(len(self.classifier)-1):
            if isinstance(self.classifier[pos], nn.Linear):
                self.threshold[prev+pos] = round(thresholds.pop(0) * self.scaling_threshold  + self.reset_threshold * default_threshold, 2)
                print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))
                
        return self.threshold
    
    def counting_spikes(cur_time, layer, spikes):
        self.spike_count
        
    def _make_layers(self, cfg):
        layers 		= []
        in_channels = 3
        
        for x in (cfg):
            stride = 1
            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(0.2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=stride, bias=False),nn.ReLU(inplace=True)]
                in_channels = x
                
        features = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Linear(512*9, 4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.2)]
        layers += [nn.Linear(4096,4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.2)]
        layers += [nn.Linear(4096, self.labels, bias=False)]
        
        classifer = nn.Sequential(*layers)
        return (features, classifer)
    
    def network_init(self, update_interval):
        self.update_interval = update_interval
        
    def neuron_init(self, x):
        self.batch_size = x.size(0)
        self.width 		= 60
        self.height 	= 60
        
        self.mem 		= {}
        self.spike 		= {}
        self.mask 		= {}
        self.spike_count= {}
        
        for l in range(len(self.features)):
            if isinstance(self.features[l], nn.Conv2d):
                self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height)
                self.spike_count[l] = torch.zeros(self.mem[l].size())
            elif isinstance(self.features[l], nn.Dropout):
                self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape))
            elif isinstance(self.features[l], nn.AvgPool2d):
                self.width = self.width//2
                self.height = self.height//2
                
        prev = len(self.features)
        for l in range(len(self.classifier)):
            if isinstance(self.classifier[l], nn.Linear):
                self.mem[prev+l] 			= torch.zeros(self.batch_size, self.classifier[l].out_features)
                self.spike_count[prev+l] 	= torch.zeros(self.mem[prev+l].size())
                
            elif isinstance(self.classifier[l], nn.Dropout):
                self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape))
        self.spike = copy.deepcopy(self.mem)
        for key, values in self.spike.items():
            for value in values:
                value.fill_(-1000)
                
    def forward(self, x, cur_time, mem=[], spike=[], mask=[], spike_count=[], find_max_mem=False, max_mem_layer=0):
        if cur_time == 0:
            self.neuron_init(x)
        else:
            self.batch_size = x.size(0)
            self.mem 		= {}
            self.spike 		= {}
            self.mask 		= {}
            for key, values in mem.items():
                self.mem[key] = values.detach()
            for key, values in spike.items():
                self.spike[key] = values.detach()
            for key, values in mask.items():
                self.mask[key] = values.detach()
            for key,values in spike_count.items():
                self.spike_count[key] = values.detach()
         
        #dct-encoding and threshold selection for input layer
        g=self.process(x)
        th_n=np.percentile(g.cpu(), 6.5)
        th_p=np.percentile(g.cpu(), 93.5)
        mem=torch.zeros(g.shape[1],g.shape[2],g.shape[3],g.shape[4])  
        features_max_layer 	= len(self.features)
        max_mem 			= 0.0
        for t in range(cur_time, cur_time+self.update_interval):
            #spike-generator encoding part
            mem=mem+g[t%16]
            spike_inp = torch.zeros_like(mem).cuda()
            
            spike_inp[mem >th_p] = 1.0
            spike_inp[mem < th_n] = -1.0
            rst = torch.zeros_like(mem).cuda()
            c = (mem >th_p)
            rst[c] = torch.ones_like(mem)[c]*th_p
            e = (mem < th_n)
            rst[e] = torch.ones_like(mem)[e]*th_n
            mem=mem-rst
            out_prev = spike_inp
            for l in range(len(self.features)):
                if isinstance(self.features[l], (nn.Conv2d)):
                    mem_thr 					= (self.mem[l]/self.threshold[l]) - 1.0
                    out 						= self.act_func(mem_thr)
                    rst 						= self.threshold[l] * (mem_thr>0).float()
                    self.spike[l] 				= self.spike[l].masked_fill(out.bool(),t-1)
                    self.spike_count[l][out.bool()] 	= self.spike_count[l][out.bool()] + 1
                    
                    if find_max_mem and l==max_mem_layer:
                        if (self.features[l](out_prev)).max()>max_mem:
                            #max_mem = (self.features[l](out_prev)).max()
                            max_mem=percentile((self.features[l](out_prev)), 99.9) 
                            #max_mem = np.percentile((self.features[l](out_prev)).cpu(), 99.9)
                        break
                    self.mem[l] 	= self.leak_mem*self.mem[l] + self.features[l](out_prev) - rst
                    out_prev  		= out.clone()
                    
                elif isinstance(self.features[l], nn.AvgPool2d):
                    out_prev 		= self.features[l](out_prev)
                elif isinstance(self.features[l], nn.Dropout):
                    out_prev 		= out_prev * self.mask[l]
            if find_max_mem and max_mem_layer<features_max_layer:
                continue
            out_prev       	= out_prev.reshape(self.batch_size, -1)
            prev = len(self.features)
            
            for l in range(len(self.classifier)-1):
                if isinstance(self.classifier[l], (nn.Linear)):
                    mem_thr 					= (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
                    out 						= self.act_func(mem_thr)
                    rst 						= self.threshold[prev+l] * (mem_thr>0).float()
                    self.spike[prev+l] 			= self.spike[prev+l].masked_fill(out.bool(),t-1)
                    self.spike_count[prev+l][out.bool()] 	= self.spike_count[prev+l][out.bool()] + 1
                    
                    if find_max_mem and (prev+l)==max_mem_layer:
                        if (self.classifier[l](out_prev)).max()>max_mem:
                            #max_mem = (self.classifier[l](out_prev)).max()
                            max_mem=percentile((self.classifier[l](out_prev)), 99.9) 
                            #max_mem = np.percentile((self.classifier[l](out_prev)).cpu(), 99.9)
                       
                        break
                    self.mem[prev+l] 	= self.leak_mem*self.mem[prev+l] + self.classifier[l](out_prev) - rst
                    out_prev  		= out.clone()
                elif isinstance(self.classifier[l], nn.Dropout):
                    out_prev 		= out_prev * self.mask[prev+l]
                    
            if not find_max_mem:
                self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
        if find_max_mem:
            return max_mem
        
        return self.mem[prev+l+1], self.mem, self.spike, self.mask, self.spike_count
        
                
                    
            
                
    
		

            
        
        
        


        
        
        
        
        
        
        
        