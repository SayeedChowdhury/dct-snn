
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn
from   tensorboard_logger import configure, log_value
import torchvision
#import torchvision.transforms as transforms
import numpy as np


import os
import argparse
import math
from vgg_ann_models import *
#from utils import progress_bar
import time


def rin(input,b=4,s=2):
  x=int(((input.shape[2]-b)/s)+1)*b
  y=int(((input.shape[3]-b)/s)+1)*b
  output = torch.zeros(input.shape[0],input.shape[1],x,y)
  m=-1
  
  for i in range(0, input.shape[2] - b + 1, s):
    m=m+1
    n=-1
    for j in range(0, input.shape[3] - b + 1, s):
      n=n+1
      output[:,:,m*b : (m+1)*b,n*b : (n+1)*b]=input[:, :, i:i+b, j:j+b]
      
  return output


class DCT2(nn.Module):
    def __init__(self, block_size=4, p=0, mode = 'random', mean = None, std=None, device = 'cpu'):

      super(DCT2, self).__init__()
      ### forming the cosine transform matrix
      self.block_size = block_size
      self.device = device
      self.mean =mean
      self.std =std
      self.Q = torch.zeros((self.block_size,self.block_size)).to(self.device)
      
      self.Q[0] = math.sqrt( 1.0/float(self.block_size) )
      for i in range (1,self.block_size,1):
        for j in range(self.block_size):
          self.Q[i,j] = math.sqrt( 2.0/float(self.block_size) ) * math.cos( float((2*j+1)*math.pi*i) /float(2.0*self.block_size) )

      

    def rgb_to_ycbcr(self,input):
        
        # input is mini-batch N x 3 x H x W of an RGB image
        #output = Variable(input.data.new(*input.size())).to(self.device)
        output = torch.zeros_like(input).to(self.device)
        input = (input * 255.0)
        output[:, 0, :, :] = input[:, 0, :, :] * 0.299+ input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114 
        output[:, 1, :, :] = input[:, 0, :, :] * -0.168736 - input[:, 1, :, :] *0.331264+ input[:, 2, :, :] * 0.5 + 128
        output[:, 2, :, :] = input[:, 0, :, :] * 0.5 - input[:, 1, :, :] * 0.418688- input[:, 2, :, :] * 0.081312+ 128
        return output/255.0

    def ycbcr_to_freq(self,input): 
 
        
        output = torch.zeros_like(input).to(self.device)
        a=int(input.shape[2]/self.block_size)
        b=int(input.shape[3]/self.block_size)
       
        # Compute DCT in block_size x block_size blocks 
        for i in range(a):
            for j in range(b):
                output[:,:,i*self.block_size : (i+1)*self.block_size,j*self.block_size : (j+1)*self.block_size] = torch.matmul(torch.matmul(self.Q, input[:, :, i*self.block_size : (i+1)*self.block_size, j*self.block_size : (j+1)*self.block_size]), self.Q.permute(1,0).contiguous() )
               
        return output 

    def forward(self, x):
        #return self.ycbcr_to_freq( self.rgb_to_ycbcr(x) )
        if (x.shape[1]==3):
          return self.ycbcr_to_freq( self.rgb_to_ycbcr(x) )
        else:
          return self.ycbcr_to_freq(x )  


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



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.001 ** (epoch // 100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        


parser = argparse.ArgumentParser(description='PyTorch tinyimagenet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
#parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed',                  default=0,         type=int,   help='Random seed')

parser.add_argument('--ckpt_dir',              default=None,      type=str,   help='Checkpoint dir. If set to none, default dir is used')
parser.add_argument('--ckpt_intrvl',           default=1,         type=int,   help='Number of epochs between successive checkpoints')
parser.add_argument('--num_epochs',            default=312,       type=int,   help='Number of epochs for backpropagation')
parser.add_argument('--resume_from_ckpt',      default=0,         type=int,   help='Resume from checkpoint?')
parser.add_argument('--tensorboard',           default=0,         type=int,   help='Log progress to TensorBoard')
global args
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
num_train = 50000
num_test  = 10000
img_size  = 32
inp_maps  = 3
num_cls   = 10
test_error_best = 100 
start_epoch     = 0
num_epochs      = args.num_epochs
end_epoch       = start_epoch+num_epochs
batch_size      = args.batch_size
ckpt_dir         = args.ckpt_dir
ckpt_intrvl      = args.ckpt_intrvl
resume_from_ckpt = True if args.resume_from_ckpt else False
#model_str_use    = 'vgg11_cifar100_ann'+'_bs'+str(batch_size)+'_new_'+str(args.lr)+'lrby5_every30epoch'
model_str_use    = 'vgg9_cifar10_ann_lr.1_.1by100'+'_bs'+str(batch_size)+'_pixelexpanded_4avgpool'
#model_str_use    = 'vgg13_tinyimgnet_4*4dctbnmaxpool_ann_lr.01_.1by100'+'_bs'+str(batch_size)+'_wd1e-4'
if(ckpt_dir is None):
   ckpt_dir = '/home/vgg9_snn_surrgrad_backprop/CHECKPOINTS/'+model_str_use
   ckpt_dir = os.path.expanduser(ckpt_dir)
   if(ckpt_intrvl > 0):
      if(not os.path.exists(ckpt_dir)):
         os.mkdir(ckpt_dir)
ckpt_fname  = ckpt_dir+'/ckpt.pth'
# Use TensorBoard?
tensorboard = True if args.tensorboard else False

# Data
print('==> Preparing data..')

#dataset             = 'tinyIMAGENET' # {'CIFAR10', 'CIFAR100', 'IMAGENET'}
dataset             = 'CIFAR10'
#usual
normalize       = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

# usual imgnet stat from repos
#normalize       = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

# calculated itiny-mgnet stat 
#normalize       = transforms.Normalize(mean = [0.48, 0.448, 0.3975], std = [0.277, 0.269, 0.282])


transform_train = transforms.Compose([
                     transforms.RandomCrop(32, padding=4),
                     transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     normalize])
transform_test  = transforms.Compose([transforms.ToTensor(), normalize])

if dataset == 'CIFAR10':
    trainset    = datasets.CIFAR10(root = './cifar_data', train = True, download = True, transform = transform_train)
    testset     = datasets.CIFAR10(root='./cifar_data', train=False, download=True, transform=                               transform_test)
    labels      = 10

elif dataset == 'CIFAR100':
    trainset    = datasets.CIFAR100(root = './cifar_data', train = True, download = True, transform = transform_train)
    testset     = datasets.CIFAR100(root='./cifar_data', train=False, download=True, transform=                               transform_test)
    labels      = 100

elif dataset == 'IMAGENET':
    labels      = 1000
    traindir    = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'train')
    valdir      = os.path.join('/local/scratch/a/imagenet/imagenet2012/', 'val')
    trainset    = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    testset     = datasets.ImageFolder(
                        valdir,
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])) 
elif dataset == 'tinyIMAGENET':
    labels      = 200
    # adding the tinyimagenet directory
    traindir    = os.path.join('/home/nano01/a/banerj11/srinivg_BackProp_CIFAR10/sayeed/tiny-imagenet-200/', 'train')
    valdir      = os.path.join('/home/nano01/a/banerj11/srinivg_BackProp_CIFAR10/sayeed/tiny-imagenet-200/', 'val')
   
#    traindir    = os.path.join('/local/scratch/a/chowdh23/data/tiny-imagenet-200/', 'train')
#    valdir      = os.path.join('/local/scratch/a/chowdh23/data/tiny-imagenet-200/', 'val')
    trainset    = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(64),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    testset     = datasets.ImageFolder(
                        valdir,
                        transforms.Compose([
                            #transforms.Resize(256),
                            #transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))

trainloader    = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader     = DataLoader(testset, batch_size=batch_size, shuffle=False)



# Model
print('==> Building model..')
model = VGG('VGG9', labels=labels)
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
model = model.cuda()
model = torch.nn.DataParallel(model).cuda()

use_cuda =torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
m=DCT2(block_size=4, device = device).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=False)
if(resume_from_ckpt):
   ckpt            = torch.load(ckpt_fname)
   start_epoch     = ckpt['start_epoch']
   end_epoch       = start_epoch+num_epochs
   test_error_best = ckpt['test_error_best']
   epoch_best      = ckpt['epoch_best']
#   train_time      = ckpt['train_time']
   model.load_state_dict(ckpt['model_state_dict'])
   optimizer.load_state_dict(ckpt['optim_state_dict'])
   print('##### Loaded ANN_VGG from {}\n'.format(ckpt_fname))


print('********** ANN training and evaluation **********')
for epoch in range(start_epoch, end_epoch):
    train_loss = AverageMeter()
    test_loss = AverageMeter()
#    model.use_max_out_over_time = use_max_out_over_time
#    model.module.updt_tend(t_end_updt)
    model.train()
    adjust_learning_rate(optimizer, epoch)
    


    for i, data in enumerate(trainloader):
#        print("Epoch: {}/{};".format(epoch+1, end_epoch), "Training batch:{}/{};".format(i+1, math.ceil(num_train/batch_size)))

#        start_time = time.time()
        # Load the inputs and targets
        inputs, targets = data
        #targets=torch.from_numpy(np.eye(num_cls)[targets])
        
        inputs, targets = inputs.cuda(), targets.cuda()
        if dataset=='CIFAR10' or dataset=='CIFAR100':
            inputs =rin(inputs)
        #inputs =m(inputs)
        
        

        # Reset the gradients
        optimizer.zero_grad()

        # Perform forward pass and compute the target loss
        output = model(inputs)
        #output= F.softmax(output,dim=1)
      
        #b=targets.float()
        loss   = criterion(output, targets)
        train_loss.update(loss.item(), targets.size(0))

        # Perform backward pass and update the weights
        loss.backward()
        optimizer.step()
#        end_time = time.time()
#        train_time += (end_time-start_time)/3600
        
    
    # Print error measures and log progress to TensorBoard
    train_loss_per_epoch = train_loss.avg
#    print("Epoch: {}/{};".format(epoch+1, end_epoch), "########## Training loss: {}".format(train_loss_per_epoch))
#    if(tensorboard):
#       log_value('train_loss', train_loss_per_epoch, epoch)

    # Evaluate classification accuracy on the test set
#    model.use_max_out_over_time = False
#    model.module.updt_tend(t_end)
    correct_pred_top1 = 0
    correct_pred_topk = 0
    model.eval()
    with torch.no_grad():
         for j, data in enumerate(testloader, 0):
#             print("Epoch: {}/{};".format(epoch+1, end_epoch), "Test batch: {}/{}".format(j+1, math.ceil(num_test/batch_size)))
             images, labels = data

             images, labels = images.cuda(), labels.cuda()
             if dataset=='CIFAR10' or dataset=='CIFAR100':
                 images =rin(images)
                      
             
             #images =m(images)
             
             
             out     = model(images)
             loss1   = criterion(out, labels)
             test_loss.update(loss1.item(), labels.size(0))
             _, predicted = out.max(1)
#             total += targets.size(0)
             correct_pred_top1 += predicted.eq(labels).sum().item()
             #print(correct_pred_top1)
#             _, pred = out.topk(topk, 1, True, True)
#             pred    = pred.t()
#             correct = pred.eq(labels.view(1, -1).expand_as(pred))
#             correct_pred_top1 += correct[:1].view(-1).float().sum(0, keepdim=True)
#             correct_pred_topk += correct[:topk].view(-1).float().sum(0, keepdim=True)

    test_loss_per_epoch = test_loss.avg        
#    print("Epoch: {}/{};".format(epoch+1, end_epoch), "########## Test loss: {}".format(test_loss_per_epoch))
    if(tensorboard):
       log_value('test_loss', test_loss_per_epoch, epoch)
    # Print error measures and log progress to TensorBoard
    test_error_top1 = (1-(correct_pred_top1/num_test))*100
#    test_error_topk = (1-(correct_pred_topk/num_test))*100
    test_error_chgd = False
    if(test_error_top1 < test_error_best):
       test_error_best = test_error_top1
       epoch_best      = epoch
       test_error_chgd = True
    print("Epoch: {}/{};".format(epoch_best+1, end_epoch), "########## Test error (top1-best): {:.2f}%".format(test_error_best))
    print("Epoch: {}/{};".format(epoch+1,      end_epoch), "########## Test error (top1-cur) : {:.2f}%".format(test_error_top1))
#    print("Epoch: {}/{};".format(epoch+1,      end_epoch), "########## Test error (top"+str(topk)+"-cur) : {:.2f}%".format(test_error_topk[0]))
    if(tensorboard):
       log_value('test_error (top1-best)', test_error_best, epoch)
       log_value('test_error (top1)', test_error_top1, epoch)
#       log_value('test_error (top'+str(topk)+')', test_error_topk, epoch)

    # Checkpoint SNN training and evaluation states
    if((ckpt_intrvl > 0) and ((epoch == end_epoch-1) or test_error_chgd)):
       print('=========== Checkpointing ANN training and evaluation states')
       ckpt = {'model_state_dict': model.state_dict(),
               'optim_state_dict': optimizer.state_dict(),
               'start_epoch'     : epoch+1,
               'test_error_best' : test_error_best,
               'epoch_best'      : epoch_best}
#               'train_time'      : train_time}
       if(test_error_chgd):
          torch.save(ckpt, ckpt_fname)





