
import torch
import torch.nn as nn
import math
torch.manual_seed(0)

cfg = {
	'VGG5' : [64, 'A', 128, 'D', 128, 'A'],
	'VGG9': [64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'A'],
	'VGG11': [64, 'A', 128, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'A'],
	'VGG13': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'A', 512, 'D', 512, 'A', 512, 'D', 512, 'A'],
    'VGG16': [64, 'D', 64, 'A', 128, 'D', 128, 'A', 256, 'D', 256, 'D', 256, 'A', 512, 'D', 512, 'D', 512, 'A', 512, 'D', 512, 'D', 512, 'D']
}




class VGG(nn.Module):
    def __init__(self, vgg_name, labels=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.labels 		= labels
        self.vgg_name = vgg_name
        self.classifier = self._make_fc_layers()
        
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
               #m.threshold = 0.999#0.75 #1.0
               n1 = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
               variance1 = math.sqrt(1. / (n1))  # math.sqrt(6. / (n + n1))
               m.weight.data.normal_(0, variance1)
               #m.bias.data.zero_()
               
            
            elif(isinstance(m, nn.Linear)):
               #m.threshold = 0.999               
               size = m.weight.size()
               fan_in = size[1]  # number of columns
               variance2 = math.sqrt(1.0 / (fan_in))  # + fan_out)) #math.sqrt(6.0 / (fan_in + fan_out))
               m.weight.data.normal_(0.0, variance2)
        
  
                
        
        

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            stride = 1
            if x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            elif x == 'D':
                layers += [nn.Dropout(0.2)]
            elif x=='M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, stride=stride, bias=False),
							nn.ReLU(inplace=True)
							]
                in_channels = x
        
        return nn.Sequential(*layers)

    
    def _make_fc_layers(self):
        layers = []
#        if self.vgg_name=='VGG16' & self.labels==1000:
        if self.vgg_name=='VGG9':
            layers += [nn.Linear(512*9, 4096, bias=False)]
        elif self.vgg_name=='VGG11':
            layers += [nn.Linear(512*9, 4096, bias=False)]
        elif self.vgg_name=='VGG13':
            layers += [nn.Linear(512*4, 4096, bias=False)]
        else:
            layers += [nn.Linear(128*8*8, 4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(4096, 4096, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Dropout(0.5)]
        layers += [nn.Linear(4096, self.labels, bias=False)]
        
        return nn.Sequential(*layers)

        
