import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights, ResNet152_Weights

import os

class feat_model(nn.Module):
    def __init__(self, backbone, pretrained):
        super(feat_model, self).__init__()
        self.backbone = backbone
        self.pretrained = pretrained

        if backbone == 'resnet50':
            model = getattr(torchvision.models, backbone)(weights=ResNet50_Weights.IMAGENET1K_V2)
        elif backbone == 'resnet152':
            model = getattr(torchvision.models, backbone)(weights=ResNet152_Weights.IMAGENET1K_V2)
        else:
            model = getattr(torchvision.models, backbone)()

        # model = getattr(torchvision.models, backbone)()
        module = list(model.children())
        if backbone[:3] == 'res':
            self.CNN = nn.Sequential(*module[:-2])
            self.out_features = module[-1].in_features
        elif backbone[:3] == 'den':
            self.CNN = nn.Sequential(*module[:-1])
            self.out_features = module[-1].in_features
        elif backbone[:3] == 'mob':
            self.CNN = nn.Sequential(*module[:-1])
            self.out_features = module[-1][1].in_features
        else:
            raise('BackboneError: wrong input backbone name: %s.'%backbone)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.CNN(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class classifier(nn.Module):
    def __init__(self, in_features, n_class):
        super(classifier, self).__init__()
        self.in_features = in_features
        self.n_class = n_class

        self.dropout = nn.Dropout()
        self.fc = nn.Linear(in_features, n_class, bias=False)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x

class create_model(nn.Module):
    def __init__(self, backbone, n_class, pretrained=True):
        super(create_model, self).__init__()
        self.backbone = backbone
        self.n_class = n_class
        self.pretrained = pretrained

        self.f_model = feat_model(backbone, pretrained)
        self.c_model = classifier(self.f_model.out_features, n_class)

    def forward(self, x):
        x = self.f_model(x)
        x = self.c_model(x)
        return x


if __name__ == "__main__":
    from tqdm import tqdm

    backbone = ['resnet50', 
                'resnet101',
                'resnet152',
                'resnext50_32x4d',
                'resnext101_32x8d',
                'densenet121',
                'densenet161',
                'densenet169',
                'densenet201',
                'mobilenet_v2'
                ]

    n_class = 200
    x = torch.randn((4, 3, 224, 224), device='cpu')
    model = create_model(backbone[0], n_class).to('cpu')

    W = model.c_model.fc.weight
    print('W size: {}'.format(W.size()))
    What = torch.mm(W, W.transpose(0, 1))
    I = torch.eye(What.size(0))
    print('I size: {}'.format(I.size()))
    print(I)
    

    '''
    pbar = tqdm(backbone)
    for b in pbar:
        pbar.set_description('Backbone: {}, x: {}, y: {}'.format(b, list(x.size()), list(y.size())))
    '''

    '''
    model = create_model(backbone[0], n_class).to('cuda')
    y = model(x)
    print('x: {}, y: {}'.format(list(x.size()), list(y.size())))
    '''
