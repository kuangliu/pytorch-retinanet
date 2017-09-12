'''Init RestinaNet50 with pretrained ResNet50 model.'''
import torch

from fpn import FPN50
from retinanet import RetinaNet


# Download pretrained ResNet50 params from:
#   'https://download.pytorch.org/models/resnet50-19c8e357.pth',
print('Loading pretrained ResNet50 model..')
d = torch.load('./model/resnet50.pth')

print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
net = RetinaNet()
net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), 'net.pth')
