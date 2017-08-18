import torch
import torch.nn as nn

from fpn import RetinaFPN101
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    num_classes = 21

    def __init__(self):
        super(RetinaNet, self).__init__()
        self.fpn = RetinaFPN101()
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = [self.loc_head(fm) for fm in fms]
        cls_preds = [self.cls_head(fm) for fm in fms]
        return loc_preds, cls_preds

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)


def test():
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(1,3,224,224)))
    for (a,b) in zip(loc_preds, cls_preds):
        print(a.size())
        print(b.size())

# test()
