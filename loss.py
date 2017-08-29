from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    num_classes = 21

    def __init__(self):
        super(FocalLoss, self).__init__()

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.75
        gamma = 2
        logp = F.log_softmax(x)
        p = logp.exp()
        w = alpha*(y>0).float() + (1-alpha)*(y==0).float()
        wp = w.view(-1,1) * (1-p).pow(gamma) * logp
        return F.nll_loss(wp, y, size_average=False)

    def focal_loss_alt(self,x,y):
        '''Focal loss alternate described in appendix.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.75
        gamma = 2
        beta = 1
        x[y.view(-1,1).expand_as(x)==0] *= -1
        w = alpha*(y>0).float() + (1-alpha)*(y==0).float()
        wp = w.view(-1,1)*F.log_softmax(gamma*x+beta)
        return F.nll_loss(wp, y, size_average=False) / gamma

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, num_classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        print('loc_loss: %.3f | cls_loss: %.3f' % (loc_loss.data[0], cls_loss.data[0]/num_pos), end=' | ')
        loss = loc_loss + cls_loss/num_pos
        return loss
