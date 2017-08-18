'''Encode target locations and class labels.'''
import math
import torch

from utils import meshgrid


class DataEncoder:
    def __init__(self):
        self.anchor_areas = [32*32., 64*64., 128*128., 256*256., 512*512.]  # p3 -> p7
        self.aspect_ratios = [1/2., 1/1., 2/1.]
        self.scale_ratios = [1., pow(2,1/3.), pow(2,2/3.)]
        self.anchor_wh = self._get_anchor_wh()

    def _get_anchor_wh(self):
        '''Compute anchor width and height for each feature map.

        Returns:
          anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
        '''
        anchor_wh = []
        for s in self.anchor_areas:
            for ar in self.aspect_ratios:  # w/h = ar
                h = math.sqrt(s/ar)
                w = ar * h
                for sr in self.scale_ratios:  # scale
                    anchor_h = h*sr
                    anchor_w = w*sr
                    anchor_wh.append([anchor_w, anchor_h])
        num_fms = len(self.anchor_areas)
        return torch.Tensor(anchor_wh).view(num_fms, -1, 2)

    def _get_anchor_boxes(self, input_size):
        '''Compute anchor boxes for each feature map.

        Args:
          input_size: (int) model input size.

        Returns:
          boxes: (list) anchor boxes for each feature map. Each of size [fmh,fmw,9,4].
        '''

        num_fms = len(self.anchor_areas)
        fm_sizes = [(1.*input_size)/pow(2,i+3) for i in range(num_fms)]  # p3 -> p7 feature map sizes
        fm_sizes = [int(math.ceil(x)) for x in fm_sizes]
        # TODO: make sure computed fm_sizes is the same as feature_map sizes

        boxes = []
        for i in range(num_fms):
            fm_size = fm_sizes[i]
            xy = meshgrid(fm_size, swap_dims=True) + 0.5  # [fm_size*fm_size,2]
            xy = xy.view(fm_size,fm_size,1,2).expand(fm_size,fm_size,9,2)
            wh = self.anchor_wh[i].view(1,1,9,2).expand(fm_size,fm_size,9,2)
            box = torch.cat([xy,wh], 3)
            boxes.append(box)
        return boxes

    def encode(self, boxes, labels, input_size):
        '''Encode target bounding boxes and class labels.

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax) in range [0,1], sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          input_size: (int) model input size.

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [5,4,fmh,fmw].
          cls_targets: (tensor) encoded class labels, sized [5,20,fmh,fmw].
          box_targets: (tensor) truth boxes, sized [#obj,4].
        '''
        anchor_boxes = self._get_anchor_boxes(input_size)
        return anchor_boxes


encoder = DataEncoder()
boxes = encoder.encode(1, 2, 224)
box = boxes[0]
box
