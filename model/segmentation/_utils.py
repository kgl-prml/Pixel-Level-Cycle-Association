from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        result["feat"] = features["out"]
        result["aux_feat"] = features["aux"]

        x = features["out"]
        x = self.classifier(x)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            result["aux"] = x

        return result
