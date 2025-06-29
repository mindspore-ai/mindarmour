import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import math
import numpy as np

class SimCLRLoss(nn.Cell):
    """Modified from https://github.com/wvangansbeke/Unsupervised-Classification."""

    def __init__(self, temperature, reduction="mean"):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.matmul = ops.MatMul()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.sum = ops.ReduceSum(keep_dims=True)
        self.mean = ops.ReduceMean()
        self.cast = ops.Cast()
        self.scatter = ops.ScatterNd()
        self.eye = ops.Eye()

    def construct(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """
        b, n, dim = features.shape
        assert n == 2
        
        # Create mask
        mask = self.eye(b, b, ms.float32)
        
        # Concatenate features
        contrast_features = ops.Concat(axis=0)(ops.Unstack(axis=1)(features))
        anchor = features[:, 0]
        
        # Dot product
        dot_product = self.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max = ops.ReduceMax(keep_dims=True)(dot_product, 1)
        logits = dot_product - logits_max
        
        # Create logits mask
        mask = ops.Tile()(mask, (1, 2))
        indices = ops.Reshape()(ops.arange(ms.Tensor([0, b], ms.int32), (-1, 1)))
        updates = ops.ZerosLike()(indices)
        logits_mask = ops.OnesLike()(mask) - self.scatter(indices, updates, ops.Shape()(mask))
        mask = mask * logits_mask
        
        # Log-softmax
        exp_logits = self.exp(logits) * logits_mask
        log_prob = logits - self.log(self.sum(exp_logits, 1))
        
        # Compute loss
        if self.reduction == "mean":
            loss = -self.mean((mask * log_prob).sum(1) / mask.sum(1))
        elif self.reduction == "none":
            loss = -((mask * log_prob).sum(1) / mask.sum(1))
        else:
            raise ValueError("The reduction must be mean or none!")

        return loss

class RCELoss(nn.Cell):
    """
    Reverse Cross Entropy Loss.
    prob_min=1e-13, one_hot=1e-20
    """
    def __init__(self, prob_min=1e-7, one_hot_min=1e-2, num_classes=10, reduction="mean"):
        super(RCELoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.prob_min = prob_min
        self.one_hot_min = one_hot_min
        self.softmax = nn.Softmax(axis=-1)
        self.log = ops.Log()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.clip = ops.clip_by_value
        self.onehot = nn.OneHot(depth=num_classes)

    def construct(self, x, target, weight=None):
        prob = self.softmax(x)
        prob = self.clip(prob, self.prob_min, 1.0)
        one_hot = self.onehot(target)
        one_hot = self.clip(one_hot, self.one_hot_min, 1.0)
        loss = -1 * self.sum(prob * self.log(one_hot), -1)
        
        if weight is not None:
            loss = loss * weight
        
        if self.reduction == "mean":
            loss = self.mean(loss)
            
        return loss

class SCELoss(nn.Cell):
    """Symmetric Cross Entropy."""
    def __init__(self, alpha=0.1, beta=1.0, one_hot_min=1.0e-2, num_classes=10, reduction="mean"):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.one_hot_min = one_hot_min
        self.num_classes = num_classes
        self.reduction = reduction
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction=reduction)
        self.rce = RCELoss(one_hot_min=self.one_hot_min, 
                          num_classes=self.num_classes, 
                          reduction=self.reduction)

    def construct(self, x, target):
        ce_loss = self.ce(x, target)
        rce_loss = self.rce(x, target)
        loss = self.alpha * ce_loss + self.beta * rce_loss
        return loss

def get_criterion(criterion_config):
    if "ce" in criterion_config:
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, **criterion_config["ce"])
    elif "simclr" in criterion_config:
        criterion = SimCLRLoss(**criterion_config["simclr"])
    elif "sce" in criterion_config:
        criterion = SCELoss(**criterion_config["sce"])
    else:
        raise ValueError("Criterion {} is not supported.".format(criterion_config))
    return criterion