# Code was partial adapted and modified from https://github.com/Tramac/Fast-SCNN-pytorch/blob/master/utils/loss.py

"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_label)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, input, target, **kwargs):
        inputs = tuple(list(input) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs)


class SoftmaxCrossEntropyOHEMLoss(nn.Module):
    def __init__(self, ignore_label=-1, thresh=0.7, min_kept=256, use_weight=False, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            print("w/ class balance")
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
                                        1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                                        1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)
        else:
            print("w/o class balance")
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))

        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = input_prob[:, valid_flag]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]

        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        valid_flag_new = input_label != self.ignore_label
        # print(np.sum(valid_flag_new))
        target = Variable(torch.from_numpy(input_label.reshape(target.size())).long().cuda())

        return self.criterion(predict, target)


class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
    def __init__(self, aux=False, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_label=ignore_index, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, input, target, **kwargs):
        inputs = tuple(list(input) + [target])
        if self.aux:
            return self._aux_forward(*inputs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs)
        

# hack for u2net
class MultiScaleCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weight=None, reduction='mean', ignore_index=-1):
        super(MultiScaleCrossEntropyLoss, self).__init__(weight=weight, reduction=reduction, ignore_index=ignore_index)
    
    def forward(self, inputs, target):
        """
        Args:
            inputs: List of torch.Tensor (d0, d1, ..., d6) 
                    Each tensor shape: (B, C, H, W) where C=num_classes (e.g., 2 for binary).
            target: torch.Tensor (B, H, W) with class indices (0 or 1).
        Returns:
            loss0: Loss from the first scale (d0).
            total_loss: Sum of losses across all scales.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        losses = []
        for i, pred in enumerate(inputs):
            loss = super().forward(pred, target)
            losses.append(loss)

        return losses[0], sum(losses)
    

def boundary_loss(pred:torch.Tensor, target:torch.Tensor):
    """"create edge mask (1 at edges, 0 elsewhere)"""
    if target.dim() == 3:
        target = target.unsqueeze(1) # [B, 1, H, W]
    kernel = torch.ones(1,1,3,3).to(pred.device)
    padded_target = F.pad(target.float(), (1,1,1,1), mode='reflect')
    edge_mask = F.conv2d(padded_target, kernel, stride=1, padding=0) - target.float()
    edge_mask = (edge_mask > 0).float()

    ce_loss = F.cross_entropy(pred, target.squeeze(1).long(), reduction='none')
    return (ce_loss*edge_mask.squeeze(1)).mean()

def gradient_loss(pred_softmax, target):
    # pred_softmax: [B, C=2, H, W] (after softmax)
    # target: [B, H, W] (values 0 or 1)
    
    # Compute gradients
    pred_grad_x = torch.abs(pred_softmax[:, 1, :, 1:] - pred_softmax[:, 1, :, :-1])  # Sky class
    pred_grad_y = torch.abs(pred_softmax[:, 1, 1:, :] - pred_softmax[:, 1, :-1, :])
    
    target_grad_x = torch.abs(target[:, :, 1:] - target[:, :, :-1])
    target_grad_y = torch.abs(target[:, 1:, :] - target[:, :-1, :])
    
    grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return grad_loss


class MixedEdgeAwareCrossEntropyLoss(nn.Module):
    def __init__(self, ce_weight=1.0, boundary_weight=0.3, grad_weight=0.2):
        super().__init__()
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.grad_weight = grad_weight

    def forward(self, pred, target):
        # pred: Raw logits [B, 2, H, W]
        # target: Ground truth [B, H, W] (values 0 or 1)

        ce_loss = F.cross_entropy(pred, target.long())
        bd_loss = boundary_loss(pred, target)
        pred_softmax = F.softmax(pred, dim=1)
        grad_loss = gradient_loss(pred_softmax, target)

        # Weighted sum
        total_loss = (
            self.ce_weight * ce_loss + 
            self.boundary_weight * bd_loss + 
            self.grad_weight * grad_loss
        )
        return total_loss