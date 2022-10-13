# Third party import
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np

class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)

        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
        weights = self.kernel(all_labels, all_labels) # [2N, 2N]
        weights = weights * (1 - np.eye(2*N)) # puts 0 on the diagonal
        weights /= weights.sum(axis=1)
        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = -1./N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__, self.temperature,
                                                         self.kernel.__name__, self.sigma)


class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)

        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)

        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss