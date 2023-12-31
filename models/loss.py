import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
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

def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):

    b_ = 0.5 * int(features.size(0))

    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, labels


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def kl_div(a:torch.Tensor, b:torch.Tensor):
    return torch.sum(a*torch.log(a/b))

def sup_constrative_loss(x:torch.Tensor, y:torch.Tensor,label,args):
    mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).bool()
    size = x.shape[0]
    loss = torch.tensor(0.0).to(args.device)
    for i in range(size):
        for j in range(size):
            if mask[i][j]:
                loss += kl_div(x[i],y[j])
    return loss/torch.sum(mask.float())

def contrastive_loss(pos, neg, gamma,alpha):
        pos_sg = pos.clone().detach()
        pos_sg_all = pos_sg
        neg_sg = neg.clone().detach()
        neg_sg_all = neg_sg

        if gamma == 1:
            loss_1 = -pos.mean()
        else:
            e_pos_1 = ((gamma-1)*pos_sg).exp()
            e_pos_all_1 = ((gamma-1) * pos_sg_all).exp()
            denom_1 = e_pos_all_1.mean()
            loss_1 = -torch.mean(pos * e_pos_1) / denom_1

        e_pos_2 = (gamma * pos_sg).exp()
        e_neg_2 = (gamma * neg_sg).exp()
        e_pos_all_2 = (gamma * pos_sg_all).exp()
        e_neg_all_2 = (gamma * neg_sg_all).exp()
        denom_2 = alpha * e_pos_all_2.mean() + (1-alpha)*e_neg_all_2.mean()
        num_1 = torch.mean(pos * e_pos_2.detach())
        num_2 = torch.mean(neg * e_neg_2.detach())
        loss_2 = (alpha * num_1 + (1 - alpha) * num_2) / denom_2
        loss = loss_1 + loss_2
        return loss

def renyi_div(a:torch.Tensor, b:torch.Tensor,alpha=2.0):
    return torch.log(torch.sum(torch.pow(a,alpha)*torch.pow(b,1-alpha)))/(alpha-1)

def renyi_loss(x:torch.Tensor, y:torch.Tensor):
    size = x.shape[0]
    loss = 0.0
    for i in range(size):
        loss += renyi_div(x[i],y[i])
    return loss/size

def sup_renyi_loss(x:torch.Tensor, y:torch.Tensor,label):
    mask = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).bool()
    size = x.shape[0]
    loss = 0.0
    for i in range(size):
        for j in range(size):
            if mask[i][j]:
                loss += renyi_div(x[i],y[j]) + renyi_div(1-x[i],1-y[j])
    return loss/torch.sum(mask.float())

class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1,beta = 1.0):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.beta = beta    
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # student_out = student_output / self.student_temp
        student_out = (student_output / self.student_temp).softmax(dim=-1)
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                # loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss = renyi_loss( q,student_out[v]) + self.beta*renyi_loss(1-q,1-student_out[v])
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
