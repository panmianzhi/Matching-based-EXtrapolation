import logging
from typing import Optional
import random

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ocpmodels.common import distutils


class L2MAELoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)

class AtomwiseL2Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        assert natoms.shape[0] == input.shape[0] == target.shape[0]
        assert len(natoms.shape) == 1  # (nAtoms, )

        dists = torch.norm(input - target, p=2, dim=-1)
        loss = natoms * dists

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        
########## quantileloss #########
#################################
class QuantileLoss(nn.Module):
    def __init__(self, tau: float=0.5, reduction: str = "sum"):
        super().__init__()
        self.tau = tau
        self.reduction = reduction
        assert reduction in ["mean", "sum"]
        logging.info(f'Use quantile loss, tau is {tau}.')

    def forward(self, outputs, targets):
        residuals = outputs - targets
        loss = torch.where(residuals < 0, (self.tau - 1) * residuals, self.tau * residuals)
        
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)


############## RankSim ###############
# https://github.com/BorealisAI/ranksim-imbalanced-regression/blob/main/imdb-wiki-dir/ranksim.py
######################################
def rank(seq):
    return torch.argsort(torch.argsort(seq).flip(1))

def rank_normalised(seq):
    return (rank(seq) + 1).float() / seq.size()[1]

class TrueRanker(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequence, lambda_val):
        rank = rank_normalised(sequence)
        ctx.lambda_val = lambda_val
        ctx.save_for_backward(sequence, rank)
        return rank

    @staticmethod
    def backward(ctx, grad_output):
        sequence, rank = ctx.saved_tensors
        assert grad_output.shape == rank.shape
        sequence_prime = sequence + ctx.lambda_val * grad_output
        rank_prime = rank_normalised(sequence_prime)
        gradient = -(rank - rank_prime) / (ctx.lambda_val + 1e-8)
        return gradient, None

class RankSimLoss(nn.Module):
    def __init__(
            self, 
            reduction: str = "sum",
            lambda_val: int=2
        ) -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum", "none"]
        self.lambda_val = lambda_val
        logging.info(f'Use ranksim loss, lambda_val is {lambda_val}.')
    
    def forward(
        self, 
        features: torch.Tensor, 
        targets: torch.Tensor, 
    ):
        loss = 0

        # Reduce ties and boost relative representation of infrequent labels by computing the 
        # regularizer over a subset of the batch in which each label appears at most once
        batch_unique_targets = torch.unique(targets)
        if len(batch_unique_targets) < len(targets):
            sampled_indices = []
            for target in batch_unique_targets:
                sampled_indices.append(random.choice((targets == target).nonzero()[:,0]).item())
            x = features[sampled_indices]
            y = targets[sampled_indices]
        else:
            x = features
            y = targets

        # Compute feature similarities
        xxt = torch.matmul(
            F.normalize(x.view(x.size(0),-1)), 
            F.normalize(x.view(x.size(0),-1)).permute(1,0)
        )

        # Compute ranking similarity loss
        for i in range(len(y)):
            label_ranks = rank_normalised(-torch.abs(y[i] - y).transpose(0,1))
            feature_ranks = TrueRanker.apply(xxt[i].unsqueeze(dim=0), self.lambda_val)
            loss += F.mse_loss(feature_ranks, label_ranks, reduction=self.reduction)
        
        return loss


############# CONR: CONTRASTIVE REGULARIZER FOR DEEP IMBALANCED REGRESSION ########
# source from https://github.com/BorealisAI/ConR/blob/main/imdb-wiki-dir/loss.py
###################################################################################
class ConRLoss(nn.Module):
    def __init__(
        self, 
        reduction: str = "sum",
        threshold=1,
        temperature=0.07,
        eta=0.01,
        global_con=False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]
        self.threshold = threshold
        self.temp = temperature
        self.eta = eta
        self.global_con = global_con
        self.all_gather = distutils.AllGather.apply
        logging.info(f'ConRLoss, threshold:{threshold}. temp:{temperature}. eta:{eta}, global_con:{global_con}')

    # To support the global contrastive learning proposed by: 
    # FLAVA: A Foundational Language And Vision Alignment Model.
    # if want to use origin CONR, just let:
    # self.global_con = False
    def forward(self, 
                feat1, 
                targets1, 
                preds1):
        '''

        feat1: (B1, D)d
        targets1: (B1, 1)
        preds1: (B1, 1)
        '''
        if self.global_con:
            feat2 = self.all_gather(feat1)
            targets2 = self.all_gather(targets1)
            preds2 = self.all_gather(preds1)
        else:
            feat2 = feat1
            targets2 = targets1
            preds2 = preds1

        assert len(feat1) == len(targets1) and len(feat1) == len(preds1)
        assert len(feat2) == len(targets2) and len(feat2) == len(preds2)
        assert len(feat1) <= len(feat2)

        l_dist= torch.abs(targets1 - targets2.flatten()[None,:]) # (B1, B2)
        p_dist= torch.abs(preds1 - preds2.flatten()[None,:]) # (B1, B2)

        pos_i = l_dist.le(self.threshold) # y dist small enough is pos
        neg_i = ((~ (l_dist.le(self.threshold))) * (p_dist.le(self.threshold))) # y dist big & pred dist small 

        # for i in range(pos_i.shape[0]):
        #     pos_i[i][i] = 0 # neglect self
        pos_i.fill_diagonal_(0)

        prod = F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).t() / self.temp # (B1, B1)
        pos = prod * pos_i
        neg = prod * neg_i
        
        pushing_w = torch.exp(l_dist * self.eta) # (B, B)
        neg_exp_dot=(pushing_w * torch.exp(neg) * neg_i).sum(1) # (B)

        # For each query sample, if there is no negative pair, zero-out the loss.
        no_neg_flag = neg_i.sum(1).bool()
        # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
        denom=pos_i.sum(1)

        loss = (
                -torch.log(
                    torch.div(
                        torch.exp(pos),
                        (torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)
                    )
                ) * pos_i
        ).sum(1) / (denom + 1e-6) # (B1,)

        loss = (loss * no_neg_flag).unsqueeze(-1) # (B1, 1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError


################# balanced MSE ##################
### modified to global version by panmz #########
### https://github.com/jiawei-ren/BalancedMSE ###
#################################################
class BMCLoss(nn.Module):
    def __init__(
        self, 
        reduction: str = "sum",
        init_noise_sigma: float = 1.0,
        global_con=True,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        self.global_con = global_con
        self.all_gather = distutils.AllGather.apply
        logging.info(f'BMC - init noise_sigma: {init_noise_sigma}, global con: {global_con}')

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        '''
        pred: (B, 1)
        target: (B, 1)
        '''
        if self.global_con:
            target_all = self.all_gather(target) # (B', 1)

            local_bs = torch.tensor([pred.shape[0]], device=pred.device)
            all_bses = [torch.zeros_like(local_bs) for _ in range(distutils.get_world_size())]
            dist.all_gather(all_bses, local_bs) # batch size over all GPUs: [bs_1, bs_2, ..., bs_n]
            all_bses = torch.tensor(all_bses).view(-1)
            cumsum_bses = torch.cumsum(all_bses, dim=0) # [bs_1, bs_1+bs_2, bs_1+bs_2+bs_3, ...]

            cumsum_bses = [0] + cumsum_bses.tolist()
            assert cumsum_bses[-1] == target_all.shape[0]

            cls_tgt = torch.arange(pred.shape[0]).to(pred.device) + cumsum_bses[distutils.get_rank()]
        else:
            target_all = target
            cls_tgt = torch.arange(pred.shape[0]).to(pred.device)

        noise_var = (self.noise_sigma ** 2).to(pred.device)
        logits = -(pred - target_all.T).pow(2) / (2 * noise_var) # (B, B')

        loss = F.cross_entropy(logits, cls_tgt, reduction=self.reduction)
        loss = loss * (2 * noise_var).detach()

        return loss
    

### cross domain contrastive learning ####
##########################################
class CrossDomainCon(nn.Module):
    def __init__(
        self, 
        reduction: str = "sum",
        threshold=0.5,
        temperature=0.07,
        global_con=False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]
        self.threshold = threshold
        self.temp = temperature
        self.global_con = global_con
        self.all_gather = distutils.AllGather.apply
        logging.info(f'CrossDomainCon, threshold:{threshold}. temp:{temperature}. global_con:{global_con}')

    def compute_loss(self, repr_main, pseudo_label, repr_aux, aux_label):
        '''
        repr_main: (B1, h)
        pseudo_label: (B1, 1)
        repr_aux: (B2, h)
        aux_label: (B2, 1)
        '''
        if self.global_con:
            repr_aux_tmp = self.all_gather(repr_aux)
            aux_label_tmp = self.all_gather(aux_label)
        else:
            repr_aux_tmp = repr_aux
            aux_label_tmp  = aux_label

        assert len(repr_main) == len(pseudo_label)
        assert len(repr_aux_tmp) == len(aux_label_tmp)

        l_dist = torch.abs(pseudo_label - aux_label_tmp.flatten()[None,:]) # (B1, B2)
        con_label = l_dist.le(self.threshold) # y dist small enough is pos

        sim_mat = F.normalize(repr_main, dim=1) @ F.normalize(repr_aux_tmp, dim=1).t() / self.temp # (B1, B2)
        
        exp_sim = torch.exp(sim_mat)
        exp_positives = torch.sum(exp_sim * con_label, dim=1) + 1e-6 # (B1,)
        exp_negatives = torch.sum(exp_sim * (~ con_label), dim=1) # (B1,)
        
        # 计算损失
        log_prob = torch.log(exp_positives / (exp_positives + exp_negatives))
        loss = -log_prob.view(-1, 1) # (B1, 1)
        assert len(loss) == len(repr_main)

        if self.reduction == 'mean':
            raise NotImplementedError
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError
        
    def forward(self, repr_main, pseudo_label, repr_aux, aux_label, sample_weight=None):
        loss = self.compute_loss(repr_main, pseudo_label, repr_aux, aux_label)
        
        if self.reduction == 'sum':
            num_samples = repr_main.shape[0]
            num_samples = distutils.all_reduce(
                num_samples, device=loss.device
            ) # sum num_samples of all processes when DDP
            return loss * distutils.get_world_size() / num_samples
        else:
            return loss


class DDPLoss(nn.Module):
    def __init__(
        self, loss_fn, loss_name: str = "mae", reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_name = loss_name
        self.reduction = reduction
        assert reduction in ["mean", "mean_all", "sum"]

        # for forces, we want to sum over xyz errors and average over batches/atoms (mean)
        # for other metrics, we want to average over all axes (mean_all) or leave as a sum (sum)
        if reduction == "mean_all":
            self.loss_fn.reduction = "mean"
        else:
            self.loss_fn.reduction = "sum"

    # remove the support of atom-wise prediction
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        embeddings: torch.Tensor = None,
        batch_size: Optional[int] = None,
        sample_weights: Optional[torch.Tensor] = None
    ):
        '''
        input: prediction of samples: (B, 1)
        target: target of samples: (B, 1)
        embeddings: embeddings of samples: (B, D)
        '''
        assert input.shape == target.shape, \
            f"Mismatched shapes: {input.shape} and {target.shape}"
        if embeddings is not None:
            assert len(input) == len(embeddings), \
                f"Mismatched length: {input.shape} and {embeddings.shape}"
        
        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)
        
        # if we need to reweight each sample's loss
        if sample_weights is not None:
            origin_loss_reduction = self.loss_fn.reduction
            self.loss_fn.reduction = 'none'

        # different loss needs different call
        if self.loss_name == 'ranksim':
            assert embeddings is not None
            loss = self.loss_fn(embeddings, target)
        elif self.loss_name == 'conr':
            loss = self.loss_fn(feat1=embeddings, targets1=target, preds1=input)
        else:
            loss = self.loss_fn(input, target)

        # reweight !
        if sample_weights is not None:
            assert len(sample_weights) == len(loss), \
                f'sample_weights: {sample_weights.shape}, loss: {loss.shape}'
            loss *= sample_weights.view_as(loss)
            if origin_loss_reduction == 'sum': loss = loss.sum()
            elif origin_loss_reduction == 'mean': loss = loss.mean()
            else: raise ValueError

            self.loss_fn.reduction = origin_loss_reduction

        # avg across gpus
        if self.reduction == "mean":
            num_samples = (
                batch_size
                if self.loss_name.startswith("atomwise")
                else input.shape[0]
            )
            num_samples = distutils.all_reduce(
                num_samples, device=input.device
            ) # sum num_samples of all processes when DDP
            # Multiply by world size since gradients are averaged across DDP replicas
            return loss * distutils.get_world_size() / num_samples
        else:
            # if reduction is sum or mean over all axes, no other operations are needed
            return loss
