"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import Optional
import math
import time
import json

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv
from scipy.stats import spearmanr, gmean
from scipy import linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch_geometric
from tqdm import tqdm

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.loss import RankSimLoss
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer

###### semantic autoencoder ########
def SAE(X, S, lamb: float=1.):
    '''
    X: sample repr (d1, N)
    S: label repr (d2, N)
    lamb: trade-off parameter
    '''
    A=S.dot(S.T)
    B=lamb*(X.dot(X.T))
    C=(1+lamb)*(S.dot(X.T))
    W=linalg.solve_sylvester(A,B,C)
    return W


####### sample y' for edm-nce #######
####################################
def gauss_density_centered(x, std):
    return torch.exp(-0.5*(x / std)**2) / (math.sqrt(2*math.pi)*std)

def gmm_density_centered(x, std):
    """
    Assumes dim=-1 is the component dimension and dim=-2 is feature dimension. Rest are sample dimension.
    """
    if x.dim() == std.dim() - 1:
        x = x.unsqueeze(-1)
    elif not (x.dim() == std.dim() and x.shape[-1] == 1):
        raise ValueError('Last dimension must be the gmm stds.')
    return gauss_density_centered(x, std).prod(-2).mean(-1)

def sample_gmm_centered(std, num_samples=1):
    num_components = std.shape[-1]
    num_dims = std.numel() // num_components

    std = std.view(1, num_dims, num_components) # (1, 1, 2)

    # Sample component ids
    k = torch.randint(num_components, (num_samples,), dtype=torch.int64)
    std_samp = std[0,:,k].t()

    # Sample from gaussion
    x_centered = std_samp * torch.randn(num_samples, num_dims)

    prob_dens = gmm_density_centered(x_centered, std) # density at x_centered
    prob_dens_zero = gmm_density_centered(torch.zeros_like(x_centered), std) # density at 0

    return x_centered, prob_dens, prob_dens_zero


####### inference for ebm #######
#################################
class DerivativeFreeOptimizer:
    """A simple derivative-free optimizer. Great for up to 5 dimensions."""
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        device: torch.device,
        noise_scale: float = 0.16,
        noise_shrink: float = 0.5,
        iters: int = 5,
        train_samples: int = 256,
        inference_samples: int = 500,
        infer_type: str = 'cosine',
        infer_cos_weight: float = 1.,
    ):
        self.device = device
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples

        assert infer_type in ["cosine", "energy", "mix"], f"bad infer type: {infer_type}"
        self.infer_type = infer_type
        self.infer_cos_weight = infer_cos_weight

        print(f'[EBM] lower bound: {self.lower_bound}, upper_bound: {self.upper_bound}, infer_samples: {self.inference_samples}, infer_type: {self.infer_type}', flush=True)

    def _sample(self, num_samples: int) -> torch.Tensor:
        """Helper method for drawing samples from the uniform random distribution."""
        size = (num_samples, 1)
        samples = np.random.uniform(self.lower_bound, self.upper_bound, size=size)
        return torch.as_tensor(samples, dtype=torch.float32, device=self.device)

    def sample(self, batch_size: int) -> torch.Tensor:
        samples = self._sample(batch_size * self.train_samples)
        return samples.reshape(batch_size, self.train_samples)

    def get_score(self, 
        x_embed: torch.Tensor, 
        samples: torch.Tensor,
        ebm: nn.Module
    ) -> torch.Tensor:
        assert len(x_embed) == len(samples)

        if self.infer_type == 'cosine':
            _, x_feat, y_feat = ebm.get_energy(x_embed, samples, ret_rep=True)
            cosine_similarity = F.cosine_similarity(x_feat, y_feat) # (batch_size*num_samples, )
            score = cosine_similarity.view(samples.shape[0], samples.shape[1])
        elif self.infer_type == 'energy':
            score = ebm.get_energy(x_embed, samples) # (B, inference_samples)
        else:
            energies, x_feat, y_feat = ebm.get_energy(x_embed, samples, ret_rep=True)
            cosine_similarity = F.cosine_similarity(
                x_feat, y_feat).view(samples.shape[0], samples.shape[1]) # (batch_size, num_samples)
            score = (1 - self.infer_cos_weight) * F.softmax(energies, dim=-1) + self.infer_cos_weight * F.softmax(cosine_similarity, dim=-1)
        return score

    @torch.no_grad()
    def infer(
        self, 
        x_embed: torch.Tensor, 
        ebm: nn.Module) -> torch.Tensor:
        """Optimize for the best action given a trained EBM."""
        '''
        x: (B, dim)
        '''

        noise_scale = self.noise_scale
        samples = self._sample(x_embed.size(0) * self.inference_samples) # (B * inference_samples, 1)
        samples = samples.view(x_embed.size(0), self.inference_samples) # (B, inference_samples)

        for _ in range(self.iters):
            # Compute energies.
            scores = self.get_score(x_embed, samples, ebm)
            # below is for numerical stability
            scores = scores.double()
            probs = F.softmax(scores, dim=-1).cpu()
            probs[torch.isnan(probs)] = 1e-6

            # Resample with replacement.
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True) # (B, inference_samples)
            samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs] # (B, inference_samples)

            # Add noise and clip to target bounds.
            samples = samples + torch.randn_like(samples) * noise_scale
            samples = samples.clamp(min=self.lower_bound, max=self.upper_bound)

            noise_scale *= self.noise_shrink

        # Return target with highest probability.
        scores = self.get_score(x_embed, samples, ebm)
        best_idxs = scores.argmax(dim=-1)
        return samples[torch.arange(samples.size(0)), best_idxs] # (B, 1)

@registry.register_trainer("mex")
class MEXTrainer(BaseTrainer):
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="wandb",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        name="ocp",
        early_stop=30,
        linear_probe=False,
    ):
        super().__init__(
            task=task,
            model=model,
            outputs=outputs,
            dataset=dataset,
            optimizer=optimizer,
            loss_fns=loss_fns,
            eval_metrics=eval_metrics,
            identifier=identifier,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            slurm=slurm,
            noddp=noddp,
            name=name,
            early_stop=early_stop,
            linear_probe=linear_probe,
        )
        
        self.neg_samples = 500
        self.infer_samples = 2000

        self.train_max_y = self.normalizers['y'].norm(
            self.config["dataset"].get('train_max_y', None)
        ).item()
        self.train_min_y = self.normalizers['y'].norm(
            self.config["dataset"].get('train_min_y', None)
        ).item()

        lower_bound, upper_bound = -10, 10 # inference label interval
        self.lower_bound = self.normalizers['y'].norm(lower_bound).item()
        self.upper_bound = self.normalizers['y'].norm(upper_bound).item()
        assert self.lower_bound <= self.train_min_y and self.upper_bound >= self.train_max_y

        self.stochastic_optimizer = DerivativeFreeOptimizer(
            lower_bound=self.lower_bound, 
            upper_bound=self.upper_bound, 
            device=self.device,
            noise_scale=0.5, noise_shrink=0.5, iters=5, 
            inference_samples=self.infer_samples, 
            train_samples=self.neg_samples,
            infer_type='cosine',
            infer_cos_weight=1.,
        )

    def load_loss(self) -> None:
        self.use_nce = False
        self.use_euc = False
        self.use_infonce = False
        for loss in self.config["loss_fns"]:
            # loss is a dict, e.g. {target_name: {fn:..., coefficient:...}, target_name: {fn:..., coefficient:...}}
            for target in loss:
                loss_name = loss[target].get("fn", None)
                coefficient = loss[target].get("coefficient", 1.)

                logging.info(f'loss_name: {loss_name}, coeff: {coefficient}')
                if loss_name == 'nce':
                    self.use_nce = True
                    self.nce_coeff = coefficient
                elif loss_name == 'info-nce':
                    self.use_infonce = True
                    self.infonce_coeff = coefficient
                elif loss_name == 'euc':
                    self.use_euc = True
                    self.euc_coeff = coefficient
                else:
                    raise ValueError
        
        assert not (self.use_nce and self.use_infonce), 'cannot use nce and infonce together'

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ): # here
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        if not hasattr(self, "early_stop") or self.early_stop is None: 
            self.early_stop = self.config["early_stop"]
        logging.info(f'Early stop patience is {self.early_stop}')

        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader) # self.step is load from checkpoint else 0

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(
                epoch_int, skip_steps
            )

            train_loader_iter = iter(self.train_loader)

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    loss_dict = self._compute_loss(batch)
                    loss = loss_dict['loss']
                    
                    # get train mae
                    # with torch.no_grad():
                    #     out_tr = self._forward(batch)
                # self.metrics = self._compute_metrics(out_tr, batch, self.evaluator, self.metrics)

                for k, v in loss_dict.items():
                    self.metrics = self.evaluator.update(
                        k, v.item(), self.metrics
                    )

                loss = self.scaler.scale(loss) if self.scaler else loss
                self._backward(loss)

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        # e.g. val_metrics: 
                        # {'dielectric_mae': {'total': 691.7076916694641, 
                        #                     'numel': 953, 
                        #                     'metric': 0.725821292412869}, 
                        #  'loss': {'total': 171.08978879451752, 
                        #           'numel': 60, 
                        #           'metric': 2.8514964799086253}}
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

            if self.early_stop == 0:
                logging.info(f'Epoch {epoch_int}, early stop!')
                break

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch):
        if hasattr(self.model, 'module'):
            model_cls = self.model.module
        else:
            model_cls = self.model
        x = model_cls.get_mol_rep(batch.to(self.device))
        y_pred = self.stochastic_optimizer.infer(x, model_cls) # (B, 1)

        assert len(self.output_targets) == 1
        target_name = list(self.output_targets.keys())[0]
        return {target_name: y_pred}

    def _compute_loss(self, batch):
        loss = 0.
        loss_info = {}

        model_cls = self.model.module if hasattr(self.model, 'module') else self.model

        batch = batch.to(self.device)
        ys = batch['y'].view(-1, 1) # ys is unnormalized target !
        ys = self.normalizers['y'].norm(ys)

        mol_embeddings = model_cls.get_mol_rep(batch) # average atom repr and pass to fc_x
        scores_gt, x_feat, y_feat = model_cls.get_energy(mol_embeddings, ys, ret_rep=True) # here x_feat is the same as mol_embedddings
        scores_gt = scores_gt.view(-1) # (B,)

        # euclidean loss
        if self.use_euc:
            x_feature_norm = F.normalize(x_feat, dim=-1)
            y_feature_norm = F.normalize(y_feat, dim=-1)
            euc_loss = torch.pow(x_feature_norm - y_feature_norm, 2).sum(1)
            euc_loss = euc_loss[torch.isfinite(euc_loss)] # neglect nan or inf
            euc_loss = self.ddp_loss(euc_loss)

            assert hasattr(euc_loss, "grad_fn")

            loss += self.euc_coeff * euc_loss
            loss_info['euc_loss'] = euc_loss
        
        # NCE loss
        if self.use_nce:
            stds = torch.zeros((1,3))
            stds[0, 0] = 0.075
            stds[0, 1] = 0.15
            stds[0, 2] = 0.3

            y_samples_zero, q_y_samples, q_ys = sample_gmm_centered(stds, num_samples=self.neg_samples)
            y_samples_zero = y_samples_zero.squeeze(1).to(self.device) # (num_samples, )
            y_samples = ys + y_samples_zero.unsqueeze(0) # (B, num_samples)
            scores_samples = model_cls.get_energy(mol_embeddings, y_samples) # (B, num_samples)

            q_y_samples = q_y_samples.to(self.device) # (num_samples, )       
            q_y_samples = q_y_samples.unsqueeze(0) * torch.ones(y_samples.size()).to(self.device)
            q_ys = q_ys[0]*torch.ones(ys.size(0)).to(self.device) # (B,)

            nce_loss = -(
                scores_gt-torch.log(q_ys) 
                - torch.log(
                    torch.exp(scores_gt-torch.log(q_ys)) + 
                    torch.sum(torch.exp(scores_samples-torch.log(q_y_samples)), dim=1)
                )
            ) # the shape is (B, )
            nce_loss = nce_loss[torch.isfinite(nce_loss)] # neglect nan or inf
            nce_loss = self.ddp_loss(nce_loss)

            assert hasattr(nce_loss, "grad_fn")

            loss += self.nce_coeff * nce_loss
            loss_info['nce_loss'] = nce_loss
        
        elif self.use_infonce:
            negatives = self.stochastic_optimizer.sample(ys.shape[0]) # (B, neg_samples)
            targets = torch.cat([ys, negatives], dim=1) # (B, 1+neg_samples)

            # CE based infoNCE, donot support pushing weight
            # Generate a random permutation of the positives and negatives.
            permutation = torch.rand(targets.size(0), targets.size(1)).argsort(dim=1)
            targets = targets[torch.arange(targets.size(0)).unsqueeze(-1), permutation]
            # Get the original index of the positive. This will serve as the class label for the loss.
            ground_truth = (permutation == 0).nonzero()[:, 1].to(self.device) # (B,)

            energy = model_cls.get_energy(mol_embeddings, targets) # (B, 1+neg+samples)

            # Interpreting the energy as a logit / negative logits, we can apply a cross entropy loss
            # to train the EBM.
            logits = 1.0 * energy
            infonce_loss = F.cross_entropy(logits, ground_truth, reduction='none')
            
            infonce_loss = infonce_loss[torch.isfinite(infonce_loss)] # neglect nan or inf
            infonce_loss = self.ddp_loss(infonce_loss)
            assert hasattr(infonce_loss, "grad_fn")

            loss += self.infonce_coeff * infonce_loss
            loss_info['infonce_loss'] = infonce_loss

        # Sanity check to make sure the compute graph is correct.
        assert hasattr(loss, "grad_fn")
        loss_info['loss'] = loss

        return loss_info

    def ddp_loss(self, loss_local):
        '''
        loss: (B,)
        '''
        loss = torch.sum(loss_local)
        bs = loss_local.shape[0]
        bs = distutils.all_reduce(bs, device=loss_local.device)
        loss = loss * distutils.get_world_size() / bs
        
        return loss

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        evaluator = Evaluator(
            task=self.name,
            eval_metrics=self.evaluation_metrics.get(
                "metrics", Evaluator.task_metrics.get(self.name, {})
            ),
        ) # e.g. eval_metrics: {elasticity: [mae]}

        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                batch.to(self.device)
                out = self._forward(batch)

            # Compute metrics.
            # become {'prop_fn': {'total': no., 'numel': no., 'metric': no.}}
            # prop_fn e.g. dielectric_mae 
            metrics = self._compute_metrics(out, batch, evaluator, metrics)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics


    def _compute_metrics(self, out, batch, evaluator, metrics={}):
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for target_name in self.output_targets: # e.g. target_name: elasticity
            target = batch[target_name]

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            target = target.view(batch_size, -1)

            targets[target_name] = target # target is unnormamlized
            if self.normalizers.get(target_name, False):
                out[target_name] = self.normalizers[target_name].denorm(
                    out[target_name]
                ) # out is normalized. so need to be denorm first

        targets["natoms"] = natoms
        out["natoms"] = natoms

        metrics = evaluator.eval(out, targets, prev_metrics=metrics)
        return metrics

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: Optional[str] = None,
        disable_tqdm: bool = False,
    ):
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)

        for i, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)

            for target_key in self.config["outputs"]:
                pred = out[target_key]
                if self.normalizers.get(target_key, False):
                    pred = self.normalizers[target_key].denorm(pred)

                if per_image:
                    pred = pred.cpu().detach().to(torch.float32)
                    ### Assumes system level properties are of the same dimension
                    per_image_pred = pred.numpy().reshape(-1).tolist()
                    predictions[f"{target_key}"].extend(per_image_pred)
                    predictions[f"{target_key}_gts"].extend(batch[target_key].cpu().numpy().reshape(-1).tolist())
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist()
                if isinstance(batch.sid, torch.Tensor)
                else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        for key in predictions:
            predictions[key] = np.array(predictions[key])

        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions


    @torch.no_grad()
    def test_validation(
        self,
    ):
        self.stochastic_optimizer.infer_type = 'cosine' # use cosine for inference
        print(f'reset optimizer type: {self.stochastic_optimizer.infer_type}', flush=True)

        rank = distutils.get_rank()
        self.model.eval()

        cur_prop = 'y'
        labels, predictions = [], []

        loaders = [self.test_loader]

        start_time = time.time()
        for data_loader in loaders:
            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                position=rank,
                desc="device {}".format(rank),
            ):
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    labels.extend(batch[cur_prop].view(-1).tolist())

                    pred = self._forward(batch.to(self.device))[cur_prop]
                    pred = self.normalizers[cur_prop].denorm(pred)
                    predictions.extend(pred.tolist())
        
        consume_time = time.time() - start_time
        time_average = consume_time / len(predictions)

        print(f'inference time per sample is {time_average}') # 0.006394689384356303

        labels = np.array(labels).reshape(-1)
        predictions = np.array(predictions).reshape(-1)

        errors = np.abs(predictions - labels)
        mae = np.mean(errors)
        print(f'mae = {mae}')
        exit(0)

        # errors_wo_zero = np.where(errors == 0, 1e-10, errors)
        # egm = gmean(errors_wo_zero)

        # dataset_name = self.config["dataset"].get('src').split('/')[-2]
        # with open('analyse_res/rmr_test_res_energy_infer.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)         
        #     writer.writerow([dataset_name, str(mae), str(egm)])

        # compute recall rate
        if 'small' in self.config["dataset"].get('src'):
            recall = np.mean(predictions < np.max(labels))
            print(f'------- extrapolate to bottom ----------\\recall: {round(recall, 4)}')
        elif 'large' in self.config["dataset"].get('src'):
            recall = np.mean(predictions > np.min(labels))
            print(f'------- extrapolate to top ----------\\recall: {round(recall, 4)}')
        else:
            raise ValueError

        dataset_name = self.config["dataset"].get('src').split('/')[-2]
        with open('analyse_res/ebm_recall.csv', mode='a', newline='') as file:
            writer = csv.writer(file)         
            writer.writerow([dataset_name, str(recall)])
            
    @torch.no_grad()
    def encode_x(self, save_path: str):
        rank = distutils.get_rank()

        self.model.eval()
        loaders = {
            'train': self.train_loader, 
            'val': self.val_loader, 
            'test': self.test_loader
        }
        split2labels = defaultdict(list)
        split2repr = defaultdict(list)

        for split, data_loader in loaders.items():
            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                position=rank,
                desc="device {}".format(rank),
            ):
                batch = batch.to(self.device)

                split2labels[split].extend(batch['y'].view(-1).tolist())
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    reprs = self.model.get_mol_rep(batch) # (B, D)
                    split2repr[split].append(reprs.cpu().numpy())
            
        np.savez(
            os.path.join(save_path, 'coverted_dataset.npz'), 
            train_repr=np.concatenate(split2repr['train'], axis=0),
            val_repr=np.concatenate(split2repr['val'], axis=0),
            test_repr=np.concatenate(split2repr['test'], axis=0),
            train_labels=np.array(split2labels['train']),
            val_labels=np.array(split2labels['val']),
            test_labels=np.array(split2labels['test']),
        )

    @torch.no_grad()
    def SAE_inference(self, save_path):
        rank = distutils.get_rank()
        self.model.eval()

        if hasattr(self.model, 'module'):
            model_cls = self.model.module
        else:
            model_cls = self.model

        loaders = {
            'train': self.train_loader, 
            'val': self.val_loader, 
            'test': self.test_loader
        }
        split2labels = defaultdict(list)
        split2repr = defaultdict(list)
        split2labelrepr = defaultdict(list)

        for split, data_loader in loaders.items():
            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                position=rank,
                desc="device {}".format(rank),
            ):
                batch = batch.to(self.device)
                split2labels[split].extend(batch['y'].tolist())
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    reprs = model_cls.get_mol_rep(batch) # (B, D)
                    split2repr[split].append(reprs)

                    label = self.normalizers['y'].norm(batch['y'].view(-1, 1))
                    label_reprs = model_cls.get_y_rep(label) # (B, d)
                    split2labelrepr[split].append(label_reprs)
        
        train_repr=torch.concat(split2repr['train'], dim=0)
        train_label_repr=torch.concat(split2labelrepr['train'], dim=0)

        W = SAE(
            train_repr.cpu().numpy().T, 
            train_label_repr.cpu().numpy().T, 
            lamb=1) # (d_label, d_sample)
        W = torch.from_numpy(W).to(self.device)

        self.stochastic_optimizer.infer_type = 'cosine' # use cosine for inference

        preds = []
        for x_repr in tqdm(split2repr['test']):
            x_repr = x_repr @ W # (B, d_label), note that we have d_label == d_sample
            y_pred = self.stochastic_optimizer.infer(x_repr, model_cls) # (B, 1)
            y_pred = self.normalizers['y'].denorm(y_pred)
            preds.extend(y_pred.view(-1).tolist())

        preds = np.array(preds).reshape(-1)
        gts = np.array(split2labels['test']).reshape(-1)

        np.savez(f'{save_path}/sae_predictions.npz', y_gts=gts, y=preds)
