"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import os
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.stats import gmean
import torch
import torch.nn as nn
import torch_geometric
from tqdm import tqdm
import time

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import cg_change_mat, check_traj_files, irreps_sum, \
    get_loss_module
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.scaling.util import ensure_fitted
from ocpmodels.trainers.base_trainer import BaseTrainer
from ocpmodels.modules.loss import DDPLoss

@registry.register_trainer("ocp")
@registry.register_trainer("energy")
@registry.register_trainer("forces")
class OCPTrainer(BaseTrainer):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        outputs (dict): Output property configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        loss_fns (dict): Loss function configuration.
        eval_metrics (dict): Evaluation metrics configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`wandb`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
        noddp (bool, optional): Run model without DDP.
    """

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

    def load_loss(self) -> None:
        self.loss_fns = []
        for _, loss in enumerate(self.config["loss_fns"]):
            # loss is a dict, e.g. {target_name: {fn:..., coefficient:...}}
            for target in loss:
                loss_name = loss[target].get("fn", "mae")
                coefficient = loss[target].get("coefficient", 1)
                loss_reduction = loss[target].get("reduction", "mean")

                ### if torch module name provided, use that directly
                if hasattr(nn, loss_name):
                    loss_fn = getattr(nn, loss_name)()
                ### otherwise, retrieve the correct module based off old naming
                elif loss_name == 'ranksim':
                    loss_fn = get_loss_module(
                        'ranksim',
                        lambda_val=loss[target].get("lambda_val", 2),
                    )
                elif loss_name == 'conr':
                    loss_fn = get_loss_module(
                        'conr', 
                        threshold=loss[target].get("threshold", 0.1),
                        global_con=loss[target].get("global_con", False),
                    )
                elif loss_name == 'quantile':
                    loss_fn = get_loss_module(
                        'quantile',
                        tau=loss[target].get("tau", 0.5),
                    )
                elif loss_name == 'bmc':
                    loss_fn = get_loss_module(
                        'bmc',
                        init_noise_sigma=loss[target].get("init_noise_sigma", 1.0),
                        global_con=loss[target].get("global_con", False),
                    )
                else:
                    loss_fn = get_loss_module(loss_name)
                loss_fn = DDPLoss(loss_fn, loss_name, loss_reduction)

                self.loss_fns.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

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
                with torch.amp.autocast('cuda',enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item(), self.metrics
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
                        

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

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
        out = self.model(batch.to(self.device))

        outputs = {}
        batch_size = batch.natoms.numel()
        num_atoms_in_batch = batch.natoms.sum()

        for target_key in self.output_targets:
            ### Target property is a direct output of the model
            if target_key in out:
                pred = out[target_key]
            else:
                raise ValueError(f'{target_key} not in out: {out.keys()}')

            ### not all models are consistent with the output shape
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_key]["level"] == "atom":
                pred = pred.view(num_atoms_in_batch, -1)
            else:
                pred = pred.view(batch_size, -1)

            outputs[target_key] = pred

        if "embeddings" in out:
            if self.output_targets[target_key]["level"] != "atom":
                outputs["embeddings"] = out["embeddings"].view(batch_size, -1)
            else:
                raise NotImplementedError

        return outputs

    def _compute_loss(self, out, batch):
        batch_size = batch.natoms.numel()

        loss = []
        for loss_fn in self.loss_fns:
            target_name, loss_info = loss_fn
            
            pred = out[target_name]
            target = batch[target_name]
            if f'{target_name}_embeddings' in out: 
                embeddings = out[f'{target_name}_embeddings']
            elif 'embeddings' in out:
                embeddings = out['embeddings']
            else: embeddings = None

            if self.normalizers.get(target_name, False):
                target = self.normalizers[target_name].norm(target)

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                raise NotImplementedError
            else:
                target = target.view(batch_size, -1)

            mult = loss_info["coefficient"]
            loss.append(
                mult
                * loss_info["fn"](
                        pred,
                        target,
                        embeddings,
                        batch_size=batch_size,
                        sample_weights=batch.weight if hasattr(batch, 'weight') else None
                    )
            )
            
        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

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
            num_atoms_in_batch = batch.natoms.sum()

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["eval_on_free_atoms"]
            ):
                target = target[mask]
                out[target_name] = out[target_name][mask]
                num_atoms_in_batch = natoms.sum()

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            targets[target_name] = target
            if self.normalizers.get(target_name, False):
                out[target_name] = self.normalizers[target_name].denorm(
                    out[target_name]
                )

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
        if self.is_debug and per_image:
            raise FileNotFoundError(
                "Predictions require debug mode to be turned off."
            )

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

            with torch.amp.autocast('cuda',enabled=self.scaler is not None):
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

    def run_relaxations(self, split="val"):

        ensure_fitted(self._unwrapped_model)

        # When set to true, uses deterministic CUDA scatter ops, if available.
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms
        # Only implemented for GemNet-OC currently.
        registry.register(
            "set_deterministic_scatter",
            self.config["task"].get("set_deterministic_scatter", False),
        )

        logging.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        # Need both `pos_relaxed` and `y_relaxed` to compute val IS2R* metrics.
        # Else just generate predictions.
        if (
            hasattr(self.relax_dataset[0], "pos_relaxed")
            and self.relax_dataset[0].pos_relaxed is not None
        ) and (
            hasattr(self.relax_dataset[0], "y_relaxed")
            and self.relax_dataset[0].y_relaxed is not None
        ):
            split = "val"
        else:
            split = "test"

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                logging.info(
                    f"Skipping batch: {batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid}"
                )
                continue

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=self.config["task"].get("save_full_traj", True),
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                sid_list = (
                    relaxed_batch.sid.tolist()
                    if isinstance(relaxed_batch.sid, torch.Tensor)
                    else relaxed_batch.sid
                )
                systemids = [str(sid) for sid in sid_list]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]

                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                logging.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    logging.info(metrics)

        if self.ema:
            self.ema.restore()

        registry.unregister("set_deterministic_scatter")

    # analyse model. Modify this function freely for any analysis !
    @torch.no_grad()
    def analyse(
        self,
        save_path: str,
    ):
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        from sklearn.manifold import TSNE
        import os.path as osp

        cur_prop = 'y'
        ensure_fitted(self._unwrapped_model, warn=True)

        rank = distutils.get_rank()

        self.model.eval()

        predictions = []
        labels = []
        embeds = []
        for data_loader in [self.train_loader, self.val_loader, self.test_loader]:
            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                position=rank,
                desc="device {}".format(rank),
                disable=False,
            ):
                with torch.amp.autocast('cuda',enabled=self.scaler is not None):
                    out = self.model(batch.to(self.device))
                
                # proposs predictions
                pred = out[cur_prop]
                if self.normalizers.get(cur_prop, False):
                    pred = self.normalizers[cur_prop].denorm(pred)

                pred = pred.cpu().detach().to(torch.float32).view(-1)
                predictions.extend(pred.tolist())
                labels.extend(batch[cur_prop].view(-1).tolist())
                embeds.append(out['embeddings'])

        
        '''
        # draw embeddings of train, val, test
        embeds = torch.concat(embeds, dim=0)
        len_train, len_val, len_test = len(self.train_loader.dataset), len(self.val_loader.dataset), len(self.test_loader.dataset)
        assert len_test + len_val + len_train == len(embeds)

        train_embeds = embeds[:len_train]
        val_embeds = embeds[len_train : len_train + len_val]
        test_embeds = embeds[len_train+len_val : len_train+len_val+len_test]

        mean_train_embeds = torch.mean(train_embeds, dim=0)
        mean_val_embeds = torch.mean(val_embeds, dim=0)
        mean_test_embeds = torch.mean(test_embeds, dim=0)

        test_train_dist = torch.norm(mean_train_embeds - mean_test_embeds, p=2)
        val_train_dist = torch.norm(mean_train_embeds - mean_val_embeds, p=2)

        print(f'test-train-dist: {test_train_dist}, val-train-dist: {val_train_dist}')
        '''

        # save pred res
        assert len(predictions) == len(labels), \
            f'bad len! pred: {len(predictions)}, label: {len(labels)}'
        min_val = min(predictions + labels)
        max_val = max(predictions + labels)
        plt.plot(np.linspace(min_val, max_val, 100), np.linspace(min_val, max_val, 100), linestyle='--', color='red')
        plt.scatter(predictions, labels, s=10)
        plt.ylabel('real')
        plt.xlabel('prediction')
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.savefig(osp.join(save_path, f'pred_res.png'))
        plt.clf()

        embeds = torch.concat(embeds, dim=0)
        print(f'embed shape is {embeds.shape}')
        # tsne = TSNE(n_components=2)
        # X_tsne = tsne.fit_transform(embeds.cpu().numpy())
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='plasma', s=10)
        # plt.colorbar()
        # plt.savefig(osp.join(save_path, f'x_repr.png'))
        # plt.clf()
        # np.savez(osp.join(saved_path, f'{cur_prop}_reprs_labels_preds.npz'), reprs=embeds.cpu().detach().numpy(), labels=labels, preds=predictions)

        predictions = np.array(predictions).reshape(-1)
        labels = np.array(labels).reshape(-1)
        assert len(embeds) == len(labels), f'embeds: {embeds.shape}, labels: {len(labels)}'
        assert len(predictions) == len(labels), f'labels: {labels.shape}, predictions: labels: {predictions.shape}'
        print(f'mae: {np.mean(np.abs(labels - predictions))}')
        
        # re-sort according to labels
        sorted_indices = np.argsort(labels)
        labels = labels[sorted_indices]
        predictions = predictions[sorted_indices]
        embeds = embeds[sorted_indices]

        # analyse
        label_similarity_matrix = -np.abs(labels[:, np.newaxis] - labels)
        prediction_similarity_matrix = -np.abs(predictions[:, np.newaxis] - predictions)

        x_normalized = embeds / embeds.norm(dim=1, keepdim=True)
        cosine_similarity_matrix = []
        for x in tqdm(x_normalized):
            sim = torch.mm(x.unsqueeze(0), x_normalized.t())
            cosine_similarity_matrix.append(sim)
        cosine_similarity_matrix = torch.concat(cosine_similarity_matrix, dim=0).cpu().detach().numpy()

        rep_label_spearman, _ = spearmanr(
            label_similarity_matrix[np.triu_indices(len(labels), k=1)], 
            cosine_similarity_matrix[np.triu_indices(len(labels), k=1)]
        )
        rep_pred_spearman, _ = spearmanr(
            prediction_similarity_matrix[np.triu_indices(len(labels), k=1)],
            cosine_similarity_matrix[np.triu_indices(len(labels), k=1)]
        )
        pred_label_spearman, _ = spearmanr(
            prediction_similarity_matrix[np.triu_indices(len(labels), k=1)],
            label_similarity_matrix[np.triu_indices(len(labels), k=1)]
        )

        # ================ draw =================
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(label_similarity_matrix, cmap='viridis', aspect='auto')
        axes[0].set_title('label')

        # 绘制第三个热力图
        axes[1].imshow(cosine_similarity_matrix, cmap='viridis', aspect='auto')
        axes[1].set_title('representation')

        axes[2].imshow(prediction_similarity_matrix, cmap='viridis', aspect='auto')
        axes[2].set_title('prediction')

        fig.suptitle(f'r&l: {round(rep_label_spearman, 2)}, r&p: {round(rep_pred_spearman, 2)}, p&l: {round(pred_label_spearman, 2)}')
        plt.tight_layout()
        plt.savefig(osp.join(save_path, f'heatmap_{cur_prop}.png'))

    @torch.no_grad()
    def test_validation(
        self,
    ):
        import csv

        rank = distutils.get_rank()
        self.model.eval()

        cur_prop = 'y'
        labels, predictions = [], []

        loaders = [self.val_loader, self.test_loader]

        start_time = time.time()
        for data_loader in loaders:
            for i, batch in tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                position=rank,
                desc="device {}".format(rank),
            ):
                with torch.amp.autocast('cuda',enabled=self.scaler is not None):
                    out = self.model(batch.to(self.device))
                
                # proposs predictions
                assert len(self.config["outputs"]) >= 1, f"config.output is {self.config['outputs']}"
                pred = out[cur_prop]
                if self.normalizers.get(cur_prop, False):
                    pred = self.normalizers[cur_prop].denorm(pred)

                pred = pred.cpu().detach().view(-1)
                predictions.extend(pred.tolist())
                labels.extend(batch[cur_prop].view(-1).tolist())
        
        # consume_time = time.time() - start_time
        # time_average = consume_time / len(predictions)
        
        # labels = np.array(labels).reshape(-1)
        # predictions = np.array(predictions).reshape(-1)

        # errors = np.abs(predictions - labels)
        # mae = np.mean(errors)
        
        # errors_wo_zero = np.where(errors == 0, 1e-10, errors)
        # egm = gmean(errors_wo_zero)

        # print(f'inference time per sample is {time_average}. MAE={round(mae, 4)}, GM={round(egm, 4)}') # 0.006394689384356303

        dataset_name = self.config["dataset"].get('src').split('/')[-2]
        if 'small' in dataset_name:
            recall = np.mean(predictions < np.max(labels))
            print(f'------- extrapolate to bottom ----------\\recall: {round(recall, 4)}')
        elif 'large' in dataset_name:
            recall = np.mean(predictions > np.min(labels))
            print(f'------- extrapolate to top ----------\\recall: {round(recall, 4)}')
        else:
            raise ValueError
        
        dataset_name = self.config["dataset"].get('src').split('/')[-2]
        with open('analyse_res/conr_recall.csv', mode='a', newline='') as file:
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
                split2labels[split].extend(batch['y'].view(-1).tolist())
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    reprs = self.model(batch.to(self.device))['embeddings'] # (B, D)
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