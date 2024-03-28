"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

"""
Reference : ocp/ocpmodels/trainers/forces_trainer.py

The following items are modified and they can be claimed as properties of Samsung Electronics. 

(1) Support more MLFF models (BPNN, NequIP, Allegro, and MACE)
(2) Support simulation indicators for the benchmark evaluation on simulations (RDF, ADF, EoS, PEW)
(3) Support more loss functions and metrics (loss.py and metric_evaluator.py in src/modules/)
(4) Support more learning rate schedulers (scheduler.py in src/modules/)
(5) Support normalization of per-atom energy (NormalizerPerAtom in src/modules/normalizer.py)
(6) Some different featurs are as follows:
    (a) Print training results using PrettyTable
    (b) Use a benchmark logger (named bm_logging) instead of the root logger (named logging in OCP)
    (c) Remove features that includes to save prediction results and make the corresponding directory named 'results'
    (d) Remove features related to HPO
    (e) Set the identifier of an experiment using the starting time
"""

"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
import json 
import time

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from prettytable import PrettyTable

import numpy as np
import torch
import torch_geometric

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.modules.scaling.util import ensure_fitted

from src.common.utils import bm_logging 
from src.common.logger import parse_logs
from src.trainers.base_trainer import BaseTrainer
from src.modules.normalizer import NormalizerPerAtom, log_and_check_normalizers
from src.modules.scheduler import LRScheduler


import wandb
import datetime
import numpy as np

from copy import deepcopy

@registry.register_trainer("forces")
class ForcesTrainer(BaseTrainer):
    """
    Trainer class for the S2EF (Structure to Energy & Force) task,
    and this class is especially used to train models implemented in OCP and BPNN models.
    """
    def _set_normalizer(self):
        self.normalizers = {}
        if self.normalizer.get("normalize_labels", False):
            if self.mode in ["validate", "fit-scale"]:
                # just empty normalizer (which will be loaded from the given checkpoint)
                if self.normalizer.get("per_atom", False):
                    self.normalizers["target"] = NormalizerPerAtom(mean=0.0, std=1.0, device=self.device,)
                else:
                    self.normalizers["target"] = Normalizer(mean=0.0, std=1.0, device=self.device,)
                self.normalizers["grad_target"] = Normalizer(mean=0.0, std=1.0, device=self.device)
                if self.mode == "fit-scale":
                    bm_logging.info(f"Normalizers are not set")
                return

            # force normalizer
            if "grad_target_std" in self.normalizer:
                # Load precomputed mean and std of training set labels (specified in a configuration file)
                if "grad_target_mean" in self.normalizer:
                    bm_logging.info("`grad_target_mean` is ignored and set as 0 explicitly.")
                scale = self.normalizer["grad_target_std"]
            elif "normalize_labels_json" in self.normalizer:
                # Load precomputed mean and std of training set labels (specified in a json file outside from a configuration file)
                normalize_stats = json.load(open(self.normalizer["normalize_labels_json"], 'r'))
                if "force_mean":
                    bm_logging.info("`force_mean` is ignored and set as 0 explicitly.")
                scale = normalize_stats.get("force_std", 1.0)
            else:
                # Compute mean and std of training set labels.
                # : force is already tensor (which can have different shapes)
                forces_train = torch.concat([data.force for data in self.train_loader.dataset])
                scale = torch.std(forces_train)
            self.normalizers["grad_target"] = Normalizer(mean=0.0, std=scale, device=self.device)

            # energy normalizer
            if "target_mean" in self.normalizer:
                shift = self.normalizer["target_mean"]
                scale = self.normalizer.get("target_std", self.normalizers["grad_target"])
                if scale != self.normalizers["grad_target"].std:
                    bm_logging.warning(f"Scaling factors of energy and force are recommended to be equal")
            elif "normalize_labels_json" in self.normalizer:
                if self.normalizer.get("per_atom", False):
                    # per-atom energy mean
                    shift = normalize_stats["energy_per_atom_mean"]
                else:
                    shift = normalize_stats["energy_mean"]
                if "energy_std":
                    bm_logging.info("`energy_std` is ignored and set as the value of `force_std` explicitly.")
                scale = self.normalizers["grad_target"].std # energy scale factor should be force std
            else:
                
                if self.normalizer.get("per_atom", False):
                    # per-atom energy mean
                    energy_per_atom_train = torch.tensor([data.y / data.force.shape[0] for data in self.train_loader.dataset])
                    shift = torch.mean(energy_per_atom_train)
                else:
                    # total energy mean
                    energy_train = torch.tensor([data.y for data in self.train_loader.dataset])
                    shift = torch.mean(energy_train)
                scale = self.normalizers["grad_target"].std # energy scale factor should be force std

            if self.normalizer.get("per_atom", False):
                # per-atom energy 
                self.normalizers["target"] = NormalizerPerAtom(mean=shift, std=scale, device=self.device)
            else:
                self.normalizers["target"] = Normalizer(mean=shift, std=scale, device=self.device)

            # logging the status of normalizers
            log_and_check_normalizers(self.normalizers["target"], self.normalizers["grad_target"], loaded=False)
            
    def _set_task(self):
        # most models have a scaler energy output (meaning that num_targets = 1)
        self.num_targets = 1

        # this benchmark focuses on s2ef task, so regress_forces should be true
        if "regress_forces" in self.config["model_attributes"]:
            assert self.config["model_attributes"]["regress_forces"], "Correct `regress_forces` to be true"
        else:
            self.config["model_attributes"]["regress_forces"] = True

    def update_best(self, primary_metric, val_metrics):
        curr_metric = val_metrics[primary_metric]["metric"]
        if "mae" in primary_metric or "mse" in primary_metric:
            if curr_metric >= self.best_val_metric:
                return
        else:
            if curr_metric <= self.best_val_metric:
                return
                
        self.best_val_metric = curr_metric
        self.save(
            metrics=val_metrics,
            checkpoint_file="best_checkpoint.pt",
            training_state=False,
        )
    
    def set_active_dataset(self):
        self.train_dataset_active = torch.utils.data.Subset(
            self.train_dataset,
            self.al_dataset_idx
        )
        self.train_sampler_active = self.get_sampler(
            dataset=self.train_dataset_active,
            batch_size=self.train_local_batch_size,
            shuffle=True, 
        )
        self.train_loader = self.get_dataloader(
            dataset=self.train_dataset_active,
            sampler=self.train_sampler_active,
            collater=self.parallel_collater,
        )
        self.config["optim"]["num_train"] = len(self.train_dataset_active)
        
        self._set_optimizer_and_lr_scheduler()
        self._set_ema()
        self._set_extras()
    
    def set_uncertainty_dataset(self, al_dataset_remaining_idx_arr):
        self.uncertainty_dataset = torch.utils.data.Subset(
            self.train_dataset,
            al_dataset_remaining_idx_arr
        )
        self.train_sampler_uncertainty = self.get_sampler(
            dataset=self.uncertainty_dataset,
            batch_size=self.train_local_batch_size * 2,
            shuffle=True, 
        )
        self.train_loader = self.get_dataloader(
            dataset=self.uncertainty_dataset,
            sampler=self.train_sampler_uncertainty,
            collater=self.parallel_collater,
        )
        
        return
    
    def init_active_distance(self):
        distance = []
        pbar = tqdm(
            range(self.org_dataset_size),
            total=self.org_dataset_size,
            desc="Atom-wise distance calculation for edges",
        )
        
        for idx in pbar:
            data = self.train_dataset[idx]
            pos = data.pos.to(self.device)
            edge = data.edge_index.to(self.device)
            node_pos_1 = pos[edge[0]].unsqueeze(1)
            node_pos_2 = pos[edge[1]].unsqueeze(1)
            data_distance = torch.cdist(node_pos_1, node_pos_2, p=2).squeeze().mean()
            distance.append(data_distance)
        
        distance = torch.stack(distance)
        idx_distance_descending = torch.argsort(distance, descending=False)
        self.al_dataset_idx = idx_distance_descending[:self.al_dataset_start_size].to('cpu')
        self.al_dataset_remaining_idx = set(idx_distance_descending[self.al_dataset_start_size:].tolist())
        
        if len(self.al_dataset_idx) + len(self.al_dataset_remaining_idx) != self.org_dataset_size:
            print(len(self.al_dataset_idx))
            print(len(self.al_dataset_remaining_idx))
            print(self.org_dataset_size)
            raise ValueError("Error, dataset index not matching")
        return
     
    def init_active_dataset(self):
        self.train_local_batch_size = self.config["optim"]["batch_size"]
        self.org_train_loader = self.train_loader
        self.org_dataset_size = len(self.org_train_loader.dataset)
        self.al_dataset_start_size = int(self.org_dataset_size * self.config["active"]["init_size"])
        self.al_dataset_update_size = int(self.org_dataset_size * self.config["active"]["update_size"])
        
        if self.config["active"]["init_method"] == "random":
            self.al_dataset_rand_idx = torch.randperm(len(self.org_train_loader.dataset))
            self.al_dataset_idx = self.al_dataset_rand_idx[:self.al_dataset_start_size]
            self.al_dataset_remaining_idx = set(self.al_dataset_rand_idx[self.al_dataset_start_size:].tolist())
        elif self.config["active"]["init_method"] == "distance":
            self.init_active_distance()
        else:
            raise NotImplementedError(f"Active learning initialization method {self.config['active']['init_method']} is not supported yet")

        self.set_active_dataset()
        
        self.analysis = {}
        self.analysis["peratom_force_error"] = []
        self.analysis["peratom_energy_error"] = []
    
    def uncertainty_random(self):
        al_dataset_remaining_idx_arr = np.array(list(self.al_dataset_remaining_idx))
        selected_idx = torch.randperm(len(al_dataset_remaining_idx_arr))[:self.al_dataset_update_size]
        al_dataset_update_idx = al_dataset_remaining_idx_arr[selected_idx]
        self.al_dataset_idx = torch.concat([self.al_dataset_idx, torch.from_numpy(al_dataset_update_idx)])
        self.al_dataset_remaining_idx.difference_update(set(al_dataset_update_idx))
        
        return al_dataset_update_idx
    
    def uncertainty_prediction_variation(self, variance_target, predictions_batch):
        if len(predictions_batch) > 0:
            raise ValueError(f"Prediction batch is empty")
        
        if variance_target == "forces":
            variation = torch.var(torch.stack(predictions_batch, dim=0), axis=0)
            variation = torch.sum(variation, dim=1)
            variation = variation.reshape(-1, 96)
            variation = torch.sum(variation, dim=1)
            # variation = torch.var(variation, dim=1) 
        elif variance_target == "energy":
            variation = torch.var(torch.stack(predictions_batch, dim=0), axis=0)
        else:
            raise NotImplementedError(f"Active learning method {variance_target} is not supported yet")
                
        return variation    
    
    def uncertainty_mcdropout(self):
        variance_target = self.config["active"]["variance_target"]
        variance_sample_number = self.config["active"].get("variance_sample_number", 4)

        al_dataset_remaining_idx_arr = np.array(list(self.al_dataset_remaining_idx))
        set_uncertainty_dataset(al_dataset_remaining_idx_arr)

        ensure_fitted(self._unwrapped_model, warn=True)
        rank = distutils.get_rank()
        self.model.eval()
        variance = []
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            position=rank,
            desc=f"Device {rank}, selecting samples for active learning",
        )
        for i, batch in pbar:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                variance_batch = []
                for _ in range(variance_sample_number):
                    out = self._forward(batch)
                    variance_batch.append(out[variance_target])

            prediction_variation = uncertainty_prediction_variation(variance_target, variance_batch)
            variance.append(prediction_variation)
            
            if i == 20 and self.config["active"].get("debug", False):
                break
                    
        variance = torch.cat(variance)
        selected_idx = torch.argsort(variance, descending=True)[:self.al_dataset_update_size]
        al_dataset_update_idx = al_dataset_remaining_idx_arr[selected_idx.detach().cpu().numpy()]
        self.al_dataset_idx = torch.concat([self.al_dataset_idx, torch.from_numpy(al_dataset_update_idx)])
        self.al_dataset_remaining_idx.difference_update(set(al_dataset_update_idx))
        
        return al_dataset_update_idx
    
    def uncertainty_noise(self):
        def add_noise(batch, noise_scale, update_edge=False):
            gaussian_noise = torch.normal(0, noise_scale, size=batch[0].pos.shape, device=batch[0].total_energy.device)
            edge_index = batch[0].edge_index
            distance = torch.cdist(batch[0].pos[edge_index[0]].unsqueeze(1), batch[0].pos[edge_index[1]].unsqueeze(1), p=2).squeeze()
            max_distance = torch.max(distance)
            batch[0].pos = batch[0].pos + gaussian_noise
            if update_edge:
                atom_wise_distance = torch.cdist(batch[0].pos.unsqueeze(0), batch[0].pos.unsqueeze(0), p=2).squeeze()
                raise NotImplementedError(f"Updaing edges when graph given is not supported yet")
            
            return batch
        
        noise_scale = self.config["active"].get("noise_scale", 0.1)
        variance_target = self.config["active"].get("variance_target", "energy")
        variance_sample_number = self.config["active"].get("variance_sample_number", 4)
        al_dataset_remaining_idx_arr = np.array(list(self.al_dataset_remaining_idx))
        self.set_uncertainty_dataset(al_dataset_remaining_idx_arr)

        ensure_fitted(self._unwrapped_model, warn=True)
        rank = distutils.get_rank()
        self.model.eval()
        variance = []
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            position=rank,
            desc=f"Device {rank}, selecting samples for active learning",
        )
        for i, batch in pbar:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                variance_batch = []
                out = self._forward(batch)
                variance_batch.append(out[variance_target])
                for _ in range(variance_sample_number):
                    batch = add_noise(batch, noise_scale, self.config["active"]["noise_update_edge"])
                    out = self._forward(batch)
                    variance_batch.append(out[variance_target])
                    
            prediction_variation = self.uncertainty_prediction_variation(variance_target, variance_batch)
            variance.append(prediction_variation)
            
            if i == 20 and self.config["active"].get("debug", False):
                break
                    
        variance = torch.cat(variance)
        selected_idx = torch.argsort(variance, descending=True)[:self.al_dataset_update_size]
        al_dataset_update_idx = al_dataset_remaining_idx_arr[selected_idx.detach().cpu().numpy()]
        self.al_dataset_idx = torch.concat([self.al_dataset_idx, torch.from_numpy(al_dataset_update_idx)])
        self.al_dataset_remaining_idx.difference_update(set(al_dataset_update_idx))
        
        return al_dataset_update_idx
    
    def uncertainty_edgedrop(self):
        def remove_edge(batch, edgedrop_scale):
            edge_index = batch[0].edge_index
            edge_numbers = edge_index.shape[0]
            idx_to_remove = torch.randperm(edge_numbers)[:int(edge_numbers * (1 - edgedrop_scale))]
            batch[0].edge_index = edge_index[idx_to_remove, :]
            return batch
        
        edgedrop_scale = self.config["active"].get("edgedrop_scale", 0.01)
        variance_target = self.config["active"].get("variance_target", "energy")
        variance_sample_number = self.config["active"].get("variance_sample_number", 4)
        al_dataset_remaining_idx_arr = np.array(list(self.al_dataset_remaining_idx))
        self.set_uncertainty_dataset(al_dataset_remaining_idx_arr)

        ensure_fitted(self._unwrapped_model, warn=True)
        rank = distutils.get_rank()
        self.model.eval()
        variance = []
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            position=rank,
            desc=f"Device {rank}, selecting samples for active learning",
        )
        for i, batch in pbar:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                variance_batch = []
                out = self._forward(batch)
                variance_batch.append(out[variance_target])
                for _ in range(variance_sample_number):
                    batch = remove_edge(batch, edgedrop_scale)
                    out = self._forward(batch)
                    variance_batch.append(out[variance_target])
                    
            prediction_variation = self.uncertainty_prediction_variation(variance_target, variance_batch)
            variance.append(prediction_variation)
            
            if i == 20 and self.config["active"].get("debug", False):
                break
                    
        variance = torch.cat(variance)
        selected_idx = torch.argsort(variance, descending=True)[:self.al_dataset_update_size]
        al_dataset_update_idx = al_dataset_remaining_idx_arr[selected_idx.detach().cpu().numpy()]
        self.al_dataset_idx = torch.concat([self.al_dataset_idx, torch.from_numpy(al_dataset_update_idx)])
        self.al_dataset_remaining_idx.difference_update(set(al_dataset_update_idx))
        
        return al_dataset_update_idx
    
    def uncertainty_latticeshift(self):
        x_shift = self.config["active"].get("x_shift", 0)
        y_shift = self.config["active"].get("y_shift", 0)
        z_shift = self.config["active"].get("z_shift", 0)
        variance_target = self.config["active"].get("variance_target", "energy")
        al_dataset_remaining_idx_arr = np.array(list(self.al_dataset_remaining_idx))
        self.set_uncertainty_dataset(al_dataset_remaining_idx_arr)

        ensure_fitted(self._unwrapped_model, warn=True)
        rank = distutils.get_rank()
        self.model.eval()
        variance = []
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            position=rank,
            desc=f"Device {rank}, selecting samples for active learning",
        )
        for i, batch in pbar:
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                variance_batch = []
                out = self._forward(batch)
                variance_batch.append(out[variance_target])
                # NOTE: Lattice shift
                batch_lattice_shifted = lattice_shift(batch, x_shift, y_shift, z_shift)
                out_shifted = self._forward(batch_lattice_shifted)
                variance_batch.append(out_shifted[variance_target])
                
            prediction_variation = self.uncertainty_prediction_variation(variance_target, variance_batch)
            variance.append(prediction_variation)
            
            if i == 20 and self.config["active"].get("debug", False):
                break
                    
        variance = torch.cat(variance)
        selected_idx = torch.argsort(variance, descending=True)[:self.al_dataset_update_size]
        al_dataset_update_idx = al_dataset_remaining_idx_arr[selected_idx.detach().cpu().numpy()]
        self.al_dataset_idx = torch.concat([self.al_dataset_idx, torch.from_numpy(al_dataset_update_idx)])
        self.al_dataset_remaining_idx.difference_update(set(al_dataset_update_idx))
        
        return al_dataset_update_idx
    
    def uncertainty_evidential(self):
        raise NotImplementedError(f"Active learning method {self.config['active']['update_method']} is not supported yet")
        
    def uncertainty_gp(self):
        raise NotImplementedError(f"Active learning method {self.config['active']['update_method']} is not supported yet")
        
    def update_active_dataset(self, round_current):
        if len(self.al_dataset_remaining_idx) == 0:
            raise ValueError(f"Remaining dataset is empty")
        
        if self.config["active"]["update_method"] == "random":
            al_dataset_update_idx = self.uncertainty_random()
        elif self.config["active"]["update_method"] == "mcdropout":
            al_dataset_update_idx = self.uncertainty_mcdropout()
        elif self.config["active"]["update_method"] == "noise":
            al_dataset_update_idx = self.uncertainty_noise()
        elif self.config["active"]["update_method"] == "edgedrop":
            al_dataset_update_idx = self.uncertainty_edgedrop()
        elif self.config["active"]["update_method"] == "latticeshift":
            al_dataset_update_idx = self.uncertinaty_latticeshift()
        elif self.config["active"]["update_method"] == "evidential":
            al_dataset_update_idx = self.uncertainty_evidential()
        elif self.config["active"]["update_method"] == "gaussian":
            al_dataset_update_idx = self.uncertainty_gp()
        else:
            raise NotImplementedError(f"Active learning method {self.config['active']['update_method']} is not supported yet")
        
        if len(self.al_dataset_idx) + len(self.al_dataset_remaining_idx) != self.org_dataset_size:
            print(len(self.al_dataset_idx))
            print(len(al_dataset_update_idx))
            print(len(self.al_dataset_remaining_idx))
            print(self.org_dataset_size)
            raise ValueError("Error, dataset index not matching")
        
        self.set_active_dataset()
    
    def train(self):
        if self.config["active"].get("use", False):
            self.active_train()
        else:
            # Configurations
            ensure_fitted(self._unwrapped_model, warn=True)
            if self.logger:
                self.logger.log_model_training_info(self._unwrapped_model)
            eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
            checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
            primary_metric = self.config["task"].get("primary_metric", self.evaluator.task_primary_metric[self.task_name])
            if (
                not hasattr(self, "primary_metric")
                or self.primary_metric != primary_metric
            ):
                self.best_val_metric = 1e9 if ("mae" in primary_metric or "mse" in primary_metric) else -1.0
            else:
                primary_metric = self.primary_metric
            self.metrics = {}
            
            # Calculate start_epoch from step instead of loading the epoch number
            # to prevent inconsistencies due to different batch size in checkpoint.
            start_train_time = time.time()
            start_epoch = self.step // len(self.train_loader)
            for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
                start_epoch_time = time.time()
                self.train_sampler.set_epoch(epoch_int) # shuffle
                skip_steps = self.step % len(self.train_loader)
                train_loader_iter = iter(self.train_loader)            

                for i in range(skip_steps, len(self.train_loader)):
                    self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                    self.step = epoch_int * len(self.train_loader) + i + 1
                    self.model.train()
                    
                    # Get a batch.
                    batch = next(train_loader_iter)

                    # Forward, loss, backward.
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)
                    loss = self.scaler.scale(loss) if self.scaler else loss
                    self._backward(loss)
                    scale = self.scaler.get_scale() if self.scaler else 1.0

                    # Compute metrics.
                
                    self.metrics = self._compute_metrics(
                        out,
                        batch,
                        self.evaluator,
                        self.metrics,
                    )

                    # update local metrics (which will be aggregated across all ranks at print_every steps)
                    self.metrics = self.evaluator.update(
                        "loss", loss.item() / scale, self.metrics
                    )

                    if (self.step % self.config["cmd"]["print_every"] == 0 or
                        self.step % len(self.train_loader) == 0
                    ):
                        # 1) aggregate training results so far
                        # 2) print logging
                        # 3) reset metrics
                        aggregated_metrics = self.evaluator.aggregate(self.metrics)

                        log_dict = {k: aggregated_metrics[k]["metric"] for k in aggregated_metrics}
                        log_dict.update(
                            {
                                "lr": self.scheduler.get_lr(),
                                "epoch": self.epoch,
                                "step": self.step,
                            }
                        )
                        # stdout logging
                        bm_logging.info("[train] " + parse_logs(log_dict))
                        # logger logging
                        if self.logger:
                            self.logger.log(log_dict, step=self.step, split="train")
                        
                        # wandb logging
                        if self.config["wandb"]:
                            wandb.log(
                                {
                                    **{"train/"+k: log_dict[k] for k in log_dict},
                                    **{"train/dataset_size": {len(self.train_loader.dataset)}}
                                },
                                step=self.step,
                            )

                        # reset metrics after logging
                        self.metrics = {}

                    if (checkpoint_every != -1 and self.step % checkpoint_every == 0):
                        self.save(checkpoint_file="checkpoint.pt", training_state=True)

                    # Evaluate on val set every `eval_every` iterations.
                    if self.step % eval_every == 0:
                        if self.val_loader is not None:
                            val_metrics = self.validate(split="val")
                            self.update_best(primary_metric, val_metrics)
                            if self.config["wandb"]:
                                wandb.log(
                                    {"val/"+k: val_metrics[k]["metric"] for k in val_metrics},
                                    step=self.step
                                )

                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        if self.step % eval_every == 0:
                            self.scheduler.step(metrics=val_metrics[primary_metric]["metric"])
                    else:
                        self.scheduler.step()

                torch.cuda.empty_cache()

                if checkpoint_every == -1:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)
            
                if (self.config["save_ckpt_every_epoch"] and 
                    (epoch_int+1) % self.config["save_ckpt_every_epoch"] == 0
                ):
                    # evaluation checkpoint (for benchmarking models during training)
                    self.save(
                        metrics=val_metrics,
                        checkpoint_file=f"ckpt_ep{epoch_int+1}.pt",
                        training_state=False,
                    )

                bm_logging.info(f"{epoch_int+1} epoch elapsed time: {time.time()-start_epoch_time:.1f} sec")

            train_elapsed_time = time.time()-start_train_time
            bm_logging.info(f"training elapsed time: {train_elapsed_time:.1f} sec")

            # final evaluation
            bm_logging.info("Performing the final evaluation (last model)")
            metric_table = self.create_metric_table(display_meV=True)
            bm_logging.info(f"\n{metric_table}")
            if self.logger:
                self.logger.log_final_metrics(metric_table, train_elapsed_time)

            # end procedure of train()
            if self.config["wandb"]:
                wandb.finish()
            self._end_train()

    def active_train(self):
        # Configurations
        ensure_fitted(self._unwrapped_model, warn=True)
        if self.logger:
            self.logger.log_model_training_info(self._unwrapped_model)
        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.config["task"].get("primary_metric", self.evaluator.task_primary_metric[self.task_name])
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if ("mae" in primary_metric or "mse" in primary_metric) else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}
        
        # Active learning configurations
        self.init_active_dataset()
        global_step = 0
        initial_weights = deepcopy(self.model.state_dict())
        start_train_time = time.time()
        
        # NOTE: Active learning round loop
        for round_current in range(0, self.config["active"]["rounds"]):
            eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
            checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
            if round_current != 0:
                bm_logging.info(f"Model initiliazed to the initial weights")
                self.model.load_state_dict(initial_weights)
            bm_logging.info(f"\n>> Active learning - round {round_current} \n - Current dataset size: {len(self.train_loader.dataset)}\n - Remaining dataset size: {len(self.al_dataset_remaining_idx)}\n - Steps taken: {self.step}")
            
            # Training loop
            start_round_time = time.time()
            for epoch_int in range(0, self.config["optim"]["max_epochs"]):
                global_epoch = round_current * self.config["optim"]["max_epochs"] + epoch_int
                start_epoch_time = time.time()
                self.train_sampler.set_epoch(epoch_int) # shuffle
                train_loader_iter = iter(self.train_loader)      

                for i in range(0, len(self.train_loader)):
                    self.global_epoch = global_epoch
                    self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                    self.step = global_step + epoch_int * len(self.train_loader) + i + 1
                    self.model.train()
                    
                    # Get a batch.
                    batch = next(train_loader_iter)

                    # Forward, loss, backward.
                    with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                        out = self._forward(batch)
                        loss = self._compute_loss(out, batch)
                    loss = self.scaler.scale(loss) if self.scaler else loss
                    self._backward(loss)
                    scale = self.scaler.get_scale() if self.scaler else 1.0

                    # Compute metrics.
                    self.metrics = self._compute_metrics(
                        out,
                        batch,
                        self.evaluator,
                        self.metrics,
                    )

                    # update local metrics (which will be aggregated across all ranks at print_every steps)
                    self.metrics = self.evaluator.update(
                        "loss", loss.item() / scale, self.metrics
                    )

                    if (
                        self.step % self.config["cmd"]["print_every"] == 0 or
                        self.step % len(self.train_loader) == 0
                    ):
                        # 1) aggregate training results so far
                        # 2) print logging
                        # 3) reset metrics
                        aggregated_metrics = self.evaluator.aggregate(self.metrics)

                        log_dict = {k: aggregated_metrics[k]["metric"] for k in aggregated_metrics}
                        log_dict.update(
                            {
                                "lr": self.scheduler.get_lr(),
                                "epoch": self.epoch,
                                "step": self.step,
                            }
                        )
                        # stdout logging
                        bm_logging.info("[train] " + parse_logs(log_dict))
                        # logger logging
                        if self.logger:
                            self.logger.log(log_dict, step=self.step, split="train")
                        # wandb logging
                        if self.config["wandb"]:
                            wandb.log(
                                {
                                    **{"train/"+k: log_dict[k] for k in log_dict}, 
                                    **{
                                        "train/active_expanded": round_current,
                                        "train/active_dataset_size": len(self.train_loader.dataset),
                                        "train/global_epoch": global_epoch
                                    }
                                },
                                step=self.step,
                            )

                        # reset metrics after logging
                        self.metrics = {}

                    if (checkpoint_every != -1 and self.step % checkpoint_every == 0):
                        self.save(checkpoint_file="checkpoint.pt", training_state=True)

                    # Evaluate on val set every `eval_every` iterations. (Validation)
                    if self.step % eval_every == 0:
                        if self.val_loader is not None:
                            val_metrics = self.validate(split="val")
                            self.update_best(primary_metric, val_metrics)
                            if self.config["wandb"]:
                                wandb.log(
                                    {"val/"+k: val_metrics[k]["metric"] for k in val_metrics},
                                    step=self.step
                                )

                    if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                        if self.step % eval_every == 0:
                            self.scheduler.step(metrics=val_metrics[primary_metric]["metric"])
                    else:
                        self.scheduler.step()

                torch.cuda.empty_cache()

                if checkpoint_every == -1:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)
            
                if (self.config["save_ckpt_every_epoch"] and 
                    (epoch_int+1) % self.config["save_ckpt_every_epoch"] == 0
                ):
                    # evaluation checkpoint (for benchmarking models during training)
                    self.save(
                        metrics=val_metrics,
                        checkpoint_file=f"ckpt_ep{epoch_int+1}.pt",
                        training_state=False,
                    )

                bm_logging.info(f"{epoch_int+1} epoch elapsed time: {time.time()-start_epoch_time:.1f} sec")

            round_elapsed_time = time.time() - start_round_time
            bm_logging.info(f"Round {round_current} elapsed time: {round_elapsed_time:.1f} sec")
            self.save(
                metrics=val_metrics,
                checkpoint_file=f"ckpt_round{round_current}.pt",
                training_state=False,
            )
            
            # Evaluation done on each round
            self.round_current = round_current
            if not self.config["active"].get("debug", False):
                metric_table = self.create_metric_table(display_meV=True)
                bm_logging.info(f'Evaluation for round {round_current}')
                bm_logging.info(f"\n{metric_table}")
            
            # Update dataset
            global_step = self.step
            dataset_update_time = time.time()
            self.update_active_dataset(round_current)
            dataset_update_time = time.time()-dataset_update_time
            bm_logging.info(f"Dataset update time: {dataset_update_time:.1f} sec")
            if self.config["wandb"]:
                wandb.log(
                    {
                        "round": round_current,
                        "active.dataset_update_time": dataset_update_time,
                    },
                    step=self.step,
                )
        
        train_elapsed_time = time.time()-start_train_time
        bm_logging.info(f"training elapsed time: {train_elapsed_time:.1f} sec")

        # Final evaluation
        bm_logging.info("Performing the final evaluation (last model)")
        metric_table = self.create_metric_table(display_meV=True)
        bm_logging.info(f"\n{metric_table}")
        if self.logger:
            self.logger.log_final_metrics(metric_table, train_elapsed_time)

        # end procedure of train()
        if self.config["wandb"]:
            wandb.finish()
        self._end_train()
        
    def _forward(self, batch_list):
        _out = self.model(batch_list)
        # energy
        out = {"energy": _out[0].view(-1) if _out[0].shape[-1] == 1 else _out[0]}
        if len(_out) >= 2:
            out["forces"] = _out[1]
        if len(_out) == 3:
            out["stress"] = _out[2]
        return out

    def _compute_loss(self, out, batch_list):
        loss = []

        # Energy loss
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            if self.normalizer.get("per_atom", False):
                # normalization for per-atom energy
                N = torch.cat(
                    [batch.natoms.to(self.device) for batch in batch_list], dim=0
                )
                energy_target = self.normalizers["target"].norm(energy_target, N)
            else:
                # normalization for total energy
                energy_target = self.normalizers["target"].norm(energy_target)

        if "per_atom" in self.config["optim"].get("loss_energy", "energy_per_atom_mse"):
            natoms = torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
            )
            energy_loss = self.loss_fn["energy"](
                input=out["energy"], 
                target=energy_target, 
                natoms=natoms,
                batch_size=batch_list[0].natoms.shape[0],
            )
        else:
            energy_loss = self.loss_fn["energy"](
                input=out["energy"], 
                target=energy_target,
            )
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(energy_mult * energy_loss)

        # Force loss
        force_mult = self.config["optim"].get("force_coefficient", 30)        
        force_target = torch.cat(
            [batch.force.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            force_target = self.normalizers["grad_target"].norm(force_target)        

        if self.config["task"].get("train_on_free_atoms", False):
            # set a mask to filter out fixed atoms (for OC20)
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list], dim=0
            )
            free_mask = fixed == 0

            if (self.config["optim"].get("loss_force", "mse").startswith("atomwise")):
                force_mult = self.config["optim"].get("force_coefficient", 1)
                natoms = torch.cat(
                    [batch.natoms.to(self.device) for batch in batch_list], dim=0
                )
                natoms = torch.repeat_interleave(natoms, natoms)
                force_loss = self.loss_fn["force"](
                    input=out["forces"][free_mask],
                    target=force_target[free_mask],
                    natoms=natoms[free_mask],
                    batch_size=batch_list[0].natoms.shape[0],
                )
            else:
                force_loss = self.loss_fn["force"](
                    input=out["forces"][free_mask], 
                    target=force_target[free_mask],
                )
        else:
            force_loss = self.loss_fn["force"](
                input=out["forces"], 
                target=force_target,
            )
        loss.append(force_mult * force_loss)
        
        # NOTE: For analysis
        # self.peratom_energy_error.append([self.loss_fn["energy"](input=out["energy"][i], target=energy_target[i], natoms=1, batch_size=batch_list[0].natoms.shape[0],).item() for i in range(0, energy_target.shape[0])])
        # self.peratom_force_error.append([self.loss_fn["force"](input=out["forces"][i], target=force_target[i]).item() for i in range(0, force_target.shape[0])])

        if self.use_stress:
            # Stress loss
            stress_mult = self.config["optim"].get("stress_coefficient", 30)
            stress_target = torch.cat(
                [batch.stress.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                stress_target = self.normalizers["grad_target"].norm(stress_target)

            stress_loss = self.loss_fn["stress"](
                input=out["stress"], 
                target=stress_target,
            )
            loss.append(stress_mult * stress_loss)

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        
        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": torch.cat(
                [batch.natoms.to(self.device) for batch in batch_list], dim=0
            ),
        }
        if self.use_stress:
            target["stress"] = torch.cat(
                [batch.stress.to(self.device) for batch in batch_list], dim=0
            )
        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask_free = fixed == 0
            out["forces"] = out["forces"][mask_free]
            target["forces"] = target["forces"][mask_free]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(torch.sum(mask_free[s_idx : s_idx + natoms]).item())
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
        
        out["natoms"] = target["natoms"]

        # To calculate metrics, model output values are in real units
        if self.normalizer.get("normalize_labels", False):
            if self.normalizer.get("per_atom", False):
                N = torch.cat(
                    [batch.natoms.to(self.device) for batch in batch_list], dim=0
                )
                out["energy"] = self.normalizers["target"].denorm(out["energy"], N)
            else:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics