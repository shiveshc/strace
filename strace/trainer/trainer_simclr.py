import omegaconf
import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from strace.trainer.trainer import Trainer
from strace.trainer.utils import get_param_schedule


from typing import Callable, Tuple


class TrainerSimCLR(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        max_step = len(self.train_dataloader)*self.hparams['train']['epochs']
        self.hparams['loss']['tau']['max_step'] = max_step
        self.tau = get_param_schedule(self.hparams['loss']['tau'])
        
    
    def add_embedding(self, step: int, num_samples: int = 500) -> None:
        pass

    
    def train_step(self,
                   batch: Tuple[Tensor, Tensor],
                   step:int) -> Tensor:

        tau = self.tau(step)
        
        self.optimizer.zero_grad()
        (x, x_prime), batch_metadata = batch['data'], batch['metadata']
        x, x_prime = x.to(self.device), x_prime.to(self.device)
        h, h_prime, z, z_prime = self.model(x, x_prime)
        loss = self.loss_fn(z, z_prime, tau) 
        loss.backward()
        if type(self.hparams['train']['clip_grad']) != type(None):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams['train']['clip_grad'])
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss
    
    
    def val_step(self,
                 step:int,
                 data_size: int = 500) -> Tensor:
        
        tau = self.tau(step)
        
        loss = 0
        cnt = 0
        for n, batch in enumerate(self.val_dataloader):
            (x, x_prime), batch_metadata = batch['data'], batch['metadata']
            x, x_prime = x.to(self.device), x_prime.to(self.device)
            cnt += x.shape[0]
            with torch.no_grad():
                h, h_prime, z, z_prime = self.model(x, x_prime)
                loss += self.loss_fn(z, z_prime, tau)
            if cnt > data_size:
                break
        loss = loss/(n + 1)
        return loss