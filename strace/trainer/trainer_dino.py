import numpy as np
import omegaconf
import torch
from torch import Tensor

from strace.trainer.trainer import Trainer
from strace.trainer.utils import get_lr_scheduler, chk_lr_scheduler_params, get_param_schedule

from typing import Callable, Tuple, Union


class TrainerDino(Trainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        
        max_step = len(self.train_dataloader)*self.hparams['train']['epochs']
        self.hparams['loss']['tps']['max_step'] = max_step
        self.hparams['loss']['tpt']['max_step'] = max_step
        self.hparams['loss']['l']['max_step'] = max_step
        self.hparams['loss']['m']['max_step'] = max_step

        self.tps = get_param_schedule(self.hparams['loss']['tps'])
        self.tpt = get_param_schedule(self.hparams['loss']['tpt'])
        self.l = get_param_schedule(self.hparams['loss']['l'])
        self.m = get_param_schedule(self.hparams['loss']['m'])
        self.C = torch.tensor([0], device=self.device)

    
    def logging_loss(self,
                     epoch:int,
                     train_loss: Tensor,
                     val_loss: Tensor) -> None:
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.writer.add_scalar('tps', self.tps(epoch), epoch)
        self.writer.add_scalar('tpt', self.tpt(epoch), epoch)
        self.writer.add_scalar('l', self.l(epoch), epoch)
        self.writer.add_scalar('m', self.m(epoch), epoch)
        print(f'epoch {epoch}, train loss {train_loss}, val loss {val_loss}')

    
    def add_embedding(self, step: int, num_samples: int = 500) -> None:
        pass
        
    
    def train_step(self,
                   batch: Tuple[Tensor, Tensor],
                   step: int) -> Tensor:
        
        tps = self.tps(step)
        tpt = self.tpt(step)
        l = self.l(step)
        m = self.m(step)
        C = self.C
        
        self.optimizer.zero_grad()
        (x, x_prime), batch_metadata = batch['data'], batch['metadata']
        x, x_prime = x.to(self.device), x_prime.to(self.device)
        s1, s2, t1, t2, s1_p, s2_p, t1_p, t2_p = self.model(x, x_prime)
        loss = (self.loss_fn(s1_p, t2_p, tps, tpt, C) + self.loss_fn(s2_p, t1_p, tps, tpt, C))/2
        loss.backward()
        if type(self.hparams['train']['clip_grad']) != type(None):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams['train']['clip_grad'])
        self.optimizer.step()
        self.lr_scheduler.step()

        with torch.no_grad():
            for gt_backbone_param, gs_backbone_param in zip(self.model.gt_backbone.parameters(), self.model.gs_backbone.parameters()):
                gt_backbone_param.data = (l * gt_backbone_param.data) + (1 - l) * gs_backbone_param.data

            if hasattr(self.model, 'gs_projector') and hasattr(self.model, 'gt_projector'):
                for gt_projector_param, gs_projector_param in zip(self.model.gt_projector.parameters(), self.model.gs_projector.parameters()):
                    gt_projector_param.data = (l * gt_projector_param.data) + (1 - l) * gs_projector_param.data
                
                # gs_projector_param = self.model.gs_projector.state_dict()
                # gt_projector_param = self.model.gt_projector.state_dict()
                # for p in gt_projector_param:
                #     gt_projector_param[p] = l*gt_projector_param[p] + (1 - l)*gs_projector_param[p]
                # self.model.gt_projector.load_state_dict(gt_projector_param)

            self.C = m*C + (1 - m)*torch.mean(torch.cat([t1_p, t2_p], dim=0), dim=0, keepdim=True)
        return loss
    
    
    def val_step(self,
                 step: int,
                 data_size: int = 500) -> Tensor:
        
        tps = self.tps(step)
        tpt = self.tpt(step)
        C = self.C
        
        loss = 0
        cnt = 0
        for n, batch in enumerate(self.val_dataloader):
            (x, x_prime), batch_metadata = batch['data'], batch['metadata']
            x, x_prime = x.to(self.device), x_prime.to(self.device)
            cnt += x.shape[0]
            with torch.no_grad():
                s1, s2, t1, t2, s1_p, s2_p, t1_p, t2_p = self.model(x, x_prime)
                loss += (self.loss_fn(s1_p, t2_p, tps, tpt, C) + self.loss_fn(s2_p, t1_p, tps, tpt, C))/2
            if cnt > data_size:
                break
        loss = loss/(n + 1)
        return loss