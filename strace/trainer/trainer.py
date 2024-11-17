import omegaconf
import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from strace.inference.utils import restore_model_ckpt
from strace.trainer import metrics
from strace.trainer.utils import flatten_cfg, get_lr_scheduler, chk_lr_scheduler_params


from typing import Callable, Tuple


class Trainer(ABC):
    def __init__(self,
                 hparams: omegaconf.dictconfig.DictConfig,
                 model_name: str,
                 model: nn.Module,
                 loss_fn: Callable,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 ckpt_dir: str,
                 tensorboard_dir: str,
                 embeddings_dir: str,
                 restore_ckpt: bool,
                 **kwargs) -> None:
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.hparams = hparams
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.model_hparams = hparams['model']
        self.model_save_path = os.path.join(ckpt_dir, model_name)
        if os.path.isdir(self.model_save_path) == False:
            os.mkdir(self.model_save_path)
        self.model = model(**self.model_hparams).to(self.device)
        
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams['train']['lr'])
        chk_lr_scheduler_params(self.hparams)
        self.lr_scheduler = get_lr_scheduler(self.hparams, self.optimizer)
        
        if restore_ckpt:
            self.model, self.optimizer, self.lr_scheduler = restore_model_ckpt(self.model, self.optimizer, self.hparams)
            
        self.loss_fn = loss_fn
        self.writer = SummaryWriter(os.path.join(tensorboard_dir, model_name))


        self.metrics_obj = getattr(metrics, self.hparams['trainer']['metrics'])(
            self.hparams,
            self.train_dataloader,
            self.device,
            self.hparams['trainer']['master_adata']
        )
        
        
    
    def get_tb_hparams(self) -> dict:
        # save_hparams = {}
        # for p in ['model', 'train', 'loss']:
        #     for k, v in self.hparams[p].items():
        #         save_hparams[f'{p}/{k}'] = v
        save_hparams = flatten_cfg(self.hparams)
        return save_hparams
            
        
    def logging_loss(self,
                     epoch:int,
                     train_loss: Tensor,
                     val_loss: Tensor) -> None:
        self.writer.add_scalar('train_loss', train_loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        print(f'epoch {epoch}, train loss {train_loss}, val loss {val_loss}')

    
    def logging_metrics(self,
                        epoch:int,
                        metrics:dict) -> None:
        for k in metrics:
            self.writer.add_scalar(k, metrics[k], epoch)    
    
    
    def save_ckpt(self, step: int) -> None:
        save_name = os.path.join(self.model_save_path, f'model_{step}.pt')
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'hparams': self.hparams}, save_name)

    
    @abstractmethod
    def add_embedding(self, step: int, num_samples: int = 500) -> None:
        features = []
        cnt = 0
        for n, batch in enumerate(self.train_dataloader):
            (x, x_prime), batch_metadata = batch['data'], batch['metadata']
            x, x_prime = x.to(self.device), x_prime.to(self.device)
            cnt += x.shape[0]
            with torch.no_grad():
                h, h_prime, z, z_prime = self.model(x, x_prime)
                features.append(h)
    
            if cnt > num_samples:
                break
                
        features = torch.concatenate(features, axis=0)
        self.writer.add_embedding(features, global_step=step)
        
        
    @abstractmethod
    def train_step(self,
                   batch: Tuple[Tensor, Tensor],
                   step:int) -> Tensor:
        
        pass
    
    @abstractmethod
    def val_step(self,
                 step:int,
                 data_size: int = 500) -> Tensor:
        pass
    
    
    def train(self) -> None:
        epochs = self.hparams['train']['epochs']
        for i in range(epochs):
            for n, batch in enumerate(self.train_dataloader):
                step = i*len(self.train_dataloader) + n
                train_loss = self.train_step(batch, step)
    
                if step%1 == 0:
                    val_loss = self.val_step(step)
                    self.logging_loss(step, train_loss, val_loss)

                # if step%250 == 0:
                #     metrics, adata = self.metrics_obj(self.model)
                #     self.logging_metrics(step, metrics)
    
                if step%500 == 0:
                    self.save_ckpt(step)
                    self.add_embedding(step)
    
        self.save_ckpt(step)
        # self.writer.add_hparams(self.get_tb_hparams(), {'train_loss':train_loss, 'val_loss':val_loss})