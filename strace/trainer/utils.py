import functools
import numpy as np
import omegaconf
import os
import random
import shutil
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from typing import Callable, List, Optional, Union


def set_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def flatten_cfg(cfg: omegaconf.dictconfig.DictConfig) -> dict:
    '''
    flatten omegaconf to 1 depth dict.
    key names in flat config are names like
    /model/in_ch, /data/mask_cell/ etc.
    '''
    def dfs(d, pname):

        if type(d) == omegaconf.listconfig.ListConfig:
            return {pname: list(d)}
        elif type(d) != omegaconf.dictconfig.DictConfig:
            return {pname: d}
        
        curr_dic = {}
        nbrs = [k for k in d]
        for e in nbrs:
            e_dict = dfs(d[e], e)
            for k in e_dict:
                curr_dic[f'{pname}/{k}'] = e_dict[k]
                
        return curr_dic

    flat_cfg = dfs(cfg, '')
    return flat_cfg


def heirarchical_cfg(cfg: omegaconf.dictconfig.DictConfig) -> dict:
    '''
    converts cfg to a heirarchical dict
    '''
    def dfs(d):
        if type(d) == omegaconf.listconfig.ListConfig:
            return list(d)
        elif type(d) != omegaconf.dictconfig.DictConfig:
            return d
        else:
            new_d = {}
            nbrs = [k for k in d]
            for e in nbrs:
                val = dfs(d[e])
                new_d[e] = val
            return new_d
    return dfs(cfg)


def set_params_cfg(cfg: omegaconf.dictconfig.DictConfig,
                  param_chain: List[str],
                  value: Union[bool, float, str, int]
                  ) -> omegaconf.dictconfig.DictConfig:
    
    def dfs(dic: dict, i: int) -> dict:
        if i == len(param_chain):
            return value
        else:
            for k in dic:
                if k == param_chain[i]:
                    dic[k] = dfs(dic[k], i + 1)
                else:
                    dic[k] = dic[k]
        return dic

    cfg = OmegaConf.to_object(cfg)
    cfg = dfs(cfg, 0)
    cfg = OmegaConf.create(cfg)
    return cfg

def save_to_path_cfg(cfg: omegaconf.dictconfig.DictConfig,
                     path: str,
                     name: str) -> None:
    if os.path.isdir(path) == False:
        os.mkdir(path)

    if name.endswith('.yaml') == False:
        name = f'{name}.yaml'
        
    OmegaConf.save(cfg, os.path.join(path, name))


def write_hparam_slurm_jobs(path: str,
                            cfg_paths: List[str]) -> None:

    shutil.copy('/home/schaudhary/mibi_segmentation/Contrastive_Learning/slurm_jobs/training.sh', os.path.join(path, 'training.sh'))
    with open(os.path.join(path, 'submit_training.sh'), 'w') as file:
        file.write('#!/bin/bash\n')
        file.write('\n')
        for i in range(len(cfg_paths)):
            file.write(f'sbatch training.sh {cfg_paths[i]}\n')


def chk_lr_scheduler_params(cfg: Optional[omegaconf.dictconfig.DictConfig] = None) -> None:
    if type(cfg) == type(None):
        instructions = '''
        ----------------------------------
        Here are the parameters needed for each LR scheduler.
        
        lr_scheduler.ExponentialLR(optimizer, gamma)
        lr_scheduler.StepLR(optimizer, step_size, gamma)
        lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
        lr_scheduler.CosineAnnealingLR(optimizer, T_max)
        
        LinearWarmupCosine:
            scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
            scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max)
        LinearWarmupStep:
            scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
            scheduler2 = lr_scheduler.StepLR(optimizer, step_size, gamma)
        LinearWarmupExponential:
            scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
            scheduler2 = lr_scheduler.ExponentialLR(optimizer, gamma)
        LinearWarmupDecreasingLinear:
            scheduler1 = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
            scheduler2 = lr_scheduler.LinearLR(optimizer, start_factor, end_factor, total_iters)
        '''
        print(instructions)
    else:
        if 'lr_scheduler' not in cfg['train']:
            pass
        else:
            lr_scheduler_name = cfg['train']['lr_scheduler']['type']
            if lr_scheduler_name == 'ExponentialLR':
                assert 'gamma' in cfg['train']['lr_scheduler']
            elif lr_scheduler_name == 'StepLR':
                assert 'step_size' in cfg['train']['lr_scheduler'] and 'gamma' in cfg['train']['lr_scheduler']
            elif lr_scheduler_name == 'LinearLR':
                assert 'start_factor' in cfg['train']['lr_scheduler'] and 'end_factor' in cfg['train']['lr_scheduler'] and 'total_iters' in cfg['train']['lr_scheduler']
            elif lr_scheduler_name == 'DecreasingLinearLR':
                assert 'start_factor' in cfg['train']['lr_scheduler'] and 'end_factor' in cfg['train']['lr_scheduler'] and 'total_iters' in cfg['train']['lr_scheduler']
            elif lr_scheduler_name == 'CosineAnnealingLR':
                assert 'T_max' in cfg['train']['lr_scheduler']
            elif lr_scheduler_name == 'LinearWarmupCosine':
                scheduler1 = cfg['train']['lr_scheduler']['scheduler1']
                scheduler2 = cfg['train']['lr_scheduler']['scheduler2']
                assert 'start_factor' in scheduler1 and 'end_factor' in scheduler1 and 'total_iters' in scheduler1
                assert 'T_max' in scheduler2
            elif lr_scheduler_name == 'LinearWarmupStep':
                scheduler1 = cfg['train']['lr_scheduler']['scheduler1']
                scheduler2 = cfg['train']['lr_scheduler']['scheduler2']
                assert 'start_factor' in scheduler1 and 'end_factor' in scheduler1 and 'total_iters' in scheduler1
                assert 'step_size' in scheduler2 and 'gamma' in scheduler2
            elif lr_scheduler_name == 'LinearWarmupExponential':
                scheduler1 = cfg['train']['lr_scheduler']['scheduler1']
                scheduler2 = cfg['train']['lr_scheduler']['scheduler2']
                assert 'start_factor' in scheduler1 and 'end_factor' in scheduler1 and 'total_iters' in scheduler1
                assert 'gamma' in scheduler2
            elif lr_scheduler_name == 'LinearWarmupDecreasingLinear':
                scheduler1 = cfg['train']['lr_scheduler']['scheduler1']
                scheduler2 = cfg['train']['lr_scheduler']['scheduler2']
                assert 'start_factor' in scheduler1 and 'end_factor' in scheduler1 and 'total_iters' in scheduler1
                assert 'start_factor' in scheduler2 and 'end_factor' in scheduler2 and 'total_iters' in scheduler2
            else:
                raise ValueError
                
                

def get_lr_scheduler(cfg: omegaconf.dictconfig.DictConfig,
                     optimizer: nn.Module,
                     **kwargs) -> nn.Module:

    if 'lr_scheduler' not in cfg['train']:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)
    else:
        lr_scheduler_name = cfg['train']['lr_scheduler']['type']
        lr_scheduler_params = {k:v for k, v in cfg['train']['lr_scheduler'].items() if k != 'type'}
        if lr_scheduler_name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler_name == 'LinearLR':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler_name == 'DecreasingLinearLR':
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler_name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **lr_scheduler_params)
        elif lr_scheduler_name == 'LinearWarmupCosine':
            milestone = lr_scheduler_params['scheduler1']['total_iters']
            scheduler1_params = {k:v for k, v in lr_scheduler_params['scheduler1'].items() if k != 'type'}
            scheduler2_params = {k:v for k, v in lr_scheduler_params['scheduler2'].items() if k != 'type'}
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler1_params)
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler2_params)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[milestone])
        elif lr_scheduler_name == 'LinearWarmupStep':
            milestone = lr_scheduler_params['scheduler1']['total_iters']
            scheduler1_params = {k:v for k, v in lr_scheduler_params['scheduler1'].items() if k != 'type'}
            scheduler2_params = {k:v for k, v in lr_scheduler_params['scheduler2'].items() if k != 'type'}
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler1_params)
            scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler2_params)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[milestone])
        elif lr_scheduler_name == 'LinearWarmupExponential':
            milestone = lr_scheduler_params['scheduler1']['total_iters']
            scheduler1_params = {k:v for k, v in lr_scheduler_params['scheduler1'].items() if k != 'type'}
            scheduler2_params = {k:v for k, v in lr_scheduler_params['scheduler2'].items() if k != 'type'}
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler1_params)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler2_params)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[milestone])
        elif lr_scheduler_name == 'LinearWarmupDecreasingLinear':
            milestone = lr_scheduler_params['scheduler1']['total_iters']
            scheduler1_params = {k:v for k, v in lr_scheduler_params['scheduler1'].items() if k != 'type'}
            scheduler2_params = {k:v for k, v in lr_scheduler_params['scheduler2'].items() if k != 'type'}
            scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler1_params)
            scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler2_params)
            scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[milestone])
        else:
            raise ValueError
    return scheduler


def get_param_schedule(schedule:Union[dict, omegaconf.dictconfig.DictConfig]) -> Callable:
    '''
    get parameter schedule for any hyperparameter
    e.g. scheduler of tau in simclr loss, or tps in
    dino loss, or momentum parameter in dino loss
    '''
    
    def schedule_constant(step:int,
                          min_val:int,
                          max_val:int,
                          max_step:int,
                          **kwargs) -> int:
        return max_val

    def schedule_linear(step:int,
                        min_val:int,
                        max_val:int,
                        max_step:int,
                        **kwargs) -> int:
        if step > max_step:
            return max_val
        else:
            val = min_val + (step/max_step)*(max_val - min_val)
            return val

    def schedule_cosine(step:int,
                        min_val:int,
                        max_val:int,
                        max_step:int,
                        **kwargs) -> int:
        if step > max_step:
            return max_val
        else:
            val = min_val + (max_val - min_val)*0.5*(1 + np.cos(np.pi*step/max_step - np.pi))
            return val
    
    if schedule['type'] == 'constant':
        fn = functools.partial(schedule_constant, **schedule)
    elif schedule['type'] == 'linear':
        fn = functools.partial(schedule_linear, **schedule)
    elif schedule['type'] == 'cosine':
        fn = functools.partial(schedule_cosine, **schedule)
    else:
        raise ValueError
    
    return fn
        
