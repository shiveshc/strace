import numpy as np
import omegaconf
import os
import pandas as pd
import random
import scanpy as sc
import time
import torch

import sys
sys.path.append('/home/schaudhary/mibi_segmentation/Contrastive_Learning')

from strace.data.dataset import CellDataPixel
from strace.inference.utils import chk_config, get_latest_ckpt
from strace.model import models
from strace.trainer import metrics


from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from typing import Union, Optional


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        

def run_metrics(cfg_path: str,
                master_adata: Optional[Union[str, sc.AnnData]] = None,
                step: Optional[int] = None) -> pd.DataFrame:
    
    curr_metrics_df = []
    
    cfg = OmegaConf.load(cfg_path)
    cfg_metadata = flatten_cfg(cfg)

    
    cfg = chk_config(cfg)
    cfg['data']['transforms'] = 'none'
    data = CellDataPixel(**cfg['data'])
    dataloader = DataLoader(data, batch_size=128, shuffle=True)

    
    ckpt_dir = cfg['trainer']['ckpt_dir']
    model_name = cfg['trainer']['model_name']
    checkpoint, ckpt_name = get_latest_ckpt(ckpt_dir, model_name, step=step)
    model_hparams = checkpoint['hparams']['model']
    model = getattr(models, [cfg['model']['model']])(**model_hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    if type(master_adata) == str:
        master_adata = sc.read_h5ad(master_adata)
    
    
    metrics_obj = getattr(metrics, cfg['trainer']['metrics'])(
        cfg,
        dataloader,
        device,
        master_adata
    )

    metrics, adata = metrics_obj(model)
    
    curr_metrics_df.append([
        cfg['trainer']['model_name'],
        cfg_path,
        ckpt_name, 
        *[v for k, v in metrics.items()],
        *[v for k, v in cfg_metadata.items()]
    ])
    
    adata.write_h5ad(os.path.join(cfg['trainer']['embeddings_dir'], f'adata_{m}'))
    
    curr_metrics_df = pd.DataFrame(curr_metrics_df, columns=[
        'model_name',
        'cfg_path',
        'ckpt_name',
        *[k for k, v in metrics.items()],
        *[k for k, v in cfg_metadata.items()]
    ])
    return curr_metrics_df
                                       
                               

if __name__ == '__main__':

    # metrics_track_df = pd.read_pickle()

    model_cfg_names = [f'dino_hp_3_config_cond_{i}' for i in range(16)]
    # model_names = [
    #     'simclr_exp_vit_b_16_tau_e1',
    #     'simclr_exp_vit_b_16_tau_e2',
    #     'simclr_exp_vit_b_16_tau_e1_maskcell',
    #     'simclr_exp_vit_b_16_tau_e2_maskcell',
    #     'simclr_raw_exp_vit_b_16_tau_e1',
    #     'simclr_raw_exp_vit_b_16_tau_e2',
    #     'simclr_raw_exp_vit_b_16_tau_e1_maskcell',
    #     'simclr_raw_exp_vit_b_16_tau_e2_maskcell',
    #     'simclr_raw_exp_vit_b_16_tau_e1_dropout',
    #     'simclr_raw_exp_vit_b_16_tau_e2_dropout',
    #     'simclr_raw_exp_vit_b_16_tau_e1_dropout_maskcell',
    #     'simclr_raw_exp_vit_b_16_tau_e2_dropout_maskcell',
        
    #     # 'simclr_exp_mlp_tau_e2_mean_cell',
    #     # 'simclr_exp_mlp_tau_e2_mean_pixel',
    #     # 'simclr_exp_mlp_tau_e2_mean_cell_dropout',
    #     # 'simclr_exp_mlp_tau_e2_mean_pixel_dropout',
    #     # 'simclr_raw_exp_mlp_tau_e2_mean_cell',
    #     # 'simclr_raw_exp_mlp_tau_e2_mean_pixel',
    #     # 'simclr_raw_exp_mlp_tau_e2_mean_cell_dropout',
    #     # 'simclr_raw_exp_mlp_tau_e2_mean_pixel_dropout',
    # ]

    # Manually setting the best step for model
    best_step = {
    # 'simclr_exp_vit_b_16_tau_e1': 6500,
    # 'simclr_exp_vit_b_16_tau_e1_maskcell': 3300,
    # 'simclr_raw_exp_vit_b_16_tau_e2_dropout_maskcell': 3500,
    'hp_1_model_cond_6': 7100,
    'hp_1_model_cond_11': 4000,
    'hp_1_model_cond_13': 6000
    }

    master_adata = sc.read_h5ad('/home/schaudhary/mibi_segmentation/MIL_senescence/analysis/n2v_denoised_all_imgs_YXC_per_channel_batch_correction/mutiple_marker_in_immune_cells/adata.h5')
    
    all_metrics_df = []
    for m in model_cfg_names:        
        print(f'running metrics for {m}')
        cfg_path = os.path.join('/home/schaudhary/mibi_segmentation/Contrastive_Learning/configs/dino_hp_3/', f'{m}.yaml')
        curr_metrics_df = run_metrics(cfg_path, master_adata, step=best_step[m] if m in best_step else None)
        all_metrics_df.append(curr_metrics_df)
    
    all_metrics_df = pd.concat(all_metrics_df, axis=0)
    all_metrics_df.to_pickle(os.path.join('/home/schaudhary/mibi_segmentation/Contrastive_Learning/outputs/dino_hp_3', f'model_metrics_df_{str(datetime.now())}.pkl'))
    
            



        
    