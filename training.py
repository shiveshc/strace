import argparse
import torch
import scanpy as sc
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from strace.data.dataset import CellDataPixel
from strace.model import models
from strace.inference.track_metrics import run_metrics
from strace.trainer.trainer_simclr import TrainerSimCLR
from strace.trainer.trainer_dino import TrainerDino
from strace.trainer.losses import simclr_loss, dino_loss
from strace.trainer.utils import set_seed


from typing import Optional


ALL_MODELS = {
    'SimCLR': models.SimCLR,
    'Dino': models.Dino,
}

ALL_LOSS_FN = {
    'simclr': simclr_loss,
    'Dino': dino_loss,
}

ALL_TRAINERS = {
    'SimCLR': TrainerSimCLR,
    'Dino': TrainerDino,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, help="Configs of yaml file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    set_seed()
    
    args = get_args()
    print(args.cfg_path)
    cfg = OmegaConf.load(args.cfg_path)
    
    data = CellDataPixel(**cfg['data'])
    train_data, val_data = torch.utils.data.random_split(data, [int(0.8*len(data)), len(data) - int(0.8*len(data))])
    train_dataloader = DataLoader(train_data, batch_size=cfg['train']['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg['train']['batch_size'], shuffle=True)
    
    trainer_cfg = {
        'hparams': cfg,
        'model_name': cfg['trainer']['model_name'],
        'model': ALL_MODELS[cfg['model']['model']],
        'loss_fn': ALL_LOSS_FN[cfg['loss']['method']],
        'train_dataloader': train_dataloader,
        'val_dataloader': val_dataloader,
        'ckpt_dir': cfg['trainer']['ckpt_dir'],
        'tensorboard_dir': cfg['trainer']['tensorboard_dir'],
        'embeddings_dir': cfg['trainer']['embeddings_dir'],
        'restore_ckpt': cfg['trainer']['restore_ckpt'],
    }

    trainer = ALL_TRAINERS[cfg['model']['model']](**trainer_cfg)
    trainer.train()

    
    master_adata = sc.read_h5ad('/home/schaudhary/mibi_segmentation/MIL_senescence/analysis/n2v_denoised_all_imgs_YXC_per_channel_batch_correction/mutiple_marker_in_immune_cells/adata.h5')
    model_metrics_df = run_metrics(args.cfg_path, master_adata)
    model_metrics_df.to_pickle(os.path.join(trainer_cfg['embeddings_dir'], f"model_metrics_df_{trainer_cfg['model_name']}_{str(datetime.now())}.pkl"))