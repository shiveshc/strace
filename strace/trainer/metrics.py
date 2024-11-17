import numpy as np
import omegaconf
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from tqdm import tqdm

from strace.inference.utils import get_clustering_scores, get_classifier_scores
from strace.trainer.utils import heirarchical_cfg

from typing import Optional, Union, Tuple


class AccuracyMetrics(ABC):
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 dataloader: DataLoader,
                 device: torch.device,
                 master_adata: Optional[Union[str, sc.AnnData]] = None) -> None:

        self.cfg = cfg
        self.dataloader = dataloader
        if type(master_adata) == str:
            self.master_adata = sc.read_h5ad(master_adata)
        else:
            self.master_adata = master_adata
        self.device = device

    
    def generate_embeddings(self, model: nn.Module) -> sc.AnnData:
        torch.manual_seed(42)
        features = []
        metadata = []
        for n, batch in tqdm(enumerate(self.dataloader)):
            (x, x_prime), batch_metadata = batch['data'], batch['metadata']
            x, x_prime = x.to(self.device), x_prime.to(self.device)
            with torch.no_grad():
                if self.cfg['model']['model'] == 'SimCLR':
                    h, h_prime, z, z_prime = model(x, x_prime)
                elif self.cfg['model']['model'] == 'Dino':
                    s1, s2, h, g2, s1_p, s2_p, g1_p, g2_p = model(x, x_prime)
                else:
                    raise ValueError
            features.append(h.cpu().numpy())
            metadata.append(pd.DataFrame(batch_metadata))
            if len(features) > 150:
                break
        features = np.concatenate(features, axis=0)
        metadata = pd.concat(metadata, axis=0).reset_index().drop(columns=['index'])
    
        if type(self.master_adata) == sc.AnnData:
            tmp = self.master_adata.obs[['ind_ct_cd45_final', 'ind_ct_tumor_final', 'annotated_marker_v3', 'region_id', 'sample_img_name']]
            metadata = metadata.merge(tmp, on=['sample_img_name', 'region_id'], how='left')
    
        adata = sc.AnnData(features)
        adata.obs = metadata
        adata.uns['cfg'] = heirarchical_cfg(self.cfg)
        sc.pp.pca(adata, n_comps=20, use_highly_variable=False, svd_solver='arpack')
        sc.pp.neighbors(adata, metric='cosine', n_neighbors=5)
        sc.tl.umap(adata)
        for res in [0.5, 1, 2]:
            sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')
        return adata
    
    
    @abstractmethod
    def __call__(self, model: nn.Module) -> dict:
        pass


class ImmAccuracy(AccuracyMetrics):
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 dataloader: DataLoader,
                 device: torch.device,
                 master_adata: Optional[Union[str, sc.AnnData]] = None) -> None:
        
        super().__init__(cfg, dataloader, device, master_adata)

    
    def __call__(self, model: nn.Module) -> Tuple[dict, sc.AnnData]:
        adata = super().generate_embeddings(model)
        lr_imm_acc = get_classifier_scores(adata, 'ind_ct_cd45_final', 'logistic')
        metrics = {'lr_imm_acc': lr_imm_acc}
        return metrics, adata


class ImmTmrCTAccuracy(AccuracyMetrics):
    def __init__(self,
                 cfg: omegaconf.dictconfig.DictConfig,
                 dataloader: DataLoader,
                 device: torch.device,
                 master_adata: Optional[Union[str, sc.AnnData]] = None) -> None:
        
        super().__init__(cfg, dataloader, device, master_adata)

    
    def __call__(self, model: nn.Module) -> Tuple[dict, sc.AnnData]:
        adata = super().generate_embeddings(model)
        lr_imm_acc = get_classifier_scores(adata, 'ind_ct_cd45_final', 'logistic')
        lr_tmr_acc = get_classifier_scores(adata, 'ind_ct_tumor_final', 'logistic')
        knn_acc = get_classifier_scores(adata, 'annotated_marker_v3', 'knn')
        lr_ct_acc = get_classifier_scores(adata, 'annotated_marker_v3', 'one_vs_rest')
        metrics = {
            'lr_imm_acc': lr_imm_acc,
            'lr_tmr_acc': lr_tmr_acc,
            'knn_acc': knn_acc,
            'lr_ct_acc': lr_ct_acc,
        }
        return metrics, adata
        
    

