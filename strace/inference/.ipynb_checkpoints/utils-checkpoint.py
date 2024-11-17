import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import os
import pandas as pd
import random
import scanpy as sc
import torch
import torch.nn as nn
from matplotlib.axes import Axes
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from strace.model import models
from strace.trainer.utils import flatten_cfg

from typing import Union, Optional, Tuple

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def chk_config(cfg: omegaconf.dictconfig.DictConfig) -> omegaconf.dictconfig.DictConfig:
    '''
    some parameters were not set in old config files
    this functions makes config file consisted wrt
    the latest config
    '''
    if 'mask_cell' not in cfg['data']:
        cfg['data']['mask_cell'] = False

    if 'data_mode' not in cfg['data']:
        cfg['data']['data_mode'] = 'patch'

    if 'embeddings_dir' not in cfg['trainer']:
        cfg['trainer']['embeddings_dir'] = '/home/schaudhary/mibi_segmentation/Contrastive_Learning/outputs/embeddings'

    if 'model' not in cfg['model']:
        cfg['model']['model'] = models.SimCLR

    cfg['data']['mode'] = 'inference'
    return cfg


def get_latest_ckpt(ckpt_dir: str,
                    model_name: str,
                    step: Optional[int] = None) -> Tuple[dict, str]:
    '''
    Function to load model ckpt
    '''
    if step == None:
        ckpts = [c for c in os.listdir(os.path.join(ckpt_dir, model_name)) if c.endswith('.pt')]
        if len(ckpts) == 0:
            print('No ckpts found')
            return {}, ''
        else:
            ckpts_t = []
            for c in ckpts:
                t = c.split('_')[1]
                t = int(t[0:len(t) - 3])
                ckpts_t.append(t)
            ckpts_t = sorted(ckpts_t)
            last_ckpt = f'model_{ckpts_t[-1]}.pt'
            print(last_ckpt)
            checkpoint = torch.load(os.path.join(ckpt_dir, model_name, last_ckpt), weights_only=False)
            return checkpoint, last_ckpt
    else:
        print(f'model_{step}.pt')
        checkpoint = torch.load(os.path.join(ckpt_dir, model_name, f'model_{step}.pt'), weights_only=False)
        return checkpoint, f'model_{step}.pt'


def restore_model_ckpt(model: nn.Module,
                       optimizer: nn.Module,
                       lr_scheduler: nn.Module,
                       cfg: omegaconf.dictconfig.DictConfig) -> Tuple[nn.Module, nn.Module, nn.Module]:
    '''
    restore model to the latest ckpt in case of
    training interruptions
    '''
    ckpt_dir = cfg['trainer']['ckpt_dir']
    model_name = cfg['trainer']['model_name']
    checkpoint, _ = get_latest_ckpt(ckpt_dir, model_name)
    if len(checkpoint) == 0:
        return model, optimizer, lr_scheduler
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        return model, optimizer, lr_scheduler
    

def get_clustering_scores(adata: sc.AnnData,
                          clust_key: str,
                          sample_size: Optional[int] = None) -> Tuple[float, float, float]:
    '''
    Function to evaluate clustering of embeddings using unsupervised cluster
    scoring methods
    '''
    if sample_size == None:
        silhouette = silhouette_score(adata.X, adata.obs[clust_key], metric='cosine')
        db_index = davies_bouldin_score(adata.X, adata.obs[clust_key])
        ch_index = calinski_harabasz_score(adata.X, adata.obs[clust_key])
    else:
        random.seed(42)
        select = random.sample(range(adata.shape[0]), min(sample_size, len(adata)))
        adata = adata[select]
        silhouette = silhouette_score(adata.X, adata.obs[clust_key], metric='cosine')
        db_index = davies_bouldin_score(adata.X, adata.obs[clust_key])
        ch_index = calinski_harabasz_score(adata.X, adata.obs[clust_key])
    return silhouette, db_index, ch_index


def get_classifier_scores(adata: sc.AnnData,
                          label: str,
                          method: str,
                          sample_size: Optional[int] = None) -> float:
    
    X = np.asarray(adata.X)
    y = adata.obs[label].values
    if sample_size != None:
        random.seed(42)
        select = random.select(range(X.shape[0]), sample_size)
        X = X[select, :]
        y = y[select, :]
    
    if method == 'logistic':
        clf = LogisticRegression()
        scores = cross_validate(clf, X, y, cv=5, scoring=['accuracy'])
        acc = np.mean(scores['test_accuracy'])
    elif method == 'knn':
        clf = KNeighborsClassifier(n_neighbors=10, metric='cosine')
        acc = clf.fit(X, y).score(X, y)
    elif method == 'one_vs_rest':
        clf = OneVsRestClassifier(LogisticRegression())
        scores = cross_validate(clf, X, y, cv=5, scoring=['accuracy'])
        acc = np.mean(scores['test_accuracy'])
    else:
        raise ValueError(f'{method} not implemented')
    return acc


def plot_spatial_cluster(adata: sc.AnnData,
                         clust_key: str,
                         clust_id: str,
                         ax: Optional[Union[np.ndarray, Axes]] = None) -> None:
    '''
    Function to plot cells colored by cluster id
    '''
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    tmp_adata = adata[adata.obs[clust_key] == clust_id]
    spatial = tmp_adata.obsm['spatial']
    ax.scatter(spatial[:, 0], spatial[:, 1], s=4, c=tmp_adata.uns[f'{clust_key}_colors'][0])
    ax.axis('off')
    

def plot_all_spatial_cluster(adata: sc.AnnData,
                             clust_key: str,
                             ax: Optional[Axes] = None) -> None:
    '''
    Function to plot cells colored by all cluster ids
    '''
    cmap1 = sns.color_palette()
    cmap2 = sns.color_palette('pastel')
    cmap3 = sns.color_palette('hls')
    colors = cmap1[0:7] + cmap1[8::] + cmap2 + cmap3
    
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
    uniq_clust = list(adata.obs[clust_key].unique())
    for i, n in enumerate(uniq_clust):
        plot_spatial_cluster(adata, clust_key, n, ax)


def plot_spatial_annotation(adata: sc.AnnData,
                            key: str,
                            ax: Optional[Axes] = None) -> None:
    '''
    Function to plot cells colored by cell type annotation
    '''
    cmap1 = sns.color_palette()
    cmap2 = sns.color_palette('pastel')
    cmap3 = sns.color_palette('hls')
    colors = cmap1[0:7] + cmap1[8::] + cmap2 + cmap3

    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    tmp_adata = adata.copy()
    for i, m in enumerate(key):
        tmp_adata = tmp_adata[tmp_adata.obs[f'ind_ct_{m}'] == 0]
    spatial = tmp_adata.obsm['spatial']
    ax.scatter(spatial[:, 0], spatial[:, 1], s=4, c='grey')
    
    for i, m in enumerate(key):
        tmp_adata = adata[adata.obs[f'ind_ct_{m}'] == 1]
        spatial = tmp_adata.obsm['spatial']
        ax.scatter(spatial[:, 0], spatial[:, 1], s=4, c=colors[i], label=m)
    ax.axis('off')


def plot_cluster_example_images(adata: sc.AnnData,
                                clust_key: str,
                                clust_id: str,
                                ch: str,
                                num_samples: Optional[int] = 8,
                                ax: Optional[Axes] = None) -> None:
    '''
    Function to plot random samples of cells in a cluster
    where clustering is generated based on embeddings
    '''
    img_crop_path = adata.uns['cfg']['/data/img_crop_path']
    tmp_adata = adata[adata.obs[clust_key] == clust_id]
    select = random.sample(range(len(tmp_adata)), num_samples)

    if ax == None:
        fig, ax = plt.subplots(1, num_samples, figsize=(24, 4))

    for i in range(len(select)):
        curr_img = tifffile.imread(os.path.join(img_crop_path, tmp_adata.obs.iloc[select[i]]['img_crop_name']))
        ax[i].imshow(curr_img[ch])
        ax[i].axis('off')


def run_inference_img(img_name: str,
                      cfg: omegaconf.dictconfig.DictConfig,
                      data: Dataset,
                      model: nn.Module,
                      master_adata: Optional[sc.AnnData] = None) -> sc.AnnData:
    '''
    Function to plot random samples of cells in a cluster
    where clustering is generated based on embeddings
    '''

    cfg = chk_config(cfg)
    data = data(img_name, **cfg['data'])
    print(len(data))
    dataloader = DataLoader(data, batch_size=128, shuffle=False)
    
    ckpt_dir = cfg['trainer']['ckpt_dir']
    model_name = cfg['trainer']['model_name']
    checkpoint, ckpt_name = get_latest_ckpt(ckpt_dir, model_name, step=None)
    model_hparams = checkpoint['hparams']['model']
    model = model(**model_hparams)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    
    torch.manual_seed(42)
    features = []
    metadata = []
    for n, batch in tqdm(enumerate(dataloader)):
        (x, x_prime), batch_metadata = batch['data'], batch['metadata']
        x, x_prime = x.to(device), x_prime.to(device)
        with torch.no_grad():
            h, h_prime, z, z_prime = model(x, x_prime)
        features.append(h.cpu().numpy())
        metadata.append(pd.DataFrame(batch_metadata))
    features = np.concatenate(features, axis=0)
    metadata = pd.concat(metadata, axis=0).reset_index().drop(columns=['index'])

    if type(master_adata) == sc.AnnData:
        tmp = master_adata.obs[['ind_ct_cd45_final', 'ind_ct_tumor_final', 'annotated_marker_v3', 'region_id', 'sample_img_name']]
        metadata = metadata.merge(tmp, on=['sample_img_name', 'region_id'], how='left')
        
    adata = sc.AnnData(features)
    adata.obs = metadata
    adata.uns['cfg'] = flatten_cfg(cfg)
    sc.pp.pca(adata, n_comps=20, use_highly_variable=False, svd_solver='arpack')
    sc.pp.neighbors(adata, metric='cosine', n_neighbors=5)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5, key_added='leiden_r0.5')
    sc.pl.umap(adata, color=['leiden_r0.5'])
    return adata