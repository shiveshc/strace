import itertools
import os
import omegaconf
import yaml
import omegaconf
from omegaconf import OmegaConf

from strace.trainer.utils import set_params_cfg, save_to_path_cfg, write_hparam_slurm_jobs


def create_jobs(base_config: omegaconf.dictconfig.DictConfig,
                hparams: dict,
                hparam_run_name: str) -> None:
    
    hparam_model_path = os.path.join('/home/schaudhary/mibi_segmentation/Contrastive_Learning/trained_models', hparam_run_name)
    if os.path.isdir(hparam_model_path) == False:
        os.mkdir(hparam_model_path)
        
    hparam_output_path = os.path.join('/home/schaudhary/mibi_segmentation/Contrastive_Learning/outputs', hparam_run_name)
    if os.path.isdir(hparam_output_path) == False:
        os.mkdir(hparam_output_path)

    hparam_cfgs_path = os.path.join('/home/schaudhary/mibi_segmentation/Contrastive_Learning/configs', hparam_run_name)
    if os.path.isdir(hparam_cfgs_path) == False:
        os.mkdir(hparam_cfgs_path)
    with open(os.path.join(hparam_cfgs_path, 'hparam_sweep_info.yaml'), 'w') as file:
        yaml.dump(hparams, file)
    
    cfg_names = []
    sweep = itertools.product(*[v for k, v in hparams.items()])
    for n, cond in enumerate(sweep):
        for i, k in enumerate(hparams.keys()):
            param_chain = k.split('/')[1::]
            value = cond[i]
            base_config = set_params_cfg(base_config, param_chain, value)

        model_name = f'{hparam_run_name}_model_cond_{n}'
        base_config['trainer']['model_name'] = model_name
        
        base_config['trainer']['ckpt_dir'] = hparam_model_path
        base_config['trainer']['embeddings_dir'] = hparam_output_path

        save_to_path_cfg(base_config, hparam_cfgs_path, f'{hparam_run_name}_config_cond_{n}.yaml')
        cfg_names.append(f'{hparam_run_name}_config_cond_{n}.yaml')
    
    write_hparam_slurm_jobs(hparam_cfgs_path, cfg_names)


if __name__ == '__main__':

    # hparams = {
    #     '/model/act': ['ReLU', 'GELU'],
    #     '/model/out_dim': [128, 256, 512],
    #     '/model/hidden_layer_dims': [[768], [768, 768]],
    #     '/model/use_proj': [False],
    #     '/loss/tpt': [
    #         {'type': 'constant', 'min_val': 0.07, 'max_val': 0.07},
    #         {'type': 'linear', 'min_val': 0.04, 'max_val': 0.07, 'max_steps': 3200},
    #         {'type': 'constant', 'min_val': 0.1, 'max_val': 0.1},
    #     ],
    #     '/loss/l': [
    #         {'type': 'cosine', 'min_val': 0.99, 'max_val': 1},
    #         {'type': 'linear', 'min_val': 0.99, 'max_val': 1},
    #     ],
    # }
    hparams = {
        '/model/act': ['ReLU', 'GELU'],
        '/model/out_dim': [768],
        '/model/hidden_layer_dims': [[768], [768, 768]],
        '/model/use_proj': [True],
        '/model/out_layer_dims': [256, 512, 1024],
        'train/lr': [0.001, 0.0005],
        '/train/lr_scheduler':[
            {'type': 'LinearWarmupStep',
             'scheduler1': {'type': 'LinearLR', 'start_factor': 0.1, 'end_factor': 1, 'total_iters': 1000},
             'scheduler2': {'type': 'StepLR', 'step_size': 500, 'gamma': 0.95}},
            {'type': 'LinearWarmupDecreasingLinear',
             'scheduler1': {'type': 'LinearLR', 'start_factor': 0.1, 'end_factor': 1, 'total_iters': 1000},
             'scheduler2': {'type': 'LinearLR', 'start_factor': 1, 'end_factor': 0.1, 'total_iters': 7000}},
        ],
        '/train/clip_grad': [False],
        '/loss/tpt': [
            {'type': 'constant', 'min_val': 0.06, 'max_val': 0.06},
            {'type': 'linear', 'min_val': 0.04, 'max_val': 0.07, 'max_steps':3000},
            {'type': 'linear', 'min_val': 0.04, 'max_val': 0.07, 'max_steps':6000},
        ],
        '/loss/l': [
            {'type': 'linear', 'min_val': 0.99, 'max_val': 1},
        ],
    }
    # hparams = {
    #     '/model/act': ['ReLU'],
    #     '/model/out_dim': [128, 256, 512],
    #     '/model/hidden_layer_dims': [[768], [768, 768]],
    #     '/model/use_proj': [True],
    #     'train/lr': [0.001, 0.0005],
    #     '/train/lr_scheduler':[
    #         {'type': 'LinearWarmupCosine',
    #          'scheduler1': {'type': 'LinearLR', 'start_factor': 0.1, 'end_factor': 1, 'total_iters': 1000},
    #          'scheduler2': {'type': 'CosineAnnealingLR', 'T_max': 7000}},
    #         {'type': 'LinearWarmupStep',
    #          'scheduler1': {'type': 'LinearLR', 'start_factor': 0.1, 'end_factor': 1, 'total_iters': 1000},
    #          'scheduler2': {'type': 'StepLR', 'step_size': 500, 'gamma': 0.95}},
    #         {'type': 'LinearWarmupDecreasingLinear',
    #          'scheduler1': {'type': 'LinearLR', 'start_factor': 0.1, 'end_factor': 1, 'total_iters': 1000},
    #          'scheduler2': {'type': 'LinearLR', 'start_factor': 1, 'end_factor': 0.1, 'total_iters': 7000}},
    #     ],
    #     '/train/clip_grad': [None],
    #     '/loss/tau': [
    #         {'type': 'constant', 'min_val': 0.05, 'max_val': 0.05},
    #         {'type': 'constant', 'min_val': 0.1, 'max_val': 0.05},
    #         {'type': 'constant', 'min_val': 0.05, 'max_val': 0.1}
    #     ],
    # }

    base_config = OmegaConf.load('/home/schaudhary/mibi_segmentation/Contrastive_Learning/configs/dino_base_config.yaml')
    hparam_run_name = 'dino_hp_4'
    create_jobs(base_config, hparams, hparam_run_name)

        
            
        