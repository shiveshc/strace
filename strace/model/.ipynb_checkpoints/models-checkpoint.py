import torch
import torch.nn as nn
from torch import Tensor

from strace.model.layers import Encoder, Projector

from typing import Tuple


class SimCLR(nn.Module):
    def __init__(self,
                 in_ch: int,
                 backbone: str,
                 out_dim: int,
                 dropout: bool = False,
                 **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder(in_ch, backbone, **kwargs)
        enc_dim = self.encoder.out_dim
        self.projector = Projector(enc_dim, out_dim, **kwargs)
        self.dropout = dropout

    def forward(self,
                x: Tensor, 
                x_prime: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        '''
            Args:
             x - image shape (N, C, H, W)

            Returns:
             z - shape (N, D)
             z_prime - shape (N, D)
        '''

        h = self.encoder(x)
        h_prime = self.encoder(x_prime)
        if self.dropout:
            h = nn.Dropout()(h)
            h_prime = nn.Dropout()(h_prime)
        z = self.projector(h)
        z_prime = self.projector(h_prime)
        return h, h_prime, z, z_prime


class Dino(nn.Module):
    def __init__(self,
                 in_ch: int,
                 backbone: str,
                 out_dim: int,
                 **kwargs) -> None:
        super().__init__()
        
        self.gs_backbone = Encoder(in_ch, backbone, **kwargs)
        self.gt_backbone = Encoder(in_ch, backbone, **kwargs)

        self.gt_backbone.load_state_dict(self.gs_backbone.state_dict())
        for p in self.gt_backbone.parameters():
            p.requires_grad = False
            
        if 'use_proj' in kwargs and kwargs['use_proj'] == True:
            enc_dim = self.gs_backbone.out_dim
            self.gs_projector = Projector(enc_dim, out_dim, **kwargs)
            print(self.gs_projector)
            self.gt_projector = Projector(enc_dim, out_dim, **kwargs)

            self.gt_projector.load_state_dict(self.gs_projector.state_dict())
            for p in self.gt_projector.parameters():
                p.requires_grad = False

    def forward(self,
                x: Tensor,
                x_prime: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        '''
            Args:
             x - image shape (N, C, H, W)

            Returns:
             s1 - shape (N, D)
             s2 - shape (N, D)
             g1 - shape (N, D)
             g2 - shape (N, D)
        '''
        s1, s2 = self.gs_backbone(x), self.gs_backbone(x_prime)
        g1, g2 = self.gt_backbone(x), self.gt_backbone(x_prime)

        if hasattr(self, 'gs_projector') and hasattr(self, 'gt_projector'):
            s1_p, s2_p = self.gs_projector(s1), self.gs_projector(s2)
            g1_p, g2_p = self.gt_projector(g1), self.gt_projector(g2)
        else:
            s1_p, s2_p = s1, s2
            g1_p, g2_p = g1, g2
            

        return s1, s2, g1, g2, s1_p, s2_p, g1_p, g2_p