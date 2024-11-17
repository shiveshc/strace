import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import resnet50, resnet34, vit_b_16, vit_l_16, vit_b_32, vit_l_32

from typing import Union, List, Tuple


_activations = {
    'ReLU': nn.ReLU(),
    'GELU': nn.GELU(),
}

class Resnet(nn.Module):
    def __init__(self, in_ch: int, backbone: str, **kwargs) -> None:
        super().__init__()
        if backbone == 'resnet50':
            print('using resnet50 backbone')
            model = resnet50(weights=None)
            self.hidden_dim = 2048
        elif backbone == 'resnet34':
            print('using resnet34 backbone')
            model = resnet34(weights=None)
            self.hidden_dim = 512
        
        self.input_layer = nn.Conv2d(in_ch, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(model.children())[1:-2])

    def forward(self, x: Tensor) -> Tensor:
        '''
            Args:
             x - image shape (N, in_ch, 64, 64)

            Returns:
             features(x) shape (N, hidden_dim)
        '''
        x = self.input_layer(x)
        x = self.encoder(x)
        x = torch.mean(x, dim=(2, 3))
        return x


class ViT(nn.Module):
    def __init__(self, in_ch: int, backbone: str = 'vit_b_16', **kwargs) -> None:
        super().__init__()
        
        if backbone == 'vit_b_16':
            print('using vit_b_16 backbone')
            model = vit_b_16(image_size=64, weights=None)
            self.patch_size = 16
            self.hidden_dim = 768
            self.conv_proj = nn.Conv2d(in_ch, self.hidden_dim, kernel_size=(16, 16), stride=(16, 16))
            self.encoder = nn.Sequential(*list(list(model.children())[1].children())[:-1])
        elif backbone == 'vit_b_32':
            print('using vit_b_32 backbone')
            model = vit_b_32(image_size=64, weights=None)
            self.patch_size = 32
            self.hidden_dim = 768
            self.conv_proj = nn.Conv2d(in_ch, self.hidden_dim, kernel_size=(32, 32), stride=(32, 32))
            self.encoder = nn.Sequential(*list(list(model.children())[1].children())[:-1])
        elif backbone == 'vit_l_16':
            print('using vit_l_16 backbone')
            model = vit_l_16(image_size=64, weights=None)
            self.patch_size = 16
            self.hidden_dim = 1024
            self.conv_proj = nn.Conv2d(in_ch, self.hidden_dim, kernel_size=(16, 16), stride=(16, 16))
            self.encoder = nn.Sequential(*list(list(model.children())[1].children())[:-1])
        elif backbone == 'vit_l_32':
            print('using vit_l_32 backbone')
            model = vit_l_32(image_size=64, weights=None)
            self.patch_size = 32
            self.hidden_dim = 1024
            self.conv_proj = nn.Conv2d(in_ch, self.hidden_dim, kernel_size=(32, 32), stride=(32, 32))
            self.encoder = nn.Sequential(*list(list(model.children())[1].children())[:-1])

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

    def _process_input(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        # torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        # torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        '''
            Args:
             x - image shape (N, in_ch, 64, 64)

            Returns:
             features(x) shape (N, hidden_dim)
        '''
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        return x


class MLP(nn.Module):
    def __init__(self, in_ch: int, hidden_layer_dims: List[int], **kwargs) -> None:
        super().__init__()

        if 'act' in kwargs:
            self.act = _activations[kwargs['act']]
        else:
            self.act = _activations['ReLU']
        
        layers = []
        layers.append(nn.Linear(in_ch, hidden_layer_dims[0]))
        layers.append(self.act)
        for n in range(1, len(hidden_layer_dims) - 1):
            layers.append(nn.Linear(hidden_layer_dims[n - 1], hidden_layer_dims[n]))
            layers.append(self.act)
        layers.append(nn.Linear(hidden_layer_dims[-2], hidden_layer_dims[-1]))
        self.encoder = nn.Sequential(*layers)
        self.hidden_dim = hidden_layer_dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        '''
            Args:
             x - shape (N, 35) 

            Returns:
             features(x) shape (N, hidden_dim)
        '''
        x = self.encoder(x)
        return x 


class Encoder(nn.Module):
    def __init__(self, in_ch: int, backbone: str, **kwargs) -> None:
        super().__init__()
        if 'resnet' in backbone:
            self.encoder = Resnet(in_ch, backbone)
            self.out_dim = self.encoder.hidden_dim
        elif 'vit' in backbone:
            self.encoder = ViT(in_ch, backbone)
            self.out_dim = self.encoder.hidden_dim
        elif 'mlp' in backbone:
            assert 'hidden_layer_dims' in kwargs
            self.encoder = MLP(in_ch, kwargs['hidden_layer_dims'])
            self.out_dim = self.encoder.hidden_dim
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        '''
            Args:
             x - image shape (N, in_ch, 64, 64)

            Returns:
             features(x) shape (N, out_dim)
        '''
        x = self.encoder(x)
        return x


class Projector(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 **kwargs) -> None:
        super().__init__()

        if 'act' in kwargs:
            self.act = _activations[kwargs['act']]
        else:
            self.act = _activations['ReLU']

        if 'hidden_layer_dims' in kwargs:
            hidden_layer_dims = kwargs['hidden_layer_dims']
        else:
            hidden_layer_dims = [in_dim]
            
        layers = []
        layers.append(nn.Linear(in_dim, hidden_layer_dims[0]))
        layers.append(self.act)
        for n in range(1, len(hidden_layer_dims)):
            layers.append(nn.Linear(hidden_layer_dims[n - 1], hidden_layer_dims[n]))
            layers.append(self.act)
        layers.append(nn.Linear(hidden_layer_dims[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

        if 'out_layer_dims' in kwargs and type(kwargs['out_layer_dims']) != type(None):
            self.out_layer = nn.Linear(out_dim, kwargs['out_layer_dims'])

    def forward(self, x: Tensor) -> Tensor:
        '''
            Args:
             x - image shape (N, in_dim)

            Returns:
             features(x) shape (N, out_dim) normalized to unit norm if
             out_layer_dims is None or (N, out_layer_dims)
        '''
        x = self.mlp(x)
        x = x/torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)

        if hasattr(self, 'out_layer'):
            x = getattr(self, 'out_layer')(x)
        return x

# Old Projector coded may be needed to run inference with
# old models
# class Projector(nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  out_dim: int,
#                  **kwargs) -> None:
#         super().__init__()
#         self.linear1 = nn.Linear(in_dim, in_dim)
#         self.linear2 = nn.Linear(in_dim, out_dim)
#         self.act = nn.ReLU()

#     def forward(self, x: Tensor) -> Tensor:
#         '''
#             Args:
#              x - image shape (N, in_dim)

#             Returns:
#              features(x) shape (N, out_dim) normalized to unit norm
#         '''
#         x = self.act(self.linear1(x))
#         x = self.linear2(x)
#         x = x/torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
#         return x