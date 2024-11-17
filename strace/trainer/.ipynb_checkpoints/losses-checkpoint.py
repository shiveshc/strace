import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional


def simclr_loss(z: Tensor,
            z_prime: Tensor,
            tau: Optional[float] = 1,
            **kwargs) -> Tensor:
    '''
        Args:
         z - image features shape (N, D). 2-norm should be 1
         z_prime - pos pairs for z shape (N, D). 2 norm should be 1
    
        Returns:
         loss
    '''
    N = z.shape[0]
    tmp = torch.concatenate([z, z_prime], axis=0)
    y = torch.matmul(tmp, tmp.T)
    y = y - torch.diag(100*torch.ones(2*N)).to(y.device)
    y = y/tau
    label = torch.tensor([i for i in range(N, 2*N)] + [i for i in range(N)]).to(y.device)
    loss = F.cross_entropy(y, label)
    return loss


def dino_loss(s: Tensor,
              t: Tensor,
              tps: float,
              tpt: float,
              center: Tensor) -> Tensor:

    t = t.detach()
    s = nn.LogSoftmax(dim=1)(s/tps)
    t = nn.Softmax(dim=1)((t - center)/tpt)
    loss = torch.mean(torch.sum(torch.mul(-t, s), dim=1))
    return loss