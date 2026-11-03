"""Small CPU residual MLP. Geometry remains the baseline, including for observed joints."""
import torch
from torch import nn


class PoseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(147))
        self.register_buffer('scale', torch.ones(147))
        self.layers = nn.Sequential(nn.Linear(147, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 36))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x, base):
        residual = self.layers((x-self.mean)/self.scale).reshape(-1, 12, 3)
        return base + residual


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('schema') != 'pose3d_mlp/1.0':
        raise ValueError('Unsupported pose model checkpoint.')
    if checkpoint.get('feature_policy') not in (None,'geometry_candidate_v1'):
        raise ValueError('Unsupported pose feature policy.')
    if type(checkpoint.get('single_hip_anchor',False)) is not bool:
        raise ValueError('Invalid hip anchor policy.')
    if type(checkpoint.get('single_view_ray',False)) is not bool:
        raise ValueError('Invalid single-view ray policy.')
    if type(checkpoint.get('reject_conflicting_view',False)) is not bool:
        raise ValueError('Invalid conflicting-view policy.')
    if checkpoint.get('reject_conflicting_view') and (not checkpoint.get('single_view_ray') or checkpoint.get('feature_policy')!='geometry_candidate_v1'):
        raise ValueError('Conflicting-view policy requires the validated candidate and ray policies.')
    model = PoseMLP()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint
