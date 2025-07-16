import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

def traj_distance_weighted_mse(noise_pred, noise, gt_positions, pred_positions, gamma=0.9):
    """
    noise_pred: (B, T, D)
    noise: (B, T, D)
    gt_positions: (B, T, 2) 真实轨迹
    pred_positions: (B, T, 2) 预测轨迹
    """
    with torch.no_grad():
        # 每个时间步与目标终点的距离    
        target = gt_positions[:, -1:, :]  # (B, 1, 2)
        dist = torch.norm(gt_positions - target, dim=-1)  # (B, T)

        # 越靠近终点，权重越大（反距离）
        time_weights = torch.exp(-gamma * dist)  # (B, T)

        # reshape for broadcasting
        time_weights = time_weights.unsqueeze(-1)  # (B, T, 1)

    mse = (noise_pred - noise) ** 2  # (B, T, D)
    weighted_mse = mse * time_weights
    return weighted_mse.mean()

def time_weighted_mse(noise_pred, noise, is_pad=None, weight_type="linear"):
    B, T, D = noise_pred.shape
    if weight_type == "linear":
        time_weights = torch.linspace(1.0, 2.0, steps=T, device=noise_pred.device)
    elif weight_type == "exp":
        time_weights = torch.exp(torch.linspace(0.0, 1.0, steps=T, device=noise_pred.device))
    elif weight_type == "last_only":
        time_weights = torch.zeros(T, device=noise_pred.device)
        time_weights[-3:] = 1.0  # 只关注最后3步
    else:
        raise ValueError("Unknown weight_type")
    time_weights = time_weights.view(1, T, 1)  
    mse = (noise_pred - noise) ** 2 * time_weights
    if is_pad is not None:
        mse = mse * (~is_pad.unsqueeze(-1))
    return mse.mean()
    noise_pred = self.model_forward(noisy_actions, timesteps, global_cond=hidden_states, states=states)
    oise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
    loss = time_weighted_mse(noise_pred, noise, is_pad, weight_type="exp")