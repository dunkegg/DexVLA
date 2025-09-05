import torch
import torch.nn.functional as F

def multik_ce_mse_loss(logits, traj_pred, tau_gt, feasible_idx, lambda_reg=1.0, lambda_cls=100.0):
    """
    logits:    [B,K]
    traj_pred: [B,K,10,3]
    tau_gt:    [B,10,3]
    feasible_idx: [B] (long)
    """
    B, K, T, D = traj_pred.shape

    # 分类：多类互斥
    cls_loss = F.cross_entropy(logits, feasible_idx)  # [B]

    # 回归：只对正类监督
    sel = feasible_idx.view(B,1,1,1).expand(B,1,T,D)
    pos_traj = traj_pred.gather(1, sel).squeeze(1)    # [B,10,3]
    reg_loss = F.mse_loss(pos_traj, tau_gt)

    loss = lambda_reg * reg_loss + lambda_cls * cls_loss
    logs = {"reg": reg_loss.item(), "cls": cls_loss.item(), "total": loss.item()}
    return loss, logs
