# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from typing import Tuple

import timm
import numpy as np
import logging

import math
from typing import Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from timm.models.vision_transformer import Mlp, use_fused_attn
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoModelForCausalLM

_logger = logging.getLogger(__name__)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale

            attn_scores = torch.matmul(q, k.transpose(-2, -1))

            # Add attention mask if provided
            if attn_mask is not None:
                attn_scores += attn_mask

            # Apply softmax to get attention weights (softmax is applied along the last dimension)
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Dropout on attention weights (if dropout is used)
            attn_weights = self.attn_drop(attn_weights)

            # Apply attention weights to value tensor (V)
            x = torch.matmul(attn_weights, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


logger = logging.getLogger(__name__)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.bfloat16) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(dtype=torch.bfloat16)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core ScaleDP Model                                #
#################################################################################

class ScaleDPBlock(nn.Module):
    """
    A ScaleDP block with adaptive layer norm zero (adaLN-Zero) conScaleDPioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask) # norm, scale&shift, attn, scale,
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of ScaleDP.
    """

    def __init__(self, hidden_size, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class AnchorClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_anchors, pooling='mean'):
        """
        :param feature_dim: 单步轨迹的特征维度 D
        :param hidden_dim: hidden_states 的维度 H
        :param num_anchors: anchor 数量
        :param pooling: pooling 类型，可选 'mean', 'max'
        """
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_anchors)  # 直接输出每个 anchor 的分数
        )
    def initialize_weights(self):
        """按照 ScaleDP 风格初始化 fc 的 Linear 层"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, hidden_states, anchors):
        """
        :param hidden_states: [B, H]
        :param anchors: [N, T, D] 轨迹库
        :return: probs [B, N] 概率分布
        """
        B, H = hidden_states.shape
        N, T, D = anchors.shape

        # 对 anchors 做时间维度聚合
        if self.pooling == 'mean':
            anchor_embed = anchors.mean(dim=1)  # [N, D]
        elif self.pooling == 'max':
            anchor_embed, _ = anchors.max(dim=1)  # [N, D]
        else:
            raise NotImplementedError(f"Pooling {self.pooling} not supported")

        # 扩展 hidden_states 与 anchor_embed 方便拼接
        hidden_exp = hidden_states.unsqueeze(1).expand(B, N, H)   # [B, N, H]
        anchor_exp = anchor_embed.unsqueeze(0).expand(B, N, D)    # [B, N, D]

        x = torch.cat([hidden_exp, anchor_exp], dim=-1)  # [B, N, H+D]

        # flatten 到 B*N, 输入 fc
        x_flat = x.view(B*N, H+D)  # [B*N, H+D]
        logits_flat = self.fc(x_flat)  # [B*N, N]

        # reshape 回 B, N, N -> 取对角线（每个 hidden 对应 N 个 anchor）
        logits = logits_flat.view(B, N, N).diagonal(dim1=1, dim2=2)  # [B, N]
        probs = F.softmax(logits, dim=-1)  # [B, N]

        return probs

from .configuration_scaledp import ScaleDPPolicyConfig
class ScaleDP(PreTrainedModel):
    """
    Diffusion models with a Transformer backbone.
    """
    config_class = ScaleDPPolicyConfig
    def __init__(
            self,
            config: ScaleDPPolicyConfig,
    ):
        super().__init__(config)
        self.train_steps = 0
        # compute number of tokens for main trunk and conScaleDPion encoder
        if config.n_obs_steps is None:
            config.n_obs_steps = config.prediction_horizon
        T = config.prediction_horizon
        T_cond = 1
        if not config.time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = config.cond_dim > 0
        if obs_as_cond:
            assert config.time_as_cond
            T_cond += config.n_obs_steps

        self.is_tinyvla = config.is_tinyvla
        if config.is_tinyvla:
            self.global_1d_pool = nn.AdaptiveAvgPool1d(1)
            self.norm_after_pool = nn.LayerNorm(config.cond_dim)
        # self.combine = nn.Linear(cond_dim+state_dim, cond_dim)
        self.combine = nn.Sequential(
            nn.Linear(config.cond_dim+config.state_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.cond_dim)
        )
        self.learn_sigma = config.learn_sigma
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim * 2 if config.learn_sigma else config.output_dim
        self.num_heads = config.num_heads

        self.x_embedder = nn.Linear(config.input_dim, config.n_emb)
        self.t_embedder = TimestepEmbedder(config.n_emb)
        self.cond_obs_emb = None
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(config.cond_dim, config.n_emb)

        self.num_anchors = 20
        # self.anchor_classifier = nn.Linear(config.n_emb, self.num_anchors)
        self.anchor_classifier = AnchorClassifier(feature_dim=config.input_dim, hidden_dim=config.cond_dim, num_anchors=self.num_anchors)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, config.prediction_horizon, config.n_emb))

        self.blocks = nn.ModuleList([
            ScaleDPBlock(config.n_emb, config.num_heads, mlp_ratio=config.mlp_ratio) for _ in range(config.depth)
        ])
        self.final_layer = FinalLayer(config.n_emb, output_dim=config.output_dim)

        # self.initialize_weights()
        # constants
        self.T = T
        self.T_cond = T_cond
        self.prediction_horizon = config.prediction_horizon
        self.time_as_cond = config.time_as_cond
        self.action_dim = config.output_dim
        self.obs_as_cond = obs_as_cond
        logger.info(
            "number of parameters in ScaleDP: %e", sum(p.numel() for p in self.parameters())
        )

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        self.num_inference_timesteps = config.num_inference_timesteps
        # self.proj_to_action = nn.Identity()
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps, # 100
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )
        self.num_queries = config.num_queries #16
        self.noise_samples = config.noise_samples # 1
        # self.num_inference_timesteps = config.num_inference_timesteps # 100

        self.anchors = None
        import h5py
        h5_path = "data/astar_paths/split/episode_0_proc_000000.hdf5"
        with h5py.File(h5_path, "r") as f:
            out = {}
            for key in [
                # "cands_xyz_resampled",
                # "cands_cluster_labels",
                "cands_cluster_centroids",
                # "cands_cluster_counts",
            ]:
                assert key in f
                out[key] = f[key][()]
            self.anchors_np = out["cands_cluster_centroids"]
        anchors_t = torch.from_numpy(self.anchors_np)
        self.register_buffer("anchors_tensor", anchors_t, persistent=False)



     


    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.cond_obs_emb.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.cond_obs_emb.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ScaleDP blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.anchor_classifier is not None:
            self.anchor_classifier.initialize_weights()

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the models into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, Attention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self,
                             learning_rate: float = 1e-4,
                             weight_decay: float = 1e-3,
                             betas: Tuple[float, float] = (0.9, 0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def get_nearest_anchor(self, actions: torch.Tensor, anchors: torch.Tensor):
        """
        轨迹级别的最近 anchor 查找 (TrackVLA 风格)，支持 batch

        Parameters
        ----------
        actions : torch.Tensor
            shape (B, T, D)，batch 中的轨迹
        anchors : torch.Tensor
            shape (N, T, D)，anchor 库

        Returns
        -------
        nearest_idx : torch.LongTensor
            shape (B,)，每个样本最近 anchor 的索引
        nearest_anchor : torch.Tensor
            shape (B, T, D)，每个样本对应的最近 anchor 轨迹
        """
        assert actions.ndim == 3, f"actions 应该是 (B, T, D)，得到 {actions.shape}"
        assert anchors.ndim == 3, f"anchors 应该是 (N, T, D)，得到 {anchors.shape}"
        assert anchors.shape[1:] == actions.shape[1:], \
            f"anchors 长度和 actions 必须一致: {anchors.shape[1:]} vs {actions.shape[1:]}"

        B, T, D = actions.shape
        N = anchors.shape[0]

        # 扩展维度计算 L2 距离
        # actions: (B, 1, T, D), anchors: (1, N, T, D)
        diff = actions.unsqueeze(1) - anchors.unsqueeze(0)  # (B, N, T, D)
        errors = (diff ** 2).mean(dim=(2, 3))  # (B, N)

        # 找到每个样本最近的 anchor
        nearest_idx = torch.argmin(errors, dim=1)  # (B,)

        # 根据索引取 anchor
        nearest_anchor = anchors[nearest_idx]  # (B, T, D)

        return nearest_idx, nearest_anchor

    def forward(self, actions, hidden_states, states, is_pad):
        """
        Forward pass for the anchor-based diffusion head (TrackVLA style).
        
        :param actions: target actions, shape [B, Ta, D]
        :param hidden_states: hidden states from encoder, shape [B, Tokens, D]
        :param states: robot states, shape [B, D]
        :param is_pad: padding mask, shape [B, Ta]
        :return: total loss (mse + bce)
        """
        if actions is not None:  # training time
            B = actions.size(0)
            actions = actions[:, :self.num_queries]
            is_pad = is_pad[:, :self.num_queries]
            num_noise_samples = self.noise_samples

            # === 1. 找到最近 anchor 并算 residual ===
            with torch.no_grad():
                ref_dtype = self.anchor_classifier.fc[0].weight.dtype
                ref_device = actions.device
                anchors = self.anchors_tensor.to(device=ref_device, dtype=ref_dtype)
                indices , nearest_anchor= self.get_nearest_anchor(actions, anchors)
                # one-hot label
                anchor_labels = F.one_hot(indices, num_classes=self.num_anchors).float()  # [B, Ta, Na]
            residual = actions - nearest_anchor                        # 学 residual 而不是 action 本身

            # === 2. 采样噪声（和标准 diffusion 一样） ===
            noise = torch.randn([num_noise_samples] + list(residual.shape),
                                device=residual.device, dtype=residual.dtype)  # [num_noise, B, Ta, D]

            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=residual.device
            ).long()

            timesteps, noise = timesteps.to(residual.device), noise.to(residual.device)

            # === 3. forward diffusion：residual 加噪 ===
            noisy_residual = torch.cat([
                self.noise_scheduler.add_noise(residual, noise[i], timesteps)
                for i in range(len(noise))
            ], dim=0)  # [num_noise * B, Ta, D]

            noisy_residual = noisy_residual.to(dtype=residual.dtype)

            # === 4. repeat conds ===
            hidden_states = hidden_states.repeat(num_noise_samples, 1, 1)
            timesteps = timesteps.repeat(num_noise_samples)
            is_pad = is_pad.repeat(num_noise_samples, 1)
            states = states.repeat(num_noise_samples, 1)

            # === 5. 模型预测噪声 ===
            noise_pred = self.model_forward(noisy_residual, timesteps,
                                            global_cond=hidden_states, states=states)

            # === 6. Diffusion MSE loss ===
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            mse_loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
            mse_loss = (mse_loss * ~is_pad.unsqueeze(-1)).mean()

            # === 7. Anchor BCE loss ===
            bce_loss = 0.0
            if anchor_labels is not None:        # :param anchor_labels: one-hot anchor index labels, shape [B, num_anchors]
                ref_dtype = self.anchor_classifier.fc[0].weight.dtype
                ref_device = hidden_states.device
                anchors = self.anchors_tensor.to(device=ref_device, dtype=ref_dtype)
                anchor_probs = self.anchor_classifier(hidden_states.mean(dim=1), anchors)  # [B*num_noise, num_anchors]
                # 注意 anchor_labels 只重复 B 次，需要扩展到 [B*num_noise, num_anchors]
                anchor_labels = anchor_labels.repeat(num_noise_samples, 1)
                bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    anchor_probs, anchor_labels.float()
                )

            # === 8. 总 loss ===
            self.beta = 1
            loss = mse_loss + self.beta * bce_loss
            # loss_dict['loss'] = loss
            # return {'loss': loss}
            output = {'loss': loss}
            output["reconstructed_action"] = None
            output["noise_pred"] = noise_pred
            output["steps"] = self.train_steps
            if self.train_steps % 250 ==0:
                with torch.no_grad():
                    naction = self.inference(hidden_states=hidden_states, states=states)
                    output["reconstructed_action"] = naction  # 可视化用
            self.train_steps+=1
            return output
        else:  # inference time

            naction = self.inference(hidden_states=hidden_states, states=states)
            return naction

    def model_forward(self, x, t, global_cond, states):
        """
        Forward pass of ScaleDP.
        x: (N, T, input_dim) noisy actions
        t: (N,) tensor of diffusion timesteps
        global_cond: (N, n_obs_steps, D) tensor of conScaleDPions: image embeddings
        """
        if self.is_tinyvla:
            global_cond = self.global_1d_pool(global_cond.permute(0, 2, 1)).squeeze(-1)
            global_cond = self.norm_after_pool(global_cond)
        else:
            global_cond = global_cond.squeeze(1)
        global_cond = torch.cat([global_cond, states], dim=-1) if states is not None else global_cond

        # if states is not None:   #wzjjj
        #     if states.dim() == 2:
        #         states = states.unsqueeze(1).expand(-1, global_cond.size(1), -1)  # [B, T, D2]
        #     global_cond = torch.cat([global_cond, states], dim=-1)


        global_cond = self.combine(global_cond)

        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.long, device=x.device)
        elif torch.is_tensor(t) and len(t.shape) == 0:
            t = t[None].to(x.device)
        t = t.expand(t.shape[0])

        x = self.x_embedder(x) + self.pos_embed.to(device=x.device, dtype=x.dtype)  # (N, T, D), where T = prediction_horizon
        t = self.t_embedder(t)  # (N, D)
        if self.obs_as_cond:
            global_cond = self.cond_obs_emb(global_cond)  # (N, D)
        # c = t + global_cond.sum(dim=1)  # (N, D)
        c = t + global_cond  # (N, D)
        for block in self.blocks:
            # x = block(x, c, attn_mask=self.mask)  # (N, T, D)
            x = block(x, c, attn_mask=None)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, output_dim)
        return x

    def inference(self, hidden_states, states):
        B = 1
        Tp = self.num_queries
        action_dim = self.action_dim
        ref_dtype = self.anchor_classifier.fc[0].weight.dtype
        ref_device = hidden_states.device
        anchors = self.anchors_tensor.to(device=ref_device, dtype=ref_dtype)
        anchor_probs = self.anchor_classifier(hidden_states.mean(dim=1), anchors) 
        anchor_idx = torch.argmax(anchor_probs, dim=-1) 
        selected_anchor = anchors[anchor_idx]

        # initialize action from Guassian noise
        noisy_residual = torch.randn((B, Tp, action_dim)).cuda()

        n_residual = noisy_residual.to(dtype=hidden_states.dtype)
        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_timesteps)

        # 原始 timesteps，比如：tensor([90, 80, 70, ..., 20, 10, 0])
        timesteps = self.noise_scheduler.timesteps.tolist()
        #----------------------------------------------------------------------------------------------------------
        # 找到10的位置
        if 10 in timesteps and 0 in timesteps:
            idx_10 = timesteps.index(10)
            idx_0 = timesteps.index(0)

            # 替换 [10, 0] 为细化版本
            refined = list(range(10, -1, -1))  # [10, 9, ..., 0]
            timesteps = timesteps[:idx_10] + refined + timesteps[idx_0+1:]

            # 更新 scheduler
            self.noise_scheduler.timesteps = torch.tensor(timesteps, device=hidden_states.device)
        #----------------------------------------------------------------------------------------------------------

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = self.model_forward(n_residual, k, global_cond=hidden_states, states=states)

            # inverse diffusion step (remove noise)
            step_result = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=n_residual
            )                        
            if not torch.all(torch.isfinite(step_result.prev_sample)):
                print(f"NaN in n_residual at timestep {k}")
                break
            n_residual = step_result.prev_sample

        naction = selected_anchor + noisy_residual
        return naction
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   ScaleDP Configs                                  #
#################################################################################

def ScaleDP_H(**kwargs):
    return ScaleDP(depth=32, n_emb=1280, num_heads=16, **kwargs)

def ScaleDP_L(**kwargs):
    return ScaleDP(depth=24, n_emb=1024, num_heads=16, **kwargs)




ScaleDP_models = {
    'ScaleDP-L': ScaleDP_L, 'ScaleDP-H': ScaleDP_H,
}

AutoModel.register(ScaleDPPolicyConfig, ScaleDP)
