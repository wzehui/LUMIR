import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from collections import OrderedDict

class CoMMFusionModule(nn.Module):
    """
    CoMM-style fusion:
      - Each modality -> latent_converter -> (tokens)
      - [CLS] + concat(modality tokens) -> Transformer Encoder
      - Take [CLS] -> projection -> Z (normalized)
      - Random hard masking for augmentations (epoch-scheduled)
      - Independent sigmoid gates per modality (no softmax)
    """
    def __init__(self, embedding_dim, fusion_dim, projection_dim,
                 num_layers=4, num_heads=8, modality_token_per_modality=None,
                 dropout=0.1, fusion_schedule=None, gate_reg_lambda=1e-3, use_gating=False):
        super().__init__()

        assert modality_token_per_modality is not None, "modality_token_per_modality must be provided!"
        self.modality_token_per_modality = modality_token_per_modality
        self.modalities = list(modality_token_per_modality.keys())
        self.token_per_modality = modality_token_per_modality

        # === Fusion schedule (for random hard masking) ===
        assert fusion_schedule is not None, "fusion_schedule must be provided!"
        self.warmup_epochs = fusion_schedule['warmup_epochs']
        self.max_epochs = fusion_schedule['max_epochs']
        self.min_p_full = fusion_schedule.get('min_p_full', 0.1)
        self.current_epoch = 0
        self.use_gating = use_gating

        # __init__
        self.latent_converter = nn.ModuleDict({
            modality: nn.Linear(embedding_dim, token_per_modality * fusion_dim)
            for modality, token_per_modality in
            modality_token_per_modality.items()
        })
        self.token_ln = nn.ModuleDict({
            modality: nn.LayerNorm(fusion_dim)
            for modality in self.modalities
        })

        # === Independent modality gates (NO softmax) ===
        self.gate_layer = nn.Sequential(
            nn.Linear(len(self.modalities), len(self.modalities)),
            nn.Sigmoid()
        )
        self.gate_reg_lambda = gate_reg_lambda

        # === Learnable [CLS] token ===
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.cls_ln = nn.LayerNorm(fusion_dim)

        # === Transformer Fusion Block ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=num_heads,
            dim_feedforward=fusion_dim * 4,
            batch_first=True,
            dropout=dropout
        )
        self.fusion_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Projection Head ===
        # self.projection_head = nn.Sequential(OrderedDict([
        #     ("layer1", nn.Linear(fusion_dim, projection_dim)),
        #     ("relu1", nn.ReLU(inplace=True)),
        #     ("layer2", nn.Linear(projection_dim, projection_dim))
        # ]))
        self.projection_head = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(fusion_dim, fusion_dim)),
            ("bn1", nn.BatchNorm1d(fusion_dim)),
            ("act", nn.GELU()),
            ("fc2", nn.Linear(fusion_dim, projection_dim)),
        ]))

        print("=== CoMMFusionModule Initialized ===")
        print("Modalities:", self.modalities)
        print("Token per modality:", self.token_per_modality)
        print(f"Total tokens per sample: {self.total_tokens()}")

    def total_tokens(self):
        return sum(self.token_per_modality.values())

    def forward(self, modality_embeddings, mask_modalities):
        """
        modality_embeddings: list of [B, embedding_dim], order matches self.modalities
        mask_modalities: list of list[bool] with shape [B, n_mod] (hard keep/drop)
        """
        batch_size = modality_embeddings[0].shape[0]
        n_modalities = len(self.modalities)

        # === Independent gates per modality ===
        gate_input = torch.ones(batch_size, n_modalities, device=modality_embeddings[0].device)
        retain_weights = self.gate_layer(gate_input)  # (B, n_modalities)

        # === Convert embeddings to tokens ===
        token_seqs = []
        for i, embed in enumerate(modality_embeddings):
            modality = self.modalities[i]
            Tm = self.token_per_modality[modality]
            converted = self.latent_converter[modality](embed)  # [B, Tm*D]
            converted = converted.view(batch_size, Tm, -1)  # [B, Tm, D]
            converted = self.token_ln[modality](converted)  # token LN(D)
            token_seqs.append(converted)

        # --- 门控开关：关闭时直接全 1 ---
        if self.use_gating:
            gate_input = torch.ones(batch_size, n_modalities,
                                    device=modality_embeddings[0].device)
            retain_weights = self.gate_layer(gate_input)  # [B, M]
        else:
            retain_weights = torch.ones(batch_size, n_modalities,
                                        device=modality_embeddings[
                                            0].device)  # << 全 1

        # === Apply hard mask and gates ===
        masked_tokens = []
        for b in range(batch_size):
            tokens_b = []
            for m, modality in enumerate(self.modalities):
                if mask_modalities[b][m]:
                    weighted_tokens = token_seqs[m][b] * retain_weights[b, m]
                    tokens_b.append(weighted_tokens)
                else:
                    tokens_b.append(torch.zeros_like(token_seqs[m][b]))
            masked_tokens.append(torch.cat(tokens_b, dim=0))  # [sum Tm, fusion_dim]
        masked_tokens = torch.stack(masked_tokens, dim=0)  # [B, sum Tm, fusion_dim]

        # === Insert CLS ===
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens_with_cls = torch.cat([cls_tokens, masked_tokens], dim=1)  # [B, 1+sum Tm, fusion_dim]

        # === Transformer fusion ===
        fused = torch.utils.checkpoint.checkpoint(self.fusion_block, tokens_with_cls, use_reentrant=False)

        # Z = fused[:, 0, :]
        Z = self.cls_ln(fused[:, 0, :])
        Z_proj = self.projection_head(Z)
        # Z_proj = F.normalize(Z_proj, dim=-1)

        # === Gate regularization (optional) ===
        if self.use_gating:
            gate_loss = self.gate_reg_lambda * retain_weights.mean()
        else:
            gate_loss = Z_proj.new_zeros(())  # 标量 0

        return Z_proj, retain_weights, gate_loss

    def gen_random_mask(self, batch_size, min_keep=1, max_keep=None):
        """
        Hard random mask with epoch scheduling:
          - During warmup: keep all modalities (p_full = 1.0)
          - After: decay p_full linearly to min_p_full
        """
        n_modalities = len(self.modalities)
        if max_keep is None:
            max_keep = n_modalities

        if self.current_epoch < self.warmup_epochs:
            p_full = 1.0
        else:
            decay_ratio = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            p_full = max(self.min_p_full, 1.0 - decay_ratio)

        masks = []
        for _ in range(batch_size):
            if torch.rand(1).item() < p_full:
                mask = [True] * n_modalities
            else:
                keep_n = torch.randint(min_keep, max_keep + 1, (1,)).item()
                perm = torch.randperm(n_modalities)
                keep_indices = perm[:keep_n].tolist()
                mask = [m in keep_indices for m in range(n_modalities)]
            masks.append(mask)
        return masks

    def project_modality(self, modality_embeddings, modality_name):
        """
        Modality-specific representation (for modality-to-global CoMM terms):
          linear -> reshape to tokens -> token-wise LayerNorm -> avg-pool -> projection -> normalize
        """
        B = modality_embeddings.size(0)
        Tm = self.token_per_modality[modality_name]

        # [B, Tm*D]
        x = self.latent_converter[modality_name](modality_embeddings)
        # [B, Tm, D]
        x = x.view(B, Tm, -1)
        # token-wise LN over D
        x = self.token_ln[modality_name](x)

        # average over tokens of this modality
        pooled = x.mean(dim=1)  # [B, D]

        # projection head + L2 normalize
        projected = self.projection_head(pooled)  # [B, projection_dim]
        # projected = F.normalize(projected, dim=-1)
        return projected