import torch
import torch.nn as nn
import torch.nn.functional as F

class CoMMLoss(nn.Module):
    """
    CoMM loss aligned with Eq.(7):
      L = - I_NCE(Zv1, Zv2) [ + symmetric ]
          - sum_i 0.5 * [ I_NCE(Z_i^v1, Zv1/Zv2) ]   (modality-to-global)

    - Global term uses memory bank for strong negatives.
    - Modality terms default to in-batch negatives to save VRAM (toggle by flag).
    """
    def __init__(self, tau=0.07, memory_size=4096, embedding_dim=256,
                 use_symmetric_global=True, use_inbatch_for_modalities=True,
                 use_inbatch_for_global=True):
        super().__init__()
        self.tau = tau
        self.memory_size = memory_size
        self.use_symmetric_global = use_symmetric_global
        self.use_inbatch_for_modalities = use_inbatch_for_modalities
        self.use_inbatch_for_global = use_inbatch_for_global

        # fp16 bank to reduce VRAM
        bank = torch.randn(memory_size, embedding_dim)
        bank = F.normalize(bank, dim=1).float()
        self.register_buffer("memory_bank", bank)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_memory(self, features):
        feats = F.normalize(features, dim=1).to(self.memory_bank.dtype)
        bsz = feats.shape[0]
        ptr = int(self.ptr)
        end = ptr + bsz
        if end <= self.memory_size:
            self.memory_bank[ptr:end] = feats
        else:
            first = self.memory_size - ptr
            self.memory_bank[ptr:] = feats[:first]
            self.memory_bank[:end - self.memory_size] = feats[first:]
        self.ptr[0] = end % self.memory_size

    def _info_nce_with_memory(self, q, k):
        bank = self.memory_bank  # [M, D] fp16
        q = F.normalize(q, dim=1).to(bank.dtype)
        k = F.normalize(k, dim=1).to(bank.dtype)

        pos = (q * k).sum(dim=-1, keepdim=True) / self.tau                 # [B,1] fp16
        neg = (q @ bank.T) / self.tau                                      # [B,M] fp16
        logits = torch.cat([pos, neg], dim=1).float()                      # fp32 for CE
        labels = logits.new_zeros(q.size(0), dtype=torch.long)
        return F.cross_entropy(logits, labels)

    def _info_nce_inbatch(self, q, k):
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        logits = (q @ k.T) / self.tau
        labels = torch.arange(q.size(0), device=q.device)
        return F.cross_entropy(logits, labels)

    def forward(self, outputs):
        # Expect:
        # outputs = {
        #   'aug1_embed': [Z_v1, Z_meta_v1, Z_rev_v1, ...],
        #   'aug2_embed': [Z_v2, Z_meta_v2, Z_rev_v2, ...]
        # }
        aug1 = outputs['aug1_embed']
        aug2 = outputs['aug2_embed']
        Z_v1, Z_v2 = aug1[0], aug2[0]

        # # 1) Global-to-global
        # global_loss = self._info_nce_with_memory(Z_v1, Z_v2)
        # if self.use_symmetric_global:
        #     global_loss = global_loss + self._info_nce_with_memory(Z_v2, Z_v1)

        # === Global-to-global ===
        if self.use_inbatch_for_global:
            global_loss = self._info_nce_inbatch(Z_v1, Z_v2)
            if self.use_symmetric_global:
                global_loss = global_loss + self._info_nce_inbatch(Z_v2, Z_v1)
        else:
            global_loss = self._info_nce_with_memory(Z_v1, Z_v2)
            if self.use_symmetric_global:
                global_loss = global_loss + self._info_nce_with_memory(Z_v2, Z_v1)

        # 2) Modality-to-global
        n_mod = min(len(aug1), len(aug2)) - 1
        if n_mod > 0:
            m_losses = []
            for i in range(1, n_mod + 1):
                Zi_v1, Zi_v2 = aug1[i], aug2[i]
                if self.use_inbatch_for_modalities:
                    m_loss = 0.5 * (
                        self._info_nce_inbatch(Zi_v1, Z_v1) +
                        self._info_nce_inbatch(Zi_v1, Z_v2)
                    )
                else:
                    m_loss = 0.5 * (
                        self._info_nce_with_memory(Zi_v1, Z_v1) +
                        self._info_nce_with_memory(Zi_v1, Z_v2)
                    )
                m_losses.append(m_loss)
            modality_loss = torch.stack(m_losses).mean()
        else:
            modality_loss = Z_v1.new_zeros(())

        total_loss = global_loss + modality_loss

        return {
            "loss": total_loss,
            "global_loss": global_loss.detach(),
            "modality_loss": modality_loss.detach()
        }