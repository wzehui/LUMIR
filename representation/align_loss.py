import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Dict, List, Tuple, Set, Iterable, Optional


class MemoryBankAlignLoss(nn.Module):
    """
    Session alignment loss (InfoNCE) with a CPU-side memory bank.

    Overview
    --------
    - Each anchor is a POI embedding (from the full-modality view).
    - Positives are selected from the same sessions where the anchor appears,
      according to `pos_strategy`:
        * "window": neighbors within ±K steps (K = pos_window, default 1)
        * "next":   only the immediate next item (index+1)
    - Negatives are sampled from the memory bank, excluding:
        * the anchor itself,
        * all positives,
        * all POIs that appear in the same sessions as the anchor (to avoid
          treating same-trajectory items as negatives).
    - With multiple positives, we aggregate their logits via log-sum-exp so the
      loss still has a single "positive" position for cross-entropy.
    - If an anchor has no valid positives or negatives in the bank, we skip it.
    - Optionally update the memory bank with the anchor embedding after each step.

    Memory Bank
    -----------
    - Implemented as a CPU dict: {poi_id: 1D Tensor[dim]}.
    - On forward, tensors are moved to the current device/dtype as needed.
    - New/updated entries can be L2-normalized before storage.

    Args
    ----
    memory_bank: Dict[int, Tensor]
        CPU dictionary mapping POI id to its embedding vector.
    session_dict: Dict[session_id, List[int]]
        Session → ordered list of POI ids in that session.
    poi_to_sessions: Dict[int, List[Tuple[session_id, index_in_session]]]
        For each POI id, all (session_id, index) occurrences.
    temperature: float
        Temperature for InfoNCE.
    neg_sample_size: int
        Maximum number of negatives per anchor.
    pos_window: int
        Window size K for the "window" strategy (use ±K neighbors, K >= 1).
        Ignored if pos_strategy="next".
    pos_strategy: str
        "window" or "next" (only the immediate next POI).
    normalize_store: bool
        If True, L2-normalize embeddings before storing to the memory bank.
    update_bank: bool
        If True, update the memory bank with the current anchor embedding.
    """

    def __init__(
        self,
        memory_bank: Dict[int, torch.Tensor],
        session_dict: Dict[int, List[int]],
        poi_to_sessions: Dict[int, List[Tuple[int, int]]],
        temperature: float = 0.05,
        neg_sample_size: int = 1000,
        pos_window: int = 1,
        pos_strategy: str = "window",
        normalize_store: bool = True,
        update_bank: bool = True,
    ):
        super().__init__()
        self.memory_bank = memory_bank
        self.session_dict = session_dict
        self.poi_to_sessions = poi_to_sessions
        self.temperature = float(temperature)
        self.neg_sample_size = int(neg_sample_size)
        self.pos_window = int(pos_window)
        self.pos_strategy = str(pos_strategy).lower()
        self.normalize_store = bool(normalize_store)
        self.update_bank = bool(update_bank)

        # Basic parameter checks
        assert self.pos_strategy in ("window", "next"), "pos_strategy must be 'window' or 'next'."
        if self.pos_strategy == "window":
            assert self.pos_window >= 1, "pos_window must be >= 1 when pos_strategy='window'."

    def _collect_positives_and_excludes(self, anchor_id: int) -> Tuple[Set[int], Set[int]]:
        """
        Collect positive POI ids and the set of ids to exclude from negatives.

        Returns
        -------
        positives: Set[int]
            The set of positive POI ids (deduplicated).
        exclude: Set[int]
            The set of ids to exclude from negative sampling, consisting of:
            all positives, all POIs in the same sessions as anchor, and the anchor itself.
        """
        positives: Set[int] = set()
        same_session_pois: Set[int] = set()

        # Iterate through all occurrences (sid, idx) of the anchor
        for sid, idx in self.poi_to_sessions.get(anchor_id, []):
            seq = self.session_dict.get(sid, [])
            if not seq:
                continue

            # Exclude all items from the same session from negatives
            same_session_pois.update(seq)

            if self.pos_strategy == "next":
                # Only the immediate next item (idx+1) is considered positive
                j = idx + 1
                if 0 <= j < len(seq):
                    cand = seq[j]
                    if cand != anchor_id:
                        positives.add(cand)
            else:
                # Window strategy: use neighbors within ±pos_window (excluding itself)
                for d in range(-self.pos_window, self.pos_window + 1):
                    if d == 0:
                        continue
                    j = idx + d
                    if 0 <= j < len(seq):
                        cand = seq[j]
                        if cand != anchor_id:
                            positives.add(cand)

        # Exclusion set for negatives = positives ∪ all same-session items ∪ {anchor}
        exclude = same_session_pois.union(positives)
        exclude.add(anchor_id)
        return positives, exclude

    def _gather_from_bank(
        self,
        ids: Iterable[int],
        device: torch.device,
        dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """
        Fetch vectors for the given ids from the CPU memory bank and move them to (device, dtype).
        Returns None if none of the ids is found.
        """
        vecs = []
        for pid in ids:
            v = self.memory_bank.get(pid, None)
            if v is None:
                continue
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            v = v.to(device=device, dtype=dtype, non_blocking=True)
            vecs.append(v)
        if len(vecs) == 0:
            return None
        return torch.stack(vecs, dim=0)

    def _sample_neg_ids(self, exclude: Set[int]) -> List[int]:
        """
        Sample up to `neg_sample_size` negative ids uniformly from the memory bank,
        excluding the given set.
        """
        all_ids = list(self.memory_bank.keys())
        candidates = [pid for pid in all_ids if pid not in exclude]
        if len(candidates) == 0:
            return []
        if len(candidates) <= self.neg_sample_size:
            return candidates
        return random.sample(candidates, self.neg_sample_size)

    def forward(self, anchor_ids: List[int], anchor_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute the alignment loss over a batch of anchors.

        Args
        ----
        anchor_ids: List[int]
            The POI ids for the anchors (length = B).
        anchor_embeddings: Tensor[B, dim]
            The corresponding anchor embeddings (full-modality view).

        Returns
        -------
        loss: Tensor (scalar)
            Mean loss over valid anchors. If no valid anchors are found, returns 0 (graph-friendly).
        """
        device = anchor_embeddings.device
        dtype = anchor_embeddings.dtype

        losses: List[torch.Tensor] = []

        for i, anchor_id in enumerate(anchor_ids):
            anchor_emb = anchor_embeddings[i]  # [dim]

            # 1) Gather positive ids and the exclusion set for negatives
            positives_ids, exclude_ids = self._collect_positives_and_excludes(anchor_id)
            if len(positives_ids) == 0:
                # No positives available → skip this anchor
                continue

            # 2) Pull positive vectors from memory bank
            positives = self._gather_from_bank(positives_ids, device, dtype)
            if positives is None or positives.numel() == 0:
                # Positives exist by id but not yet in bank (cold-start) → skip
                continue

            # 3) Sample negative ids and pull vectors from memory bank
            neg_ids = self._sample_neg_ids(exclude_ids)
            if len(neg_ids) == 0:
                # No negatives available → skip
                continue
            negatives = self._gather_from_bank(neg_ids, device, dtype)
            if negatives is None or negatives.numel() == 0:
                continue

            # 4) Normalize all embeddings
            a = F.normalize(anchor_emb.unsqueeze(0), dim=-1)  # [1, dim]
            P = F.normalize(positives, dim=-1)                # [n_pos, dim]
            N = F.normalize(negatives, dim=-1)                # [n_neg, dim]

            # 5) Compute logits:
            #    - Multiple positives: aggregate via log-sum-exp
            pos_logits = (a @ P.T) / self.temperature         # [1, n_pos]
            neg_logits = (a @ N.T) / self.temperature         # [1, n_neg]
            pos_agg = torch.logsumexp(pos_logits, dim=1, keepdim=True)  # [1, 1]

            logits = torch.cat([pos_agg, neg_logits], dim=1)  # [1, 1 + n_neg]
            labels = torch.zeros(1, dtype=torch.long, device=device)  # positive index = 0

            # 6) Cross-entropy loss
            loss = F.cross_entropy(logits, labels)
            losses.append(loss)

            # 7) (Optional) update memory bank with the anchor embedding
            if self.update_bank:
                v = a.squeeze(0).detach().to("cpu")
                if self.normalize_store:
                    v = F.normalize(v, dim=0)
                self.memory_bank[anchor_id] = v

        # If no valid anchors in this batch, return a zero scalar that still participates in the graph
        if len(losses) == 0:
            return anchor_embeddings.sum() * 0.0

        return torch.stack(losses).mean()