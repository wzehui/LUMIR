import torch
import torch.nn.functional as F

def DropEntropyLoss(retain_matrix: torch.Tensor):
    """
    Entropy regularization for softmax-based modality retain weights.

    Args:
        retain_matrix (Tensor): [B, M] retain weights after softmax.

    Returns:
        Tensor: Negative entropy loss (to be minimized).
    """
    # Add epsilon to avoid log(0)
    eps = 1e-8
    retain_matrix = torch.clamp(retain_matrix, min=eps, max=1.0)
    entropy = -torch.sum(retain_matrix * torch.log(retain_matrix), dim=1)  # [B]
    return -entropy.mean()  # negative to encourage higher entropy

def InfoNCELoss(x_full, x_tilde, temperature=0.05):
    """
    Compute the InfoNCE-based compression loss to minimize I(x, x_tilde).

    Args:
        x_full (Tensor): Full modality concatenated embedding. Shape (batch_size, dim)
        x_tilde (Tensor): Drop-augmented compressed embedding. Shape (batch_size, dim)
        temperature (float): Temperature hyperparameter for scaling similarities.

    Returns:
        Tensor: Compression loss (scalar)
    """
    x_tilde = x_tilde.to(x_full.dtype)
    x_full = F.normalize(x_full, dim=-1)
    x_tilde = F.normalize(x_tilde, dim=-1)

    sim_matrix = torch.matmul(x_full, x_tilde.T)  # [B, B]
    logits = sim_matrix / temperature

    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def SeqAlignLoss(embeddings, sequence_pairs, temperature=0.05):
    """
    Sequence-aware contrastive loss (InfoNCE) - pair version:
    Each (anchor_idx, next_idx) pair is a positive pair.

    Args:
        embeddings (Tensor): [batch_size, dim], batch 中的 POI embedding
        sequence_pairs (List[List[int]]): list of [anchor_idx, next_idx]
        temperature (float): scaling temperature

    Returns:
        loss (Tensor)
    """
    embeddings = F.normalize(embeddings, dim=-1)

    total_loss = 0.0
    num_pairs = 0

    all_indices = set(range(embeddings.size(0)))

    for anchor_idx, pos_idx in sequence_pairs:
        # Positive sim
        pos_sim = torch.matmul(embeddings[anchor_idx], embeddings[pos_idx])

        # Negatives: all other embeddings not pos
        neg_indices = list(all_indices - {pos_idx})

        neg_sim_list = []
        for neg_idx in neg_indices:
            neg_sim = torch.matmul(embeddings[anchor_idx], embeddings[neg_idx])
            neg_sim_list.append(neg_sim)

        neg_sim_vec = torch.stack(neg_sim_list)

        # InfoNCE loss
        numerator = torch.exp(pos_sim / temperature)
        denominator = numerator + torch.sum(torch.exp(neg_sim_vec / temperature))
        loss_i = -torch.log(numerator / (denominator + 1e-8))

        total_loss += loss_i
        num_pairs += 1

    if num_pairs == 0:
        print("⚠️ Warning: No valid sequence pairs found in batch. Skipping loss.")
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)

    return total_loss / num_pairs


def IBLoss(x_full, x_tilde, sequences, lambda_compress=1.0, lambda_align=1.0, temperature=0.1):
    """
    Compute total information bottleneck loss: compression + sequence alignment.

    Args:
        x_full (Tensor): Original full embeddings. [B, D]
        x_tilde (Tensor): Compressed embeddings. [B, D]
        sequences (List[List[int]]): Index list of sequences.
        lambda_compress (float): Weight for compress loss.
        lambda_align (float): Weight for align loss.
        temperature (float): Temperature hyperparameter.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: total_loss, compress_loss, align_loss
    """
    compress_loss = InfoNCELoss(x_full, x_tilde, temperature)
    align_loss = SeqAlignLoss(x_tilde, sequences, temperature)
    total_loss = lambda_compress * compress_loss + lambda_align * align_loss
    return total_loss, compress_loss, align_loss