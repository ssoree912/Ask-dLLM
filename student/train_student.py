"""Block-conditioned scoring and one student optimization step."""
import torch
from torch import nn
from torch.nn import functional as F


class BlockScorer(nn.Module):
    """Independent layer scorers with a shared readout trunk across KV heads."""

    def __init__(self, layers, hidden_dim, proj_dim=256, mlp_dim=512, heads=1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "token": nn.Linear(hidden_dim, proj_dim),
                "block": nn.Linear(hidden_dim, proj_dim),
                "readout": nn.Sequential(nn.Linear(3 * proj_dim, mlp_dim),
                                         nn.GELU(), nn.Linear(mlp_dim, heads)),
            }) for _ in range(layers)
        ])

    def score(self, layer, hidden, candidates, block):
        """hidden: [sequence, width]; output: [1 or kv_heads, candidates]."""
        net = self.layers[layer]
        token = net["token"](hidden[candidates])
        context = net["block"](hidden[block].mean(0)).expand_as(token)
        features = torch.cat((token, context, token * context), dim=-1)
        return net["readout"](features).transpose(0, 1)


def ranking_loss(scores, targets, pairs=4096, lambda_list=1.0):
    """KL to normalized attention labels plus pairwise logistic ranking loss."""
    if scores.shape != targets.shape or targets.ndim != 2 or targets.shape[-1] == 0:
        raise ValueError("scores and targets must have matching [heads, candidates] shapes")
    if pairs < 1:
        raise ValueError("pairs must be positive")
    usable = torch.isfinite(targets).all(-1) & (targets.sum(-1) > 0)
    if not usable.any():
        return None
    scores, targets = scores[usable], targets[usable]
    distribution = targets / targets.sum(-1, keepdim=True)
    # Sum over candidates; mean over heads, so the loss scale is head-count invariant.
    kl = F.kl_div(scores.log_softmax(-1), distribution, reduction="none").sum(-1).mean()
    i = torch.randint(targets.shape[-1], (pairs,), device=scores.device)
    j = torch.randint(targets.shape[-1], (pairs,), device=scores.device)
    sign = (targets[:, i] - targets[:, j]).sign()
    unequal = sign != 0
    loss = lambda_list * kl
    if unequal.any():
        difference = scores[:, i] - scores[:, j]
        loss = loss + F.softplus(-sign[unequal] * difference[unequal]).mean()
    return loss


def train_step(student, record, replay, optimizer, pairs=4096, lambda_list=1.0):
    """Replay the frozen model at selection time, then update each layer scorer.

    replay(tokens, block_start) returns [sequence, width] hidden states per
    layer from the full-sequence step-1 forward, before revealing any tokens.
    The caller keeps the base model in eval mode and matches teacher decoding.
    """
    student.train()
    device = next(student.parameters()).device
    with torch.no_grad():
        hidden = replay(record["x_at_block_start"].to(device), record["block_start"])
    candidates = record["candidate_indices"].to(device)
    start = record["block_start"]
    block = torch.arange(start, start + record["block_length"], device=device)
    targets = record["label_final_rowmax"].to(device).float()
    losses = []
    for layer in range(len(student.layers)):
        scores = student.score(layer, hidden[layer].detach().float(), candidates, block)
        loss = ranking_loss(scores, targets[layer], pairs, lambda_list)
        if loss is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return sum(losses) / max(1, len(losses))
