"""
Ledoit-Wolf shrinkage and correlation-aware channel permutation for GReS.

Provides:
- ledoit_wolf_shrinkage_intensity: Compute optimal shrinkage toward diagonal target.
- apply_shrinkage_to_block: Apply pre-computed shrinkage to a block Gram.
- correlation_aware_permutation: Find channel permutation minimizing block correlation.
- FullGramWrapper: Layer wrapper accumulating full Gram matrix with shrinkage support.

Shrinkage formula:
    G_hat = (1 - rho) * G + rho * diag(G)
where rho is the closed-form Ledoit-Wolf estimator.
"""

import torch
import torch.nn as nn


def ledoit_wolf_shrinkage_intensity(G, T):
    """
    Compute the Ledoit-Wolf shrinkage intensity for shrinking G toward diag(G).

    The LW estimator minimizes E[||G_hat - G_pop||_F^2] where G_pop is the
    population Gram. The closed form for the diagonal target is:
    rho = sum of off-diagonal sampling variances / sum of off-diagonal energies.

    Args:
        G: (d, d) empirical Gram matrix (unnormalized sum of outer products).
        T: Number of tokens used to estimate G.

    Returns:
        rho: Scalar in [0, 1], the optimal shrinkage intensity.
        G_hat: (d, d) the shrunk Gram matrix.
    """
    d = G.shape[0]
    if d <= 1:
        return 0.0, G.clone()

    diag_G = torch.diag(G)

    # Off-diagonal energy
    off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=G.device)
    off_diag_energy = (G[off_diag_mask] ** 2).sum()

    if off_diag_energy < 1e-12:
        return 1.0, G.clone()

    # Approximate off-diagonal sampling variance:
    # var(G_jk) ~ (1/T) * (G_jj * G_kk + G_jk^2)
    diag_outer = diag_G.unsqueeze(0) * diag_G.unsqueeze(1)
    numerator = ((diag_outer[off_diag_mask] + G[off_diag_mask] ** 2) / T).sum()

    rho = (numerator / off_diag_energy).item()
    rho = max(0.0, min(1.0, rho))

    G_hat = (1.0 - rho) * G + rho * torch.diag(diag_G)

    return rho, G_hat


def apply_shrinkage_to_block(G_block, rho):
    """
    Apply pre-computed shrinkage intensity to a block Gram.

    Args:
        G_block: (M, M) block Gram matrix.
        rho: Shrinkage intensity in [0, 1].

    Returns:
        G_shrunk: (M, M) shrunk block Gram.
    """
    diag_G = torch.diag(torch.diag(G_block))
    return (1.0 - rho) * G_block + rho * diag_G


def correlation_aware_permutation(G, W, M, N_keep):
    """
    Find a channel permutation that reduces total N:M reconstruction error
    by grouping channels to minimize within-block correlation.

    For large dimensions (d > 2000), uses spectral interleaving: sort channels
    by the dominant eigenvector of the correlation matrix, then interleave
    into blocks. This puts dissimilar channels together.

    For smaller dimensions, uses greedy vectorized assignment based on
    accumulated block affinities.

    Args:
        G: (d, d) full Gram matrix (or shrunk Gram).
        W: (o, d) weight matrix.
        M: Block size.
        N_keep: Number of weights to keep per block.

    Returns:
        perm: (d,) permutation indices.
        inv_perm: (d,) inverse permutation.
    """
    d = G.shape[0]
    n_blocks = d // M
    remainder = d - n_blocks * M
    device = G.device

    if n_blocks <= 1:
        perm = torch.arange(d, device=device)
        return perm, perm

    diag_G = torch.diag(G).clone()
    diag_G[diag_G < 1e-10] = 1e-10

    # Normalized correlation matrix
    D_inv_sqrt = 1.0 / torch.sqrt(diag_G)
    corr_abs = (G * (D_inv_sqrt.unsqueeze(0) * D_inv_sqrt.unsqueeze(1))).abs()
    corr_abs.fill_diagonal_(0)

    if d > 2000:
        # Spectral interleaving for large dimensions
        try:
            _, eigenvectors = torch.linalg.eigh(corr_abs)
            ev = eigenvectors[:, -1]
            spectral_order = torch.argsort(ev)

            perm = torch.empty(n_blocks * M, dtype=torch.long, device=device)
            for slot in range(M):
                start = slot * n_blocks
                end = start + n_blocks
                perm[slot::M] = spectral_order[start:end]

            if remainder > 0:
                extra = spectral_order[n_blocks * M:]
                perm = torch.cat([perm, extra])
        except Exception:
            perm = torch.arange(d, device=device)
    else:
        # Greedy vectorized assignment for smaller dimensions
        block_sizes = torch.zeros(n_blocks, dtype=torch.long, device=device)
        block_members = [[] for _ in range(n_blocks)]
        assigned = torch.zeros(d, dtype=torch.bool, device=device)
        block_affinity = torch.zeros(n_blocks, d, device=device)

        # Process channels in order of decreasing importance
        wanda_scores = (W.abs() * torch.sqrt(diag_G).unsqueeze(0)).mean(dim=0)
        channel_order = torch.argsort(wanda_scores, descending=True).tolist()

        for ch in channel_order:
            if assigned[ch]:
                continue
            affinities = block_affinity[:, ch].clone()
            affinities[block_sizes >= M] = float('inf')
            best_block = affinities.argmin().item()
            if affinities[best_block] == float('inf'):
                break
            block_members[best_block].append(ch)
            block_sizes[best_block] += 1
            assigned[ch] = True
            block_affinity[best_block] += corr_abs[ch]

        perm_list = []
        for b in range(n_blocks):
            perm_list.extend(block_members[b])
        for ch in range(d):
            if not assigned[ch]:
                perm_list.append(ch)
        perm = torch.tensor(perm_list, dtype=torch.long, device=device)

    inv_perm = torch.empty(d, dtype=torch.long, device=device)
    inv_perm[perm] = torch.arange(d, device=device)

    return perm, inv_perm


class FullGramWrapper:
    """
    Wraps a linear layer to accumulate the full input Gram matrix G = XX^T.

    Used for shrinkage and correlation-aware permutation variants.
    After calibration, call compute_shrinkage() to get the shrunk Gram
    and optionally compute_permutation() for channel reordering.

    Args:
        layer: nn.Linear layer to wrap.
        layer_id: Layer index.
        layer_name: Layer name.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0

        self.scaler_row = torch.zeros((self.columns,), device=self.dev)
        self.gram = torch.zeros((self.columns, self.columns), device=self.dev)

        self.rho = None
        self.gram_shrunk = None
        self.perm = None
        self.inv_perm = None

    def add_batch(self, inp, out):
        """Accumulate diagonal and full Gram from a batch of activations."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        inp = inp.type(torch.float32)

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + tmp)

        self.gram *= self.nsamples / (self.nsamples + tmp)
        self.gram += (inp @ inp.t()) / (self.nsamples + tmp)

        self.nsamples += tmp

    def compute_shrinkage(self):
        """Compute Ledoit-Wolf shrinkage and store shrunk Gram."""
        G = self.gram * self.nsamples
        T = self.nsamples
        self.rho, self.gram_shrunk = ledoit_wolf_shrinkage_intensity(G, T)
        return self.rho

    def compute_permutation(self, W, M, N_keep):
        """Compute correlation-aware channel permutation."""
        G = self.gram_shrunk if self.gram_shrunk is not None else self.gram * self.nsamples
        self.perm, self.inv_perm = correlation_aware_permutation(G, W, M, N_keep)
        return self.perm, self.inv_perm

    def get_block_gram_shrunk(self, block_idx, M):
        """Get shrunk Gram for a specific block (after permutation if applied)."""
        G = self.gram_shrunk if self.gram_shrunk is not None else self.gram * self.nsamples
        if self.perm is not None:
            G = G[self.perm][:, self.perm]
        start = block_idx * M
        end = min(start + M, self.columns)
        return G[start:end, start:end]

    def get_full_gram_shrunk(self):
        """Get the full shrunk Gram (with permutation applied if available)."""
        G = self.gram_shrunk if self.gram_shrunk is not None else self.gram * self.nsamples
        if self.perm is not None:
            G = G[self.perm][:, self.perm]
        return G
