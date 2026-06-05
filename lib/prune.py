"""
GReS: Gram-based Removal Saliency
One-shot pruning across unstructured, semi-structured (N:M), and structured granularities.

Key equations from the paper:
- Unstructured (Eq 5): Sal(P) = theta_P^T G_PP theta_P  (diagonal => Wanda)
- N:M (Eq 6): P* = argmin_{|P|=M-N} theta_P^T G_BB,PP theta_P  (enumerate keep-patterns)
- Compensation (Eq 7): Sal_comp(P) = theta_P^T [G_BB^{-1}]_PP^{-1} theta_P
- Structured (Eq 9): Sal(P) = 1_P^T (M . C)_PP 1_P  where M=V^T V, C=ZZ^T
"""

import time
import torch
import torch.nn as nn
from itertools import combinations

from .layerwrapper import WrappedGPT, PermutedWrappedGPT, StructuredWrapper
from .shrinkage import FullGramWrapper, apply_shrinkage_to_block
from .data import get_loaders


# ============================================================================
# Utility functions
# ============================================================================

def find_layers(module, layers=[nn.Linear], name=''):
    """Recursively find all layers of specified types in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    """
    Check the actual sparsity of all linear layers in the model.

    Returns:
        Overall sparsity ratio (fraction of zero weights).
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()
            sub_count += (W == 0).sum().item()
            sub_params += W.numel()
        print(f"  layer {i} sparsity {float(sub_count) / sub_params:.6f}")
    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    """
    Run calibration data through the embedding layer to get inputs for the
    first transformer layer. Uses a Catcher module to intercept activations.

    Args:
        model: HuggingFace causal LM.
        dataloader: List of (input_ids, targets) tuples.
        device: Torch device.

    Returns:
        inps: (nsamples, seqlen, hidden_size) tensor of layer-0 inputs.
        outs: Zero tensor of same shape (buffer for outputs).
        attention_mask: Attention mask from first batch.
        position_ids: Position IDs from first batch.
        position_embeddings: Position embeddings (for rotary models).
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if hasattr(model, 'hf_device_map') and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    nsamples = 128 if model.config.hidden_size < 8192 else 64
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=device
    )
    inps.requires_grad = False
    cache = {
        'i': 0, 'attention_mask': None,
        'position_ids': None, 'position_embeddings': None
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask')
            cache['position_ids'] = kwargs.get('position_ids')
            cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids, position_embeddings


# ============================================================================
# N:M pattern selection (fully vectorized)
# ============================================================================

def select_nm_fully_vectorized(W, G_blocks, M, N_keep, compensate=False, damp=1e-4):
    """
    Fully vectorized N:M pruning for an entire layer at once.

    Reshapes the weight matrix into (o, n_blocks, M) and processes all blocks
    simultaneously via batched tensor operations. Evaluates all C(M, N_keep)
    keep-patterns and selects the one minimizing block reconstruction error.

    Args:
        W: (o, i) weight matrix.
        G_blocks: (n_blocks, M, M) block Gram matrices.
        M: Block size.
        N_keep: Number of weights to keep per block.
        compensate: If True, use compensated saliency (Eq. 7) and compute
            weight updates for kept weights.
        damp: Dampening for matrix inversion in compensation mode.

    Returns:
        mask: (o, i) boolean, True = prune.
        updates: (o, i) weight updates (compensation deltas, or zeros).
    """
    o, i = W.shape
    n_blocks = i // M
    device = W.device
    n_prune = M - N_keep

    W_blocks = W[:, :n_blocks * M].reshape(o, n_blocks, M)

    keep_patterns = list(combinations(range(M), N_keep))
    n_patterns = len(keep_patterns)
    prune_patterns = [[j for j in range(M) if j not in kp] for kp in keep_patterns]

    if not compensate:
        # Cost[p, r, b] = theta_{r,b,P_p}^T G_{b,P_p,P_p} theta_{r,b,P_p}
        all_costs = torch.zeros(n_patterns, o, n_blocks, device=device)
        for p_idx, prune_idx in enumerate(prune_patterns):
            theta_P = W_blocks[:, :, prune_idx]
            G_PP = G_blocks[:n_blocks, :, :][:, prune_idx, :][:, :, prune_idx]
            tmp = torch.einsum('obn,bnm->obm', theta_P, G_PP)
            all_costs[p_idx] = (tmp * theta_P).sum(dim=2)

        best_pattern = all_costs.argmin(dim=0)

        mask_blocks = torch.zeros(o, n_blocks, M, dtype=torch.bool, device=device)
        for p_idx, prune_idx in enumerate(prune_patterns):
            match = (best_pattern == p_idx)
            for pi in prune_idx:
                mask_blocks[:, :, pi] |= match

        mask = torch.zeros(o, i, dtype=torch.bool, device=device)
        mask[:, :n_blocks * M] = mask_blocks.reshape(o, n_blocks * M)
        updates = torch.zeros_like(W)

    else:
        # Compensated mode: use Schur complement for saliency
        G_BB = G_blocks[:n_blocks].clone()
        diag_mean = torch.diagonal(G_BB, dim1=1, dim2=2).mean(dim=1, keepdim=True)
        G_BB += damp * diag_mean.unsqueeze(2) * torch.eye(M, device=device).unsqueeze(0)
        G_BB_inv = torch.linalg.inv(G_BB)

        all_costs = torch.zeros(n_patterns, o, n_blocks, device=device)

        schur_list = []
        comp_list = []
        for p_idx, prune_idx in enumerate(prune_patterns):
            keep_idx = list(keep_patterns[p_idx])
            G_BB_inv_PP = G_BB_inv[:, prune_idx, :][:, :, prune_idx]
            S_P = torch.linalg.inv(G_BB_inv_PP)
            schur_list.append(S_P)

            G_KK = G_BB[:, keep_idx, :][:, :, keep_idx]
            G_KP = G_BB[:, keep_idx, :][:, :, prune_idx]
            G_KK_inv = torch.linalg.inv(G_KK)
            comp_list.append(torch.bmm(G_KK_inv, G_KP))

        for p_idx, prune_idx in enumerate(prune_patterns):
            theta_P = W_blocks[:, :, prune_idx]
            S_P = schur_list[p_idx]
            tmp = torch.einsum('obn,bnm->obm', theta_P, S_P)
            all_costs[p_idx] = (tmp * theta_P).sum(dim=2)

        best_pattern = all_costs.argmin(dim=0)

        mask_blocks = torch.zeros(o, n_blocks, M, dtype=torch.bool, device=device)
        update_blocks = torch.zeros(o, n_blocks, M, device=device)

        for p_idx, prune_idx in enumerate(prune_patterns):
            keep_idx = list(keep_patterns[p_idx])
            match = (best_pattern == p_idx)
            for pi in prune_idx:
                mask_blocks[:, :, pi] |= match

            theta_P = W_blocks[:, :, prune_idx]
            comp_mat = comp_list[p_idx]
            delta_K = torch.einsum('bnk,obk->obn', comp_mat, theta_P)

            for ki_idx, ki in enumerate(keep_idx):
                update_blocks[:, :, ki] += match.float() * delta_K[:, :, ki_idx]

        mask = torch.zeros(o, i, dtype=torch.bool, device=device)
        mask[:, :n_blocks * M] = mask_blocks.reshape(o, n_blocks * M)
        updates = torch.zeros_like(W)
        updates[:, :n_blocks * M] = update_blocks.reshape(o, n_blocks * M)

    return mask, updates


def select_nm_vectorized_pp(W, G_blocks, M, N_keep, compensate=False, damp=1e-4,
                            skip_threshold=0.1, use_bf16=True):
    """
    Vectorized++ N:M pruning with two efficiency improvements:

    1. Wanda-bound skipping: For blocks where max within-block correlation
       rho_B < threshold, Wanda's diagonal selection is provably near-optimal
       (Theorem 1). Those blocks skip the full C(M,N) enumeration.

    2. bf16 precision: Evaluate quadratic forms in bf16 for the full-search
       blocks. The ranking is stable since candidates differ by much more
       than bf16 relative error.

    Args:
        W: (o, i) weight matrix.
        G_blocks: (n_blocks, M, M) block Gram matrices.
        M: Block size.
        N_keep: Number of weights to keep per block.
        compensate: If True, fall back to full vectorized (no skipping).
        damp: Dampening for compensation.
        skip_threshold: Correlation threshold for Wanda-bound skipping.
        use_bf16: Use bf16 for full-search blocks.

    Returns:
        mask: (o, i) boolean, True = prune.
        updates: (o, i) weight updates.
        stats: Dict with skip_ratio, n_full, n_skip.
    """
    o, i = W.shape
    n_blocks = i // M
    device = W.device
    n_prune = M - N_keep

    W_blocks = W[:, :n_blocks * M].reshape(o, n_blocks, M)

    keep_patterns = list(combinations(range(M), N_keep))
    n_patterns = len(keep_patterns)
    prune_patterns = [[j for j in range(M) if j not in kp] for kp in keep_patterns]

    # Wanda baseline for all blocks
    G_diag = torch.diagonal(G_blocks[:n_blocks], dim1=1, dim2=2)
    wanda_scores = W_blocks.abs() * G_diag.sqrt().unsqueeze(0)
    _, wanda_order = wanda_scores.sort(dim=2)
    wanda_prune_idx = wanda_order[:, :, :n_prune]

    # Per-block max correlation for skip decision
    G_diag_safe = G_diag.clamp(min=1e-10)
    D_inv_sqrt = 1.0 / G_diag_safe.sqrt()
    G_norm = G_blocks[:n_blocks] * (D_inv_sqrt.unsqueeze(2) * D_inv_sqrt.unsqueeze(1))
    rho_B = (G_norm.abs() - torch.eye(M, device=device).unsqueeze(0)).amax(dim=(1, 2))

    skip_mask = (rho_B < skip_threshold)
    n_skip = skip_mask.sum().item()
    n_full = n_blocks - n_skip

    if not compensate:
        # Start with Wanda mask for ALL blocks
        mask_blocks = torch.zeros(o, n_blocks, M, dtype=torch.bool, device=device)
        mask_blocks.scatter_(2, wanda_prune_idx, True)

        if n_full > 0:
            # Full search on non-skipped blocks
            full_idx = (~skip_mask).nonzero(as_tuple=True)[0]
            G_full = G_blocks[full_idx].bfloat16() if use_bf16 else G_blocks[full_idx]
            W_full = W_blocks[:, full_idx, :].bfloat16() if use_bf16 else W_blocks[:, full_idx, :]

            all_costs = torch.zeros(n_patterns, o, n_full, device=device)
            for p_idx, prune_idx in enumerate(prune_patterns):
                theta_P = W_full[:, :, prune_idx]
                G_PP = G_full[:, prune_idx, :][:, :, prune_idx]
                tmp = torch.einsum('obn,bnm->obm', theta_P, G_PP)
                all_costs[p_idx] = (tmp * theta_P).sum(dim=2).float()

            best_pattern = all_costs.argmin(dim=0)
            full_mask = torch.zeros(o, n_full, M, dtype=torch.bool, device=device)
            for p_idx, prune_idx in enumerate(prune_patterns):
                match = (best_pattern == p_idx)
                for pi in prune_idx:
                    full_mask[:, :, pi] |= match
            mask_blocks[:, full_idx, :] = full_mask

        mask = torch.zeros(o, i, dtype=torch.bool, device=device)
        mask[:, :n_blocks * M] = mask_blocks.reshape(o, n_blocks * M)
        updates = torch.zeros_like(W)
    else:
        # Compensated mode: fall back to full vectorized
        mask, updates = select_nm_fully_vectorized(
            W, G_blocks, M, N_keep, compensate=True, damp=damp
        )
        return mask, updates, {'skip_ratio': 0.0, 'n_full': n_blocks, 'n_skip': 0}

    return mask, updates, {
        'skip_ratio': n_skip / max(n_blocks, 1),
        'n_full': n_full, 'n_skip': n_skip,
    }


# ============================================================================
# Cheap activation-sorted channel permutation
# ============================================================================

def cheap_channel_permutation(G_diag, M):
    """
    Fast activation-sorted channel permutation for N:M pruning.

    Sort channels by activation magnitude (sqrt of Gram diagonal), then
    interleave high and low into each block of M. This ensures each block
    has a mix of important and unimportant channels, reducing the chance
    of pruning a critical cluster.

    Cost: O(d log d) for the sort -- negligible vs calibration.

    Args:
        G_diag: (d,) diagonal of the Gram matrix (activation norms squared).
        M: Block size for N:M sparsity.

    Returns:
        perm: (d,) permutation indices.
        inv_perm: (d,) inverse permutation.
    """
    d = G_diag.shape[0]
    n_blocks = d // M
    device = G_diag.device

    if n_blocks <= 1:
        perm = torch.arange(d, device=device)
        return perm, perm

    sorted_idx = torch.argsort(G_diag)  # ascending by activation norm

    # Interleave: deal channels round-robin across blocks
    usable = n_blocks * M
    perm = torch.empty(usable, dtype=torch.long, device=device)
    for slot in range(M):
        start = slot * n_blocks
        end = start + n_blocks
        perm[slot::M] = sorted_idx[start:end]

    # Handle remainder columns
    if usable < d:
        extra = sorted_idx[usable:]
        perm = torch.cat([perm, extra])

    inv_perm = torch.empty(d, dtype=torch.long, device=device)
    inv_perm[perm] = torch.arange(d, device=device)

    return perm, inv_perm


# ============================================================================
# GReS v2 N:M selection (outlier protection, optional hybrid metric, rescaling)
# ============================================================================

def select_nm_v2(W, G_blocks, M, N_keep, outlier_sigma=3.0,
                 hybrid_alpha=0.0, rescale=True, damp=0.01):
    """
    GReS v2 N:M selection with improvements over v1:

    1. Outlier channel protection: Channels with activation norm > mean + sigma*std
       get a large saliency boost so they are never pruned.

    2. Hybrid metric (optional): Combine Gram-based saliency with Wanda scores.
       Final cost = Gram_cost * (1 + alpha * wanda_score_normalized)

    3. Row-wise weight rescaling after pruning: Adjust kept weights via
       closed-form least-squares to minimize block reconstruction error.
       delta_K = G_KK^{-1} G_KP theta_P (same as block-OBS).

    Pattern selection uses the UNDAMPED Gram to preserve the exact same
    ranking as base GReS. Dampening is only applied for the G_KK inversion
    in the rescaling step (numerical stability).

    Args:
        W: (o, i) weight matrix.
        G_blocks: (n_blocks, M, M) block Gram matrices (undamped).
        M: Block size.
        N_keep: Weights to keep per block.
        outlier_sigma: Std multiplier for outlier detection (999 = disabled).
        hybrid_alpha: Weight of Wanda term (0 = pure Gram, default).
        rescale: Whether to apply row-wise weight rescaling.
        damp: Dampening factor for G_KK inversion only.

    Returns:
        mask: (o, i) boolean, True = prune.
        updates: (o, i) weight updates from rescaling.
    """
    o, i = W.shape
    n_blocks = i // M
    device = W.device

    W_blocks = W[:, :n_blocks * M].reshape(o, n_blocks, M)

    keep_patterns = list(combinations(range(M), N_keep))
    n_patterns = len(keep_patterns)
    prune_patterns = [[j for j in range(M) if j not in kp] for kp in keep_patterns]

    G_sel = G_blocks[:n_blocks]

    # Outlier protection
    block_diag = torch.diagonal(G_sel, dim1=1, dim2=2)
    col_norms = block_diag.sqrt().clamp(min=1e-10)
    col_mean = col_norms.mean(dim=1, keepdim=True)
    col_std = col_norms.std(dim=1, keepdim=True).clamp(min=1e-8)
    outlier_mask = col_norms > (col_mean + outlier_sigma * col_std)

    # Wanda scores per block
    wanda_block = W_blocks.abs() * col_norms.unsqueeze(0)

    all_costs = torch.zeros(n_patterns, o, n_blocks, device=device)

    for p_idx, prune_idx in enumerate(prune_patterns):
        theta_P = W_blocks[:, :, prune_idx]
        G_PP = G_sel[:, prune_idx, :][:, :, prune_idx]
        tmp = torch.einsum('obn,bnm->obm', theta_P, G_PP)
        gram_cost = (tmp * theta_P).sum(dim=2)

        if hybrid_alpha > 0:
            wanda_cost = wanda_block[:, :, prune_idx].sum(dim=2)
            wanda_total = wanda_block.sum(dim=2).clamp(min=1e-10)
            wanda_frac = wanda_cost / wanda_total
            all_costs[p_idx] = gram_cost * (1.0 + hybrid_alpha * wanda_frac)
        else:
            all_costs[p_idx] = gram_cost

        # Outlier penalty
        for pi in prune_idx:
            outlier_in_pattern = outlier_mask[:, pi]
            all_costs[p_idx, :, outlier_in_pattern] += 1e12

    best_pattern = all_costs.argmin(dim=0)

    mask_blocks = torch.zeros(o, n_blocks, M, dtype=torch.bool, device=device)
    for p_idx, prune_idx in enumerate(prune_patterns):
        match = (best_pattern == p_idx)
        for pi in prune_idx:
            mask_blocks[:, :, pi] |= match

    mask = torch.zeros(o, i, dtype=torch.bool, device=device)
    mask[:, :n_blocks * M] = mask_blocks.reshape(o, n_blocks * M)

    # Row-wise weight rescaling (block-OBS compensation)
    updates = torch.zeros_like(W)
    if rescale:
        G_inv = G_blocks[:n_blocks].clone()
        diag_mean = torch.diagonal(G_inv, dim1=1, dim2=2).mean(dim=1, keepdim=True)
        G_inv += damp * diag_mean.unsqueeze(2) * torch.eye(M, device=device).unsqueeze(0)

        update_blocks = torch.zeros(o, n_blocks, M, device=device)

        for p_idx, prune_idx in enumerate(prune_patterns):
            keep_idx = list(keep_patterns[p_idx])
            match = (best_pattern == p_idx)
            if not match.any():
                continue

            G_KK = G_inv[:, keep_idx, :][:, :, keep_idx]
            G_KP = G_inv[:, keep_idx, :][:, :, prune_idx]
            G_KK_inv = torch.linalg.inv(G_KK)
            comp_mat = torch.bmm(G_KK_inv, G_KP)

            theta_P = W_blocks[:, :, prune_idx]
            delta_K = torch.einsum('bnk,obk->obn', comp_mat, theta_P)

            for ki_idx, ki in enumerate(keep_idx):
                update_blocks[:, :, ki] += match.float() * delta_K[:, :, ki_idx]

        updates[:, :n_blocks * M] = update_blocks.reshape(o, n_blocks * M)

    return mask, updates


# ============================================================================
# Main GReS pruning function
# ============================================================================

@torch.no_grad()
def prune_gres(args, model, tokenizer, device=torch.device("cuda:0"),
               prune_n=0, prune_m=0):
    """
    GReS pruning: unified across unstructured and N:M sparsity.

    For unstructured: equivalent to Wanda (diagonal of Gram, Proposition 1).
    For N:M: two-pass approach with cheap channel permutation.
      Pass 1: Accumulate diagonal Gram to compute activation-sorted permutation.
      Pass 2: Accumulate block Grams in permuted order for pattern selection.
      The permutation groups dissimilar channels into each block, improving
      the Gram-based N:M pattern selection.

    Args:
        args: Namespace with nsamples, seed, sparsity_ratio, compensate.
        model: HuggingFace causal LM with model.seqlen attribute.
        tokenizer: HuggingFace tokenizer.
        device: Torch device.
        prune_n: Number of weights to prune per block (0 = unstructured).
        prune_m: Block size M for N:M sparsity.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer
    )
    print("Dataset loading complete")

    inps, outs, attention_mask, position_ids, position_embeddings = \
        prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    compensate = getattr(args, 'compensate', False)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            )
            if position_embeddings is not None:
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        if prune_n != 0:
            # ==== N:M sparsity: two-pass with cheap channel permutation ====
            M = prune_m
            N_keep = prune_m - prune_n

            # --- Pass 1: diagonal Gram for permutation ---
            diag_wrappers = {}
            for name in subset:
                diag_wrappers[name] = WrappedGPT(
                    subset[name], gram_mode="diagonal", block_size=0
                )

            def add_batch_diag(name):
                def tmp(_, inp, out):
                    diag_wrappers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in diag_wrappers:
                handles.append(subset[name].register_forward_hook(add_batch_diag(name)))

            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            # Compute permutation per sublayer
            perms = {}
            inv_perms = {}
            for name in subset:
                G_diag = diag_wrappers[name].scaler_row.float() * diag_wrappers[name].nsamples
                perm, inv_perm = cheap_channel_permutation(G_diag, M)
                perms[name] = perm
                inv_perms[name] = inv_perm

            del diag_wrappers
            torch.cuda.empty_cache()

            # --- Pass 2: block Grams in permuted order ---
            block_wrappers = {}
            for name in subset:
                block_wrappers[name] = PermutedWrappedGPT(
                    subset[name], perm=perms[name],
                    gram_mode="block", block_size=M
                )

            def add_batch_block(name):
                def tmp(_, inp, out):
                    block_wrappers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in block_wrappers:
                handles.append(subset[name].register_forward_hook(add_batch_block(name)))

            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            # --- Prune with permuted block Grams, then unpermute ---
            for name in subset:
                print(f"Pruning layer {i} name {name}")
                W = subset[name].weight.data.clone().float()
                o_dim, i_dim = W.shape
                n_blocks = i_dim // M

                W_perm = W[:, perms[name]]
                G_all = block_wrappers[name].block_grams[:n_blocks, :M, :M].float() \
                    * block_wrappers[name].nsamples

                tick = time.time()
                n_patterns = len(list(combinations(range(M), N_keep)))

                if n_patterns <= 10:
                    W_mask, W_updates = select_nm_fully_vectorized(
                        W_perm, G_all, M, N_keep, compensate=compensate
                    )
                else:
                    W_mask, W_updates, stats = select_nm_vectorized_pp(
                        W_perm, G_all, M, N_keep, compensate=compensate,
                        skip_threshold=0.3, use_bf16=True
                    )
                print(f"  N:M selection took {time.time() - tick:.2f}s")

                if compensate:
                    W_perm += W_updates
                W_perm[W_mask] = 0

                subset[name].weight.data = W_perm[:, inv_perms[name]].to(
                    subset[name].weight.data.dtype
                )

        else:
            # ==== Unstructured: diagonal Gram = Wanda (Proposition 1) ====
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedGPT(
                    subset[name], gram_mode="diagonal", block_size=0
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0),
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    position_embeddings=position_embeddings)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f"Pruning layer {i} name {name}")
                W = subset[name].weight.data.clone().float()
                W_metric = torch.abs(W) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
                subset[name].weight.data[W_mask] = 0

        # Forward through pruned layer to get inputs for next layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ============================================================================
# GReS with Ledoit-Wolf shrinkage
# ============================================================================

@torch.no_grad()
def prune_gres_shrink(args, model, tokenizer, device=torch.device("cuda:0"),
                      prune_n=0, prune_m=0):
    """
    GReS with Ledoit-Wolf Gram shrinkage and optional channel permutation.

    Accumulates the full Gram matrix, applies Ledoit-Wolf shrinkage
    G_hat = (1-rho)*G + rho*diag(G), then extracts block Grams for N:M
    pattern selection.

    Args:
        args: Namespace with nsamples, seed, sparsity_ratio, compensate,
            use_permutation.
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        device: Torch device.
        prune_n: Number of weights to prune per block (0 = unstructured).
        prune_m: Block size M.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer
    )
    print("Dataset loading complete")

    inps, outs, attention_mask, position_ids, position_embeddings = \
        prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    compensate = getattr(args, 'compensate', False)
    use_permutation = getattr(args, 'use_permutation', False)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            )
            if position_embeddings is not None:
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = FullGramWrapper(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"Pruning layer {i} name {name}")
            W = subset[name].weight.data.clone().float()
            o_dim, i_dim = W.shape
            wrapper = wrapped_layers[name]

            rho = wrapper.compute_shrinkage()
            print(f"  LW shrinkage rho = {rho:.4f}")

            if prune_n != 0:
                M = prune_m
                N_keep = prune_m - prune_n

                if use_permutation:
                    perm, inv_perm = wrapper.compute_permutation(W, M, N_keep)
                    W = W[:, perm]

                n_blocks = i_dim // M
                G_shrunk = wrapper.get_full_gram_shrunk()
                G_blocks = torch.zeros(n_blocks, M, M, device=G_shrunk.device)
                for b in range(n_blocks):
                    s = b * M
                    G_blocks[b] = G_shrunk[s:s+M, s:s+M]

                tick = time.time()
                W_mask, W_updates = select_nm_fully_vectorized(
                    W, G_blocks, M, N_keep, compensate=compensate
                )
                print(f"  N:M selection took {time.time() - tick:.2f}s")

                if compensate:
                    W += W_updates
                W[W_mask] = 0

                if use_permutation:
                    subset[name].weight.data = W[:, inv_perm].to(subset[name].weight.data.dtype)
                else:
                    subset[name].weight.data = W.to(subset[name].weight.data.dtype)

            else:
                W_metric = torch.abs(W) * torch.sqrt(
                    wrapper.scaler_row.reshape((1, -1))
                )
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)
                subset[name].weight.data[W_mask] = 0

            # Free memory
            wrapper.gram = None
            wrapper.gram_shrunk = None
            torch.cuda.empty_cache()

        # Forward through pruned layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


# ============================================================================
# GReS structured pruning (heads, neurons, GQA groups)
# ============================================================================

def get_gqa_config(model):
    """
    Extract GQA configuration from model.

    Returns:
        num_heads: Total query heads.
        num_kv_heads: Number of KV heads (= num_heads for MHA).
        head_dim: Dimension per head.
        num_q_per_kv: Query heads per KV head.
    """
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)
    head_dim = cfg.hidden_size // num_heads
    num_q_per_kv = num_heads // num_kv_heads
    return num_heads, num_kv_heads, head_dim, num_q_per_kv


def _zero_attention_heads(subset, q_heads_remove, kv_heads_remove, head_dim):
    """
    Zero out attention heads, handling GQA correctly.

    For q_proj/o_proj: zero the specified query heads.
    For k_proj/v_proj: zero the specified KV heads.
    """
    for proj_name in subset:
        if 'q_proj' in proj_name:
            for h in q_heads_remove:
                s = slice(h * head_dim, (h + 1) * head_dim)
                subset[proj_name].weight.data[s, :] = 0
        elif 'o_proj' in proj_name:
            for h in q_heads_remove:
                s = slice(h * head_dim, (h + 1) * head_dim)
                subset[proj_name].weight.data[:, s] = 0
        elif 'k_proj' in proj_name:
            for kv in kv_heads_remove:
                s = slice(kv * head_dim, (kv + 1) * head_dim)
                subset[proj_name].weight.data[s, :] = 0
        elif 'v_proj' in proj_name:
            for kv in kv_heads_remove:
                s = slice(kv * head_dim, (kv + 1) * head_dim)
                subset[proj_name].weight.data[s, :] = 0


def _compute_head_scores(W_o, sw, num_heads, head_dim):
    """Diagonal head saliency: ||W_O[:, S_h]||_F^2 * ||z_h||^2."""
    scores = torch.zeros(num_heads, device=W_o.device)
    for h in range(num_heads):
        s = slice(h * head_dim, (h + 1) * head_dim)
        scores[h] = torch.norm(W_o[:, s], 'fro') ** 2 * sw.activation_norms[s].sum()
    return scores


def _compute_head_gram(W_o, sw, num_heads, head_dim):
    """Full head Gram: Gamma_{h1,h2} = <V_{h1}^T V_{h2}, Z_{h1} Z_{h2}^T>_F."""
    gram = torch.zeros(num_heads, num_heads, device=W_o.device)
    for h1 in range(num_heads):
        s1 = slice(h1 * head_dim, (h1 + 1) * head_dim)
        for h2 in range(h1, num_heads):
            s2 = slice(h2 * head_dim, (h2 + 1) * head_dim)
            wg = W_o[:, s1].t() @ W_o[:, s2]
            ag = sw.activation_gram[s1, s2] * sw.nsamples
            val = (wg * ag).sum()
            gram[h1, h2] = val
            gram[h2, h1] = val
    return gram


def _greedy_remove(gram, n_remove):
    """
    Greedy selection of units to remove using marginal cost (Eq. 10).

    At each step, removes the unit with smallest marginal saliency:
    marginal(u) = Gamma_{uu} + 2 * sum_{v in removed} Gamma_{uv}
    """
    n = gram.shape[0]
    removed = set()
    for _ in range(n_remove):
        best_u, best_cost = None, float('inf')
        for u in range(n):
            if u in removed:
                continue
            marginal = gram[u, u] + 2 * sum(gram[u, v].item() for v in removed)
            if marginal < best_cost:
                best_cost = marginal
                best_u = u
        removed.add(best_u)
    return removed


def _apply_structured_compensation(W_out, gram, keep_set, prune_set, head_dim, damp=1e-2):
    """
    Structured compensation via Schur complement (Proposition 2).

    After selecting which units to prune, adjusts remaining weights:
    delta_K = G_KK^{-1} G_KP (compensation matrix)
    Then W_out[:, S_k] += sum_p comp[k,p] * W_out[:, S_p]
    """
    keep = sorted(keep_set)
    prune = sorted(prune_set)
    if not prune or not keep:
        return

    n_k, n_p = len(keep), len(prune)

    G_KK = gram[keep][:, keep].clone()
    G_KP = gram[keep][:, prune]
    G_KK += damp * G_KK.diag().mean() * torch.eye(n_k, device=gram.device)
    comp = torch.linalg.solve(G_KK, G_KP)

    for ki, k in enumerate(keep):
        sk = slice(k * head_dim, (k + 1) * head_dim)
        update = torch.zeros_like(W_out[:, sk])
        for pi, p in enumerate(prune):
            sp = slice(p * head_dim, (p + 1) * head_dim)
            update += comp[ki, pi] * W_out[:, sp]
        W_out[:, sk] += update


def _prune_heads_per_layer(subset, wrapped, attn_proj, num_heads, num_kv_heads,
                           head_dim, num_q_per_kv, ratio, use_correlation, compensate):
    """Prune individual attention heads with proper GQA handling."""
    name, proj = attn_proj
    sw = wrapped[name]
    W_o = proj.weight.data.float()

    n_remove = int(num_heads * ratio)
    if n_remove == 0:
        return

    head_scores = _compute_head_scores(W_o, sw, num_heads, head_dim)

    if use_correlation and sw.activation_gram is not None:
        head_gram = _compute_head_gram(W_o, sw, num_heads, head_dim)
        removed_q = _greedy_remove(head_gram, n_remove)

        if compensate:
            kept_q = set(range(num_heads)) - removed_q
            W_comp = proj.weight.data.float().clone()
            _apply_structured_compensation(W_comp, head_gram, kept_q, removed_q, head_dim)
            proj.weight.data = W_comp.to(proj.weight.data.dtype)
    else:
        _, remove_indices = torch.topk(head_scores, n_remove, largest=False)
        removed_q = set(remove_indices.tolist())

    # Only remove KV heads if ALL query heads in the group are removed
    removed_kv = set()
    for kv in range(num_kv_heads):
        q_in_group = set(range(kv * num_q_per_kv, (kv + 1) * num_q_per_kv))
        if q_in_group.issubset(removed_q):
            removed_kv.add(kv)

    print(f"    heads: removing {len(removed_q)}/{num_heads} q-heads, "
          f"{len(removed_kv)}/{num_kv_heads} kv-heads")
    _zero_attention_heads(subset, removed_q, removed_kv, head_dim)


def _prune_kv_groups_per_layer(subset, wrapped, attn_proj, num_heads, num_kv_heads,
                               head_dim, num_q_per_kv, ratio, use_correlation, compensate):
    """Prune entire KV groups (GQA group pruning)."""
    name, proj = attn_proj
    sw = wrapped[name]
    W_o = proj.weight.data.float()

    n_remove = max(1, int(num_kv_heads * ratio))
    if n_remove >= num_kv_heads:
        n_remove = num_kv_heads - 1
    if n_remove == 0:
        return

    head_scores = _compute_head_scores(W_o, sw, num_heads, head_dim)

    # Aggregate to group level
    group_scores = torch.zeros(num_kv_heads, device=W_o.device)
    for g in range(num_kv_heads):
        q_start = g * num_q_per_kv
        group_scores[g] = head_scores[q_start:q_start + num_q_per_kv].sum()

    if use_correlation and sw.activation_gram is not None:
        head_gram = _compute_head_gram(W_o, sw, num_heads, head_dim)
        group_gram = torch.zeros(num_kv_heads, num_kv_heads, device=W_o.device)
        for g1 in range(num_kv_heads):
            for g2 in range(num_kv_heads):
                for h1 in range(g1 * num_q_per_kv, (g1 + 1) * num_q_per_kv):
                    for h2 in range(g2 * num_q_per_kv, (g2 + 1) * num_q_per_kv):
                        group_gram[g1, g2] += head_gram[h1, h2]

        removed_groups = _greedy_remove(group_gram, n_remove)

        if compensate:
            kept_groups = set(range(num_kv_heads)) - removed_groups
            W_comp = proj.weight.data.float().clone()
            _apply_structured_compensation(
                W_comp, group_gram,
                kept_groups, removed_groups, head_dim * num_q_per_kv)
            proj.weight.data = W_comp.to(proj.weight.data.dtype)
    else:
        _, remove_indices = torch.topk(group_scores, n_remove, largest=False)
        removed_groups = set(remove_indices.tolist())

    removed_q = set()
    removed_kv = set()
    for g in removed_groups:
        for h in range(g * num_q_per_kv, (g + 1) * num_q_per_kv):
            removed_q.add(h)
        removed_kv.add(g)

    print(f"    kv_groups: removing {len(removed_groups)}/{num_kv_heads} groups "
          f"({len(removed_q)} q-heads, {len(removed_kv)} kv-heads)")
    _zero_attention_heads(subset, removed_q, removed_kv, head_dim)


def _prune_neurons_per_layer(subset, wrapped, mlp_proj, ratio, compensate):
    """Prune MLP neurons based on diagonal saliency."""
    name, proj = mlp_proj
    sw = wrapped[name]
    W_down = proj.weight.data.float()
    n_neurons = W_down.shape[1]

    n_remove = int(n_neurons * ratio)
    if n_remove == 0:
        return

    weight_norms = torch.norm(W_down, dim=0) ** 2
    act_norms = sw.activation_norms * sw.nsamples
    neuron_scores = weight_norms * act_norms

    _, remove_indices = torch.topk(neuron_scores, n_remove, largest=False)
    remove_mask = torch.zeros(n_neurons, dtype=torch.bool, device=W_down.device)
    remove_mask[remove_indices] = True

    print(f"    neurons: removing {n_remove}/{n_neurons}")
    for proj_name in subset:
        if 'down_proj' in proj_name:
            subset[proj_name].weight.data[:, remove_mask] = 0
        elif any(p in proj_name for p in ['gate_proj', 'up_proj']):
            subset[proj_name].weight.data[remove_mask, :] = 0


@torch.no_grad()
def prune_gres_structured(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    GReS structured pruning with GQA support.

    Supports multiple modes via args.struct_mode:
    - "head": Prune individual attention heads (GQA-aware).
    - "neuron": Prune MLP neurons only.
    - "gqa_group": Prune entire KV groups (most natural for GQA models).
    - "combined": Prune both heads and neurons (default).

    With args.compensate=True, applies Schur complement compensation
    to adjust remaining weights after pruning (Proposition 2).

    Args:
        args: Namespace with nsamples, seed, sparsity_ratio, struct_mode,
            use_correlation, compensate.
        model: HuggingFace causal LM.
        tokenizer: HuggingFace tokenizer.
        device: Torch device.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("Loading calibration data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer
    )
    print("Dataset loading complete")

    inps, outs, attention_mask, position_ids, position_embeddings = \
        prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    num_heads, num_kv_heads, head_dim, num_q_per_kv = get_gqa_config(model)
    struct_ratio = args.sparsity_ratio
    use_correlation = getattr(args, 'use_correlation', False)
    compensate = getattr(args, 'compensate', False)
    mode = getattr(args, 'struct_mode', 'combined')

    print(f"Structured pruning: mode={mode}, ratio={struct_ratio}, "
          f"heads={num_heads}, kv_heads={num_kv_heads}, GQA={num_heads != num_kv_heads}")

    for i in range(len(layers)):
        layer = layers[i]

        if hasattr(model, 'hf_device_map') and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = (
                inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            )
            if position_embeddings is not None:
                position_embeddings = tuple(pe.to(dev) for pe in position_embeddings)

        subset = find_layers(layer)
        attn_proj = None
        mlp_proj = None
        for name in subset:
            if 'o_proj' in name:
                attn_proj = (name, subset[name])
            elif 'down_proj' in name:
                mlp_proj = (name, subset[name])

        # Wrap output projections for activation statistics
        wrapped = {}
        if attn_proj and mode in ('head', 'gqa_group', 'combined'):
            name, proj = attn_proj
            sw = StructuredWrapper(proj, layer_name=name, unit_type="head")
            if use_correlation:
                sw.enable_full_gram()
            wrapped[name] = sw
        if mlp_proj and mode in ('neuron', 'combined'):
            name, proj = mlp_proj
            sw = StructuredWrapper(proj, layer_name=name, unit_type="neuron")
            wrapped[name] = sw

        def add_batch(wname):
            def tmp(_, inp, out):
                wrapped[wname].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for wname in wrapped:
            handles.append(subset[wname].register_forward_hook(add_batch(wname)))

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()

        print(f"Pruning layer {i}")

        if attn_proj and mode in ('head', 'combined'):
            _prune_heads_per_layer(subset, wrapped, attn_proj, num_heads, num_kv_heads,
                                   head_dim, num_q_per_kv, struct_ratio,
                                   use_correlation, compensate)
        elif attn_proj and mode == 'gqa_group':
            _prune_kv_groups_per_layer(subset, wrapped, attn_proj, num_heads, num_kv_heads,
                                       head_dim, num_q_per_kv, struct_ratio,
                                       use_correlation, compensate)

        if mlp_proj and mode in ('neuron', 'combined'):
            _prune_neurons_per_layer(subset, wrapped, mlp_proj, struct_ratio, compensate)

        # Forward through pruned layer
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
