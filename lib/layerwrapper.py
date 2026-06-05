"""
Layer wrappers for accumulating input Gram statistics during calibration.

Provides three wrappers for different pruning granularities:
- WrappedGPT: Diagonal and/or block Gram accumulation (unstructured, N:M).
- PermutedWrappedGPT: Like WrappedGPT but permutes channels before accumulating
  (used by GReS two-pass N:M pruning).
- StructuredWrapper: Activation statistics for structured pruning (heads, neurons).
"""

import torch
import torch.nn as nn


class WrappedGPT:
    """
    Wraps a linear layer to accumulate input Gram statistics for GReS pruning.

    Supports two modes controlled by gram_mode:
    - "diagonal": Accumulates G_jj = ||X_j||^2 per channel (for unstructured pruning).
    - "block": Additionally accumulates (M x M) block Gram matrices G_BB for each
      block of M consecutive channels (for N:M semi-structured pruning).

    All statistics use online updates to avoid storing full activation matrices.

    Args:
        layer: nn.Linear layer to wrap.
        layer_id: Layer index (for logging).
        layer_name: Layer name (for logging).
        gram_mode: "diagonal" or "block".
        block_size: Block size M for N:M pruning (only used when gram_mode="block").
    """

    def __init__(self, layer, layer_id=0, layer_name="none",
                 gram_mode="diagonal", block_size=0):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name
        self.gram_mode = gram_mode
        self.block_size = block_size

        # Diagonal: G_jj = ||X_j||^2 (always accumulated)
        self.scaler_row = torch.zeros((self.columns,), device=self.dev)

        # Block Gram: G_BB for each block (only for N:M mode)
        if gram_mode == "block" and block_size > 0:
            n_blocks = (self.columns + block_size - 1) // block_size
            self.block_grams = torch.zeros(
                (n_blocks, block_size, block_size), device=self.dev
            )

    def add_batch(self, inp, out):
        """
        Accumulate Gram statistics from a batch of activations.

        Args:
            inp: Input activations, shape (batch, seq_len, channels) or (seq_len, channels).
            out: Output activations (unused, kept for hook API compatibility).
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # (channels, T)

        inp = inp.type(torch.float32)

        # Online update for diagonal: G_jj = ||X_j||^2
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + tmp)

        # Block Gram accumulation: G_BB = X_B @ X_B^T for each block
        if self.gram_mode == "block" and self.block_size > 0:
            M = self.block_size
            n_full_blocks = self.columns // M
            self.block_grams *= self.nsamples / (self.nsamples + tmp)
            if n_full_blocks > 0:
                inp_blocks = inp[:n_full_blocks * M, :].reshape(n_full_blocks, M, -1)
                grams = torch.bmm(inp_blocks, inp_blocks.transpose(1, 2)) / (self.nsamples + tmp)
                self.block_grams[:n_full_blocks, :M, :M] += grams
            # Handle last partial block
            if n_full_blocks * M < self.columns:
                start = n_full_blocks * M
                block_inp = inp[start:, :]
                actual_size = block_inp.shape[0]
                gram = block_inp @ block_inp.t() / (self.nsamples + tmp)
                self.block_grams[n_full_blocks, :actual_size, :actual_size] += gram

        self.nsamples += tmp

    def get_block_gram(self, block_idx):
        """Return the unnormalized Gram matrix for a specific block."""
        M = self.block_size
        start = block_idx * M
        end = min(start + M, self.columns)
        actual_size = end - start
        return self.block_grams[block_idx, :actual_size, :actual_size] * self.nsamples


class PermutedWrappedGPT(WrappedGPT):
    """
    Like WrappedGPT but applies a channel permutation to inputs before
    accumulating Gram statistics.

    Used by GReS two-pass N:M pruning: Pass 1 computes the permutation
    from diagonal statistics, Pass 2 accumulates block Grams in the
    permuted channel order.

    Args:
        layer: nn.Linear layer to wrap.
        perm: (columns,) tensor of permutation indices.
        layer_id: Layer index.
        layer_name: Layer name.
        gram_mode: "diagonal" or "block".
        block_size: Block size M.
    """

    def __init__(self, layer, perm, layer_id=0, layer_name="none",
                 gram_mode="block", block_size=0):
        super().__init__(layer, layer_id=layer_id, layer_name=layer_name,
                         gram_mode=gram_mode, block_size=block_size)
        self.perm = perm

    def add_batch(self, inp, out):
        """Accumulate statistics on permuted activations."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # (channels, T)

        inp = inp.type(torch.float32)

        # Apply channel permutation
        inp = inp[self.perm]

        # Online update for diagonal
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + tmp)

        # Block Gram on permuted input
        if self.gram_mode == "block" and self.block_size > 0:
            M = self.block_size
            n_full_blocks = self.columns // M
            self.block_grams *= self.nsamples / (self.nsamples + tmp)
            if n_full_blocks > 0:
                inp_blocks = inp[:n_full_blocks * M, :].reshape(n_full_blocks, M, -1)
                grams = torch.bmm(inp_blocks, inp_blocks.transpose(1, 2)) / (self.nsamples + tmp)
                self.block_grams[:n_full_blocks, :M, :M] += grams
            if n_full_blocks * M < self.columns:
                start = n_full_blocks * M
                block_inp = inp[start:, :]
                actual_size = block_inp.shape[0]
                gram = block_inp @ block_inp.t() / (self.nsamples + tmp)
                self.block_grams[n_full_blocks, :actual_size, :actual_size] += gram

        self.nsamples += tmp


class StructuredWrapper:
    """
    Wraps output projections (o_proj / down_proj) to accumulate structured
    pruning statistics for GReS.

    For neurons: V = W_down, z_n(t) = a_n(t)
    For heads:   V = W_O[:, S_h], z_h(t) = o_h(t)

    Accumulates:
    - activation_norms: ||z_u||^2 (diagonal of Z Z^T)
    - activation_gram: Z Z^T (optional, for correlation-aware selection)

    The weight Gram V^T V is computed directly from weights at pruning time.

    Args:
        layer: nn.Linear output projection layer.
        layer_name: Name for logging.
        unit_type: "neuron" or "head".
    """

    def __init__(self, layer, layer_name="none", unit_type="neuron"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.unit_type = unit_type
        self.layer_name = layer_name
        self.n_units = layer.weight.shape[1]

        # Diagonal of activation Gram: ||z_u||^2
        self.activation_norms = torch.zeros(self.n_units, device=self.dev)
        # Full activation Gram (optional, enabled via enable_full_gram)
        self.activation_gram = None
        self.nsamples = 0

    def add_batch(self, inp, out):
        """
        Accumulate activation statistics.

        Args:
            inp: Input to the output projection (pre-projection activations).
            out: Output (unused).
        """
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()  # (n_units, T)

        inp = inp.type(torch.float32)

        # Online update for activation norms
        self.activation_norms *= self.nsamples / (self.nsamples + tmp)
        self.activation_norms += torch.norm(inp, p=2, dim=1) ** 2 / (self.nsamples + tmp)

        # Full activation Gram (if enabled)
        if self.activation_gram is not None:
            self.activation_gram *= self.nsamples / (self.nsamples + tmp)
            self.activation_gram += (inp @ inp.t()) / (self.nsamples + tmp)

        self.nsamples += tmp

    def enable_full_gram(self):
        """Enable full activation Gram accumulation for correlation-aware selection."""
        self.activation_gram = torch.zeros(
            (self.n_units, self.n_units), device=self.dev
        )
