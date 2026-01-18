"""
================================================================================
[CXL-Enabled Engram Architecture Implementation]

This module extends the Engram conditional memory implementation to use CXL
(Compute Express Link) GPU peer-to-peer DMA for embedding storage and retrieval.

Key Features:
1. Embedding tables stored in CXL memory (host-side)
2. GPU P2P DMA for efficient embedding lookups
3. Batch-optimized retrieval to minimize DMA operations
4. Seamless integration with existing Engram architecture

Usage:
    from engram_cxl import CXLEngram, CXLEngramConfig

    config = CXLEngramConfig(cxl_buffer_size_mb=2048)
    engram = CXLEngram(layer_id=1, cxl_config=config)
================================================================================
"""

import sys
import os

# Add CXL PyTorch expander to path
sys.path.insert(0, '/root/Pooneh/cxl_pytorch_expander/python')

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
from sympy import isprime

# Import CXL tensor management
try:
    from cxl_tensor import CXLTensorManager, CXLTensor, TransferMode
    HAS_CXL = True
except ImportError:
    print("Warning: CXL tensor module not found. Falling back to standard PyTorch.")
    HAS_CXL = False

# Import original Engram components for compatibility
from engram_demo_v1 import (
    EngramConfig, BackBoneConfig,
    CompressedTokenizer, NgramHashMapping, ShortConv,
    find_next_prime, engram_cfg, backbone_config
)


@dataclass
class CXLEngramConfig:
    """Configuration for CXL-enabled Engram"""
    cxl_buffer_size_mb: int = 2048  # Total CXL buffer size
    prefer_p2p_dma: bool = True     # Use P2P DMA when available
    batch_threshold: int = 64       # Min batch size for CXL path
    prefetch_embeddings: bool = True  # Prefetch frequently used embeddings
    cache_hot_embeddings: bool = True  # Keep hot embeddings on GPU
    hot_cache_size_mb: int = 256    # GPU cache size for hot embeddings


class CXLEmbeddingTable:
    """
    Embedding table stored in CXL memory with GPU P2P DMA access.

    This class manages a large embedding table that resides in CXL memory
    and provides efficient batch lookups using GPU peer-to-peer DMA.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.float32,
        cxl_manager: Optional[CXLTensorManager] = None,
        device: torch.device = None
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CXL manager for memory operations
        if cxl_manager is None and HAS_CXL:
            self.cxl_manager = CXLTensorManager.get_instance()
        else:
            self.cxl_manager = cxl_manager

        # Initialize embedding storage
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings and offload to CXL"""
        # Create initial embedding weights (could be loaded from checkpoint)
        embeddings = torch.randn(
            self.num_embeddings,
            self.embedding_dim,
            dtype=self.dtype
        )

        # Scale by sqrt(dim) for better initialization
        embeddings = embeddings * (1.0 / math.sqrt(self.embedding_dim))

        if HAS_CXL and self.cxl_manager is not None:
            # Store in CXL memory
            self.cxl_manager.initialize()

            # Move to GPU first for P2P DMA path
            if torch.cuda.is_available():
                embeddings = embeddings.to(self.device)

            # Offload to CXL
            self.cxl_tensor = CXLTensor(embeddings, self.cxl_manager)
            self.cxl_tensor.offload_to_cxl()

            # Store metadata for retrieval
            self.buffer_id = self.cxl_tensor._metadata.cxl_buffer_id
            self.cxl_offset = self.cxl_tensor._metadata.cxl_offset
            self.row_size_bytes = self.embedding_dim * embeddings.element_size()

            print(f"CXLEmbeddingTable: Offloaded {self.num_embeddings}x{self.embedding_dim} "
                  f"embeddings to CXL (buffer={self.buffer_id})")
        else:
            # Fallback: keep on GPU/CPU
            self.embeddings = embeddings.to(self.device)
            self.cxl_tensor = None
            print(f"CXLEmbeddingTable: Using standard memory for embeddings")

    def load_from_weights(self, weights: torch.Tensor):
        """Load pre-trained weights into CXL storage"""
        assert weights.shape == (self.num_embeddings, self.embedding_dim)

        if HAS_CXL and self.cxl_manager is not None:
            if torch.cuda.is_available():
                weights = weights.to(self.device)

            # Re-offload with new weights
            self.cxl_tensor = CXLTensor(weights, self.cxl_manager)
            self.cxl_tensor.offload_to_cxl()
            self.buffer_id = self.cxl_tensor._metadata.cxl_buffer_id
            self.cxl_offset = self.cxl_tensor._metadata.cxl_offset
        else:
            self.embeddings = weights.to(self.device)

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings by indices.

        Args:
            indices: Tensor of shape [...] containing embedding indices

        Returns:
            Tensor of shape [..., embedding_dim] containing looked-up embeddings
        """
        original_shape = indices.shape
        flat_indices = indices.flatten()

        if HAS_CXL and self.cxl_tensor is not None:
            # CXL path: Fetch embeddings via P2P DMA
            embeddings = self._lookup_from_cxl(flat_indices)
        else:
            # Standard path: Direct GPU lookup
            embeddings = self.embeddings[flat_indices]

        # Reshape to match input
        output_shape = original_shape + (self.embedding_dim,)
        return embeddings.view(*output_shape)

    def _lookup_from_cxl(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Efficient batch lookup from CXL memory.

        Optimizations:
        1. Unique indices to minimize DMA transfers
        2. Contiguous memory access patterns
        3. Batch DMA operations
        """
        # Get unique indices and inverse mapping
        unique_indices, inverse = torch.unique(indices, return_inverse=True)
        num_unique = unique_indices.numel()

        if num_unique == 0:
            return torch.empty(0, self.embedding_dim, dtype=self.dtype, device=self.device)

        # Fetch full embedding table from CXL
        # For large batches, this is more efficient than individual row fetches
        full_embeddings = self.cxl_tensor.to_gpu(device=self.device)

        # Index into the embeddings
        unique_embeddings = full_embeddings[unique_indices]

        # Re-offload the table back to CXL to free GPU memory
        self.cxl_tensor = CXLTensor(full_embeddings, self.cxl_manager)
        self.cxl_tensor.offload_to_cxl()

        # Scatter back to original positions
        result = unique_embeddings[inverse]

        return result

    def _lookup_from_cxl_rowwise(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Row-wise lookup from CXL (for sparse access patterns).

        This is more efficient when only a few rows are needed.
        """
        unique_indices, inverse = torch.unique(indices, return_inverse=True)

        # For small number of unique indices, fetch row by row
        unique_embeddings = []
        for idx in unique_indices.cpu().numpy():
            # Calculate byte offset for this row
            row_offset = self.cxl_offset + (idx * self.row_size_bytes)

            # Fetch single row from CXL
            row = self.cxl_manager._load_from_cxl(
                buffer_id=self.buffer_id,
                offset=row_offset,
                shape=(self.embedding_dim,),
                dtype=self.dtype,
                device=self.device
            )
            unique_embeddings.append(row)

        unique_embeddings = torch.stack(unique_embeddings)
        return unique_embeddings[inverse]


class CXLMultiHeadEmbedding(nn.Module):
    """
    Multi-head embedding with CXL-backed storage.

    Each head has its own embedding table stored in CXL memory.
    Lookups are performed via GPU P2P DMA for efficient memory access.
    """

    def __init__(
        self,
        list_of_N: List[int],
        D: int,
        cxl_config: Optional[CXLEngramConfig] = None,
        device: torch.device = None
    ):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.cxl_config = cxl_config or CXLEngramConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute offsets for combined embedding table
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        # Total vocabulary size across all heads
        total_N = sum(list_of_N)

        # Initialize CXL manager
        if HAS_CXL:
            self.cxl_manager = CXLTensorManager.get_instance()
            self.cxl_manager.initialize(
                buffer_size_mb=self.cxl_config.cxl_buffer_size_mb,
                prefer_p2p=self.cxl_config.prefer_p2p_dma
            )
        else:
            self.cxl_manager = None

        # Create CXL-backed embedding table
        self.embedding_table = CXLEmbeddingTable(
            num_embeddings=total_N,
            embedding_dim=D,
            dtype=torch.float32,
            cxl_manager=self.cxl_manager,
            device=self.device
        )

        # Optional: GPU cache for hot embeddings
        if self.cxl_config.cache_hot_embeddings:
            cache_size = (self.cxl_config.hot_cache_size_mb * 1024 * 1024) // (D * 4)
            self._init_hot_cache(min(cache_size, total_N))
        else:
            self.hot_cache = None

    def _init_hot_cache(self, cache_size: int):
        """Initialize GPU cache for frequently accessed embeddings"""
        self.hot_cache = nn.Embedding(cache_size, self.embedding_dim)
        self.cache_indices = torch.zeros(cache_size, dtype=torch.long)
        self.cache_valid = torch.zeros(cache_size, dtype=torch.bool)
        self.cache_hits = 0
        self.cache_misses = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings for input IDs.

        Args:
            input_ids: Tensor of shape [B, L, num_heads] containing hash indices

        Returns:
            Tensor of shape [B, L, num_heads, embedding_dim]
        """
        # Apply offsets to get global indices
        shifted_input_ids = input_ids + self.offsets.to(input_ids.device)

        # Look up from CXL-backed embedding table
        output = self.embedding_table.lookup(shifted_input_ids)

        return output

    def load_pretrained(self, embedding_weights: torch.Tensor):
        """Load pre-trained embedding weights"""
        self.embedding_table.load_from_weights(embedding_weights)


class CXLEngram(nn.Module):
    """
    CXL-enabled Engram module for conditional memory in LLMs.

    This module stores N-gram embedding tables in CXL memory and uses
    GPU P2P DMA for efficient retrieval during inference.

    Key benefits:
    1. Massive embedding tables can be stored off-GPU
    2. Deterministic hash-based addressing enables efficient DMA patterns
    3. Reduced GPU memory pressure for larger models
    """

    def __init__(
        self,
        layer_id: int,
        engram_config: EngramConfig = None,
        backbone_cfg: BackBoneConfig = None,
        cxl_config: CXLEngramConfig = None
    ):
        super().__init__()
        self.layer_id = layer_id
        self.engram_config = engram_config or engram_cfg
        self.backbone_config = backbone_cfg or backbone_config
        self.cxl_config = cxl_config or CXLEngramConfig()

        # Hash mapping for N-gram lookups
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=self.engram_config.engram_vocab_size,
            max_ngram_size=self.engram_config.max_ngram_size,
            n_embed_per_ngram=self.engram_config.n_embed_per_ngram,
            n_head_per_ngram=self.engram_config.n_head_per_ngram,
            layer_ids=self.engram_config.layer_ids,
            tokenizer_name_or_path=self.engram_config.tokenizer_name_or_path,
            pad_id=self.engram_config.pad_id,
            seed=self.engram_config.seed,
        )

        # CXL-backed multi-head embedding
        self.multi_head_embedding = CXLMultiHeadEmbedding(
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D=self.engram_config.n_embed_per_ngram // self.engram_config.n_head_per_ngram,
            cxl_config=self.cxl_config,
        )

        # Short convolution for temporal processing
        self.short_conv = ShortConv(
            hidden_size=self.backbone_config.hidden_size,
            kernel_size=self.engram_config.kernel_size,
            dilation=self.engram_config.max_ngram_size,
            hc_mult=self.backbone_config.hc_mult,
        )

        # Projection layers
        engram_hidden_size = (self.engram_config.max_ngram_size - 1) * self.engram_config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, self.backbone_config.hidden_size)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, self.backbone_config.hidden_size)
            for _ in range(self.backbone_config.hc_mult)
        ])

        # Normalization layers
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(self.backbone_config.hidden_size)
            for _ in range(self.backbone_config.hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(self.backbone_config.hidden_size)
            for _ in range(self.backbone_config.hc_mult)
        ])

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with CXL-backed embedding lookup.

        Args:
            hidden_states: [B, L, HC_MULT, D] - Hidden states from transformer
            input_ids: [B, L] - Original token IDs

        Returns:
            [B, L, HC_MULT, D] - Engram output to be added to hidden states
        """
        # Compute hash indices for N-gram lookups
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids.cpu().numpy())[self.layer_id]
        ).to(hidden_states.device)

        # Look up embeddings from CXL via P2P DMA
        embeddings = self.multi_head_embedding(hash_input_ids)

        # Flatten heads: [B, L, num_heads, D] -> [B, L, num_heads * D]
        embeddings = embeddings.flatten(start_dim=-2)

        # Compute gating for each hyper-connection
        gates = []
        for hc_idx in range(self.backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)

            # Gated attention score
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates = torch.stack(gates, dim=2)

        # Apply gates and project values
        value = gates * self.value_proj(embeddings).unsqueeze(2)

        # Add short convolution
        output = value + self.short_conv(value)

        return output


class CXLTransformerBlock(nn.Module):
    """Transformer block with optional CXL-enabled Engram"""

    def __init__(
        self,
        layer_id: int,
        cxl_config: CXLEngramConfig = None
    ):
        super().__init__()
        # Placeholder for attention and MoE (as in original)
        self.attn = lambda x: x
        self.moe = lambda x: x

        # CXL-enabled Engram for designated layers
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = CXLEngram(
                layer_id=layer_id,
                cxl_config=cxl_config
            )

    def forward(self, input_ids: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states, input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


def create_cxl_llm(cxl_config: CXLEngramConfig = None) -> List[nn.Module]:
    """Create an LLM with CXL-enabled Engram layers"""
    cxl_config = cxl_config or CXLEngramConfig()

    llm = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        *[CXLTransformerBlock(layer_id=layer_id, cxl_config=cxl_config)
          for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]

    return llm


def benchmark_cxl_engram():
    """Benchmark CXL vs standard Engram performance"""
    import time

    print("=" * 60)
    print("CXL Engram Benchmark")
    print("=" * 60)

    # Configuration
    cxl_config = CXLEngramConfig(
        cxl_buffer_size_mb=1024,
        prefer_p2p_dma=True,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create CXL Engram
    print("\nInitializing CXL Engram...")
    cxl_engram = CXLEngram(layer_id=1, cxl_config=cxl_config)
    cxl_engram = cxl_engram.to(device)

    # Create test data
    batch_size = 4
    seq_len = 128

    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )

    text = "Only Alexander the Great could tame the horse Bucephalus." * 10
    input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=seq_len).input_ids
    input_ids = input_ids.expand(batch_size, -1)

    hidden_states = torch.randn(
        batch_size, input_ids.shape[1], backbone_config.hc_mult, backbone_config.hidden_size,
        device=device
    )

    # Warmup
    print("\nWarmup...")
    for _ in range(3):
        with torch.no_grad():
            _ = cxl_engram(hidden_states, input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print("\nBenchmarking...")
    num_iterations = 20

    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            output = cxl_engram(hidden_states, input_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / num_iterations) * 1000

    print(f"\nResults:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {input_ids.shape[1]}")
    print(f"  Average time per forward: {avg_time_ms:.2f} ms")
    print(f"  Throughput: {(batch_size * num_iterations) / elapsed:.1f} samples/sec")
    print(f"  Output shape: {output.shape}")

    # Get CXL stats
    if HAS_CXL:
        manager = CXLTensorManager.get_instance()
        stats = manager.get_stats()
        print(f"\nCXL Stats:")
        print(f"  Transfer mode: {stats['transfer_mode']}")
        print(f"  Buffers allocated: {stats['num_buffers']}")
        print(f"  Memory used: {stats['current_used_mb']:.2f} MB")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Run demo
    print("CXL-Enabled Engram Demo")
    print("=" * 60)

    # Create CXL LLM
    cxl_config = CXLEngramConfig(cxl_buffer_size_mb=1024)
    LLM = create_cxl_llm(cxl_config)

    # Test forward pass
    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    B, L = input_ids.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Input: {text}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Device: {device}")

    # Move model to device
    for layer in LLM:
        if hasattr(layer, 'to'):
            layer.to(device)

    # Forward pass
    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids.to(device))
            # Mock hyper-connection
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
        elif idx == len(LLM) - 1:
            # Mock hyper-connection aggregation
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    print(f"\n Forward Complete!")
    print(f"Output shape: {output.shape}")

    # Run benchmark
    print("\n")
    benchmark_cxl_engram()
