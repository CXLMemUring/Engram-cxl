"""
================================================================================
Advanced CXL Embedding with Optimized P2P DMA

This module provides advanced embedding table management using CXL memory
with optimized GPU P2P DMA operations:

1. Sharded storage: Large tables split across multiple CXL buffers
2. Row-wise DMA: Fetch only needed rows for sparse access patterns
3. Timed transfers: Kernel-level latency measurement
4. Async prefetch: Overlap DMA with computation
================================================================================
"""

import sys
sys.path.insert(0, '/root/Pooneh/cxl_pytorch_expander/python')

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
import torch.nn as nn

try:
    from cxl_tensor import CXLTensorManager, CXLTensor, TransferMode
    import cxl_memory
    HAS_CXL = True
except ImportError:
    HAS_CXL = False
    print("Warning: CXL modules not available")


@dataclass
class CXLEmbeddingStats:
    """Statistics for CXL embedding operations"""
    total_lookups: int = 0
    unique_lookups: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_dma_time_ns: int = 0
    total_rows_fetched: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_dma_latency_us(self) -> float:
        return (self.total_dma_time_ns / self.total_rows_fetched / 1000) if self.total_rows_fetched > 0 else 0.0


class ShardedCXLEmbedding(nn.Module):
    """
    Sharded embedding table stored across multiple CXL buffers.

    This implementation splits large embedding tables into shards,
    each stored in its own CXL buffer. This enables:
    1. Support for very large embedding tables (>16GB)
    2. Parallel DMA from different shards
    3. Better cache locality within shards
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        num_shards: int = 8,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        buffer_size_mb: int = 256
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_shards = num_shards
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Calculate shard sizes
        self.shard_size = (num_embeddings + num_shards - 1) // num_shards
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.row_size_bytes = embedding_dim * self.element_size

        # Statistics
        self.stats = CXLEmbeddingStats()

        # Initialize CXL manager
        if HAS_CXL:
            self.cxl_manager = CXLTensorManager.get_instance()
            self.cxl_manager.initialize(
                buffer_size_mb=buffer_size_mb,
                prefer_p2p=True
            )
        else:
            self.cxl_manager = None

        # Initialize shards
        self._init_shards()

    def _init_shards(self):
        """Initialize embedding shards in CXL memory"""
        self.shards = []  # List of (buffer_id, offset, num_rows) tuples
        self.shard_embeddings = []  # For non-CXL fallback

        for shard_idx in range(self.num_shards):
            start_idx = shard_idx * self.shard_size
            end_idx = min(start_idx + self.shard_size, self.num_embeddings)
            num_rows = end_idx - start_idx

            if num_rows <= 0:
                break

            # Create shard embeddings
            shard_data = torch.randn(
                num_rows, self.embedding_dim,
                dtype=self.dtype
            ) * (1.0 / math.sqrt(self.embedding_dim))

            if HAS_CXL and self.cxl_manager is not None:
                # Move to GPU for P2P path
                if torch.cuda.is_available():
                    shard_data = shard_data.to(self.device)

                # Store in CXL
                buffer_id, offset, shape, dtype, latency = \
                    self.cxl_manager.direct_gpu_to_cxl_timed(shard_data)

                self.shards.append({
                    'buffer_id': buffer_id,
                    'offset': offset,
                    'num_rows': num_rows,
                    'start_idx': start_idx,
                    'latency_ns': latency
                })

                print(f"  Shard {shard_idx}: {num_rows} rows, "
                      f"offset={offset}, latency={latency/1e6:.2f}ms")
            else:
                # Fallback: keep on GPU
                self.shard_embeddings.append(shard_data.to(self.device))
                self.shards.append({
                    'buffer_id': None,
                    'num_rows': num_rows,
                    'start_idx': start_idx,
                })

    def _get_shard_for_index(self, idx: int) -> Tuple[int, int]:
        """Get shard index and local index for a global index"""
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        return shard_idx, local_idx

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings by indices.

        Args:
            indices: Tensor of shape [...] with embedding indices

        Returns:
            Tensor of shape [..., embedding_dim]
        """
        original_shape = indices.shape
        flat_indices = indices.flatten().long()

        # Track stats
        self.stats.total_lookups += flat_indices.numel()

        if HAS_CXL and self.cxl_manager is not None:
            embeddings = self._lookup_from_cxl(flat_indices)
        else:
            embeddings = self._lookup_fallback(flat_indices)

        return embeddings.view(*original_shape, self.embedding_dim)

    def _lookup_from_cxl(self, indices: torch.Tensor) -> torch.Tensor:
        """Optimized CXL lookup with batch shard fetching"""
        # Get unique indices to minimize DMA
        unique_indices, inverse = torch.unique(indices, return_inverse=True)
        self.stats.unique_lookups += unique_indices.numel()

        # Group indices by shard
        shard_groups = {}
        unique_indices_cpu = unique_indices.cpu().numpy()

        for local_pos, global_idx in enumerate(unique_indices_cpu):
            shard_idx, local_idx = self._get_shard_for_index(global_idx)
            if shard_idx not in shard_groups:
                shard_groups[shard_idx] = ([], [])
            shard_groups[shard_idx][0].append(local_idx)
            shard_groups[shard_idx][1].append(local_pos)

        # Fetch embeddings from each shard
        unique_embeddings = torch.empty(
            unique_indices.numel(), self.embedding_dim,
            dtype=self.dtype, device=self.device
        )

        total_dma_time = 0

        for shard_idx, (local_indices, positions) in shard_groups.items():
            shard_info = self.shards[shard_idx]

            # Fetch entire shard from CXL (more efficient than row-by-row)
            shard_data, latency = self.cxl_manager.direct_cxl_to_gpu_timed(
                buffer_id=shard_info['buffer_id'],
                offset=shard_info['offset'],
                shape=(shard_info['num_rows'], self.embedding_dim),
                dtype=self.dtype,
                device=self.device
            )
            total_dma_time += latency

            # Extract needed rows
            local_indices_tensor = torch.tensor(local_indices, device=self.device)
            positions_tensor = torch.tensor(positions, device=self.device)

            unique_embeddings[positions_tensor] = shard_data[local_indices_tensor]

            # Re-offload shard
            self.cxl_manager.direct_gpu_to_cxl(shard_data)

            self.stats.total_rows_fetched += shard_info['num_rows']

        self.stats.total_dma_time_ns += total_dma_time

        # Scatter to original positions
        return unique_embeddings[inverse]

    def _lookup_fallback(self, indices: torch.Tensor) -> torch.Tensor:
        """Fallback lookup without CXL"""
        unique_indices, inverse = torch.unique(indices, return_inverse=True)

        # Allocate output
        unique_embeddings = torch.empty(
            unique_indices.numel(), self.embedding_dim,
            dtype=self.dtype, device=self.device
        )

        # Group by shard and fetch
        for global_idx_pos, global_idx in enumerate(unique_indices.cpu().numpy()):
            shard_idx, local_idx = self._get_shard_for_index(global_idx)
            unique_embeddings[global_idx_pos] = self.shard_embeddings[shard_idx][local_idx]

        return unique_embeddings[inverse]

    def get_stats(self) -> Dict:
        """Get embedding operation statistics"""
        return {
            'total_lookups': self.stats.total_lookups,
            'unique_lookups': self.stats.unique_lookups,
            'dedup_ratio': self.stats.unique_lookups / self.stats.total_lookups if self.stats.total_lookups > 0 else 0,
            'avg_dma_latency_us': self.stats.avg_dma_latency_us,
            'total_dma_time_ms': self.stats.total_dma_time_ns / 1e6,
            'rows_fetched': self.stats.total_rows_fetched,
        }


class DirectCXLEmbedding(nn.Module):
    """
    Direct GPU<->CXL embedding using low-level P2P DMA APIs.

    This class provides the most efficient CXL embedding access by:
    1. Pre-registering embedding buffer with CUDA
    2. Using direct gpu_to_cxl/cxl_to_gpu for transfers
    3. Minimizing Python overhead with batch operations
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.row_size = embedding_dim * self.element_size
        self.total_size = num_embeddings * self.row_size

        # Initialize CXL buffer
        self._init_cxl_buffer()

    def _init_cxl_buffer(self):
        """Initialize CXL buffer for embeddings"""
        if not HAS_CXL:
            # Fallback to standard embedding
            self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
            self.buffer_id = None
            return

        # Initialize CXL
        cxl_memory.init()

        # Allocate CXL buffer
        self.buffer_id = cxl_memory.alloc_buffer(self.total_size)
        self.buffer_ptr = cxl_memory.get_buffer_ptr(self.buffer_id)

        print(f"DirectCXLEmbedding: Allocated {self.total_size / 1024 / 1024:.2f} MB "
              f"in CXL buffer {self.buffer_id}")

        # Initialize with random embeddings
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize embedding weights in CXL"""
        # Create initial weights on GPU
        weights = torch.randn(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device=self.device
        ) * (1.0 / math.sqrt(self.embedding_dim))

        # Transfer to CXL
        torch.cuda.synchronize()
        gpu_ptr = weights.data_ptr()
        cxl_memory.gpu_to_cxl(self.buffer_id, gpu_ptr, 0, self.total_size)

        # Free GPU memory
        del weights
        torch.cuda.empty_cache()

    def load_weights(self, weights: torch.Tensor):
        """Load pre-trained weights into CXL"""
        assert weights.shape == (self.num_embeddings, self.embedding_dim)

        if self.buffer_id is None:
            self.embedding.weight.data.copy_(weights)
            return

        weights = weights.to(self.device).contiguous()
        torch.cuda.synchronize()

        cxl_memory.gpu_to_cxl(
            self.buffer_id,
            weights.data_ptr(),
            0,
            self.total_size
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up embeddings via CXL P2P DMA"""
        if self.buffer_id is None:
            return self.embedding(indices)

        original_shape = indices.shape
        flat_indices = indices.flatten().long()

        # Fetch full table (for now - can optimize for sparse access)
        embeddings = torch.empty(
            self.num_embeddings, self.embedding_dim,
            dtype=self.dtype, device=self.device
        )

        torch.cuda.synchronize()
        cxl_memory.cxl_to_gpu(
            self.buffer_id,
            embeddings.data_ptr(),
            0,
            self.total_size
        )
        torch.cuda.synchronize()

        # Index and return
        result = embeddings[flat_indices]

        return result.view(*original_shape, self.embedding_dim)

    def forward_sparse(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Sparse lookup - fetch only needed rows.

        More efficient when few unique indices are accessed.
        """
        if self.buffer_id is None:
            return self.embedding(indices)

        original_shape = indices.shape
        flat_indices = indices.flatten().long()

        # Get unique indices
        unique_indices, inverse = torch.unique(flat_indices, return_inverse=True)
        num_unique = unique_indices.numel()

        # Allocate output
        unique_embeddings = torch.empty(
            num_unique, self.embedding_dim,
            dtype=self.dtype, device=self.device
        )

        # Fetch each unique row
        unique_indices_cpu = unique_indices.cpu().numpy()

        for i, idx in enumerate(unique_indices_cpu):
            # Calculate byte offset for this row
            row_offset = int(idx) * self.row_size

            # Allocate single row
            row_data = torch.empty(
                self.embedding_dim,
                dtype=self.dtype, device=self.device
            )

            torch.cuda.synchronize()
            cxl_memory.cxl_to_gpu(
                self.buffer_id,
                row_data.data_ptr(),
                row_offset,
                self.row_size
            )

            unique_embeddings[i] = row_data

        torch.cuda.synchronize()

        # Scatter to original positions
        result = unique_embeddings[inverse]
        return result.view(*original_shape, self.embedding_dim)

    def cleanup(self):
        """Free CXL buffer"""
        if self.buffer_id is not None:
            torch.cuda.synchronize()
            cxl_memory.free_buffer(self.buffer_id)
            self.buffer_id = None


class CachedCXLEmbedding(nn.Module):
    """
    CXL embedding with GPU-side LRU cache.

    Frequently accessed embeddings are cached on GPU to avoid
    repeated CXL transfers.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        cache_size: int = 8192,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.cache_size = min(cache_size, num_embeddings)
        self.dtype = dtype
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # CXL-backed embedding for main storage
        self.cxl_embedding = DirectCXLEmbedding(
            num_embeddings, embedding_dim, dtype, device
        )

        # GPU cache
        self.register_buffer(
            'cache_data',
            torch.zeros(self.cache_size, embedding_dim, dtype=dtype)
        )
        self.register_buffer(
            'cache_indices',
            torch.full((self.cache_size,), -1, dtype=torch.long)
        )
        self.register_buffer(
            'cache_ages',
            torch.zeros(self.cache_size, dtype=torch.long)
        )

        self.age_counter = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Look up with caching"""
        original_shape = indices.shape
        flat_indices = indices.flatten().long()

        # Check cache
        unique_indices, inverse = torch.unique(flat_indices, return_inverse=True)

        # Find cached and uncached indices
        cached_mask = torch.zeros(unique_indices.numel(), dtype=torch.bool, device=self.device)
        cached_positions = torch.zeros(unique_indices.numel(), dtype=torch.long, device=self.device)

        for i, idx in enumerate(unique_indices):
            cache_pos = (self.cache_indices == idx).nonzero(as_tuple=True)[0]
            if len(cache_pos) > 0:
                cached_mask[i] = True
                cached_positions[i] = cache_pos[0]
                self.cache_ages[cache_pos[0]] = self.age_counter
                self.cache_hits += 1
            else:
                self.cache_misses += 1

        self.age_counter += 1

        # Allocate output
        unique_embeddings = torch.empty(
            unique_indices.numel(), self.embedding_dim,
            dtype=self.dtype, device=self.device
        )

        # Fill from cache
        if cached_mask.any():
            cached_indices = cached_mask.nonzero(as_tuple=True)[0]
            for idx in cached_indices:
                unique_embeddings[idx] = self.cache_data[cached_positions[idx]]

        # Fetch uncached from CXL
        uncached_mask = ~cached_mask
        if uncached_mask.any():
            uncached_positions = uncached_mask.nonzero(as_tuple=True)[0]
            uncached_indices = unique_indices[uncached_mask]

            # Fetch from CXL
            fetched = self.cxl_embedding.forward_sparse(uncached_indices)
            unique_embeddings[uncached_positions] = fetched

            # Update cache (LRU eviction)
            self._update_cache(uncached_indices, fetched)

        # Scatter to original positions
        result = unique_embeddings[inverse]
        return result.view(*original_shape, self.embedding_dim)

    def _update_cache(self, indices: torch.Tensor, embeddings: torch.Tensor):
        """Update cache with LRU eviction"""
        for i, (idx, emb) in enumerate(zip(indices, embeddings)):
            # Find LRU slot
            lru_pos = self.cache_ages.argmin()

            # Update cache
            self.cache_data[lru_pos] = emb
            self.cache_indices[lru_pos] = idx
            self.cache_ages[lru_pos] = self.age_counter + i

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.cache_hits + self.cache_misses
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / total if total > 0 else 0.0,
            'cache_size': self.cache_size,
            'cache_utilization': (self.cache_indices >= 0).sum().item() / self.cache_size
        }

    def cleanup(self):
        """Cleanup resources"""
        self.cxl_embedding.cleanup()


def benchmark_cxl_embeddings():
    """Benchmark different CXL embedding implementations"""
    print("=" * 70)
    print("CXL Embedding Benchmark")
    print("=" * 70)

    if not HAS_CXL:
        print("CXL not available, skipping benchmark")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test configurations
    num_embeddings = 100000
    embedding_dim = 512
    batch_size = 32
    seq_len = 128
    num_warmup = 5
    num_iterations = 20

    # Create test indices (simulating hash lookups)
    indices = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

    print(f"\nConfiguration:")
    print(f"  Embeddings: {num_embeddings} x {embedding_dim}")
    print(f"  Batch: {batch_size} x {seq_len}")
    print(f"  Device: {device}")

    # Test 1: Sharded CXL Embedding
    print(f"\n--- ShardedCXLEmbedding (8 shards) ---")
    sharded = ShardedCXLEmbedding(
        num_embeddings, embedding_dim,
        num_shards=8, device=device
    )

    # Warmup
    for _ in range(num_warmup):
        _ = sharded(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = sharded(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  Avg time: {(elapsed / num_iterations) * 1000:.2f} ms")
    print(f"  Stats: {sharded.get_stats()}")

    # Test 2: Direct CXL Embedding
    print(f"\n--- DirectCXLEmbedding ---")
    direct = DirectCXLEmbedding(num_embeddings, embedding_dim, device=device)

    # Warmup
    for _ in range(num_warmup):
        _ = direct(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark full table fetch
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = direct(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_full = time.perf_counter() - start

    print(f"  Full fetch avg: {(elapsed_full / num_iterations) * 1000:.2f} ms")

    # Benchmark sparse fetch
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = direct.forward_sparse(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_sparse = time.perf_counter() - start

    print(f"  Sparse fetch avg: {(elapsed_sparse / num_iterations) * 1000:.2f} ms")

    # Cleanup
    direct.cleanup()

    # Test 3: Cached CXL Embedding
    print(f"\n--- CachedCXLEmbedding (cache=8192) ---")
    cached = CachedCXLEmbedding(
        num_embeddings, embedding_dim,
        cache_size=8192, device=device
    )

    # Warmup (also warms cache)
    for _ in range(num_warmup):
        _ = cached(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reset stats after warmup
    cached.cache_hits = 0
    cached.cache_misses = 0

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iterations):
        output = cached(indices)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"  Avg time: {(elapsed / num_iterations) * 1000:.2f} ms")
    print(f"  Cache stats: {cached.get_cache_stats()}")

    cached.cleanup()

    print("\n" + "=" * 70)


if __name__ == '__main__':
    benchmark_cxl_embeddings()
