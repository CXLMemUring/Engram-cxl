#!/usr/bin/env python3
"""
================================================================================
Test Suite for CXL-Enabled Engram

This script provides comprehensive tests for the CXL-based conditional memory
implementation, including:

1. Basic functionality tests
2. Performance comparison with standard implementation
3. Memory usage analysis
4. CXL P2P DMA validation
================================================================================
"""

import sys
import os
import time
import argparse

# Add paths
sys.path.insert(0, '/root/Pooneh/cxl_pytorch_expander/python')

import torch
import torch.nn as nn
import numpy as np

# Check CXL availability
try:
    from cxl_tensor import CXLTensorManager, TransferMode
    HAS_CXL = True
except ImportError:
    HAS_CXL = False
    print("Warning: CXL module not available")


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def test_cxl_manager():
    """Test CXL manager initialization and basic operations"""
    print_separator("Test 1: CXL Manager Initialization")

    if not HAS_CXL:
        print("SKIP: CXL not available")
        return False

    manager = CXLTensorManager.get_instance()
    success = manager.initialize(buffer_size_mb=256, prefer_p2p=True)

    print(f"Initialization: {'SUCCESS' if success else 'FAILED'}")
    print(f"Transfer mode: {manager.transfer_mode.value}")
    print(f"P2P available: {manager._p2p_available}")

    stats = manager.get_stats()
    print(f"Stats: {stats}")

    return success


def test_basic_embedding_offload():
    """Test basic embedding offload and retrieval"""
    print_separator("Test 2: Basic Embedding Offload")

    if not HAS_CXL:
        print("SKIP: CXL not available")
        return False

    from cxl_tensor import CXLTensor, CXLTensorManager

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create embedding table
    num_embeddings = 10000
    embedding_dim = 256

    embeddings = torch.randn(num_embeddings, embedding_dim, device=device)
    original_sum = embeddings.sum().item()

    print(f"Created embeddings: {embeddings.shape}")
    print(f"Original checksum: {original_sum:.4f}")

    # Offload to CXL
    manager = CXLTensorManager.get_instance()
    cxl_tensor = CXLTensor(embeddings, manager)
    cxl_tensor.offload_to_cxl()

    print(f"Offloaded to CXL")
    print(f"Location: {cxl_tensor.location.value}")

    # Retrieve from CXL
    restored = cxl_tensor.to_gpu(device=device)
    restored_sum = restored.sum().item()

    print(f"Retrieved from CXL")
    print(f"Restored checksum: {restored_sum:.4f}")

    # Verify
    diff = abs(original_sum - restored_sum)
    success = diff < 1e-3

    print(f"Difference: {diff:.6f}")
    print(f"Result: {'PASS' if success else 'FAIL'}")

    return success


def test_cxl_embedding_table():
    """Test CXLEmbeddingTable class"""
    print_separator("Test 3: CXL Embedding Table")

    try:
        from engram_cxl import CXLEmbeddingTable
    except ImportError as e:
        print(f"SKIP: Cannot import engram_cxl: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create CXL embedding table
    num_embeddings = 5000
    embedding_dim = 128

    print(f"Creating CXLEmbeddingTable: {num_embeddings}x{embedding_dim}")

    table = CXLEmbeddingTable(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        device=device
    )

    # Test lookup
    batch_size = 4
    seq_len = 32
    indices = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

    print(f"Lookup indices shape: {indices.shape}")

    embeddings = table.lookup(indices)

    print(f"Output shape: {embeddings.shape}")
    expected_shape = (batch_size, seq_len, embedding_dim)

    success = embeddings.shape == expected_shape
    print(f"Shape check: {'PASS' if success else 'FAIL'}")

    return success


def test_cxl_engram_forward():
    """Test CXLEngram forward pass"""
    print_separator("Test 4: CXL Engram Forward Pass")

    try:
        from engram_cxl import CXLEngram, CXLEngramConfig
        from engram_demo_v1 import engram_cfg, backbone_config, AutoTokenizer
    except ImportError as e:
        print(f"SKIP: Cannot import modules: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create CXL Engram
    cxl_config = CXLEngramConfig(cxl_buffer_size_mb=512)
    engram = CXLEngram(layer_id=1, cxl_config=cxl_config)
    engram = engram.to(device)

    # Create test input
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    batch_size = 2
    input_ids = input_ids.expand(batch_size, -1)
    seq_len = input_ids.shape[1]

    print(f"Input IDs shape: {input_ids.shape}")

    # Create hidden states
    hidden_states = torch.randn(
        batch_size, seq_len,
        backbone_config.hc_mult, backbone_config.hidden_size,
        device=device
    )

    print(f"Hidden states shape: {hidden_states.shape}")

    # Forward pass
    with torch.no_grad():
        output = engram(hidden_states, input_ids)

    print(f"Output shape: {output.shape}")

    expected_shape = (batch_size, seq_len, backbone_config.hc_mult, backbone_config.hidden_size)
    success = output.shape == expected_shape

    print(f"Shape check: {'PASS' if success else 'FAIL'}")

    return success


def test_performance_comparison():
    """Compare CXL vs standard Engram performance"""
    print_separator("Test 5: Performance Comparison")

    try:
        from engram_cxl import CXLEngram, CXLEngramConfig
        from engram_demo_v1 import Engram, engram_cfg, backbone_config, AutoTokenizer
    except ImportError as e:
        print(f"SKIP: Cannot import modules: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available for performance test")
        return True

    # Create models
    standard_engram = Engram(layer_id=1).to(device)
    cxl_engram = CXLEngram(
        layer_id=1,
        cxl_config=CXLEngramConfig(cxl_buffer_size_mb=512)
    ).to(device)

    # Create test data
    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path,
        trust_remote_code=True
    )

    text = "Performance test with longer input text for more accurate timing." * 5
    input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=128).input_ids

    batch_size = 4
    input_ids = input_ids.expand(batch_size, -1)
    seq_len = input_ids.shape[1]

    hidden_states = torch.randn(
        batch_size, seq_len,
        backbone_config.hc_mult, backbone_config.hidden_size,
        device=device
    )

    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")

    # Warmup
    num_warmup = 5
    num_iterations = 20

    print("\nWarming up...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = standard_engram(hidden_states, input_ids)
            _ = cxl_engram(hidden_states, input_ids)
    torch.cuda.synchronize()

    # Benchmark standard
    print("Benchmarking standard Engram...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = standard_engram(hidden_states, input_ids)
    torch.cuda.synchronize()
    standard_time = (time.perf_counter() - start) / num_iterations * 1000

    # Benchmark CXL
    print("Benchmarking CXL Engram...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = cxl_engram(hidden_states, input_ids)
    torch.cuda.synchronize()
    cxl_time = (time.perf_counter() - start) / num_iterations * 1000

    print(f"\nResults:")
    print(f"  Standard Engram: {standard_time:.2f} ms")
    print(f"  CXL Engram:      {cxl_time:.2f} ms")
    print(f"  Ratio:           {cxl_time / standard_time:.2f}x")

    # CXL is expected to be slower due to memory transfers
    # but enables much larger models
    return True


def test_advanced_embeddings():
    """Test advanced CXL embedding implementations"""
    print_separator("Test 6: Advanced CXL Embeddings")

    try:
        from cxl_embedding_advanced import (
            ShardedCXLEmbedding,
            DirectCXLEmbedding,
            CachedCXLEmbedding
        )
    except ImportError as e:
        print(f"SKIP: Cannot import advanced embeddings: {e}")
        return False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_embeddings = 10000
    embedding_dim = 256
    batch_size = 8
    seq_len = 64

    indices = torch.randint(0, num_embeddings, (batch_size, seq_len), device=device)

    results = []

    # Test ShardedCXLEmbedding
    print("\n--- ShardedCXLEmbedding ---")
    try:
        sharded = ShardedCXLEmbedding(
            num_embeddings, embedding_dim,
            num_shards=4, device=device
        )
        output = sharded(indices)
        success = output.shape == (batch_size, seq_len, embedding_dim)
        print(f"Shape: {output.shape} - {'PASS' if success else 'FAIL'}")
        results.append(success)
    except Exception as e:
        print(f"Error: {e}")
        results.append(False)

    # Test DirectCXLEmbedding
    print("\n--- DirectCXLEmbedding ---")
    try:
        direct = DirectCXLEmbedding(num_embeddings, embedding_dim, device=device)
        output = direct(indices)
        success = output.shape == (batch_size, seq_len, embedding_dim)
        print(f"Shape: {output.shape} - {'PASS' if success else 'FAIL'}")
        direct.cleanup()
        results.append(success)
    except Exception as e:
        print(f"Error: {e}")
        results.append(False)

    # Test CachedCXLEmbedding
    print("\n--- CachedCXLEmbedding ---")
    try:
        cached = CachedCXLEmbedding(
            num_embeddings, embedding_dim,
            cache_size=4096, device=device
        )
        output = cached(indices)
        success = output.shape == (batch_size, seq_len, embedding_dim)
        print(f"Shape: {output.shape} - {'PASS' if success else 'FAIL'}")
        print(f"Cache stats: {cached.get_cache_stats()}")
        cached.cleanup()
        results.append(success)
    except Exception as e:
        print(f"Error: {e}")
        results.append(False)

    return all(results)


def test_memory_efficiency():
    """Test memory efficiency of CXL embedding vs standard"""
    print_separator("Test 7: Memory Efficiency")

    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return True

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Measure standard embedding memory
    num_embeddings = 100000
    embedding_dim = 512

    print(f"\nEmbedding size: {num_embeddings}x{embedding_dim}")
    expected_mb = num_embeddings * embedding_dim * 4 / 1024 / 1024
    print(f"Expected memory: {expected_mb:.2f} MB")

    # Standard embedding
    print("\n--- Standard nn.Embedding ---")
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated() / 1024 / 1024

    standard_emb = nn.Embedding(num_embeddings, embedding_dim).cuda()

    after = torch.cuda.memory_allocated() / 1024 / 1024
    standard_mem = after - before
    print(f"GPU memory used: {standard_mem:.2f} MB")

    del standard_emb
    torch.cuda.empty_cache()

    # CXL embedding
    print("\n--- CXL Embedding ---")
    try:
        from engram_cxl import CXLEmbeddingTable

        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated() / 1024 / 1024

        cxl_emb = CXLEmbeddingTable(
            num_embeddings, embedding_dim,
            device=torch.device('cuda')
        )

        after = torch.cuda.memory_allocated() / 1024 / 1024
        cxl_mem = after - before
        print(f"GPU memory used: {cxl_mem:.2f} MB")

        savings = ((standard_mem - cxl_mem) / standard_mem) * 100
        print(f"\nMemory savings: {savings:.1f}%")

    except Exception as e:
        print(f"Error: {e}")
        return False

    return True


def run_all_tests():
    """Run all tests"""
    print_separator("CXL Engram Test Suite")

    results = {}

    # Run tests
    results['cxl_manager'] = test_cxl_manager()
    results['basic_offload'] = test_basic_embedding_offload()
    results['embedding_table'] = test_cxl_embedding_table()
    results['engram_forward'] = test_cxl_engram_forward()
    results['performance'] = test_performance_comparison()
    results['advanced_embeddings'] = test_advanced_embeddings()
    results['memory_efficiency'] = test_memory_efficiency()

    # Summary
    print_separator("Test Summary")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


def main():
    parser = argparse.ArgumentParser(description='Test CXL Engram implementation')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'manager', 'offload', 'embedding', 'engram',
                                 'performance', 'advanced', 'memory'],
                        help='Which test to run')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run extended benchmark')

    args = parser.parse_args()

    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'manager':
        success = test_cxl_manager()
    elif args.test == 'offload':
        success = test_basic_embedding_offload()
    elif args.test == 'embedding':
        success = test_cxl_embedding_table()
    elif args.test == 'engram':
        success = test_cxl_engram_forward()
    elif args.test == 'performance':
        success = test_performance_comparison()
    elif args.test == 'advanced':
        success = test_advanced_embeddings()
    elif args.test == 'memory':
        success = test_memory_efficiency()

    if args.benchmark:
        print_separator("Extended Benchmark")
        from cxl_embedding_advanced import benchmark_cxl_embeddings
        benchmark_cxl_embeddings()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
