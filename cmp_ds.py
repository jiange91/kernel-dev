import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
import argparse
import deepspeed.inference.v2.ragged as ds_ragged
from deepspeed.ops.op_builder import RaggedOpsBuilder
from deepspeed.inference.v2.kernels.ragged_ops import BlockedFlashAttn

MAX_Q_LEN = 1
NUM_SEQ = 1
CTX_LEN = 1024
TEST_SEED = 0
BLOCK_SIZE = 64
NUM_LAYERS = 28

NUM_HEADS = 16
HEAD_SIZE = 256


@torch.inference_mode()
def run_ds_flash(
    atoms,
    query,
    kv_caches,
    scale,
    _t,
) -> torch.Tensor:
    output = torch.empty(query.shape, dtype=_t, device='cuda')
    _attn_kernel = BlockedFlashAttn(HEAD_SIZE, _t)
    # warm up
    # forward
    for i in range(NUM_LAYERS):
        kv_cache = kv_caches[i]
        k_cache, v_cache = kv_cache.unbind(dim=2)
        _attn_kernel(output, query, k_cache, v_cache, atoms, scale)
        query = output
    # torch.cuda.cudart().cudaProfilerStart() 
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record() 
    for i in range(NUM_LAYERS):
        kv_cache = kv_caches[i]
        k_cache, v_cache = kv_cache.unbind(dim=2)
        _attn_kernel(output, query, k_cache, v_cache, atoms, scale)
        query = output
    end.record()
    end.synchronize()
    print(f'elapsed time: {start.elapsed_time(end)} ms')
    return output


def test_xformer(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    _t: torch.dtype,
) -> torch.Tensor:
    # prepare tokens
    # q_lens = random.choices(range(1, MAX_Q_LEN+1), k=num_seqs)
    q_lens = [MAX_Q_LEN] * num_seqs
    num_tokens = sum(q_lens)
    
    query = torch.empty(num_tokens, num_heads * head_size, dtype=dtype, device='cuda')
    query.uniform_(-1e-3, 1e-3)
    
    assert CTX_LEN % BLOCK_SIZE == 0
    num_blocks_per_seq = CTX_LEN // BLOCK_SIZE
    kv_caches = torch.empty(size=(NUM_LAYERS, num_blocks_per_seq * num_seqs, BLOCK_SIZE, 2, num_heads, head_size),
                            dtype=dtype,
                            device='cuda')
    kv_caches.uniform_(-1e-3, 1e-3)
    
    scale = float(1.0 / (head_size**0.5))
    
    inf_module = RaggedOpsBuilder().load()
    build_atoms_kernel = inf_module.build_atoms
    atoms: torch.Tensor = torch.empty((256, 8), dtype=torch.int32, device='cpu', pin_memory=True)
    q_block_size = 64
    
    _batch_metadata_storage_shadow = torch.zeros(2, dtype=torch.int32, device='cpu', pin_memory=True)
    _batch_metadata_storage_shadow[0] = num_tokens
    _batch_metadata_storage_shadow[1] = num_seqs
    
    _inflight_seq_descriptors_shadow = torch.zeros((num_seqs, 4),
                                                    dtype=torch.int32,
                                                    device='cpu', pin_memory=True)
    start_token_idx = 0
    for i in range(num_seqs):
        _inflight_seq_descriptors_shadow[i][0] = start_token_idx
        start_token_idx += q_lens[i]
        _inflight_seq_descriptors_shadow[i][1] = q_lens[i]
        _inflight_seq_descriptors_shadow[i][2] = CTX_LEN - q_lens[i]
    
    ids_shape = (
        num_seqs,
        1, # allocation group
        num_blocks_per_seq,
    )
    all_block_ids = torch.zeros(ids_shape, dtype=torch.int32, device='cuda')
    kv_ptrs = torch.zeros((num_seqs), 
                        dtype=torch.int64,
                        device='cpu')
    for i in range(num_seqs):
        kv_cache_ids = all_block_ids[i]
        cache_block_ids = torch.arange(num_blocks_per_seq * i, num_blocks_per_seq * (i+1), dtype=torch.int32, device='cuda')
        kv_cache_ids[0].copy_(cache_block_ids)
        kv_ptrs[i] = kv_cache_ids.data_ptr()
    n_atoms = build_atoms_kernel(atoms, _batch_metadata_storage_shadow,
                                _inflight_seq_descriptors_shadow, kv_ptrs, q_block_size, 64)
    
    # last not count
    output = run_ds_flash(atoms, query, kv_caches, scale, dtype)
    print(output.size())
    
if __name__ == "__main__":
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    # for dtype in [torch.half, torch.bfloat16, torch.float]:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=int, default=None)
    parser.add_argument('-q', type=int, default=None)
    args = parser.parse_args()
    NUM_SEQ = args.s or NUM_SEQ
    MAX_Q_LEN = args.q or MAX_Q_LEN 
    dtype = torch.half
    
    print(f'Testing with '
        f'dtype={dtype}, '
        f'num_seq={NUM_SEQ}, '
        f'max_q_per_seq={MAX_Q_LEN}, '
        f'context_len={CTX_LEN}')
    test_xformer(
        num_seqs=NUM_SEQ,
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        _t=dtype   
    )