import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
import argparse
import time

from kernel import attention_ops

NUM_SEQ = 1
QS_PER_TB = 4
MAX_Q_LEN = 4
CTX_LEN = 1024
TEST_SEED = 0

assert MAX_Q_LEN % QS_PER_TB == 0


@torch.inference_mode()
def run_multi_query_att(
    query,
    key_cache,
    value_cache,
    head_mapping,
    scale,
    mul_block_tables,
    mul_context_lens,
    num_queries_per_seq,
    seq_start_idxs,
    max_queries,
    block_size,
    mul_max_context_len,
    dtype
):
    num_tokens = query.size(0)
    num_heads = query.size(1)
    head_size = query.size(2)
 
    mul_output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    # warm up
    for _ in range(10):
        attention_ops.multi_token_cached_kv_attention(
            mul_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            mul_block_tables,
            mul_context_lens,
            num_queries_per_seq,
            seq_start_idxs,
            max_queries,
            block_size,
            mul_max_context_len,
            None
        )
        mul_output, query = query, mul_output
    # torch.cuda.cudart().cudaProfilerStart() 
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record() 
    for _ in range(28):
        attention_ops.multi_token_cached_kv_attention(
            mul_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            mul_block_tables,
            mul_context_lens,
            num_queries_per_seq,
            seq_start_idxs,
            max_queries,
            block_size,
            mul_max_context_len,
            None
        )
        mul_output, query = query, mul_output
    end.record()
    end.synchronize()
    print(f'elapsed time: {start.elapsed_time(end)} ms')
    return query


def test_milti_token_multi_query_kernel(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # prepare tokens
    # q_lens = random.choices(range(1, MAX_Q_LEN+1), k=num_seqs)
    q_lens = [MAX_Q_LEN] * num_seqs
    num_tokens = sum(q_lens)
    groups = [(l + QS_PER_TB - 1) // QS_PER_TB for l in q_lens]
    grouped_q_lens = [min(l-qs, QS_PER_TB) for l in q_lens for qs in range(0, l, QS_PER_TB)]
    print(groups, grouped_q_lens)
    
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    query.uniform_(-1e-3, 1e-3)
    
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    num_blocks = num_seqs * CTX_LEN // block_size
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=(num_blocks, *key_block_shape),
                            dtype=dtype,
                            device='cuda')
    key_cache.uniform_(-1e-3, 1e-3)
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                              dtype=dtype,
                              device='cuda')
    value_cache.uniform_(-1e-3, 1e-3)
    context_lens = []
    g_idx = 0
    for l, g in zip(q_lens, groups):
        seen_tokens = CTX_LEN - l
        for i in range(g):
            context_lens.append(seen_tokens + grouped_q_lens[g_idx + i])
            seen_tokens += grouped_q_lens[g_idx + i]
        g_idx += g
    print(context_lens)
    assert len(context_lens) == sum(groups)
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')
    
    # prepare cache, oversubscript for each seq
    max_num_blocks_per_seq = (CTX_LEN + block_size - 1) // block_size
    assert(max_num_blocks_per_seq == CTX_LEN // block_size)
    # each seq will have different space
    block_tables = []
    for i, l in enumerate(q_lens):
        block_table = [[
            x + i * max_num_blocks_per_seq for x in range(max_num_blocks_per_seq)
        ]] * groups[i]
        block_tables.extend(block_table)
    assert len(block_tables) == sum(groups)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    scale = float(1.0 / (head_size**0.5))
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda") 
    
    # last not count
    query_start_ids = [0]
    for q_len in grouped_q_lens:
        query_start_ids.append(query_start_ids[-1] + q_len)
    output = run_multi_query_att(
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        torch.tensor(grouped_q_lens, dtype=torch.int, device='cuda'),
        torch.tensor(query_start_ids[:-1], dtype=torch.int, device='cuda'),
        max(grouped_q_lens),
        block_size,
        max_context_len,
        dtype
    )
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
    test_milti_token_multi_query_kernel(
        num_seqs=NUM_SEQ,
        num_heads=16,
        head_size=256,
        block_size=16,
        dtype=dtype   
    )