import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
import argparse
import time


MAX_Q_LEN = 4
NUM_SEQ = 1
CTX_LEN = 1024
TEST_SEED = 0


@torch.inference_mode()
def run_xformer(
    query,
    key,
    value,
    scale,
    q_lens,
    kv_lens,
):
    attention_bias = BlockDiagonalCausalMask.from_seqlens(q_lens, kv_lens)
    # warm up
    query = query.unsqueeze(0)
    for _ in range(10):
        out = xops.memory_efficient_attention_forward(
            query,
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attention_bias,
            p=0.0,
            scale=scale,
        )
        query = out
    # torch.cuda.cudart().cudaProfilerStart() 
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record() 
    for _ in range(28):
        out = xops.memory_efficient_attention_forward(
            query,
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attention_bias,
            p=0.0,
            scale=scale,
        )
        query = out
    end.record()
    end.synchronize()
    print(f'elapsed time: {start.elapsed_time(end)} ms')
    return out


def test_xformer(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # prepare tokens
    # q_lens = random.choices(range(1, MAX_Q_LEN+1), k=num_seqs)
    q_lens = [MAX_Q_LEN] * num_seqs
    num_tokens = sum(q_lens)
    
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype, device='cuda')
    query.uniform_(-1e-3, 1e-3)
    
    kv = torch.empty(CTX_LEN * num_seqs, 2, num_heads, head_size, dtype=dtype, device='cuda')
    key, val = kv.unbind(dim=1)
    key.uniform_(-1e-3, 1e-3)
    val.uniform_(-1e-3, 1e-3)
    
    scale = float(1.0 / (head_size**0.5))

    # last not count
    output = run_xformer(
        query,
        key,
        val,
        scale,
        q_lens,
        [CTX_LEN] * num_seqs,
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
    test_xformer(
        num_seqs=NUM_SEQ,
        num_heads=16,
        head_size=256,
        dtype=dtype   
    )