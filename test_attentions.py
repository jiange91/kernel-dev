import random
from typing import List, Optional

import torch
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from kernel import attention_ops

MAX_Q_LEN = 4
CTX_LEN = 512
TEST_SEED = 0

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    # print(attn)
    # print(torch.exp(attn - torch.max(attn, dim=-1)[0].T.repeat(1,1,attn.size(-1))))
    attn = torch.softmax(attn, dim=-1)
    # print(attn)
    out = torch.einsum('hqk,khd->qhd', attn, value)
    return out

def ref_multi_token_cached(
	query_start_ids: List[int],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
	dtype: torch.dtype, 
) -> torch.Tensor:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    scale = 1.0 / (head_size**0.5)
    num_queries = len(query_start_ids) - 1
    ref_outputs = []
    for i in range(num_queries):
        start_idx = query_start_ids[i]
        end_idx = query_start_ids[i + 1]
        query_len = end_idx - start_idx
        context_len = int(context_lens[i])
        block_table = block_tables[i]

        # Create attention mask
        attn_mask = torch.triu(torch.ones(query_len, context_len),
                               diagonal=context_len - query_len + 1) * -1e5
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            keys,
            values,
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output

@torch.inference_mode()
def test_milti_token_kernels(
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
    
    context_lens = [x + CTX_LEN - l + 1 for l in q_lens for x in range(l)]
    max_context_len = max(context_lens)
    assert(max_context_len == CTX_LEN)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')
    
    # prepare cache, oversubscript for each seq
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    assert(max_num_blocks_per_seq == CTX_LEN // block_size)
    # each seq will have different space
    block_tables = []
    for i, l in enumerate(q_lens):
        block_table = [[
            x + i * max_num_blocks_per_seq for x in range(max_num_blocks_per_seq)
        ]] * l
        block_tables.extend(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    scale = float(1.0 / (head_size**0.5))
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda") 

    mul_context_lens = [CTX_LEN] * len(q_lens)
    mul_max_context_len = max(mul_context_lens)
    mul_context_lens = torch.tensor(mul_context_lens, dtype=torch.int, device='cuda') 

    # last not count
    query_start_ids = [0]
    for q_len in q_lens:
        query_start_ids.append(query_start_ids[-1] + q_len)
    mul_block_tables = block_tables[query_start_ids[:-1]]
    
    ref_output = ref_multi_token_cached(
        query_start_ids,
		query,
        key_cache,
        value_cache,
        mul_block_tables,
        mul_context_lens,
        dtype
    )

    output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
    )
    # print(f"single: {output}")
    
    mul_output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    attention_ops.multi_token_cached_kv_attention(
        mul_output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        mul_block_tables,
        mul_context_lens,
        torch.tensor(q_lens, dtype=torch.int, device='cuda'),
        torch.tensor(query_start_ids[:-1], dtype=torch.int, device='cuda'),
        max(q_lens),
        block_size,
        mul_max_context_len,
        None
    )
    # print(f'mul: {mul_output}')
    # print(f'ref: {ref_output}')
    # ids = ~torch.isclose(mul_output, ref_output, atol=1e-4, rtol=1e-5)
    # print(ref_output[ids])
    # print(mul_output[ids])
    assert torch.allclose(output, ref_output, atol=1e-4, rtol=1e-5)
    assert torch.allclose(mul_output, ref_output, atol=1e-4, rtol=1e-5)
    
if __name__ == "__main__":
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    # for dtype in [torch.half, torch.bfloat16, torch.float]:
    for dtype in [torch.half]:
        for block_size in [16]:
            for head_size in [256]:
                print(f'Testing multi_query_cached_kv_attention with '
                    f'dtype={dtype}, block_size={block_size}, '
                    f'head_size={head_size}')
                test_milti_token_kernels(
                    num_seqs=128,
                    num_heads=16,
                    head_size=head_size,
                    block_size=block_size,
                    dtype=dtype   
                )