#include <torch/extension.h>
#include <c10/util/Optional.h>

void single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);


void multi_token_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& head_mapping,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  torch::Tensor& num_queries_per_seq,    // [num_seqs]
  torch::Tensor& seq_start_idxs,  // [num_seqs]
  int max_queries_per_seq,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_cached_kv_attention",
    &single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors");
  m.def(
    "multi_token_cached_kv_attention",
    &multi_token_cached_kv_attention,
    "Compute the attention between multiple queries and the cached key/value tensors");
}
