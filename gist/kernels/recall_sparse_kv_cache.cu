#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <tuple>
#include <vector>
#include <cub/cub.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
// #define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void compact_kv_kernel(
    const scalar_t* __restrict__ in,   // [outer, S, inner]
    scalar_t* __restrict__ out,        // [outer, S, inner], only first new_len positions in S are valid
    const uint8_t* __restrict__ keep_mask,
    const int32_t* __restrict__ prefix,
    int64_t outer,
    int64_t S,
    int64_t inner
) {
    int64_t total = outer * S * inner;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int64_t tmp = idx;
    int64_t inner_idx = tmp % inner;
    tmp /= inner;
    int64_t s = tmp % S;
    int64_t outer_idx = tmp / S;

    if (keep_mask[s]) {
        int32_t dst_s = prefix[s];
        int64_t in_offset = (outer_idx * S + s) * inner + inner_idx;
        int64_t out_offset = (outer_idx * S + dst_s) * inner + inner_idx;
        out[out_offset] = in[in_offset];
    }
}

__global__ void init_tail_kernel(
    uint8_t* keep_mask,
    int64_t G,
    int64_t S
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= S) return;
    keep_mask[idx] = (idx >= G) ? 1 : 0;
}

template <typename index_t>
__global__ void mark_recall_kernel(
    const index_t* __restrict__ recall_doc_set,
    uint8_t* __restrict__ keep_mask,
    int64_t num_recall,
    int64_t content_len,
    int64_t G
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_recall) return;

    int64_t u = static_cast<int64_t>(recall_doc_set[idx]) - content_len;
    if (u >= 0 && u < G) {
        keep_mask[u] = 1;   // idempotent write
    }
}

__global__ void compute_new_len_kernel(
    const uint8_t* keep_mask,
    const int32_t* prefix,
    int64_t S,
    int32_t* out_new_len
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (S == 0) {
            *out_new_len = 0;
        } else {
            *out_new_len = prefix[S - 1] + static_cast<int32_t>(keep_mask[S - 1]);
        }
    }
}

static inline std::tuple<int64_t, int64_t> flatten_outer_inner(
    const torch::Tensor& x
) {
    // compact on dim=3
    TORCH_CHECK(x.dim() >= 4, "cached_local_kv_caches must have at least 4 dims");
    int64_t S = x.size(3);

    int64_t outer = 1;
    for (int i = 0; i < 3; ++i) outer *= x.size(i);

    int64_t inner = 1;
    for (int i = 4; i < x.dim(); ++i) inner *= x.size(i);

    return {outer, inner};
}

__global__ void mask_u8_to_i32_kernel(
    const uint8_t* in,
    int32_t* out,
    int64_t S
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < S) out[idx] = static_cast<int32_t>(in[idx]);
}

template <typename scalar_t>
__global__ void compact_kv_to_flat_padded_kernel(
    const scalar_t* __restrict__ in,      // [outer, S, inner]
    scalar_t* __restrict__ out_flat,      // [target_numel], pre-zeroed
    const uint8_t* __restrict__ keep_mask,
    const int32_t* __restrict__ prefix,
    const int32_t* __restrict__ new_len_ptr,   // [1]
    int64_t outer,
    int64_t S,
    int64_t inner,
    int64_t target_numel
) {
    int64_t total = outer * S * inner;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int32_t new_len = new_len_ptr[0];

    int64_t tmp = idx;
    int64_t inner_idx = tmp % inner;
    tmp /= inner;
    int64_t s = tmp % S;
    int64_t outer_idx = tmp / S;

    if (keep_mask[s]) {
        int32_t dst_s = prefix[s];
        int64_t in_offset = (outer_idx * S + s) * inner + inner_idx;
        int64_t out_offset = (outer_idx * static_cast<int64_t>(new_len) + dst_s) * inner + inner_idx;

        if (out_offset < target_numel) {
            out_flat[out_offset] = in[in_offset];
        }
    }
}

std::vector<torch::Tensor> select_kv_cuda(
    torch::Tensor cached_local_kv_caches,
    torch::Tensor recall_doc_set,
    int64_t content_len,
    int64_t infer_gist_token_len,
    int64_t selected_token_len,
    int64_t target_numel
) {
    CHECK_CUDA(cached_local_kv_caches);
    CHECK_CUDA(recall_doc_set);

    TORCH_CHECK(selected_token_len == cached_local_kv_caches.size(3),
                "selected_token_len must equal cached_local_kv_caches.size(3)");
    TORCH_CHECK(infer_gist_token_len >= 0, "infer_gist_token_len must be >= 0");
    TORCH_CHECK(infer_gist_token_len <= selected_token_len,
                "infer_gist_token_len must be <= selected_token_len");

    // Respect the caller's current stream to avoid cross-stream data hazards.
    auto stream = at::cuda::getCurrentCUDAStream(cached_local_kv_caches.device().index());

    const int64_t S = selected_token_len;
    const int64_t num_recall = recall_doc_set.numel();

    auto keep_mask = torch::zeros({S},
        torch::TensorOptions().device(cached_local_kv_caches.device()).dtype(torch::kUInt8));

    auto prefix = torch::zeros({S},
        torch::TensorOptions().device(cached_local_kv_caches.device()).dtype(torch::kInt32));

    auto new_len_gpu = torch::zeros({1},
        torch::TensorOptions().device(cached_local_kv_caches.device()).dtype(torch::kInt32));

    if (S > 0) {
        int threads = 256;
        int blocks = (S + threads - 1) / threads;
        init_tail_kernel<<<blocks, threads, 0, stream>>>(
            keep_mask.data_ptr<uint8_t>(),
            infer_gist_token_len,
            S
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (num_recall > 0) {
        int threads = 256;
        auto recall_doc_contig = recall_doc_set.contiguous();  // Ensure contiguous for kernel access
        int blocks = (num_recall + threads - 1) / threads;
        if (recall_doc_contig.scalar_type() == torch::kInt64) {
            mark_recall_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                recall_doc_contig.data_ptr<int64_t>(),
                keep_mask.data_ptr<uint8_t>(),
                num_recall,
                content_len,
                infer_gist_token_len
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else if (recall_doc_contig.scalar_type() == torch::kInt32) {
            mark_recall_kernel<int32_t><<<blocks, threads, 0, stream>>>(
                recall_doc_contig.data_ptr<int32_t>(),
                keep_mask.data_ptr<uint8_t>(),
                num_recall,
                content_len,
                infer_gist_token_len
            );
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        } else {
            TORCH_CHECK(false, "recall_doc_set must be int32 or int64");
        }
    }
        // CUB exclusive scan
    size_t temp_storage_bytes = 0;
    auto keep_int = torch::zeros({S},
        torch::TensorOptions().device(cached_local_kv_caches.device()).dtype(torch::kInt32));

    if (S > 0) {
        int threads = 256;
        int blocks = (S + threads - 1) / threads;
        mask_u8_to_i32_kernel<<<blocks, threads, 0, stream>>>(
            keep_mask.data_ptr<uint8_t>(),
            keep_int.data_ptr<int32_t>(),
            S
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            temp_storage_bytes,
            keep_int.data_ptr<int32_t>(),
            prefix.data_ptr<int32_t>(),
            S,
            stream
        );
        auto temp_storage = torch::empty(
            {(long long)temp_storage_bytes},
            torch::TensorOptions().device(cached_local_kv_caches.device()).dtype(torch::kUInt8)
        );

        cub::DeviceScan::ExclusiveSum(
            temp_storage.data_ptr<uint8_t>(),
            temp_storage_bytes,
            keep_int.data_ptr<int32_t>(),
            prefix.data_ptr<int32_t>(),
            S,
            stream
        );

        compute_new_len_kernel<<<1, 1, 0, stream>>>(
            keep_mask.data_ptr<uint8_t>(),
            prefix.data_ptr<int32_t>(),
            S,
            new_len_gpu.data_ptr<int32_t>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    TORCH_CHECK(target_numel >= 0, "target_numel must be >= 0");

    auto outer_inner = flatten_outer_inner(cached_local_kv_caches);
    int64_t outer = std::get<0>(outer_inner);
    int64_t inner = std::get<1>(outer_inner);
    auto cache_contig = cached_local_kv_caches.contiguous();
    auto in_3d = cache_contig.view({outer, S, inner});

    // 直接分配最终 flat padded buffer，并预置为 0
    auto padded_out_flat = torch::zeros(
        {target_numel},
        cached_local_kv_caches.options()
    );

    {
        int64_t total = outer * S * inner;
        if (total > 0) {
            int threads = 256;
            int blocks = (total + threads - 1) / threads;

            AT_DISPATCH_SWITCH(
                cached_local_kv_caches.scalar_type(),
                "compact_kv_to_flat_padded_kernel",
                AT_DISPATCH_CASE(torch::kFloat, [&] {
                    using scalar_t = float;
                    compact_kv_to_flat_padded_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        in_3d.data_ptr<scalar_t>(),
                        padded_out_flat.data_ptr<scalar_t>(),
                        keep_mask.data_ptr<uint8_t>(),
                        prefix.data_ptr<int32_t>(),
                        new_len_gpu.data_ptr<int32_t>(),
                        outer,
                        S,
                        inner,
                        target_numel
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                })
                AT_DISPATCH_CASE(torch::kHalf, [&] {
                    using scalar_t = at::Half;
                    compact_kv_to_flat_padded_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        in_3d.data_ptr<scalar_t>(),
                        padded_out_flat.data_ptr<scalar_t>(),
                        keep_mask.data_ptr<uint8_t>(),
                        prefix.data_ptr<int32_t>(),
                        new_len_gpu.data_ptr<int32_t>(),
                        outer,
                        S,
                        inner,
                        target_numel
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                })
                AT_DISPATCH_CASE(torch::kBFloat16, [&] {
                    using scalar_t = at::BFloat16;
                    compact_kv_to_flat_padded_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        in_3d.data_ptr<scalar_t>(),
                        padded_out_flat.data_ptr<scalar_t>(),
                        keep_mask.data_ptr<uint8_t>(),
                        prefix.data_ptr<int32_t>(),
                        new_len_gpu.data_ptr<int32_t>(),
                        outer,
                        S,
                        inner,
                        target_numel
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                })
                AT_DISPATCH_CASE(torch::kByte, [&] {
                    using scalar_t = uint8_t;
                    compact_kv_to_flat_padded_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                        in_3d.data_ptr<scalar_t>(),
                        padded_out_flat.data_ptr<scalar_t>(),
                        keep_mask.data_ptr<uint8_t>(),
                        prefix.data_ptr<int32_t>(),
                        new_len_gpu.data_ptr<int32_t>(),
                        outer,
                        S,
                        inner,
                        target_numel
                    );
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                })
            );
        }
    }

    return {padded_out_flat, keep_mask, prefix, new_len_gpu};
}