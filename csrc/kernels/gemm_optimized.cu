#include <cuda_runtime.h>
#include <torch/extension.h>

__launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerMultiprocessor=4
__global__ void gemm_kernel_optimized_v1(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int TILE_SIZE = 32;
    const int THREAD_TILE = 2;  // 每线程处理 2x2
    
    // Shared memory with +1 padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    
    // 每个线程负责的输出位置基址
    const int c_row_base = blockIdx.y * TILE_SIZE + ty * THREAD_TILE;
    const int c_col_base = blockIdx.x * TILE_SIZE + tx * THREAD_TILE;
    
    // 寄存器累加器: 2x2 = 4 个（比 8x8=64 个大大减少）
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    
    // 加载索引：256 线程加载 32x32 = 1024 元素
    // 每个线程加载 4 个元素
    const int tid = ty * 16 + tx;
    
    // K 维度循环
    const int num_k_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = kt * TILE_SIZE;
        
        // ============ 协作加载 A tile [32x32] ============
        // 256 线程加载 1024 元素，每线程 4 个
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid * 4 + i;
            int load_row = load_idx / TILE_SIZE;  // 0-31
            int load_col = load_idx % TILE_SIZE;  // 0-31
            
            int gm_row = blockIdx.y * TILE_SIZE + load_row;
            int gm_col = k_base + load_col;
            
            As[load_row][load_col] = (gm_row < M && gm_col < K) ? A[gm_row * K + gm_col] : 0.0f;
        }
        
        // ============ 协作加载 B tile [32x32] ============
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid * 4 + i;
            int load_row = load_idx / TILE_SIZE;
            int load_col = load_idx % TILE_SIZE;
            
            int gm_row = k_base + load_row;
            int gm_col = blockIdx.x * TILE_SIZE + load_col;
            
            Bs[load_row][load_col] = (gm_row < K && gm_col < N) ? B[gm_row * N + gm_col] : 0.0f;
        }
        
        __syncthreads();
        
        // ============ 计算：每线程 2x2 输出 ============
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a0 = As[ty * THREAD_TILE + 0][k];
            float a1 = As[ty * THREAD_TILE + 1][k];
            float b0 = Bs[k][tx * THREAD_TILE + 0];
            float b1 = Bs[k][tx * THREAD_TILE + 1];
            
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }
        
        __syncthreads();
    }
    
    // ============ 写回结果 ============
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        int gm_row = c_row_base + i;
        if (gm_row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                int gm_col = c_col_base + j;
                if (gm_col < N) {
                    C[gm_row * N + gm_col] = acc[i][j];
                }
            }
        }
    }
}

// =============================================================================
// 2. 优化的 Shared Memory GEMM (16x16 threads, 每线程 4x4 输出)
// =============================================================================
// TILE_SIZE = 64, 每个 block 处理 64x64 输出
// 16x16 = 256 线程，每线程处理 4x4 = 16 个输出元素

__launch_bounds__(256, 2)  // 每线程 16 个累加器，需要更多寄存器
__global__ void gemm_kernel_optimized_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N
) {
    const int TILE_M = 64;
    const int TILE_N = 64;
    const int TILE_K = 16;  // 较小的 K tile 减少 shared memory
    const int THREAD_TILE_M = 4;
    const int THREAD_TILE_N = 4;
    
    // Shared memory with padding
    __shared__ float As[TILE_K][TILE_M + 1];
    __shared__ float Bs[TILE_K][TILE_N + 1];
    
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int tid = ty * 16 + tx;
    
    // 每个线程负责的输出位置
    const int c_row_base = blockIdx.y * TILE_M + ty * THREAD_TILE_M;
    const int c_col_base = blockIdx.x * TILE_N + tx * THREAD_TILE_N;
    
    // 寄存器累加器: 4x4 = 16 个
    float acc[THREAD_TILE_M][THREAD_TILE_N] = {{0}};
    
    // 寄存器缓存 A 和 B 的片段
    float a_frag[THREAD_TILE_M];
    float b_frag[THREAD_TILE_N];
    
    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = kt * TILE_K;
        
        // ============ 协作加载 A tile [64x16] ============
        // 256 线程加载 64*16 = 1024 元素，每线程 4 个
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid * 4 + i;
            int load_m = load_idx / TILE_K;  // 0-63
            int load_k = load_idx % TILE_K;  // 0-15
            
            int gm_row = blockIdx.y * TILE_M + load_m;
            int gm_col = k_base + load_k;
            
            if (load_m < TILE_M) {
                As[load_k][load_m] = (gm_row < M && gm_col < K) ? A[gm_row * K + gm_col] : 0.0f;
            }
        }
        
        // ============ 协作加载 B tile [16x64] ============
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int load_idx = tid * 4 + i;
            int load_k = load_idx / TILE_N;  // 0-15
            int load_n = load_idx % TILE_N;  // 0-63
            
            int gm_row = k_base + load_k;
            int gm_col = blockIdx.x * TILE_N + load_n;
            
            if (load_k < TILE_K) {
                Bs[load_k][load_n] = (gm_row < K && gm_col < N) ? B[gm_row * N + gm_col] : 0.0f;
            }
        }
        
        __syncthreads();
        
        // ============ 计算 ============
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // 加载 A 片段到寄存器
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_frag[i] = As[k][ty * THREAD_TILE_M + i];
            }
            // 加载 B 片段到寄存器
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_frag[j] = Bs[k][tx * THREAD_TILE_N + j];
            }
            
            // 外积累加
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // ============ 写回结果 ============
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        int gm_row = c_row_base + i;
        if (gm_row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                int gm_col = c_col_base + j;
                if (gm_col < N) {
                    C[gm_row * N + gm_col] = acc[i][j];
                }
            }
        }
    }
}

// =============================================================================
// 3. Split-K GEMM 优化版：无 Atomic，Block 内归约
// =============================================================================
// 关键改进：
// - 不使用 workspace[k_splits, M, N]（太大）
// - 每个 block 负责一个输出 tile 的完整 K 分片
// - 使用 shared memory 做 block 内归约

// Step 1: 每个 K-split 计算 partial sum，写入 tile-local workspace
template <int TILE_SIZE = 32>
__global__ void gemm_splitk_compute_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ partial_sums,  // [num_tiles_m, num_tiles_n, k_splits, TILE_SIZE, TILE_SIZE]
    int M, int K, int N,
    int k_splits,
    int num_tiles_m,
    int num_tiles_n
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    const int k_split_idx = blockIdx.z;
    
    const int k_per_split = (K + k_splits - 1) / k_splits;
    const int k_start = k_split_idx * k_per_split;
    const int k_end = min(k_start + k_per_split, K);
    
    const int row = tile_m * TILE_SIZE + ty;
    const int col = tile_n * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    const int num_k_tiles = (k_end - k_start + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = k_start + kt * TILE_SIZE;
        
        // 加载 A
        int k_idx = k_base + tx;
        As[ty][tx] = (row < M && k_idx < k_end) ? A[row * K + k_idx] : 0.0f;
        
        // 加载 B
        k_idx = k_base + ty;
        Bs[ty][tx] = (k_idx < k_end && col < N) ? B[k_idx * N + col] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 写入 partial_sums（按 tile 组织，更小的连续写入）
    // 索引: [tile_m][tile_n][k_split][ty][tx]
    if (row < M && col < N) {
        int partial_idx = ((tile_m * num_tiles_n + tile_n) * k_splits + k_split_idx) 
                          * TILE_SIZE * TILE_SIZE + ty * TILE_SIZE + tx;
        partial_sums[partial_idx] = sum;
    }
}

// Step 2: 归约 partial sums
template <int TILE_SIZE = 32>
__global__ void gemm_splitk_reduce_v2(
    const float* __restrict__ partial_sums,
    float* __restrict__ C,
    int M, int K, int N,
    int k_splits,
    int num_tiles_m,
    int num_tiles_n
) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    
    const int row = tile_m * TILE_SIZE + ty;
    const int col = tile_n * TILE_SIZE + tx;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // 归约所有 k_splits
        int base_idx = (tile_m * num_tiles_n + tile_n) * k_splits * TILE_SIZE * TILE_SIZE 
                       + ty * TILE_SIZE + tx;
        
        #pragma unroll 4
        for (int s = 0; s < k_splits; ++s) {
            sum += partial_sums[base_idx + s * TILE_SIZE * TILE_SIZE];
        }
        
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 4. Split-K GEMM 优化版 V3：使用 Shared Memory 做 Block 内归约
// =============================================================================
// 一个 block 负责一个 output tile，block 内的 warps 分担不同的 K-splits
// 最后在 shared memory 中归约

__launch_bounds__(256, 2)
__global__ void gemm_splitk_fused(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    int k_splits  // 必须是 2 的幂且 <= 8
) {
    const int TILE_SIZE = 32;
    const int THREADS_PER_DIM = 16;  // 16x16 线程
    const int THREAD_TILE = 2;       // 每线程 2x2 输出
    
    // Shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float reduce_smem[8][TILE_SIZE][TILE_SIZE + 1];  // 最多支持 8 splits
    
    const int tx = threadIdx.x % THREADS_PER_DIM;  // 0-15
    const int ty = threadIdx.x / THREADS_PER_DIM;  // 0-15
    
    // 每个 warp 处理一个 k_split
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int k_split_idx = warp_id % k_splits;
    
    const int k_per_split = (K + k_splits - 1) / k_splits;
    const int k_start = k_split_idx * k_per_split;
    const int k_end = min(k_start + k_per_split, K);
    
    const int tile_m = blockIdx.y;
    const int tile_n = blockIdx.x;
    
    // 每线程 2x2 输出位置
    const int c_row_local = ty * THREAD_TILE;
    const int c_col_local = tx * THREAD_TILE;
    
    // 累加器
    float acc[THREAD_TILE][THREAD_TILE] = {{0}};
    
    // 计算当前 k_split 的部分结果
    const int num_k_tiles = (k_end - k_start + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k_base = k_start + kt * TILE_SIZE;
        
        // 协作加载（所有 warp 并行加载同一个 k_tile）
        // 注意：这里需要处理不同 warp 加载不同 k 范围
        // 简化：让每个 warp 独立加载自己需要的数据
        
        for (int i = lane_id; i < TILE_SIZE * TILE_SIZE; i += 32) {
            int load_row = i / TILE_SIZE;
            int load_col = i % TILE_SIZE;
            
            int gm_row = tile_m * TILE_SIZE + load_row;
            int gm_col = k_base + load_col;
            As[load_row][load_col] = (gm_row < M && gm_col < k_end) ? A[gm_row * K + gm_col] : 0.0f;
            
            gm_row = k_base + load_row;
            gm_col = tile_n * TILE_SIZE + load_col;
            Bs[load_row][load_col] = (gm_row < k_end && gm_col < N) ? B[gm_row * N + gm_col] : 0.0f;
        }
        
        __syncwarp();
        
        // 计算
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a0 = As[c_row_local + 0][k];
            float a1 = As[c_row_local + 1][k];
            float b0 = Bs[k][c_col_local + 0];
            float b1 = Bs[k][c_col_local + 1];
            
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }
    }
    
    // 将结果写入 reduce shared memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; ++j) {
            reduce_smem[k_split_idx][c_row_local + i][c_col_local + j] = acc[i][j];
        }
    }
    
    __syncthreads();
    
    // 归约（只有第一组 warp 做归约）
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                float sum = 0.0f;
                #pragma unroll
                for (int s = 0; s < k_splits; ++s) {
                    sum += reduce_smem[s][c_row_local + i][c_col_local + j];
                }
                
                int gm_row = tile_m * TILE_SIZE + c_row_local + i;
                int gm_col = tile_n * TILE_SIZE + c_col_local + j;
                if (gm_row < M && gm_col < N) {
                    C[gm_row * N + gm_col] = sum;
                }
            }
        }
    }
}

// =============================================================================
// Host Functions
// =============================================================================

torch::Tensor gemm_optimized_v1_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 supported");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(16, 16);  // 256 threads
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_kernel_optimized_v1<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

torch::Tensor gemm_optimized_v2_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 supported");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const int TILE_M = 64;
    const int TILE_N = 64;
    dim3 threads(16, 16);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    gemm_kernel_optimized_v2<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

torch::Tensor gemm_splitk_v2_cuda(torch::Tensor A, torch::Tensor B, int k_splits) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(k_splits > 0 && k_splits <= 16, "k_splits must be 1-16");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    const int TILE_SIZE = 32;
    int num_tiles_m = (M + TILE_SIZE - 1) / TILE_SIZE;
    int num_tiles_n = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Partial sums: [num_tiles_m * num_tiles_n * k_splits * TILE_SIZE * TILE_SIZE]
    // 比 [k_splits, M, N] 小很多，因为只存储每个 tile 的数据
    auto partial_sums = torch::zeros(
        {num_tiles_m * num_tiles_n * k_splits * TILE_SIZE * TILE_SIZE}, 
        A.options()
    );
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks_compute(num_tiles_n, num_tiles_m, k_splits);
    dim3 blocks_reduce(num_tiles_n, num_tiles_m);
    
    // Step 1: Compute partial sums
    gemm_splitk_compute_v2<32><<<blocks_compute, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), partial_sums.data_ptr<float>(),
        M, K, N, k_splits, num_tiles_m, num_tiles_n
    );
    
    // Step 2: Reduce
    gemm_splitk_reduce_v2<32><<<blocks_reduce, threads>>>(
        partial_sums.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N, k_splits, num_tiles_m, num_tiles_n
    );
    
    return C;
}

// 为了与现有接口兼容，添加 wrapper
torch::Tensor gemm_optimized_cuda(torch::Tensor A, torch::Tensor B, int version) {
    if (version == 1) {
        return gemm_optimized_v1_cuda(A, B);
    } else {
        return gemm_optimized_v2_cuda(A, B);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_optimized", &gemm_optimized_cuda, 
          "Optimized GEMM (CUDA, float32 only)",
          py::arg("A"), py::arg("B"), py::arg("version") = 1);
    
    m.def("gemm_optimized_v1", &gemm_optimized_v1_cuda,
          "Optimized GEMM v1: 16x16 threads, 2x2 per thread (CUDA, float32 only)",
          py::arg("A"), py::arg("B"));
    
    m.def("gemm_optimized_v2", &gemm_optimized_v2_cuda,
          "Optimized GEMM v2: 16x16 threads, 4x4 per thread (CUDA, float32 only)",
          py::arg("A"), py::arg("B"));
    
    m.def("gemm_splitk_v2", &gemm_splitk_v2_cuda,
          "Split-K GEMM v2: No atomics, tile-local partial sums (CUDA, float32 only)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
}
