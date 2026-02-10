#include <cuda_runtime.h>
#include <torch/extension.h>

// =============================================================================
// 1. Simple GEMM kernel (naive implementation)
// =============================================================================

// Simple GEMM kernel: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
template <typename scalar_t>
__global__ void gemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        scalar_t sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 2. Shared Memory Tiled GEMM kernel
// =============================================================================

// Optimized GEMM kernel using shared memory
template <typename scalar_t>
__global__ void gemm_kernel_shared(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    scalar_t sum = 0;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 3. StreamK GEMM kernel
// =============================================================================
// StreamK 核心思想：
// - 传统方法：每个 block 处理一个固定的输出 tile，导致负载不均衡
// - StreamK：将所有 K 维度的 MAC 操作均匀分配给所有 blocks
// 
// 优势：
// 1. 更好的负载均衡，减少 tail effect
// 2. 适合 M*N 较小但 K 很大的场景
// 3. 更好地利用 GPU 所有 SM

template <typename scalar_t, int TILE_M = 64, int TILE_N = 64, int TILE_K = 32>
__global__ void gemm_kernel_streamk(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ workspace,  // 用于部分结果累加
    int M, int K, int N,
    int num_tiles_m,
    int num_tiles_n,
    int num_tiles_k,
    int total_tiles,       // M_tiles * N_tiles
    int iters_per_tile,    // 每个 tile 需要的 K 迭代次数
    int total_iters,       // 总迭代次数 = total_tiles * iters_per_tile
    int sk_tiles,          // 使用 StreamK 处理的 tile 数量
    int sk_iters,          // StreamK 阶段的总迭代次数
    int sk_blocks,         // 参与 StreamK 的 block 数量
    int dp_start_tile      // Data-Parallel 阶段开始的 tile 索引
) {
    // 每个 thread 处理 TILE_M/blockDim.y * TILE_N/blockDim.x 个元素
    const int THREAD_M = TILE_M / 8;  // 假设 blockDim.y = 8
    const int THREAD_N = TILE_N / 8;  // 假设 blockDim.x = 8
    
    __shared__ scalar_t As[TILE_K][TILE_M];
    __shared__ scalar_t Bs[TILE_K][TILE_N];
    
    // 寄存器存储局部累加结果
    scalar_t acc[THREAD_M][THREAD_N] = {0};
    
    int bid = blockIdx.x;
    
    // ==========================================================
    // StreamK 阶段：处理前 sk_tiles 个 tiles
    // ==========================================================
    if (bid < sk_blocks) {
        // 计算该 block 负责的迭代范围
        int iters_per_block = (sk_iters + sk_blocks - 1) / sk_blocks;
        int iter_start = bid * iters_per_block;
        int iter_end = min(iter_start + iters_per_block, sk_iters);
        
        int current_tile = -1;
        int tile_iter_start = 0;
        
        for (int iter = iter_start; iter < iter_end; ) {
            // 计算当前迭代属于哪个 tile
            int new_tile = iter / iters_per_tile;
            int iter_within_tile = iter % iters_per_tile;
            
            if (new_tile >= sk_tiles) break;
            
            // 如果切换到新 tile，保存之前的结果
            if (new_tile != current_tile) {
                if (current_tile >= 0) {
                    // 将部分结果写入 workspace
                    int tile_m = current_tile / num_tiles_n;
                    int tile_n = current_tile % num_tiles_n;
                    int base_m = tile_m * TILE_M;
                    int base_n = tile_n * TILE_N;
                    
                    for (int tm = 0; tm < THREAD_M; ++tm) {
                        for (int tn = 0; tn < THREAD_N; ++tn) {
                            int m = base_m + threadIdx.y * THREAD_M + tm;
                            int n = base_n + threadIdx.x * THREAD_N + tn;
                            if (m < M && n < N) {
                                // 使用 atomicAdd 累加到 workspace
                                atomicAdd(&workspace[m * N + n], acc[tm][tn]);
                            }
                            acc[tm][tn] = 0;
                        }
                    }
                }
                current_tile = new_tile;
                tile_iter_start = iter_within_tile;
            }
            
            // 计算 tile 坐标
            int tile_m = current_tile / num_tiles_n;
            int tile_n = current_tile % num_tiles_n;
            int base_m = tile_m * TILE_M;
            int base_n = tile_n * TILE_N;
            int base_k = iter_within_tile * TILE_K;
            
            // 加载 A tile 到 shared memory
            for (int k = threadIdx.y; k < TILE_K; k += blockDim.y) {
                for (int m = threadIdx.x; m < TILE_M; m += blockDim.x) {
                    int gm = base_m + m;
                    int gk = base_k + k;
                    As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0;
                }
            }
            
            // 加载 B tile 到 shared memory
            for (int k = threadIdx.y; k < TILE_K; k += blockDim.y) {
                for (int n = threadIdx.x; n < TILE_N; n += blockDim.x) {
                    int gk = base_k + k;
                    int gn = base_n + n;
                    Bs[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : 0;
                }
            }
            
            __syncthreads();
            
            // 计算
            for (int k = 0; k < TILE_K; ++k) {
                for (int tm = 0; tm < THREAD_M; ++tm) {
                    for (int tn = 0; tn < THREAD_N; ++tn) {
                        int m_idx = threadIdx.y * THREAD_M + tm;
                        int n_idx = threadIdx.x * THREAD_N + tn;
                        acc[tm][tn] += As[k][m_idx] * Bs[k][n_idx];
                    }
                }
            }
            
            __syncthreads();
            
            iter++;
        }
        
        // 保存最后一个 tile 的结果
        if (current_tile >= 0 && current_tile < sk_tiles) {
            int tile_m = current_tile / num_tiles_n;
            int tile_n = current_tile % num_tiles_n;
            int base_m = tile_m * TILE_M;
            int base_n = tile_n * TILE_N;
            
            for (int tm = 0; tm < THREAD_M; ++tm) {
                for (int tn = 0; tn < THREAD_N; ++tn) {
                    int m = base_m + threadIdx.y * THREAD_M + tm;
                    int n = base_n + threadIdx.x * THREAD_N + tn;
                    if (m < M && n < N) {
                        atomicAdd(&workspace[m * N + n], acc[tm][tn]);
                    }
                }
            }
        }
    }
    
    // ==========================================================
    // Data-Parallel 阶段：传统 tile-based 并行
    // ==========================================================
    int dp_bid = bid - sk_blocks;
    if (dp_bid >= 0 && dp_start_tile + dp_bid < total_tiles) {
        int tile_idx = dp_start_tile + dp_bid;
        int tile_m = tile_idx / num_tiles_n;
        int tile_n = tile_idx % num_tiles_n;
        int base_m = tile_m * TILE_M;
        int base_n = tile_n * TILE_N;
        
        // 重置累加器
        for (int tm = 0; tm < THREAD_M; ++tm) {
            for (int tn = 0; tn < THREAD_N; ++tn) {
                acc[tm][tn] = 0;
            }
        }
        
        // 遍历所有 K tiles
        for (int k_tile = 0; k_tile < num_tiles_k; ++k_tile) {
            int base_k = k_tile * TILE_K;
            
            // 加载 A tile
            for (int k = threadIdx.y; k < TILE_K; k += blockDim.y) {
                for (int m = threadIdx.x; m < TILE_M; m += blockDim.x) {
                    int gm = base_m + m;
                    int gk = base_k + k;
                    As[k][m] = (gm < M && gk < K) ? A[gm * K + gk] : 0;
                }
            }
            
            // 加载 B tile
            for (int k = threadIdx.y; k < TILE_K; k += blockDim.y) {
                for (int n = threadIdx.x; n < TILE_N; n += blockDim.x) {
                    int gk = base_k + k;
                    int gn = base_n + n;
                    Bs[k][n] = (gk < K && gn < N) ? B[gk * N + gn] : 0;
                }
            }
            
            __syncthreads();
            
            // 计算
            for (int k = 0; k < TILE_K; ++k) {
                for (int tm = 0; tm < THREAD_M; ++tm) {
                    for (int tn = 0; tn < THREAD_N; ++tn) {
                        int m_idx = threadIdx.y * THREAD_M + tm;
                        int n_idx = threadIdx.x * THREAD_N + tn;
                        acc[tm][tn] += As[k][m_idx] * Bs[k][n_idx];
                    }
                }
            }
            
            __syncthreads();
        }
        
        // 直接写入输出
        for (int tm = 0; tm < THREAD_M; ++tm) {
            for (int tn = 0; tn < THREAD_N; ++tn) {
                int m = base_m + threadIdx.y * THREAD_M + tm;
                int n = base_n + threadIdx.x * THREAD_N + tn;
                if (m < M && n < N) {
                    C[m * N + n] = acc[tm][tn];
                }
            }
        }
    }
}

// StreamK 结果合并 kernel
template <typename scalar_t>
__global__ void streamk_fixup_kernel(
    scalar_t* __restrict__ C,
    const scalar_t* __restrict__ workspace,
    int M, int N,
    int sk_tiles,
    int num_tiles_n,
    int TILE_M,
    int TILE_N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        int m = idx / N;
        int n = idx % N;
        
        // 检查是否在 StreamK 处理的 tile 范围内
        int tile_m = m / TILE_M;
        int tile_n = n / TILE_N;
        int tile_idx = tile_m * num_tiles_n + tile_n;
        
        if (tile_idx < sk_tiles) {
            C[m * N + n] = workspace[m * N + n];
        }
    }
}

// =============================================================================
// 4. Simplified StreamK-inspired kernel (easier to understand)
// =============================================================================
// 简化版 StreamK：将 K 维度分割，多个 blocks 并行处理同一个输出 tile

template <typename scalar_t, int TILE_SIZE = 32>
__global__ void gemm_kernel_streamk_simple(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M, int K, int N,
    int k_splits  // K 维度分割数
) {
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];  // +1 避免 bank conflict
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];
    
    // blockIdx.z 表示 K 维度的分片索引
    int k_split_idx = blockIdx.z;
    int k_start = k_split_idx * ((K + k_splits - 1) / k_splits);
    int k_end = min(k_start + ((K + k_splits - 1) / k_splits), K);
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    scalar_t sum = 0;
    
    // 只处理分配给当前 block 的 K 范围
    for (int t = k_start / TILE_SIZE; t < (k_end + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_base = t * TILE_SIZE;
        
        // 加载 A tile
        if (row < M && k_base + threadIdx.x < K && k_base + threadIdx.x >= k_start && k_base + threadIdx.x < k_end) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        // 加载 B tile
        if (col < N && k_base + threadIdx.y < K && k_base + threadIdx.y >= k_start && k_base + threadIdx.y < k_end) {
            Bs[threadIdx.y][threadIdx.x] = B[(k_base + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // 计算
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 使用 atomicAdd 累加结果（因为多个 blocks 处理同一个输出位置）
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

// =============================================================================
// 5. Split-K GEMM (StreamK 的简化变体)
// =============================================================================
// Split-K: 固定 K 分割，每个分割独立计算，最后 reduce

template <typename scalar_t, int TILE_SIZE = 32>
__global__ void gemm_kernel_splitk(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ workspace,  // [k_splits, M, N]
    int M, int K, int N,
    int k_splits
) {
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int k_split_idx = blockIdx.z;
    int k_per_split = (K + k_splits - 1) / k_splits;
    int k_start = k_split_idx * k_per_split;
    int k_end = min(k_start + k_per_split, K);
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    scalar_t sum = 0;
    
    int num_k_tiles = (k_end - k_start + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_k_tiles; ++t) {
        int k_base = k_start + t * TILE_SIZE;
        
        // 加载 A tile
        int k_idx = k_base + threadIdx.x;
        if (row < M && k_idx < k_end) {
            As[threadIdx.y][threadIdx.x] = A[row * K + k_idx];
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        // 加载 B tile
        k_idx = k_base + threadIdx.y;
        if (col < N && k_idx < k_end) {
            Bs[threadIdx.y][threadIdx.x] = B[k_idx * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // 写入 workspace 对应分片
    if (row < M && col < N) {
        workspace[k_split_idx * M * N + row * N + col] = sum;
    }
}

// Split-K reduce kernel
template <typename scalar_t>
__global__ void splitk_reduce_kernel(
    const scalar_t* __restrict__ workspace,
    scalar_t* __restrict__ C,
    int M, int N, int k_splits
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    
    if (idx < total) {
        scalar_t sum = 0;
        for (int s = 0; s < k_splits; ++s) {
            sum += workspace[s * total + idx];
        }
        C[idx] = sum;
    }
}

// =============================================================================
// Host Functions
// =============================================================================

// Original GEMM (naive or shared memory)
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B, bool use_shared) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    if (use_shared) {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_kernel_shared", ([&] {
            gemm_kernel_shared<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N
            );
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_kernel", ([&] {
            gemm_kernel<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, K, N
            );
        }));
    }
    
    return C;
}

// Split-K GEMM
torch::Tensor gemm_splitk_cuda(torch::Tensor A, torch::Tensor B, int k_splits) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    TORCH_CHECK(k_splits > 0, "k_splits must be positive");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 自动选择合适的 k_splits
    if (k_splits <= 0) {
        // 根据 K 维度大小自动选择
        k_splits = std::min(8, (K + 255) / 256);
        k_splits = std::max(1, k_splits);
    }
    
    auto C = torch::zeros({M, N}, A.options());
    auto workspace = torch::zeros({k_splits, M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE,
                k_splits);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_kernel_splitk", ([&] {
        // Step 1: 并行计算各个 K 分片
        gemm_kernel_splitk<scalar_t, 32><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            workspace.data_ptr<scalar_t>(),
            M, K, N, k_splits
        );
        
        // Step 2: Reduce 各分片结果
        int reduce_threads = 256;
        int reduce_blocks = (M * N + reduce_threads - 1) / reduce_threads;
        splitk_reduce_kernel<scalar_t><<<reduce_blocks, reduce_threads>>>(
            workspace.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, k_splits
        );
    }));
    
    return C;
}

// StreamK-style simple GEMM (uses atomics)
torch::Tensor gemm_streamk_simple_cuda(torch::Tensor A, torch::Tensor B, int k_splits) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    // 自动选择 k_splits
    if (k_splits <= 0) {
        k_splits = std::min(4, (K + 511) / 512);
        k_splits = std::max(1, k_splits);
    }
    
    auto C = torch::zeros({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                (M + TILE_SIZE - 1) / TILE_SIZE,
                k_splits);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "gemm_kernel_streamk_simple", ([&] {
        gemm_kernel_streamk_simple<scalar_t, 32><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N, k_splits
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda, "GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("use_shared") = true);
    
    m.def("gemm_splitk", &gemm_splitk_cuda, "Split-K GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
    
    m.def("gemm_streamk", &gemm_streamk_simple_cuda, "StreamK-style GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
}
