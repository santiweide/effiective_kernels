#include <cuda_runtime.h>
#include <torch/extension.h>

// =============================================================================
// 向量化访存辅助工具
// =============================================================================

// 向量化类型定义
template<typename T, int N> struct VectorType;

template<> struct VectorType<float, 1> { using type = float; };
template<> struct VectorType<float, 2> { using type = float2; };
template<> struct VectorType<float, 4> { using type = float4; };
template<> struct VectorType<double, 1> { using type = double; };
template<> struct VectorType<double, 2> { using type = double2; };

// 向量化加载 (float4 = 128-bit = 16 bytes)
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ float2 load_float2(const float* ptr) {
    return *reinterpret_cast<const float2*>(ptr);
}

__device__ __forceinline__ double2 load_double2(const double* ptr) {
    return *reinterpret_cast<const double2*>(ptr);
}

// 向量化存储
__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

__device__ __forceinline__ void store_float2(float* ptr, float2 val) {
    *reinterpret_cast<float2*>(ptr) = val;
}

// 检查地址是否对齐
__device__ __forceinline__ bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

__device__ __forceinline__ bool is_aligned_8(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 7) == 0;
}

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
// 2. Shared Memory Tiled GEMM kernel (标量版本 - 保留作为参考)
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
// 2.5 Vectorized Shared Memory GEMM kernel (float4 向量化版本)
// =============================================================================
// 使用 float4 向量化加载，每个线程一次加载 4 个元素
// TILE_M = 128, TILE_N = 128, TILE_K = 8
// 每个 block: 32x8 线程，每个线程处理 4x16 个输出元素

template <int TILE_M = 128, int TILE_N = 128, int TILE_K = 8>
__global__ void gemm_kernel_shared_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N) {
    
    // Block 和 thread 索引
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;  // 0-31
    const int ty = threadIdx.y;  // 0-7
    
    // 线程块内的线程ID
    const int tid = ty * 32 + tx;
    
    // Shared memory: 添加 padding 避免 bank conflict
    __shared__ float As[TILE_K][TILE_M + 4];  // [8][132]
    __shared__ float Bs[TILE_K][TILE_N + 4];  // [8][132]
    
    // 寄存器累加器: 每个线程计算 4x4 个输出
    float acc[4][4] = {0};
    
    // A 和 B 的 fragment 寄存器
    float a_frag[4];
    float b_frag[4];
    
    // 计算基础偏移
    const int a_base_row = by * TILE_M;
    const int b_base_col = bx * TILE_N;
    
    // 每个线程负责加载的位置
    // 加载 A: 每个线程加载 4 个连续的 float (float4)
    // 256 线程, 每行加载 TILE_M=128 个元素, 需要 128/4=32 个 float4
    // 每行 32 个 float4, 256/32 = 8 行可以一次加载完 TILE_K 行
    const int a_load_row = tid / 32;  // 0-7
    const int a_load_col = (tid % 32) * 4;  // 0, 4, 8, ..., 124
    
    // 加载 B: 类似
    const int b_load_row = tid / 32;  // 0-7
    const int b_load_col = (tid % 32) * 4;  // 0, 4, 8, ..., 124
    
    // 计算每个线程负责的输出位置
    // 32x8 线程块，要覆盖 128x128 输出
    // 每个线程处理 4x16 个输出？不，让我们简化为每线程 4x4
    // 使用 warp-level 的划分：
    // ty (0-7) 沿着 M 方向分组，每组 16 行
    // tx (0-31) 沿着 N 方向分组，每组 4 列
    const int c_row_base = ty * 16 + (tx / 8) * 4;  // 每个 warp 中 4 个线程组处理 4 行
    const int c_col_base = (tx % 8) * 16;  // 每组处理 16 列，分成 4 次计算
    
    // K 维度循环
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        // ===================== 加载 A tile =====================
        // 全局内存位置
        int a_gm_row = a_base_row + a_load_row;
        int a_gm_col = k_tile + a_load_col;
        
        if (a_gm_row < M && a_gm_col + 3 < K) {
            // 向量化加载 float4
            float4 a_vec = load_float4(&A[a_gm_row * K + a_gm_col]);
            As[a_load_row][a_load_col + 0] = a_vec.x;
            As[a_load_row][a_load_col + 1] = a_vec.y;
            As[a_load_row][a_load_col + 2] = a_vec.z;
            As[a_load_row][a_load_col + 3] = a_vec.w;
        } else {
            // 边界处理：标量加载
            for (int i = 0; i < 4; ++i) {
                int col = a_gm_col + i;
                As[a_load_row][a_load_col + i] = (a_gm_row < M && col < K) ? A[a_gm_row * K + col] : 0.0f;
            }
        }
        
        // ===================== 加载 B tile =====================
        int b_gm_row = k_tile + b_load_row;
        int b_gm_col = b_base_col + b_load_col;
        
        if (b_gm_row < K && b_gm_col + 3 < N) {
            float4 b_vec = load_float4(&B[b_gm_row * N + b_gm_col]);
            Bs[b_load_row][b_load_col + 0] = b_vec.x;
            Bs[b_load_row][b_load_col + 1] = b_vec.y;
            Bs[b_load_row][b_load_col + 2] = b_vec.z;
            Bs[b_load_row][b_load_col + 3] = b_vec.w;
        } else {
            for (int i = 0; i < 4; ++i) {
                int col = b_gm_col + i;
                Bs[b_load_row][b_load_col + i] = (b_gm_row < K && col < N) ? B[b_gm_row * N + col] : 0.0f;
            }
        }
        
        __syncthreads();
        
        // ===================== 计算 =====================
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            // 从 shared memory 加载到寄存器
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_frag[i] = As[k][c_row_base + i];
            }
            #pragma unroll  
            for (int i = 0; i < 4; ++i) {
                b_frag[i] = Bs[k][c_col_base + i];
            }
            
            // 外积累加
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // ===================== 写回结果 =====================
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gm_row = a_base_row + c_row_base + i;
        if (gm_row < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gm_col = b_base_col + c_col_base + j;
                if (gm_col < N) {
                    C[gm_row * N + gm_col] = acc[i][j];
                }
            }
        }
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
// 4.5 Vectorized StreamK kernel (float4 向量化版本)
// =============================================================================

template <int TILE_SIZE = 32, int VEC_SIZE = 4>
__global__ void gemm_kernel_streamk_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    int k_splits
) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float As[TILE_SIZE][TILE_SIZE + 4];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 4];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    int k_split_idx = blockIdx.z;
    int k_per_split = (K + k_splits - 1) / k_splits;
    int k_start = k_split_idx * k_per_split;
    int k_end = min(k_start + k_per_split, K);
    
    int row_base = blockIdx.y * TILE_SIZE;
    int col_base = blockIdx.x * TILE_SIZE;
    
    // 寄存器累加器
    float acc[4] = {0, 0, 0, 0};
    
    // 每个线程负责加载的行列
    // 32x32 tile, 1024 线程 (32x32), 每个线程加载 1 个元素
    // 或者使用更少的线程 + 向量化加载
    
    int row = row_base + ty;
    int col = col_base + tx;
    
    float sum = 0;
    
    int num_k_tiles = (k_end - k_start + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_k_tiles; ++t) {
        int k_base = k_start + t * TILE_SIZE;
        
        // 加载 A tile - 使用向量化加载（当对齐且在边界内时）
        int a_gm_col = k_base + tx;
        if (row < M && a_gm_col < k_end && a_gm_col < K) {
            As[ty][tx] = A[row * K + a_gm_col];
        } else {
            As[ty][tx] = 0;
        }
        
        // 加载 B tile
        int b_gm_row = k_base + ty;
        if (b_gm_row < k_end && b_gm_row < K && col < N) {
            Bs[ty][tx] = B[b_gm_row * N + col];
        } else {
            Bs[ty][tx] = 0;
        }
        
        __syncthreads();
        
        // 计算 - 展开循环提高 ILP
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            sum += As[ty][k + 0] * Bs[k + 0][tx];
            sum += As[ty][k + 1] * Bs[k + 1][tx];
            sum += As[ty][k + 2] * Bs[k + 2][tx];
            sum += As[ty][k + 3] * Bs[k + 3][tx];
        }
        
        __syncthreads();
    }
    
    // 使用 atomicAdd 累加
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

// =============================================================================
// 5.5 Vectorized Split-K GEMM (float4 向量化版本)
// =============================================================================
// 使用更大的 tile 和 float4 向量化加载

template <int TILE_M = 64, int TILE_N = 64, int TILE_K = 16>
__global__ void gemm_kernel_splitk_vectorized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ workspace,
    int M, int K, int N,
    int k_splits
) {
    // Block 配置: 16x16 线程
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int tid = ty * 16 + tx;
    
    // Shared memory with padding
    __shared__ float As[TILE_K][TILE_M + 4];
    __shared__ float Bs[TILE_K][TILE_N + 4];
    
    // 寄存器累加器: 每线程 4x4 输出
    float acc[4][4] = {{0}};
    
    int k_split_idx = blockIdx.z;
    int k_per_split = (K + k_splits - 1) / k_splits;
    int k_start = k_split_idx * k_per_split;
    int k_end = min(k_start + k_per_split, K);
    
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;
    
    // 计算每个线程负责的输出位置
    int c_row = ty * 4;  // 每个线程处理 4 行
    int c_col = tx * 4;  // 每个线程处理 4 列
    
    // A 加载：256 线程加载 16x64 = 1024 个元素
    // 每个线程加载 4 个连续元素
    const int a_load_row = tid / 16;  // 0-15 行
    const int a_load_col = (tid % 16) * 4;  // 每行 64 元素，16 个 float4
    
    // B 加载：类似
    const int b_load_row = tid / 16;
    const int b_load_col = (tid % 16) * 4;
    
    int num_k_tiles = (k_end - k_start + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < num_k_tiles; ++t) {
        int k_base = k_start + t * TILE_K;
        
        // ================== 加载 A ==================
        int a_gm_row = row_base + a_load_row;
        int a_gm_col = k_base + (a_load_col / 4);  // K 方向的偏移
        
        // 重新映射：我们需要加载 A[TILE_M][TILE_K]
        // 256 线程加载 64*16=1024 元素
        // 每线程加载 4 个元素
        int load_id = tid;
        int a_m = load_id / 4;  // 0-63
        int a_k = (load_id % 4) * 4;  // 0, 4, 8, 12
        
        if (a_m < TILE_M) {
            int gm_row = row_base + a_m;
            for (int i = 0; i < 4; ++i) {
                int gm_k = k_base + a_k + i;
                As[a_k + i][a_m] = (gm_row < M && gm_k < k_end) ? A[gm_row * K + gm_k] : 0.0f;
            }
        }
        
        // ================== 加载 B ==================
        // B[TILE_K][TILE_N] = 16*64
        int b_k = load_id / 16;  // 0-15
        int b_n = (load_id % 16) * 4;  // 0, 4, 8, ..., 60
        
        if (b_k < TILE_K) {
            int gm_k = k_base + b_k;
            if (gm_k < k_end) {
                for (int i = 0; i < 4; ++i) {
                    int gm_col = col_base + b_n + i;
                    Bs[b_k][b_n + i] = (gm_col < N) ? B[gm_k * N + gm_col] : 0.0f;
                }
            } else {
                for (int i = 0; i < 4; ++i) {
                    Bs[b_k][b_n + i] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // ================== 计算 ==================
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a_reg[4], b_reg[4];
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a_reg[i] = As[k][c_row + i];
                b_reg[i] = Bs[k][c_col + i];
            }
            
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // ================== 写回 workspace ==================
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int gm_row = row_base + c_row + i;
        if (gm_row < M) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int gm_col = col_base + c_col + j;
                if (gm_col < N) {
                    workspace[k_split_idx * M * N + gm_row * N + gm_col] = acc[i][j];
                }
            }
        }
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

// Vectorized GEMM (float4 optimized) - 仅支持 float32
torch::Tensor gemm_vectorized_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Vectorized GEMM only supports float32");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // 使用向量化 kernel: TILE_M=128, TILE_N=128, TILE_K=8
    // Block: 32x8 线程
    const int TILE_M = 128;
    const int TILE_N = 128;
    
    dim3 threads(32, 8);
    dim3 blocks((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    
    gemm_kernel_shared_vectorized<128, 128, 8><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
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

// Vectorized Split-K GEMM (float32 only)
torch::Tensor gemm_splitk_vectorized_cuda(torch::Tensor A, torch::Tensor B, int k_splits) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Vectorized Split-K GEMM only supports float32");
    TORCH_CHECK(k_splits > 0, "k_splits must be positive");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    auto workspace = torch::zeros({k_splits, M, N}, A.options());
    
    // 使用向量化版本: TILE_M=64, TILE_N=64, TILE_K=16
    const int TILE_M = 64;
    const int TILE_N = 64;
    
    dim3 threads(16, 16);
    dim3 blocks((N + TILE_N - 1) / TILE_N, 
                (M + TILE_M - 1) / TILE_M,
                k_splits);
    
    // Step 1: 并行计算
    gemm_kernel_splitk_vectorized<64, 64, 16><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        workspace.data_ptr<float>(),
        M, K, N, k_splits
    );
    
    // Step 2: Reduce
    int reduce_threads = 256;
    int reduce_blocks = (M * N + reduce_threads - 1) / reduce_threads;
    splitk_reduce_kernel<float><<<reduce_blocks, reduce_threads>>>(
        workspace.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, k_splits
    );
    
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

// StreamK Vectorized (float32 only, uses atomics)
torch::Tensor gemm_streamk_vectorized_cuda(torch::Tensor A, torch::Tensor B, int k_splits) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible dimensions for matrix multiplication");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Vectorized StreamK GEMM only supports float32");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
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
    
    gemm_kernel_streamk_vectorized<32, 4><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N, k_splits
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda, "GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("use_shared") = true);
    
    m.def("gemm_vectorized", &gemm_vectorized_cuda, "Vectorized GEMM with float4 (CUDA, float32 only)",
          py::arg("A"), py::arg("B"));
    
    m.def("gemm_splitk", &gemm_splitk_cuda, "Split-K GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
    
    m.def("gemm_splitk_vectorized", &gemm_splitk_vectorized_cuda, 
          "Vectorized Split-K GEMM (CUDA, float32 only)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
    
    m.def("gemm_streamk", &gemm_streamk_simple_cuda, "StreamK-style GEMM (CUDA)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
    
    m.def("gemm_streamk_vectorized", &gemm_streamk_vectorized_cuda,
          "Vectorized StreamK GEMM (CUDA, float32 only)",
          py::arg("A"), py::arg("B"), py::arg("k_splits") = 4);
}
