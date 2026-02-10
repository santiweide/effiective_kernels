#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// -------------------------------------------------------
// Kernel: 按照 stride 步长读取数组
// -------------------------------------------------------
// 每个线程只读取一个数： data[tid * stride]
// 这意味着 Stride 越大，线程间访问的内存地址越分散
__global__ void stride_read_kernel(float* __restrict__ data, float* __restrict__ out, int stride, int n) {
    // 计算全局线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 计算实际要访问的内存索引
    // 使用 long long 防止在大 stride 下索引溢出
    long long idx = (long long)tid * stride;

    if (idx < n) {
        // 读取 data (关键测试点) 并写入 out (防止被编译器优化掉)
        out[tid] = data[idx];
    }
}

int main(int argc, char **argv) {
    // 1. 设置实验参数
    // 我们总共使用 100 万个线程进行测试
    int num_threads_total = 1 << 20; // approx 1 Million threads
    int block_size = 256;
    int grid_size = (num_threads_total + block_size - 1) / block_size;

    // 最大 Stride 设为 32 (对应 float 的 128 字节，正好一个 L1 Cache Line)
    int max_stride = 32;
    
    // 数组需要足够大，以容纳最大 stride 下的访问范围
    // Size = threads * max_stride
    size_t n_elements = (size_t)num_threads_total * max_stride;
    size_t bytes = n_elements * sizeof(float);
    
    // 仅用于输出的数组大小 (只存结果，不用 stride)
    size_t out_bytes = num_threads_total * sizeof(float);

    printf("Array Size: %.2f MB\n", bytes / (1024.0 * 1024.0));
    printf("Total Threads: %d\n", num_threads_total);
    printf("----------------------------------------------------------------\n");
    printf("| Stride | Time (ms) | Bandwidth (GB/s) | Relative Speed |\n");
    printf("----------------------------------------------------------------\n");

    // 2. 分配内存
    float *h_data = (float*)malloc(bytes);
    float *d_data, *d_out;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_out, out_bytes));

    // 初始化数据 (简单填充)
    for (size_t i = 0; i < n_elements; i++) h_data[i] = 1.0f;
    CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // 3. 循环测试不同的 Stride
    // Stride: 1, 2, 4, 8, 16, 32
    int strides[] = {1, 2, 4, 8, 16, 32};
    float base_bandwidth = 0.0f;

    for (int i = 0; i < 6; i++) {
        int current_stride = strides[i];

        // 预热 (Warmup) - 让 GPU 频率跑起来
        stride_read_kernel<<<grid_size, block_size>>>(d_data, d_out, current_stride, n_elements);
        CHECK(cudaDeviceSynchronize());

        // 计时开始
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));
        
        // 运行多次以获得稳定数据
        int n_iter = 100;
        for(int j=0; j<n_iter; j++) {
            stride_read_kernel<<<grid_size, block_size>>>(d_data, d_out, current_stride, n_elements);
        }

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        // 4. 计算指标
        // 有效数据量：每次迭代只有 num_threads_total * 4 bytes 是真正被读取的"有用"数据
        // 注意：这是"有效带宽"，不是物理带宽。它反映了数据读取的效率。
        double total_data_read = (double)num_threads_total * sizeof(float) * n_iter;
        double gb_per_sec = (total_data_read / 1e9) / (milliseconds / 1000.0);

        if (current_stride == 1) base_bandwidth = gb_per_sec;

        printf("|   %2d   |  %7.3f  |      %6.2f      |      %3.0f%%     |\n", 
               current_stride, milliseconds, gb_per_sec, (gb_per_sec / base_bandwidth) * 100);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("----------------------------------------------------------------\n");

    // 清理
    cudaFree(d_data);
    cudaFree(d_out);
    free(h_data);

    return 0;
}
