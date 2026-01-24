# 使用指南 Usage Guide

## 目录 Table of Contents

1. [安装方法](#安装方法-installation-methods)
2. [开发工作流](#开发工作流-development-workflow)
3. [添加新内核](#添加新内核-adding-new-kernels)
4. [性能优化建议](#性能优化建议-performance-optimization)

## 安装方法 Installation Methods

### 方法 1: 静态编译（生产环境）Static Compilation (Production)

适用于部署和生产环境，编译一次即可使用。

Suitable for deployment and production environments - compile once and use.

```bash
# 克隆仓库 Clone repository
git clone https://github.com/santiweide/effiective_kernels.git
cd effiective_kernels

# 安装依赖 Install dependencies
pip install -r requirements.txt

# 编译并安装 Compile and install
pip install -e .
```

安装后可以在任何地方导入：
After installation, you can import from anywhere:

```python
import efficient_kernels as ek
```

### 方法 2: JIT 编译（开发模式）JIT Compilation (Development Mode)

适用于开发和调试，无需安装即可使用。

Suitable for development and debugging - no installation needed.

```bash
# 仅需安装依赖 Only install dependencies
pip install torch

# 直接使用，首次会自动编译 Direct use, auto-compiles on first import
cd effiective_kernels
python
>>> import efficient_kernels as ek
# JIT compilation happens automatically
```

## 开发工作流 Development Workflow

### 场景 1: 修改现有内核 Modifying Existing Kernel

使用 JIT 模式可以快速迭代：

Use JIT mode for fast iteration:

```bash
# 1. 修改 CUDA 代码
# 1. Modify CUDA code
vim csrc/kernels/gemm.cu

# 2. 删除之前的 JIT 编译缓存（如果有）
# 2. Remove previous JIT compilation cache (if any)
rm -rf ~/.cache/torch_extensions/

# 3. 重新运行测试，会自动重新编译
# 3. Re-run test, will auto-recompile
python examples/basic_gemm.py
```

### 场景 2: 性能基准测试 Performance Benchmarking

```python
import torch
import efficient_kernels as ek
import time

# 准备数据
A = torch.randn(1024, 2048, device='cuda')
B = torch.randn(2048, 512, device='cuda')

# 预热
for _ in range(10):
    _ = ek.gemm(A, B)
torch.cuda.synchronize()

# 测试
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(100):
    C = ek.gemm(A, B)
end.record()
torch.cuda.synchronize()

print(f"Average time: {start.elapsed_time(end) / 100:.4f} ms")
```

## 添加新内核 Adding New Kernels

### 步骤 Steps

#### 1. 创建 CUDA 内核文件 Create CUDA Kernel File

在 `csrc/kernels/` 中创建新的 `.cu` 文件：

Create new `.cu` file in `csrc/kernels/`:

```cuda
// csrc/kernels/my_kernel.cu
#include <torch/extension.h>

__global__ void my_kernel_impl(...) {
    // Your CUDA kernel implementation
}

torch::Tensor my_kernel(torch::Tensor input) {
    // Host function
    // ...
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel, "My custom kernel");
}
```

#### 2. 更新 setup.py

在 `sources` 列表中添加新文件：

Add new file to `sources` list:

```python
ext_modules=[
    cpp_extension.CUDAExtension(
        name='efficient_kernels._C',
        sources=[
            'csrc/kernels/gemm.cu',
            'csrc/kernels/my_kernel.cu',  # 添加这一行 Add this line
        ],
        extra_compile_args=extra_compile_args,
    )
]
```

#### 3. 更新 Python 包 Update Python Package

在 `efficient_kernels/__init__.py` 中添加：

Add to `efficient_kernels/__init__.py`:

```python
def my_kernel(input: torch.Tensor) -> torch.Tensor:
    """Your kernel documentation"""
    ext = _get_extension()
    return ext.my_kernel(input)

__all__ = ['gemm', 'my_kernel', '__version__']
```

#### 4. 更新 JIT 编译源文件列表

在 `_get_jit_extension()` 函数中：

In `_get_jit_extension()` function:

```python
source_files = [
    str(package_dir / 'csrc' / 'kernels' / 'gemm.cu'),
    str(package_dir / 'csrc' / 'kernels' / 'my_kernel.cu'),  # 添加 Add
]
```

## 性能优化建议 Performance Optimization

### 1. 使用 Shared Memory

```cuda
__shared__ float shared_data[TILE_SIZE][TILE_SIZE];
```

### 2. 内存合并访问 Coalesced Memory Access

确保线程访问内存是连续的：

Ensure threads access memory contiguously:

```cuda
// Good: 合并访问 coalesced
float val = data[threadIdx.x + blockIdx.x * blockDim.x];

// Bad: 跨步访问 strided
float val = data[threadIdx.x * stride];
```

### 3. 避免分支发散 Avoid Branch Divergence

```cuda
// Good: 所有线程执行相同路径 All threads take same path
int idx = threadIdx.x < N ? threadIdx.x : 0;

// Bad: 线程分支 Thread divergence
if (threadIdx.x < N) {
    // Some threads execute this
} else {
    // Other threads execute this
}
```

### 4. 使用编译器优化标志

setup.py 中已包含：

Already included in setup.py:

```python
'-O3',              # 最高优化级别 Highest optimization
'--use_fast_math',  # 快速数学运算 Fast math
```

### 5. Profile 性能 Profile Performance

```bash
# 使用 NVIDIA Nsight Systems
nsys profile python examples/basic_gemm.py

# 使用 NVIDIA Nsight Compute
ncu python examples/basic_gemm.py
```

## 常见问题 Common Issues

### Q: JIT 编译失败怎么办？
### Q: What if JIT compilation fails?

**A:** 检查以下几点 Check the following:
1. CUDA Toolkit 是否正确安装 Is CUDA Toolkit properly installed
2. PyTorch CUDA 版本是否匹配 Does PyTorch CUDA version match
3. 清除编译缓存: `rm -rf ~/.cache/torch_extensions/`

### Q: 如何调试 CUDA 内核？
### Q: How to debug CUDA kernels?

**A:** 使用以下方法 Use these methods:
1. `printf()` in CUDA kernel (简单但有效 simple but effective)
2. `cuda-gdb` (NVIDIA 调试器 NVIDIA debugger)
3. Nsight tools (专业工具 professional tools)

### Q: 性能不如预期？
### Q: Performance not as expected?

**A:** 检查 Check:
1. 使用 `use_shared=True` 启用共享内存优化
2. 确保输入数据在 GPU 上 Ensure input data is on GPU
3. 使用合适的矩阵尺寸（避免太小的矩阵）Appropriate matrix sizes
4. Profile 找出瓶颈 Profile to find bottlenecks

## 参考资料 References

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch C++ Extension](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
