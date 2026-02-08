# 架构说明 Architecture Documentation

## 概述 Overview

本仓库实现了一个独立于 PyTorch 主仓库的 CUDA 内核开发框架，支持两种编译模式：

This repository implements a CUDA kernel development framework independent of the main PyTorch repository, supporting two compilation modes:

1. **静态编译模式 Static Compilation**: 编译为 `.so` 文件，适合生产环境
2. **JIT 动态编译模式 JIT Dynamic Compilation**: 即时编译，适合开发调试

## 目录结构 Directory Structure

```
effiective_kernels/
├── csrc/                          # C++/CUDA 源代码目录
│   └── kernels/                   # 内核实现
│       └── gemm.cu               # GEMM 内核示例
│
├── efficient_kernels/             # Python 包目录
│   └── __init__.py               # 包初始化，JIT 支持
│
├── examples/                      # 示例脚本
│   ├── basic_gemm.py             # 基本使用示例
│   └── jit_example.py            # JIT 编译示例
│
├── tests/                         # 测试
│   └── test_gemm.py              # GEMM 内核测试
│
├── setup.py                       # 静态编译配置
├── requirements.txt               # Python 依赖
├── Makefile                       # 便捷命令
├── README.md                      # 项目说明
├── USAGE_GUIDE.md                # 使用指南
└── ARCHITECTURE.md               # 本文件
```

## 核心组件 Core Components

### 1. CUDA 内核实现 (csrc/kernels/)

CUDA 内核使用 PyTorch 的 C++ 扩展 API：

CUDA kernels use PyTorch's C++ extension API:

```cuda
// 内核函数 Kernel function
__global__ void gemm_kernel(...) {
    // CUDA implementation
}

// Host 函数 Host function
torch::Tensor gemm_cuda(torch::Tensor A, torch::Tensor B) {
    // Launch kernel
    gemm_kernel<<<blocks, threads>>>(...);
    return output;
}

// Python 绑定 Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda, "GEMM kernel");
}
```

**关键技术点 Key Technical Points:**

- 使用 `torch::Tensor` 作为参数类型，与 PyTorch 无缝集成
- 使用 `AT_DISPATCH_FLOATING_TYPES` 支持多种数据类型
- 使用 PyBind11 创建 Python 绑定
- CUDA 内核优化：shared memory, memory coalescing

### 2. Python 包 (efficient_kernels/)

`__init__.py` 实现了智能加载机制：

`__init__.py` implements intelligent loading mechanism:

```python
# 1. 尝试加载预编译的扩展 Try to load pre-compiled extension
try:
    from . import _C
    _compiled_available = True
except ImportError:
    _compiled_available = False

# 2. JIT 编译回退 JIT compilation fallback
def _get_jit_extension():
    from torch.utils.cpp_extension import load
    return load(name='...', sources=[...])

# 3. 统一接口 Unified interface
def _get_extension():
    if _compiled_available:
        return _C  # 使用预编译版本 Use pre-compiled
    else:
        return _get_jit_extension()  # JIT 编译 JIT compile
```

**设计优势 Design Advantages:**

- 用户无需关心底层实现，统一使用 `ek.gemm()`
- 开发时自动 JIT 编译，生产时使用预编译版本
- 优雅的错误处理和提示

### 3. 静态编译配置 (setup.py)

使用 PyTorch 的 `cpp_extension` 模块：

Uses PyTorch's `cpp_extension` module:

```python
from torch.utils import cpp_extension

setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='efficient_kernels._C',
            sources=['csrc/kernels/gemm.cu'],
            extra_compile_args={
                'nvcc': ['-O3', '--use_fast_math', ...]
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
```

**编译流程 Compilation Flow:**

1. `pip install -e .` 触发 `setup.py`
2. `CUDAExtension` 自动检测 CUDA 环境
3. 调用 `nvcc` 编译 `.cu` 文件
4. 生成 `efficient_kernels/_C.cpython-*.so`
5. Python 可以直接 `import efficient_kernels._C`

## 工作原理 How It Works

### 静态编译模式 Static Compilation Mode

```
┌─────────────┐
│  setup.py   │
└──────┬──────┘
       │
       ├─→ 检测 CUDA 环境 Detect CUDA
       ├─→ 编译 .cu → .o  Compile .cu → .o
       ├─→ 链接生成 .so   Link to create .so
       │
       ▼
┌─────────────────────┐
│ efficient_kernels/  │
│   _C.so             │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Python import       │
│ efficient_kernels   │
└─────────────────────┘
```

### JIT 动态编译模式 JIT Dynamic Compilation Mode

```
┌─────────────────────┐
│ import              │
│ efficient_kernels   │
└──────┬──────────────┘
       │
       ├─→ 尝试导入 _C.so  Try import _C.so
       │   (失败 Failed)
       │
       ├─→ 触发 JIT 编译   Trigger JIT compile
       │   torch.utils.cpp_extension.load()
       │
       ├─→ 编译 .cu → .so  Compile .cu → .so
       │   缓存到 ~/.cache/torch_extensions/
       │
       ▼
┌─────────────────────┐
│ JIT 编译的模块      │
│ 存储在缓存目录      │
└─────────────────────┘
```

**JIT 编译缓存位置 JIT Compilation Cache Location:**

```
~/.cache/torch_extensions/
└── py{version}_cu{cuda_version}/
    └── efficient_kernels_jit/
        ├── efficient_kernels_jit.so
        └── ...
```

## 数据流 Data Flow

```
Python
  ↓
torch.Tensor (CUDA)
  ↓
efficient_kernels.gemm(A, B)
  ↓
_C.gemm() or _jit.gemm()
  ↓
gemm_cuda() [C++ host function]
  ↓
gemm_kernel<<<>>>() [CUDA kernel]
  ↓
GPU 计算 GPU computation
  ↓
torch.Tensor (CUDA) [result]
  ↓
Python
```

## 性能考虑 Performance Considerations

### 1. 编译优化 Compilation Optimization

```python
extra_compile_args = {
    'cxx': ['-O3'],                    # C++ 最高优化
    'nvcc': [
        '-O3',                         # CUDA 最高优化
        '--use_fast_math',             # 快速数学运算
        '-lineinfo',                   # 调试信息
        '-gencode=arch=compute_XX,...' # 目标架构
    ]
}
```

### 2. 内核优化 Kernel Optimization

**基础版本 Basic Version:**
- 直接全局内存访问
- 简单实现，性能一般

**优化版本 Optimized Version:**
- Shared memory tiling
- Memory coalescing
- 减少全局内存访问

### 3. PyTorch 集成优化 PyTorch Integration Optimization

- 直接使用 `torch::Tensor` 避免数据拷贝
- `AT_DISPATCH_FLOATING_TYPES` 支持类型分发
- CUDA stream 自动管理

## 扩展性设计 Extensibility Design

### 添加新内核的模板 Template for Adding New Kernel

**1. CUDA 文件模板:**

```cuda
#include <torch/extension.h>

__global__ void my_kernel_impl(...) {
    // Implementation
}

torch::Tensor my_kernel(torch::Tensor input) {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_kernel", ([&] {
        my_kernel_impl<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), ...
        );
    }));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("my_kernel", &my_kernel, "My kernel description");
}
```

**2. Python 包装模板:**

```python
def my_kernel(input: torch.Tensor, param: int) -> torch.Tensor:
    """
    Kernel description
    
    Args:
        input: Input tensor
        param: Some parameter
        
    Returns:
        Output tensor
    """
    ext = _get_extension()
    return ext.my_kernel(input, param)
```

## 最佳实践 Best Practices

### 开发流程 Development Workflow

1. **原型开发 Prototyping**: 使用 JIT 模式快速迭代
2. **性能优化 Optimization**: 使用 profiling 工具找出瓶颈
3. **测试验证 Testing**: 编写单元测试确保正确性
4. **生产部署 Production**: 使用静态编译模式

### 调试技巧 Debugging Tips

1. **CUDA 错误检查**: 使用 `cudaGetLastError()` 和 `cudaDeviceSynchronize()`
2. **Printf 调试**: CUDA kernel 中可以使用 `printf()`
3. **Nsight Tools**: 使用专业工具进行深度分析

## 与 PyTorch 主仓库的区别 Differences from Main PyTorch

| 特性 Feature | 本仓库 This Repo | PyTorch 主仓库 Main PyTorch |
|-------------|------------------|----------------------------|
| 编译时间 Compile Time | 秒级 Seconds | 小时级 Hours |
| 独立性 Independence | 完全独立 Fully independent | 耦合 Coupled |
| 灵活性 Flexibility | 高 High | 低 Low |
| 适用场景 Use Case | 快速原型/研究 Rapid prototyping/research | 生产环境 Production |

## 未来扩展方向 Future Extensions

1. **多 GPU 支持 Multi-GPU Support**: 添加分布式计算支持
2. **更多算子 More Operators**: GEMV, Convolution, Attention 等
3. **自动调优 Auto-tuning**: 自动选择最优参数
4. **模板化 Templating**: 提供内核生成器

## 参考文献 References

- [PyTorch Custom C++ and CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyBind11 Documentation](https://pybind11.readthedocs.io/)
