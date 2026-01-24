# Efficient Kernels

独立于 PyTorch 的自定义 CUDA 内核仓库，支持静态编译和 JIT 动态加载。

A standalone repository for custom CUDA kernels independent of PyTorch, supporting both static compilation and JIT dynamic loading.

## 特性 Features

- ✅ **独立编译**: 无需重新编译整个 PyTorch，直接编译为 Python 可导入的 `.so` 文件
- ✅ **JIT 动态加载**: 支持开发模式下的即时编译，修改代码后无需重新安装
- ✅ **GEMM 示例**: 包含优化的矩阵乘法内核实现（含 shared memory 优化）
- ✅ **易于扩展**: 清晰的项目结构，方便添加新的 CUDA 内核

- ✅ **Independent Compilation**: Compile directly to Python-importable `.so` files without rebuilding PyTorch
- ✅ **JIT Dynamic Loading**: Supports just-in-time compilation in development mode - no reinstall needed after code changes
- ✅ **GEMM Example**: Includes optimized matrix multiplication kernel (with shared memory optimization)
- ✅ **Easy to Extend**: Clear project structure for adding new CUDA kernels

## 安装 Installation

### 方式 1: 静态编译安装 Static Compilation

```bash
pip install -e .
```

这会将 CUDA 内核编译为静态库，适合生产环境使用。

This compiles CUDA kernels into a static library, suitable for production use.

### 方式 2: JIT 开发模式 JIT Development Mode

直接使用即可，首次导入时会自动触发 JIT 编译：

Simply use directly - JIT compilation will be triggered automatically on first import:

```python
import efficient_kernels as ek
# JIT compilation happens here if not already compiled
```

## 快速开始 Quick Start

### 基本使用 Basic Usage

```python
import torch
import efficient_kernels as ek

# 创建测试矩阵 Create test matrices
A = torch.randn(512, 1024, device='cuda')
B = torch.randn(1024, 256, device='cuda')

# 使用自定义 GEMM 内核 Use custom GEMM kernel
C = ek.gemm(A, B, use_shared=True)

# 验证结果 Verify results
C_ref = torch.matmul(A, B)
print(f"Max difference: {torch.max(torch.abs(C - C_ref))}")
```

### 运行示例 Run Examples

```bash
# 基本 GEMM 示例和性能测试
# Basic GEMM example with performance benchmark
python examples/basic_gemm.py

# JIT 编译示例
# JIT compilation example
python examples/jit_example.py
```

## 项目结构 Project Structure

```
efficient_kernels/
├── csrc/                      # C++/CUDA 源代码 Source code
│   └── kernels/
│       └── gemm.cu           # GEMM 内核实现 GEMM kernel implementation
├── efficient_kernels/         # Python 包 Python package
│   └── __init__.py           # Python 接口和 JIT 支持 Python interface with JIT support
├── examples/                  # 示例脚本 Example scripts
│   ├── basic_gemm.py         # 基本使用示例 Basic usage example
│   └── jit_example.py        # JIT 编译示例 JIT compilation example
├── setup.py                   # 安装脚本 Installation script
└── README.md
```

## 添加新内核 Adding New Kernels

1. 在 `csrc/kernels/` 中创建 `.cu` 文件
2. 在 `setup.py` 的 `sources` 列表中添加新文件
3. 在 `efficient_kernels/__init__.py` 中添加 Python 接口
4. 更新 JIT 编译的源文件列表

1. Create `.cu` file in `csrc/kernels/`
2. Add new file to `sources` list in `setup.py`
3. Add Python interface in `efficient_kernels/__init__.py`
4. Update JIT compilation source file list

## 开发模式 Development Mode

JIT 模式的优势 / Advantages of JIT mode:
- 修改内核代码后无需重新运行 `pip install` / No need to run `pip install` after modifying kernel code
- 快速迭代和调试 / Fast iteration and debugging
- 自动重新编译 / Automatic recompilation

## 性能说明 Performance Notes

本仓库的 GEMM 实现是教学示例，主要展示如何独立编译 CUDA 内核。实际生产环境建议使用高度优化的库如 cuBLAS。

The GEMM implementation in this repository is an educational example demonstrating how to compile CUDA kernels independently. For production use, highly optimized libraries like cuBLAS are recommended.

## 要求 Requirements

- Python >= 3.7
- PyTorch >= 1.8.0
- CUDA Toolkit
- NVIDIA GPU with compute capability >= 7.0

## 许可 License

MIT License

## 贡献 Contributing

欢迎提交问题和拉取请求！

Issues and pull requests are welcome!