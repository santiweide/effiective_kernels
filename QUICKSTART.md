# Quick Start Guide 快速入门

hendrix 
```shell
module load gcc/11.2.0
```

## 5分钟上手 Get Started in 5 Minutes

### 1️⃣ 克隆并进入仓库 Clone and Enter Repository

```bash
git clone https://github.com/santiweide/effiective_kernels.git
cd effiective_kernels
```

### 2️⃣ 安装依赖 Install Dependencies

```bash
pip install torch  # 或使用 CUDA 版本: or with CUDA:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3️⃣ 选择模式 Choose Mode

#### 🔧 开发模式 (Development Mode) - JIT 编译

**不需要安装，直接使用！No installation needed, use directly!**

```bash
# 运行 JIT 示例 Run JIT example
python examples/jit_example.py
```

第一次运行会自动编译（需要几秒钟），之后就很快了。
First run will auto-compile (takes a few seconds), then it's fast.

#### 🚀 生产模式 (Production Mode) - 静态编译

**需要 CUDA 环境 Requires CUDA environment**

```bash
# 编译并安装 Compile and install
pip install -e .

# 使用 Use
python -c "import efficient_kernels as ek; print('Version:', ek.__version__)"
```

### 4️⃣ 运行示例 Run Examples

```bash
# 基本使用和性能测试 Basic usage and benchmarking
python examples/basic_gemm.py

# JIT 编译演示 JIT compilation demo
python examples/jit_example.py

# 运行测试 Run tests
python tests/test_gemm.py
```

### 5️⃣ 在代码中使用 Use in Your Code

```python
import torch
import efficient_kernels as ek

# 创建测试数据 Create test data
A = torch.randn(512, 1024, device='cuda')
B = torch.randn(1024, 256, device='cuda')

# 使用自定义 GEMM 内核 Use custom GEMM kernel
C = ek.gemm(A, B, use_shared=True)

# 验证结果 Verify result
C_ref = torch.matmul(A, B)
print(f"Difference: {torch.max(torch.abs(C - C_ref))}")
```

## 📖 详细文档 Detailed Documentation

- **README.md**: 项目概览 Project overview
- **USAGE_GUIDE.md**: 详细使用指南 Detailed usage guide
- **ARCHITECTURE.md**: 技术架构文档 Technical architecture

## 🆘 遇到问题？Troubleshooting

### CUDA 未安装 CUDA not installed
- 使用 JIT 模式需要 CUDA
- 可以先查看代码结构，在有 CUDA 的机器上运行

### 编译失败 Compilation failed
```bash
# 清除缓存 Clear cache
rm -rf ~/.cache/torch_extensions/

# 重新尝试 Try again
python examples/jit_example.py
```

### 导入错误 Import error
```bash
# 确保在正确目录 Make sure in correct directory
cd effiective_kernels

# 或安装包 Or install package
pip install -e .
```

## 🎯 下一步 Next Steps

1. 📖 阅读 `USAGE_GUIDE.md` 学习如何添加新内核
2. 🔍 查看 `csrc/kernels/gemm.cu` 了解内核实现
3. ✏️ 修改代码，立即测试（JIT 模式自动重新编译）
4. 📚 阅读 `ARCHITECTURE.md` 了解技术细节

祝你开发顺利！Happy coding! 🚀
