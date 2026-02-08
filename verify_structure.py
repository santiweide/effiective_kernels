#!/usr/bin/env python
"""
Verification script to check repository structure and configuration
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists"""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def check_content(path, patterns, description):
    """Check if file contains expected patterns"""
    if not os.path.exists(path):
        print(f"✗ {description}: File not found")
        return False
    
    with open(path, 'r') as f:
        content = f.read()
    
    all_found = True
    for pattern in patterns:
        if pattern not in content:
            print(f"  ✗ Missing pattern: {pattern}")
            all_found = False
    
    status = "✓" if all_found else "✗"
    print(f"{status} {description}")
    return all_found

def main():
    print("=" * 70)
    print("Repository Structure Verification")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    os.chdir(base_dir)
    
    checks = []
    
    # Check directory structure
    print("\n1. Directory Structure")
    print("-" * 70)
    checks.append(check_file("csrc/kernels", "CUDA kernels directory"))
    checks.append(check_file("efficient_kernels", "Python package directory"))
    checks.append(check_file("examples", "Examples directory"))
    checks.append(check_file("tests", "Tests directory"))
    
    # Check essential files
    print("\n2. Essential Files")
    print("-" * 70)
    checks.append(check_file("setup.py", "Setup script"))
    checks.append(check_file("requirements.txt", "Requirements file"))
    checks.append(check_file("README.md", "README documentation"))
    checks.append(check_file(".gitignore", "Git ignore file"))
    checks.append(check_file("Makefile", "Makefile"))
    
    # Check CUDA kernel
    print("\n3. CUDA Kernel Implementation")
    print("-" * 70)
    checks.append(check_content(
        "csrc/kernels/gemm.cu",
        ["__global__", "PYBIND11_MODULE", "torch::Tensor", "__shared__"],
        "GEMM kernel implementation"
    ))
    
    # Check Python package
    print("\n4. Python Package")
    print("-" * 70)
    checks.append(check_content(
        "efficient_kernels/__init__.py",
        ["_get_extension", "_get_jit_extension", "def gemm", "__version__"],
        "Package initialization with JIT support"
    ))
    
    # Check setup.py
    print("\n5. Build Configuration")
    print("-" * 70)
    checks.append(check_content(
        "setup.py",
        ["CUDAExtension", "efficient_kernels._C", "gemm.cu", "BuildExtension"],
        "Setup.py configuration"
    ))
    
    # Check examples
    print("\n6. Example Scripts")
    print("-" * 70)
    checks.append(check_file("examples/basic_gemm.py", "Basic GEMM example"))
    checks.append(check_file("examples/jit_example.py", "JIT compilation example"))
    
    # Check tests
    print("\n7. Tests")
    print("-" * 70)
    checks.append(check_file("tests/test_gemm.py", "GEMM tests"))
    
    # Check documentation
    print("\n8. Documentation")
    print("-" * 70)
    checks.append(check_file("USAGE_GUIDE.md", "Usage guide"))
    checks.append(check_file("ARCHITECTURE.md", "Architecture documentation"))
    
    # Try importing (syntax check)
    print("\n9. Python Syntax Validation")
    print("-" * 70)
    try:
        sys.path.insert(0, str(base_dir))
        import efficient_kernels
        print(f"✓ Package imports successfully (version {efficient_kernels.__version__})")
        checks.append(True)
    except Exception as e:
        print(f"✗ Package import failed: {e}")
        checks.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    total = len(checks)
    passed = sum(checks)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! Repository is properly structured.")
        return 0
    else:
        print(f"\n✗ {total - passed} check(s) failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
