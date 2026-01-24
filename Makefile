.PHONY: install clean test example help

help:
	@echo "Available commands:"
	@echo "  make install    - Install package in development mode"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make test       - Run tests"
	@echo "  make example    - Run basic example"
	@echo "  make jit        - Run JIT example"

install:
	pip install -e .

clean:
	rm -rf build dist *.egg-info
	find . -name "*.so" -delete
	find . -name "*.o" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

test:
	python tests/test_gemm.py

example:
	python examples/basic_gemm.py

jit:
	python examples/jit_example.py
