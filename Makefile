PYTHON ?= .venv/bin/python
CUDA_HOME ?= /usr/local/cuda
TORCH_CUDA_ARCH_LIST ?= 12.0
BENCHMARK_ARGS ?=

.PHONY: build-cuda benchmark benchmark-cuda benchmark-all

build-cuda:
	CUDA_HOME="$(CUDA_HOME)" TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		"$(PYTHON)" setup_cuda.py build_ext --inplace

benchmark:
	"$(PYTHON)" torch_transformer_benchmark.py --device cuda --dtype float32 \
		--optimized-backend pytorch --compile-user \
		--compile-mode reduce-overhead $(BENCHMARK_ARGS)

benchmark-cuda: build-cuda
	"$(PYTHON)" torch_transformer_benchmark.py --device cuda --dtype float32 \
		--optimized-backend cuda $(BENCHMARK_ARGS)

benchmark-all: build-cuda benchmark benchmark-cuda
