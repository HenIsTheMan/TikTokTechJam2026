PYTHON ?= .venv/bin/python
CUDA_HOME ?= /usr/local/cuda
TORCH_CUDA_ARCH_LIST ?= 12.0
BENCHMARK_ARGS ?=

.PHONY: build-cuda benchmark benchmark-cuda benchmark-hybrid benchmark-aggressive \
	benchmark-all tune-aggressive profile-aggressive

build-cuda:
	CUDA_HOME="$(CUDA_HOME)" TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		"$(PYTHON)" setup_cuda.py build_ext --inplace

benchmark:
	"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--optimized-backend pytorch --compile-user \
		--compile-mode reduce-overhead $(BENCHMARK_ARGS)

benchmark-cuda: build-cuda
	"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--optimized-backend cuda $(BENCHMARK_ARGS)

benchmark-hybrid: build-cuda
	"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--optimized-backend cuda-hybrid $(BENCHMARK_ARGS)

benchmark-aggressive: build-cuda
	"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--optimized-backend cuda-aggressive $(BENCHMARK_ARGS)

benchmark-all: build-cuda benchmark benchmark-cuda benchmark-hybrid benchmark-aggressive

tune-aggressive: build-cuda
	"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--search-tf32-plan --accuracy-trials 5 \
		--search-beam-width 4 --search-steps 4 \
		--tuning-output aggressive_tuning.json $(BENCHMARK_ARGS)

profile-aggressive: build-cuda
	nsys profile --trace=cuda,nvtx,cublas --force-overwrite=true \
		--output=aggressive_profile \
		"$(PYTHON)" Sandbox.py --device cuda --dtype float32 \
		--optimized-backend cuda-aggressive --no-cuda-graph --profile-ranges --warmup 10 \
		--repeats 20 --benchmark-rounds 1 $(BENCHMARK_ARGS)

benchmark-best:
	make benchmark-aggressive BENCHMARK_ARGS="--causal --padding-ratio 0.35"
