CUDA_HOME ?= /usr/local/cuda
TORCH_CUDA_ARCH_LIST ?= 12.0
CMD_LINE_ARGS ?=

.DEFAULT_GOAL := cpu

.PHONY: cuda cpu build-cuda clean

cuda: build-cuda
	python3 Sandbox.py --device cuda $(CMD_LINE_ARGS)

cpu:
	python3 Sandbox.py --device cpu $(CMD_LINE_ARGS)

build-cuda:
	CUDA_HOME="$(CUDA_HOME)" TORCH_CUDA_ARCH_LIST="$(TORCH_CUDA_ARCH_LIST)" \
		python3 setup_cuda.py build_ext --inplace

clean:
	rm -rf build
	rm -f transformer_cuda_ext*.so
	rm -rf ./*.egg-info