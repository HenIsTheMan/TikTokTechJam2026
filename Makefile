benchmark:
	.venv/bin/python torch_transformer_benchmark.py \
	--device cuda \
	--dtype float32 \
	--compile-user \
	--compile-mode reduce-overhead