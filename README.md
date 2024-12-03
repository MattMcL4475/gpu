# GPU test code

To test how many GPUs are available from a single process, run this:

`curl -sSL https://raw.githubusercontent.com/MattMcL4475/gpu/refs/heads/main/test_gpus.py | python3 -`

Example output:
```bash
$ curl -sSL https://raw.githubusercontent.com/MattMcL4475/gpu/refs/heads/main/test_gpus.py | python3 -
GPU count: 1
Launching torch.matmul [CUDA cublasSgemm()] on each GPU...
GPU [i=0, name=NVIDIA GeForce RTX 3080, uuid=f14bc10b-58cc-4372-a567-0e02b2c3d479] launched...
GPU [i=0, uuid=f14bc10b-58cc-4372-a567-0e02b2c3d479] done. [0][0]=18.236143112182617
Completed in 0.57s.
$
```

## `cublas<t>gemm`
https://docs.nvidia.com/cuda/cublas/#cublas-t-gemm

```cpp
cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
```

## `torch.matmul` calling `cublasSgemm`
https://github.com/pytorch/pytorch/blob/0f3f801fc2657cca74c4d45bc0b7ab5fd48005d5/aten/src/ATen/cuda/CUDABlas.cpp#L824

## Full async version (experimental)
*Note: unclear if `torch` is truly thread-safe. Also, the `asyncio` overhead makes this slower for one GPU.*

`curl -sSL https://raw.githubusercontent.com/MattMcL4475/gpu/refs/heads/main/test_gpus_async.py | python3 -`
