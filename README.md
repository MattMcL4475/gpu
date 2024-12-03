# GPU test code

To test how many GPUs are available from a single process, run this:

`curl -sSL https://raw.githubusercontent.com/MattMcL4475/gpu/refs/heads/main/test_gpus.py | python3 -`

Example output on an Azure [Standard_NC96ads_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series):
```bash
$ curl -sSL https://raw.githubusercontent.com/MattMcL4475/gpu/refs/heads/main/test_gpus.py | python3 -
2024-12-03 00:15:40 [INFO]: Started.
2024-12-03 00:15:40 [INFO]: Checking prerequisites...
2024-12-03 00:15:41 [INFO]: Calling CUDA to get GPU device count...
2024-12-03 00:16:07 [INFO]: GPU count: 4
2024-12-03 00:16:08 [INFO]: GPU [i=0] kernel launched...
2024-12-03 00:16:08 [INFO]: GPU [i=1] kernel launched...
2024-12-03 00:16:08 [INFO]: GPU [i=2] kernel launched...
2024-12-03 00:16:08 [INFO]: GPU [i=3] kernel launched...
2024-12-03 00:16:08 [INFO]: GPU [0][NVIDIA A100 80GB PCIe][8ef45637] [0][0]=34.22
2024-12-03 00:16:08 [INFO]: GPU [1][NVIDIA A100 80GB PCIe][d3cf470b] [0][0]=-16.10
2024-12-03 00:16:08 [INFO]: GPU [2][NVIDIA A100 80GB PCIe][8976c82e] [0][0]=2.64
2024-12-03 00:16:08 [INFO]: GPU [3][NVIDIA A100 80GB PCIe][09a9da88] [0][0]=-83.37
2024-12-03 00:16:08 [INFO]: Completed in 1.02s.
.
----------------------------------------------------------------------
Ran 1 test in 28.610s

OK
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
