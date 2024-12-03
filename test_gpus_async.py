import asyncio
import time
import os
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

try:
    import torch
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "torch"])
    import torch

try:
    import numpy
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "numpy"])
    import numpy

async def get_device_properties_async(device_id):
    device_properties = torch.cuda.get_device_properties(device_id)
    return {
        "uuid": device_properties.uuid,
        "name": device_properties.name,
    }

async def launch_matmul_on_device(device_id):
    stream = torch.cuda.Stream(device=device_id)
    try:
        with torch.cuda.device(device_id), torch.cuda.stream(stream):
            a = torch.randn(1024, 1024, device=f'cuda:{device_id}')
            result = torch.matmul(a, a)
            stream.synchronize()
            return result
    except Exception as e:
        return e

async def main():
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        logger.info("No GPUs available. Exiting.")
        return

    start_time = time.time()

    device_property_tasks = [get_device_properties_async(i) for i in range(gpu_count)]
    matmul_tasks = [launch_matmul_on_device(i) for i in range(gpu_count)]

    device_properties_list = await asyncio.gather(*device_property_tasks)
    matmul_results = await asyncio.gather(*matmul_tasks)

    for device_id, (device_properties, result) in enumerate(zip(device_properties_list, matmul_results)):
        if isinstance(result, Exception):
            logger.error(f"GPU [i={device_id}] encountered an error: {result}")
        else:
            logger.info(
                f"GPU [i={device_id}, uuid={device_properties['uuid']}, name={device_properties['name']}] "
                f"done. [0][0]={result[0][0].item()}"
            )

    logger.info(f"Completed in {time.time() - start_time:.2f}s.")

if __name__ == "__main__":
    asyncio.run(main())
