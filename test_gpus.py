import unittest
import logging
import subprocess
import sys
import os
import time

class GPUTest(unittest.TestCase):
    """Unit test class to validate GPU operations using PyTorch."""

    def test_gpu_operations(self):
        """Test GPU operations on all available GPUs."""
        self.logger.info(f"GPU count: {self.gpu_count}")
        if self.gpu_count < 1:
            self.skipTest("Skipping GPU operation tests due to no GPUs available.")

        start_time = time.time()
        for i in range(self.gpu_count):
            stream = self.torch.cuda.Stream(device=i)
            self.streams.append(stream)
            with self.torch.cuda.device(i), self.torch.cuda.stream(stream):
                a = self.torch.randn(1024, 1024, device=f"cuda:{i}")
                self.tensors.append((i, self.torch.matmul(a, a)))
                self.logger.info(f"GPU [i={i}] kernel launched...")

        for stream, (device_id, tensor) in zip(self.streams, self.tensors):
            stream.synchronize()
            device_properties = self.torch.cuda.get_device_properties(device_id)
            device_name = device_properties.name
            device_uuid_stem = str(device_properties.uuid).split("-")[0]
            self.logger.info(
                f"GPU [{device_id}][{device_name}][{device_uuid_stem}] [0][0]={tensor[0][0].item():.2f}"
            )

        self.logger.info(f"Completed in {time.time() - start_time:.2f}s.")

    @classmethod
    def setUpClass(cls):
        """Set up resources required for GPU tests, including logging, package installation, and GPU initialization."""
        cls.logger = cls.setup_logger()
        cls.logger.info(f"Started.")
        cls.logger.info(f"Checking prerequisites...")
        cls.ensure_pip_installed()
        cls.ensure_package_installed("torch")
        cls.ensure_package_installed("numpy")
        import torch
        cls.torch = torch
        cls.logger.info(f"Calling CUDA to get GPU device count...")
        cls.gpu_count = cls.torch.cuda.device_count()
        cls.streams = []
        cls.tensors = []

    @staticmethod
    def ensure_pip_installed():
        """Ensure `pip` is installed for the current Python interpreter."""
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        except subprocess.CalledProcessError:
            logging.info(f"Pip not installed. Installing...")
            subprocess.check_call(["sudo", "apt", "update"])
            subprocess.check_call(["sudo", "apt", "install", "-y", "python3-pip"])

    @staticmethod
    def ensure_package_installed(package_name):
        """Ensure the specified Python package is installed.

        Args:
            package_name (str): The name of the package to check and install if necessary.
        """
        try:
            __import__(package_name)
        except ImportError:
            logging.info(f"{package_name} not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            __import__(package_name)

    @staticmethod
    def setup_logger():
        """Set up a logger for the test suite.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return logging.getLogger()

if __name__ == "__main__":
    unittest.main()
