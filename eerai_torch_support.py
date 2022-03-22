#!/usr/bin/env python
# Copyright 2022 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
File: eerai_torch_support.py

This script checks for the HW support in pytorch
"""

# Built-in modules
import sys

# Third party modules
import torch

# Own modules

# Constants

# Functions

# Main


def info(msg):
    print("eerai_torch_support: %s" % str(msg))


def main():
    """
    Program main
    """
    info("CUDNN: Enabled: %s" % torch.backends.cudnn.enabled)
    info("CUDNN: Available: %s" % torch.backends.cudnn.is_available())
    info("CUDNN: Version: %s" % torch.backends.cudnn.version())
    info("CUDDN: Allow tf32: %s" % torch.backends.cudnn.allow_tf32)
    info("CUDDN: Benchmark: %s" % torch.backends.cudnn.benchmark)
    info("CUDA: Number of GPUs: %s" % torch.cuda.device_count())
    info("CUDA: Supported arch: %s" % torch.cuda.get_arch_list())
    info("MKL: Available: %s" % torch.backends.mkl.is_available())
    info("MKLDNN: Available: %s" % torch.backends.mkldnn.is_available())
    info("OPENMP: Available: %s" % torch.backends.openmp.is_available())
    sys.exit(0)


if __name__ == "__main__":  # run main if executed from the command line
    # and the main method exists

    if callable(locals().get("main")):
        main()
        sys.exit(0)
