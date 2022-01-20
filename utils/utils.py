"""
CUDA and other utilities
"""
import torch


def print_cuda_info():
    """
    Print CUDA information
    """
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))

    print("#gpu:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print("gpu:", i)
        print(torch.cuda.get_device_name(i))

        tot_m = torch.cuda.get_device_properties(0).total_memory
        res_m = torch.cuda.memory_reserved(0)
        all_m = torch.cuda.memory_allocated(0)
        fre_m = res_m - all_m  # free inside reserved
        print("GPU total memory:", tot_m)
        print(fre_m / 1e6)
        print("Cuda version:", torch.version.cuda)


def ensure_path(string: str) -> str:
    """
    Ensure that a string is a path
    """
    if string[-1] != "/":
        string += "/"

    return string
