#check_dgx_spark
import os
import platform
import torch
import subprocess


def main():
    print("=== System Check ===")
    print("Machine:", platform.machine())
    print("Platform:", platform.platform())
    print("Python:", platform.python_version())

    print("\n=== CUDA / Torch ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        print("device 0:", torch.cuda.get_device_name(0))

    print("\n=== nvidia-smi ===")
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True)
        print(out[:2000])
    except Exception as e:
        print("nvidia-smi failed:", e)

    print("\n=== Environment ===")
    print("HF_HOME =", os.environ.get("HF_HOME"))
    print("TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
    
    print("\n=== Kernel ===")
    print("flash_sdp:", torch.backends.cuda.flash_sdp_enabled())
    print("sdpa:", torch.backends.cuda.sdp_kernel())


if __name__ == "__main__":
    main()