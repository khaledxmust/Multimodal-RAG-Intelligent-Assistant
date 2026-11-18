from typing import Any
import ctypes
import platform
import subprocess

# ============================================================
# NVIDIA HARDWARE DETECTION
# ============================================================

def get_nvidia_gpu_info() -> dict[str, Any]:
    """
    Detect NVIDIA GPU hardware using nvidia-smi.

    This does NOT mean that PyTorch or another framework
    can necessarily use the GPU.
    """

    result = {
        "available": False,
        "name": None,
        "vram_gb": 0.0,
        "error": None,
    }

    try:

        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if process.returncode != 0:

            result["error"] = (
                process.stderr.strip()
                or "nvidia-smi returned an error."
            )

            return result

        output = (
            process.stdout
            .strip()
        )

        if not output:

            result["error"] = (
                "No NVIDIA GPU returned by nvidia-smi."
            )

            return result

        # Use first GPU for this POC.
        first_gpu = (
            output.splitlines()[0]
        )

        name, memory_mb = (
            first_gpu.rsplit(",", 1)
        )

        name = name.strip()

        memory_mb = float(
            memory_mb.strip()
        )

        result.update(
            {
                "available": True,
                "name": name,
                "vram_gb": round(
                    memory_mb / 1024,
                    2,
                ),
            }
        )

        return result

    except FileNotFoundError:

        result["error"] = (
            "nvidia-smi was not found."
        )

        return result

    except Exception as error:

        result["error"] = str(error)

        return result


# ============================================================
# PYTORCH CUDA DETECTION
# ============================================================

def get_torch_cuda_info() -> dict[str, Any]:
    """
    Check whether the installed PyTorch build can actually
    use CUDA.

    A physical NVIDIA GPU may exist while this still returns
    False, for example when torch is CPU-only.
    """

    result = {
        "available": False,
        "cuda_version": None,
        "device_name": None,
        "error": None,
    }

    try:

        import torch

        result["cuda_version"] = (
            torch.version.cuda
        )

        if not torch.cuda.is_available():

            result["error"] = (
                "PyTorch CUDA is unavailable. "
                "The installed torch build may be CPU-only "
                "or CUDA initialization may have failed."
            )

            return result

        # ----------------------------------------------------
        # Real CUDA smoke test
        # ----------------------------------------------------

        try:

            test_tensor = torch.zeros(
                1,
                device="cuda",
            )

            torch.cuda.synchronize()

            del test_tensor

            result.update(
                {
                    "available": True,
                    "device_name":
                        torch.cuda.get_device_name(0),
                }
            )

            return result

        except Exception as error:

            result["error"] = (
                f"CUDA smoke test failed: {error}"
            )

            return result

    except ImportError:

        result["error"] = (
            "PyTorch is not installed."
        )

        return result

    except Exception as error:

        result["error"] = str(error)

        return result


# ============================================================
# TORCH DEVICE SELECTION
# ============================================================

def choose_torch_device(
    min_vram_gb: float = 0.0,
) -> dict[str, Any]:
    """
    Select CUDA only when:

    1. NVIDIA hardware exists.
    2. GPU has enough VRAM.
    3. Installed PyTorch supports CUDA.
    4. A real CUDA tensor allocation succeeds.

    Otherwise CPU is returned.
    """

    gpu = get_nvidia_gpu_info()

    if not gpu["available"]:

        return {
            "device": "cpu",
            "reason": (
                "No usable NVIDIA GPU detected."
            ),
            "gpu": gpu,
        }

    if gpu["vram_gb"] < min_vram_gb:

        return {
            "device": "cpu",
            "reason": (
                f"GPU has {gpu['vram_gb']:.2f} GB VRAM; "
                f"{min_vram_gb:.2f} GB required."
            ),
            "gpu": gpu,
        }

    torch_cuda = (
        get_torch_cuda_info()
    )

    if not torch_cuda["available"]:

        return {
            "device": "cpu",
            "reason": (
                "NVIDIA GPU detected, but PyTorch CUDA "
                "is not usable."
            ),
            "gpu": gpu,
            "torch_cuda": torch_cuda,
        }

    return {
        "device": "cuda",
        "reason": (
            "GPU passed hardware, VRAM, "
            "and CUDA validation."
        ),
        "gpu": gpu,
        "torch_cuda": torch_cuda,
    }


def check_ctranslate2_cuda_runtime() -> dict[str, Any]:
    """
    Check whether the CUDA runtime libraries required by the
    current CTranslate2 installation are loadable.

    This avoids selecting CUDA just because the NVIDIA GPU
    is visible when inference would fail immediately.
    """

    result = {
        "available": False,
        "error": None,
    }

    # On non-Windows systems we do not perform the DLL test.
    if platform.system() != "Windows":

        result["available"] = True

        return result

    required_dlls = [
        "cublas64_12.dll",
    ]

    for dll_name in required_dlls:

        try:

            ctypes.WinDLL(
                dll_name
            )

        except OSError:

            result["error"] = (
                f"Required CUDA library "
                f"{dll_name} is not available."
            )

            return result

    result["available"] = True

    return result


def choose_ctranslate2_device(
    min_vram_gb: float = 1.5,
) -> dict[str, Any]:
    """
    Select CUDA for CTranslate2/Faster-Whisper when:

    1. NVIDIA hardware exists.
    2. GPU has enough VRAM.
    3. CTranslate2 can detect a CUDA device.

    Actual model loading/inference is still protected
    by runtime CPU fallback in the ASR layer.
    """

    gpu = get_nvidia_gpu_info()

    if not gpu["available"]:

        return {
            "device": "cpu",
            "compute_type": "int8",
            "reason": "No NVIDIA GPU detected.",
            "gpu": gpu,
        }

    if gpu["vram_gb"] < min_vram_gb:

        return {
            "device": "cpu",
            "compute_type": "int8",
            "reason": (
                f"GPU has {gpu['vram_gb']:.2f} GB VRAM; "
                f"{min_vram_gb:.2f} GB required for ASR GPU attempt."
            ),
            "gpu": gpu,
        }

    try:

        import ctranslate2

        cuda_count = (
            ctranslate2.get_cuda_device_count()
        )

        if cuda_count < 1:

            return {
                "device": "cpu",
                "compute_type": "int8",
                "reason": (
                    "CTranslate2 cannot detect a CUDA device."
                ),
                "gpu": gpu,
            }

        runtime = (
            check_ctranslate2_cuda_runtime()
        )

        if not runtime["available"]:

            return {
                "device": "cpu",
                "compute_type": "int8",
                "reason": (
                    "CUDA hardware was detected, but "
                    f"{runtime['error']} "
                    "Using CPU fallback."
                ),
                "gpu": gpu,
            }

        return {
            "device": "cuda",
            "compute_type": "float16",
            "reason": (
                "CTranslate2 detected CUDA and GPU VRAM "
                "passed the ASR threshold."
            ),
            "gpu": gpu,
        }

    except Exception as error:

        return {
            "device": "cpu",
            "compute_type": "int8",
            "reason": (
                f"CTranslate2 CUDA detection failed: {error}"
            ),
            "gpu": gpu,
        }


# ============================================================
# SUMMARY
# ============================================================

def print_hardware_summary() -> None:

    gpu = get_nvidia_gpu_info()

    torch_cuda = (
        get_torch_cuda_info()
    )

    print()

    print(
        "=" * 80
    )

    print(
        "HARDWARE DETECTION"
    )

    print(
        "=" * 80
    )

    if gpu["available"]:

        print(
            "NVIDIA GPU:",
            gpu["name"],
        )

        print(
            "VRAM:",
            f"{gpu['vram_gb']:.2f} GB",
        )

    else:

        print(
            "NVIDIA GPU: not available"
        )

        if gpu["error"]:

            print(
                "GPU detection:",
                gpu["error"],
            )

    print()

    print(
        "PyTorch CUDA:",
        (
            "available"
            if torch_cuda["available"]
            else "not available"
        ),
    )

    if torch_cuda["cuda_version"]:

        print(
            "PyTorch CUDA version:",
            torch_cuda["cuda_version"],
        )

    if torch_cuda["error"]:

        print(
            "PyTorch status:",
            torch_cuda["error"],
        )

    print()

    voice_tut = (
        choose_torch_device(
            min_vram_gb=4.0
        )
    )

    embeddings = (
        choose_torch_device(
            min_vram_gb=1.0
        )
    )

    print(
        "Suggested embeddings device:",
        embeddings["device"],
    )

    print(
        "Reason:",
        embeddings["reason"],
    )

    print()

    print(
        "Suggested VoiceTut device:",
        voice_tut["device"],
    )

    print(
        "Reason:",
        voice_tut["reason"],
    )

    asr = (
        choose_ctranslate2_device(
            min_vram_gb=1.5
        )
    )

    print()

    print(
        "Suggested ASR device:",
        asr["device"],
    )

    print(
        "ASR compute type:",
        asr["compute_type"],
    )

    print(
        "Reason:",
        asr["reason"],
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print_hardware_summary()