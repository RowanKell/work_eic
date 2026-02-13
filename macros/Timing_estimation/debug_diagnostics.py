"""
Diagnostic utilities for debugging NF sampling errors on GPU.
Captures GPU info, validates context tensors, and logs error states for replay.
"""
import torch
import subprocess
import json
import os
import datetime

def get_gpu_info():
    """Capture GPU name, architecture, VRAM, CUDA/driver version, and nvidia-smi output."""
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        info["cuda_version"] = torch.version.cuda
        info["torch_version"] = torch.__version__
        mem = torch.cuda.get_device_properties(0)
        info["total_memory_GB"] = round(mem.total_memory / 1e9, 2)

    # Capture nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = result.stdout
    except Exception as e:
        info["nvidia_smi"] = f"Failed to run nvidia-smi: {e}"

    # Capture SLURM node name
    info["slurm_node"] = os.environ.get("SLURMD_NODENAME", "unknown")
    info["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "unknown")

    return info


def validate_context_tensor(context, label="context"):
    """
    Check context tensor for NaN/Inf and report min/max/mean per column.

    Expected columns: [z_pos, theta, momentum]
    Training ranges: z_pos [-735, 770], theta [0, 180], momentum [0.01, 10]

    Returns dict with validation results and number of bad entries.
    """
    result = {
        "label": label,
        "shape": list(context.shape),
        "has_nan": bool(torch.isnan(context).any()),
        "has_inf": bool(torch.isinf(context).any()),
        "num_nan": int(torch.isnan(context).sum()),
        "num_inf": int(torch.isinf(context).sum()),
    }

    col_names = ["z_pos", "theta", "momentum"]
    if context.dim() == 2 and context.shape[1] >= 3:
        for i, name in enumerate(col_names[:context.shape[1]]):
            col = context[:, i]
            valid = col[~(torch.isnan(col) | torch.isinf(col))]
            if len(valid) > 0:
                result[f"{name}_min"] = float(valid.min())
                result[f"{name}_max"] = float(valid.max())
                result[f"{name}_mean"] = float(valid.mean())
                result[f"{name}_std"] = float(valid.std())
            else:
                result[f"{name}_min"] = None
                result[f"{name}_max"] = None
                result[f"{name}_mean"] = None
                result[f"{name}_std"] = None

    return result


def clamp_context_tensor(context, verbose=True):
    """
    Clamp context values to training range and remove NaN/Inf entries.

    Training ranges:
        z_pos:    [-735, 770]
        theta:    [0, 180]
        momentum: [0.01, 10]

    Returns (clamped_context, valid_mask) where valid_mask indicates non-NaN/Inf rows.
    """
    # First remove NaN/Inf rows
    nan_mask = torch.isnan(context).any(dim=1)
    inf_mask = torch.isinf(context).any(dim=1)
    bad_mask = nan_mask | inf_mask
    num_bad = int(bad_mask.sum())

    if num_bad > 0 and verbose:
        print(f"WARNING: Removing {num_bad} context entries with NaN/Inf values")

    valid_mask = ~bad_mask
    clamped = context.clone()

    # Clamp ranges (only for valid entries)
    ranges = [
        (-735.0, 770.0),   # z_pos
        (0.0, 180.0),      # theta
        (0.01, 10.0),      # momentum
    ]

    if clamped.dim() == 2:
        for i, (lo, hi) in enumerate(ranges[:clamped.shape[1]]):
            col = clamped[:, i]
            out_of_range = valid_mask & ((col < lo) | (col > hi))
            num_clamped = int(out_of_range.sum())
            if num_clamped > 0 and verbose:
                print(f"WARNING: Clamping {num_clamped} values in column {i} "
                      f"(range [{lo}, {hi}], actual [{float(col[valid_mask].min()):.2f}, {float(col[valid_mask].max()):.2f}])")
            clamped[:, i] = torch.clamp(col, lo, hi)

    return clamped, valid_mask


def log_error_state(error, context_batch, gpu_info, batch_idx=0, log_dir=None):
    """
    Save full diagnostic JSON + problematic context tensor as .pt file for replay.

    Args:
        error: The exception that was raised
        context_batch: The context tensor that caused the error
        gpu_info: Output from get_gpu_info()
        batch_idx: Which batch failed
        log_dir: Directory to save logs (default: work_eic/macros/Timing_estimation/debug_logs/)
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_logs")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save context tensor for replay — handle corrupted CUDA state gracefully
    tensor_path = os.path.join(log_dir, f"error_context_{timestamp}_batch{batch_idx}.pt")
    try:
        cpu_tensor = context_batch.detach().cpu() if context_batch.is_cuda else context_batch.detach()
        torch.save(cpu_tensor, tensor_path)
    except RuntimeError:
        # CUDA context is corrupted, cannot move tensor — save shape info only
        cpu_tensor = None
        print(f"  WARNING: CUDA corrupted, cannot save context tensor to {tensor_path}")

    # Save diagnostic JSON
    if cpu_tensor is not None:
        validation = validate_context_tensor(cpu_tensor)
    else:
        validation = {"label": "unavailable_cuda_corrupted", "shape": list(context_batch.shape)}

    diag = {
        "timestamp": timestamp,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "batch_idx": batch_idx,
        "context_shape": list(context_batch.shape),
        "context_validation": validation,
        "gpu_info": {k: v for k, v in gpu_info.items() if k != "nvidia_smi"},
        "tensor_file": tensor_path,
    }

    json_path = os.path.join(log_dir, f"error_diag_{timestamp}_batch{batch_idx}.json")
    with open(json_path, "w") as f:
        json.dump(diag, f, indent=2)

    print(f"Error diagnostics saved to {json_path}")
    print(f"Context tensor saved to {tensor_path}")

    return json_path, tensor_path
