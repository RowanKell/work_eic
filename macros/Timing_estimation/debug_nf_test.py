"""
Standalone NF reproducibility test script.

Tests both loading methods (model.load vs CPU-first) and sampling with
nominal, extreme, and replay contexts across different GPU types.

Usage:
    python3 debug_nf_test.py [--thickness 2cm] [--replay_dir debug_logs/]
"""
import argparse
import sys
import os
import traceback

import torch
import numpy as np
import normflows as nf

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from debug_diagnostics import get_gpu_info, validate_context_tensor, log_error_state


def build_model(thickness="2cm"):
    """Build NF model architecture (without loading weights)."""
    configs = {
        "1cm": {"K": 8, "hidden_layers": 26, "hidden_units": 256, "context_size": 3,
                "batch_size": 2000, "num_context": 3, "run_num": 7,
                "path_template": "/hpc/group/vossenlab/rck32/NF_time_res_models/run_{run}_3context_8flows_26hl_256hu_2000bs.pth"},
        "2cm": {"K": 8, "hidden_layers": 26, "hidden_units": 256, "context_size": 3,
                "batch_size": 20000, "num_context": 3, "run_num": 1,
                "path_template": "/hpc/group/vossenlab/rck32/NF_time_res_models/thicker_2cm/run_{run}_3context_8flows_26hl_256hu_20000bs_checkpoint_e13.pth"},
        "5.55cm": {"K": 8, "hidden_layers": 26, "hidden_units": 256, "context_size": 3,
                   "batch_size": 20000, "num_context": 3, "run_num": 1,
                   "path_template": "/hpc/group/vossenlab/rck32/NF_time_res_models/thicker_5.55cm/run_{run}_3context_8flows_26hl_256hu_20000bs.pth"},
    }

    cfg = configs[thickness]
    K = cfg["K"]
    latent_size = 1
    hidden_layers = cfg["hidden_layers"]
    hidden_units = cfg["hidden_units"]
    context_size = cfg["context_size"]

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(
            latent_size, hidden_layers, hidden_units, num_context_channels=context_size)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(1, trainable=False)
    model = nf.ConditionalNormalizingFlow(q0, flows)

    model_path = cfg["path_template"].format(run=cfg["run_num"])
    return model, model_path


def test_loading_method_original(model, model_path, gpu_info):
    """Test original loading: model.load() then model.to('cuda')."""
    print("\n=== TEST: Original loading (model.load + to(cuda)) ===")
    try:
        model.load(model_path)
        model.to(torch.device('cuda'))
        model.eval()
        print("  PASS: Model loaded successfully with original method")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_loading_method_cpufirst(model, model_path, gpu_info):
    """Test CPU-first loading: torch.load(map_location='cpu') then load_state_dict then to(cuda)."""
    print("\n=== TEST: CPU-first loading (load_state_dict + to(cuda)) ===")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(torch.device('cuda'))
        model.eval()
        # cuBLAS warmup
        with torch.no_grad():
            dummy_ctx = torch.randn(2, 3, device='cuda')
            _ = model.sample(num_samples=2, context=dummy_ctx)
        torch.cuda.empty_cache()
        print("  PASS: Model loaded successfully with CPU-first method")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_sampling_nominal(model, num_trials=10):
    """Test sampling with nominal context values (should always work)."""
    print(f"\n=== TEST: Nominal sampling ({num_trials} trials) ===")
    # Nominal: z=0, theta=90, p=2.5 (all well within training range)
    context = torch.tensor([[0.0, 90.0, 2.5]], dtype=torch.float32, device='cuda')
    successes = 0
    for trial in range(num_trials):
        try:
            with torch.no_grad():
                samples = model.sample(num_samples=1, context=context)[0]
            if torch.isnan(samples).any() or torch.isinf(samples).any():
                print(f"  Trial {trial+1}: WARNING - NaN/Inf in output")
            else:
                successes += 1
        except Exception as e:
            print(f"  Trial {trial+1}: FAIL - {type(e).__name__}: {e}")

    print(f"  Result: {successes}/{num_trials} successful")
    return successes == num_trials


def test_sampling_batch(model, batch_size=1000):
    """Test batch sampling with random valid context values."""
    print(f"\n=== TEST: Batch sampling (batch_size={batch_size}) ===")
    # Random context within training ranges
    z_pos = torch.FloatTensor(batch_size, 1).uniform_(-735, 770)
    theta = torch.FloatTensor(batch_size, 1).uniform_(10, 170)
    momentum = torch.FloatTensor(batch_size, 1).uniform_(0.5, 5.0)
    context = torch.cat([z_pos, theta, momentum], dim=1).to('cuda')

    try:
        with torch.no_grad():
            samples = model.sample(num_samples=batch_size, context=context)[0]
        nan_count = int(torch.isnan(samples).sum())
        inf_count = int(torch.isinf(samples).sum())
        print(f"  PASS: Sampled {batch_size} values (NaN: {nan_count}, Inf: {inf_count})")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False


def test_sampling_extreme(model, gpu_info):
    """Test sampling with extreme/edge-case context values."""
    print("\n=== TEST: Extreme context values ===")

    test_cases = [
        ("zero momentum", [0.0, 90.0, 0.0]),
        ("negative momentum", [0.0, 90.0, -1.0]),
        ("very high momentum", [0.0, 90.0, 100.0]),
        ("extreme z_pos", [2000.0, 90.0, 2.5]),
        ("negative z_pos extreme", [-2000.0, 90.0, 2.5]),
        ("theta=0", [0.0, 0.0, 2.5]),
        ("theta=180", [0.0, 180.0, 2.5]),
        ("theta=360", [0.0, 360.0, 2.5]),
        ("all zeros", [0.0, 0.0, 0.0]),
        ("edge of range", [-735.0, 0.0, 0.01]),
        ("just outside range", [-800.0, 185.0, 11.0]),
    ]

    results = {}
    for name, values in test_cases:
        context = torch.tensor([values], dtype=torch.float32, device='cuda')
        try:
            with torch.no_grad():
                samples = model.sample(num_samples=1, context=context)[0]
            has_nan = bool(torch.isnan(samples).any())
            has_inf = bool(torch.isinf(samples).any())
            status = "PASS" if not (has_nan or has_inf) else f"WARN (NaN:{has_nan}, Inf:{has_inf})"
            results[name] = "pass"
        except AssertionError as e:
            status = f"FAIL (AssertionError - spline discriminant)"
            results[name] = "assertion_fail"
        except RuntimeError as e:
            status = f"FAIL (RuntimeError: {e})"
            results[name] = "runtime_fail"
        except Exception as e:
            status = f"FAIL ({type(e).__name__}: {e})"
            results[name] = "other_fail"
        print(f"  {name:30s} -> {status}")

    return results


def test_replay_contexts(model, replay_dir, gpu_info):
    """Replay any saved error context tensors from debug_logs/."""
    print(f"\n=== TEST: Replay saved error contexts from {replay_dir} ===")

    if not os.path.exists(replay_dir):
        print("  No replay directory found, skipping")
        return

    pt_files = [f for f in os.listdir(replay_dir) if f.endswith('.pt')]
    if not pt_files:
        print("  No .pt files found in replay directory")
        return

    for pt_file in sorted(pt_files):
        filepath = os.path.join(replay_dir, pt_file)
        try:
            context = torch.load(filepath, map_location='cpu')
            validation = validate_context_tensor(context)
            print(f"\n  Replaying {pt_file} (shape: {list(context.shape)})")
            print(f"    Validation: NaN={validation['has_nan']}, Inf={validation['has_inf']}")

            context_gpu = context.to('cuda')
            with torch.no_grad():
                samples = model.sample(num_samples=len(context_gpu), context=context_gpu)[0]
            print(f"    PASS: Successfully sampled {len(context_gpu)} values")
        except AssertionError:
            print(f"    FAIL: AssertionError (spline discriminant < 0)")
        except RuntimeError as e:
            print(f"    FAIL: RuntimeError: {e}")
        except Exception as e:
            print(f"    FAIL: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser(description="NF reproducibility test")
    parser.add_argument("--thickness", type=str, default="2cm", choices=["1cm", "2cm", "5.55cm"])
    parser.add_argument("--replay_dir", type=str, default=None,
                        help="Directory with saved error .pt contexts")
    parser.add_argument("--skip_original_load", action="store_true",
                        help="Skip testing original model.load() method")
    args = parser.parse_args()

    print("=" * 60)
    print("NF REPRODUCIBILITY & DIAGNOSTICS TEST")
    print("=" * 60)

    # Step 1: GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('device_name', 'N/A')}")
    print(f"Capability: {gpu_info.get('device_capability', 'N/A')}")
    print(f"CUDA: {gpu_info.get('cuda_version', 'N/A')}")
    print(f"Memory: {gpu_info.get('total_memory_GB', 'N/A')} GB")
    print(f"Node: {gpu_info.get('slurm_node', 'N/A')}")
    print(f"Job ID: {gpu_info.get('slurm_job_id', 'N/A')}")

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available, exiting")
        sys.exit(1)

    # Step 2: Build model
    model, model_path = build_model(args.thickness)
    print(f"\nModel path: {model_path}")

    # Step 3: Test original loading (the method that fails on A5000)
    if not args.skip_original_load:
        model1, _ = build_model(args.thickness)
        test_loading_method_original(model1, model_path, gpu_info)
        del model1
        torch.cuda.empty_cache()

    # Step 4: Test CPU-first loading (the proposed fix)
    load_ok = test_loading_method_cpufirst(model, model_path, gpu_info)
    if not load_ok:
        print("\nFATAL: Cannot load model, aborting remaining tests")
        sys.exit(1)

    # Step 5: Nominal sampling
    test_sampling_nominal(model, num_trials=10)

    # Step 6: Batch sampling
    test_sampling_batch(model, batch_size=1000)
    test_sampling_batch(model, batch_size=5000)

    # Step 7: Extreme context values
    test_sampling_extreme(model, gpu_info)

    # Step 8: Replay saved error contexts
    replay_dir = args.replay_dir
    if replay_dir is None:
        replay_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_logs")
    test_replay_contexts(model, replay_dir, gpu_info)

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
