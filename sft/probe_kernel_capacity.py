"""Check the official convolution reference before an independent capacity probe."""

import argparse
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

from .check_activation_offload import run_trial, write_json


def check_convolution():
    import torch
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_fn, causal_conv1d_ref

    results = []
    for length, channel_last, activation in product((2048, 8192), (False, True), (None, "silu")):
        torch.manual_seed(0)
        shape = (2, length, 256) if channel_last else (2, 256, length)
        value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        if channel_last:
            value = value.transpose(1, 2)
        value.requires_grad_()
        weight = torch.randn(256, 4, device="cuda", requires_grad=True)
        bias = torch.randn(256, device="cuda", requires_grad=True)
        reference = [tensor.detach().clone().requires_grad_() for tensor in (value, weight, bias)]
        actual = causal_conv1d_fn(value, weight, bias, activation=activation)
        expected = causal_conv1d_ref(*reference, activation=activation)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        # Upstream v1.7.0 BF16 tolerances, with FP32 parameter gradients.
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=5e-2)
        torch.testing.assert_close(value.grad, reference[0].grad, rtol=1e-2, atol=5e-2)
        for actual_parameter, expected_parameter in zip((weight, bias), reference[1:]):
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad,
                                       rtol=1e-3, atol=1e-3)
        results.append({
            "length": length, "channel_last": channel_last, "activation": activation,
            "output_max_abs": (actual.float() - expected.float()).abs().max().item(),
            "input_gradient_max_abs": (value.grad.float() - reference[0].grad.float()).abs().max().item(),
        })
    return {"passed": True, "cases": results, "device": torch.cuda.get_device_name(),
            "reference": "Dao-AILab/causal-conv1d/v1.7.0/tests/test_causal_conv1d.py",
            "model_optimizer_parity_certified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "runtime", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--kernel-only", action="store_true")
    args = parser.parse_args()
    if args.kernel_only:
        write_json(args.output / "kernel-reference.json", check_convolution())
        return
    args.output.mkdir(parents=True)
    code = 1
    try:
        # Release the probe's CUDA context before the full-model worker starts.
        subprocess.run([sys.executable, "-m", "sft.probe_kernel_capacity", *sys.argv[1:], "--kernel-only"],
                       check=True)
        config = json.loads(args.config.read_text())
        run_trial(args, config, "capacity", False, config["gate_indices"])
        code = 0
    finally:
        (args.output / "exit-code").write_text(str(code) + "\n")


if __name__ == "__main__":
    main()
