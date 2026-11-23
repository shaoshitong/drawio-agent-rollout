"""Use the training processor's CPU resize path during causal SFT evaluation."""


def install():
    from sglang.srt.multimodal.processors import base_processor

    # This module-local switch only selects the fast image processor's device.
    # The model and its attention kernels continue to run on CUDA.
    base_processor._is_cpu = True
