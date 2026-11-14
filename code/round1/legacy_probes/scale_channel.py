# （本改动由Claude fable5）
"""Global amplitude calibration step.

Applies a single global complex-amplitude scale to the generated channel
tensor. The power-spectrum shape of every point (PAS over the BS angle
domain, PDP over the delay domain) is unchanged by a global scale; only the
absolute amplitude calibration of the generated channel changes. This is the
final amplitude-calibration stage of the generation pipeline
(generate_s1_residual_component_gate.py produces the unscaled tensor).

Usage:
    python scale_channel.py --input Round1_Test_Channel_unscaled.npy \
        --output Round1_Test_Channel.npy --scale 0.3
"""

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, required=True)
    args = parser.parse_args()

    channel = np.load(args.input)
    assert channel.dtype == np.complex64, channel.dtype
    assert channel.shape == (500, 256, 4, 192), channel.shape

    scaled = (channel * np.complex64(args.scale)).astype(np.complex64)
    assert np.isfinite(scaled.view(np.float32)).all()

    np.save(args.output, scaled)
    print(f"saved {args.output}: shape={scaled.shape} dtype={scaled.dtype} scale={args.scale}")


if __name__ == "__main__":
    main()
