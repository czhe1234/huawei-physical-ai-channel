# Source Notice

The propagation kernels in this directory are derived from the Sionna RT
2.0.1 files required by the production physical solver. They retain the original SPDX
copyright and Apache-2.0 headers. The corresponding license text is retained
in `LICENSE` in this directory.

Project-specific changes include package import paths, a compact scene wrapper
for the official Round 2 map, stable surface-plane normalization, resumable
batched execution, and an output adapter. The adapter preserves path geometry,
delay, angles, interactions, and complex per-path coefficients required by the
Physical-AI model, but the vendored solver does not generate the final complete
submission channel.
