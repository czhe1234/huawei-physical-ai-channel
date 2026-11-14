# Third-Party Notice

The project-local propagation kernels under
`code/round2/physical_prior/physical_rt/` are derived from the Sionna RT
2.0.1 source distribution and remain licensed under Apache License 2.0.
The original SPDX headers, source notice, and full license text are retained
in that directory.

The production pipeline imports these project-local modules directly. It does
not import or require the external `sionna` Python package. Project-specific
changes include package-path isolation, a compact scene wrapper for the
official Round 2 map, stable surface-plane normalization, resumable batched
execution, and adapters that expose only the path state required by the
Physical-AI model.

Other Python dependencies retain their respective upstream licenses.

