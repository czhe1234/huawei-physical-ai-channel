# WRF-GS+ simulation adapter: source identity and scope

This is a frequency-conditioned, position-only Gaussian-splat **power** adapter.
It is not an unchanged paper reproduction. It never supplies complex H, PDP or
NMSE. The paper's CSI case study takes query uplink CSI; that branch is excluded.

## Fixed author source

- Repository: https://github.com/wenchaozheng/WRF-GSplus
- Commit: `e564f3184463717ae08660a8b7232c6318c73ca6`.
- `utils/pos_utils.py` SHA256:
  `a7e14da60145afc6b2d341d861aaca20bc034b2bf7d32c46799b264ca08847ea`.
- `utils/sh_utils.py` SHA256:
  `7d1ff267546390635e6d1f68c4f88c4a8b052482d5c9be1d32f06bc69e9a96e7`.
- Both identities are checked before loading definitions. Original source files
  and licenses remain untouched. The original is_6dof=False DeformNetwork body
  and degree-3 spherical-harmonic evaluation are reused directly. The loader
  excludes import-time package/CUDA dependencies; it does not alter those files.
- Core source: `utils/pos_utils.py`, `gaussian_renderer/__init__.py`, and
  `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`.
- Paper: https://arxiv.org/html/2412.04832v4 , Sections III-A, V, VII-B.

## Preserved mechanism

Each site has learned 3D Gaussian centers, scales, rotations, opacities and
degree-3 SH attributes. Query position conditions the author's deformation
network. Spherical projection propagates the deformed 3D covariance into 2D.
Depth-sorted front-to-back alpha blending forms the rendered field. Every output
passes this field; there is no bypass position-to-power MLP.

The pinned renderer ignores the deformation warp of Gaussian centers. This
adapter also ignores that warp and freezes its unused head. The author's signal
phase is immediately removed by abs; the inactive phase head is frozen. No
learned or identifiable CSI phase is claimed.

## Explicit task modifications

1. Explicit frequency conditions the deformation through an additional 1D
   sinusoidal embedding with six bands. The existing 3D position encoding is
   retained. No carrier, tone spacing or frequency grid is defaulted. Forward
   requires the exact bound vector and order.
2. Eight separate SH/signal channels represent BS polarization2 by UE port4.
   All eight retain a common output scale; no per-polarization/UE power
   normalization is applied. UE ports are an ordered output dimension. Known UE
   offsets/polarization labels must be recorded by the shared geometry manifest.
3. A separate nonnegative learned readout per channel maps the rendered angular
   field to 128 **indexed DFT bins**. This readout is not an angular-image resize,
   nearest-angle lookup or physical incoherent angle-to-beam conversion. No
   claim that angle power determines coherent DFT power is made. This is an
   explicit codebook-power task readout trained against exact codebook labels.
4. The output is `[B,256,4,F]`, with 256 ordered `[polarization2,H16,V8]`, unshifted.
   The label helper applies the exact unnormalized 16x8 FFT2 and squared modulus.
   It does not normalize each polarization or average frequency/UE ports.
5. Float, unbounded power supervision replaces PNG display quantization and
   clipping. SH color uses softplus instead of the original lower clamp. The
   independently learned channel gains permit relative power differences.

## CPU renderer differences

This is dense pure-Torch CPU rendering, not the CUDA implementation. The
adaptation retains opacity alpha cap0.99, alpha cutoff1/255, near depth0.2m,
post-transmittance cutoff1e-4, and covariance floor0.3pixels squared. It uses an
equirectangular full-sphere camera and configurable raster size; the checked
default is 32x16, whereas the released example uses 360x90. It does not implement
CUDA tile bounds, radius pruning, antialiasing, GUI, densification or exposure.
It does not clip the final power field to[0,1]. Seam wrapping is not implemented.
These are disclosed architectural/numerical differences. No real-data or CUDA
equivalence error has been measured, so numerical parity is **unverified**.

## Geometry, eligibility and information boundary

Gaussian initialization may use only known site geometry and allowed training
positions. The constructor requires explicit BS centers, right-handed camera
axes, initial Gaussian locations and fit-only coordinate normalization inputs.
`geometry_identity` binds the shared geometry fingerprint. Neither constructor
nor forward reads channel files, evaluation labels, RT data or checkpoints.

Official PAS is eligible only after the caller independently verifies the
`official_fft2_power_unshifted_P2_H16_V8_unnormalized` target contract and records
its identity. The constructor token is a binding declaration, not proof. Full H,
PDP, complex NMSE and the official combined score remain unavailable. Do not
fabricate H from the power output. Native angular-spectrum replication is a
different task and must have a different row/name.

Synthetic CPU checks cover finite/nonnegative shapes, independent channels,
site/frequency conditioning, finite derivatives and invalid-spec rejection. They
do not train, predict real channels or report algorithm performance. The shared
manifest owns the verified/implementation-convention frequency provenance and
official port-coordinate/polarization mapping; this note does not certify them.
