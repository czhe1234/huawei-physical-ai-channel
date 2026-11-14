# NeWRF Round2 adapter: source and qualification contract

Scope: implement and check CPU synthetic algebra only. No training, scoring, RT,
checkpoint, official channel, held-out path, or measured-data access.

Plan and acceptance checklist:
- [x] Read hw/AGENTS and code/round2/AGENTS; inspect official model/encoder/sampler/synthesizer.
- [x] Keep fixed-BS transmitter -> moving-UE receiver direction.
- [x] Bind actual128 BS spatial elements,2 colocated BS polarizations,2 UE spatial offsets,2 UE polarizations.
- [x] Implement source-pinned conditional fields and deterministic coarse/fine sampling.
- [x] Verify synthetic complex synthesis, independent polarizations, physical-port axes, gradients and CPU execution.
- [ ] Qualify real-data training direction provenance and physical frequency source. This item is blocked, not presumed passed.

Source: LuHaofan/NeWRF, supplied snapshot commit78feb38471f766f17764a28e8518a05697944107.
The snapshot directory has no git metadata. Individual source SHA256 identities
are checked by the adapter. models.py, encoders.py, samplers.py, synthesizers.py
are loaded without the training, data, ray-search or CUDA runner packages.

Preserved: original NeWRF class, positional encoder, deterministic stratified and
hierarchical sampling, and Synthesizer complex volume rendering. The author's
exclusive-cumprod helper sets its first entry to zero; this unusual first-cell
behavior is preserved explicitly rather than silently repaired. Frequency is
passed to the author's synthesizer in FP64 GHz, as required by its fc*1e9 formula.

Declared adaptation: eight independent fields for BS-polarization x UE-port
(2 x 4). Each field conditions on sampled position, actual BS spatial TX-element
position and physical frequency. It is shared across both sites and128 BS
spatial elements. This is a substantive fixed-source-to-source-conditioned
extension, not an unchanged original experiment. BS ports are polarization then
16x8 h/v; UE ports are polarization then two vertical elements. UE spatial
origins use spec.ue_world_positions with site-dependent dynamic UE look-at-BS rotation. Colocated polarizations share geometry,
not channel coefficients. No port or polarization channel is copied.

Ray-provider signature: directions(ue_origins[B,4,3], site_ids[B],
bs_tx_positions[B,2,128,3]) -> unit directions[B,2,128,4,R,3]. Rays start at UE
and backtrace toward the scene, consistent with original mobile-STA LastXPts.
A training-ray provider must eventually bind the exact canonical FIT IDs,
split fingerprint, source geometry and per-training-UE DoA provenance. Test
rays must come from an independently qualified geometry prior or the author's
FIT-only virtual-source search. The original hardcoded80% partition and
filename-only search caches are not imported or used.

Metadata findings: Round2_Setup.json specifies shape,2 BS positions, BS2pol x
16x8, UE2pol x1x2 and192tones. It does not specify carrier, tone spacing, array
spacing or orientation. The existing V16/production code uses6.69GHz and30kHz,
which is an implementation convention, not independently certified publisher
metadata. No official Train_Channel array was opened for this audit.

Existing generator source describes theta_r,phi_r,vertices and point_ids in
path caches, but these are approximate RT priors, not published official path
truth. Their row/fold binding, actual receiver-side direction convention and
compatibility with every physical element have not been certified here. No
path-array content or held-out cache was opened. Four UE ports alone do not
justify high-accuracy UE-side multipath DoA extraction from TrainH. Accordingly,
real-data forward is blocked until the FIT-only ray contract is independently
reviewed. The disclosed V16-compatible frequency convention is not by itself
an extra gate. Synthetic mode cannot make that qualification.

No performance or paper-reproduction conclusion is attached to this adapter.

Further cache audit: V16 prior_preparation.py/source_port_common.py explicitly
write point_ids, four angles including theta_r/phi_r, vertices and interaction_count.
A correct FIT-only provider could read last valid interaction minus UE origin;
LoS uses BS minus UE. However a targeted directory inventory of retained hw
assets (excluding archives, datasets and measured channels) found no retained
prior_supports or path_tokens directories. V16_fast_repro_submission retains
models, reports and test model_inputs, not the required Train direction cache.
No cache was opened or declared absent merely because it lacks a DoA key.

The adapter exposes return_coarse=True -> (fine_H, coarse_H) so a future accepted
training implementation can apply the original joint coarse/fine objective.
A loss on only the final H would not train the coarse field through detached
hierarchical sampling, so it must not be described as the original training rule.

CPU synthetic validation: full[1,256,4,192] complex64, nonconstant frequency
response, tone-subset parity, independent BS/UE polarization outputs, finite
coarse/fine gradients, author complex-integral independent scalar arithmetic,
closed real-data gate. No channel/path arrays, optimizer, score, RT, GPU or
checkpoint were used. Tiny2-layer8-wide networks with4+2 ray samples are numerical
fixtures, not the paper capacity or selected experimental configuration.
