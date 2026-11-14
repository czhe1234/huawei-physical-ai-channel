South structured adaptation v2

This package restores the measured task's frequency-resolved foundation, path sequence/set state, relative-delay correction, real receive-array energy organization and joint complex synthesis. It is a new implementation, not pretrained V16 weights or an unchanged replica of the original multi-UE/polarization pipeline.

Engineering acceptance (CPU; synthetic checks and a bounded FIT-only interface pass; no optimization, real scores, RT, test or north channels):
```sh
CUDA_VISIBLE_DEVICES="" HW_SKIP_ENV_CHECK=1 ./run.sh python -m measured.hall_structured_v2.check
```

The explicit train module is prepared for a later authorized development run and was not invoked in this implementation stage. Its help describes bounded development options. Existing output folders are protected; interrupted runs may resume from a complete epoch only when the source/data/config identity matches.

Default geometry reuses the old provisional south cache. Missing variance is unknown. A fresh cache may be supplied through the provenance-checked cache interface, without triggering RT. A recorded convergence flag alone does not authorize a certified run: a matching successful external convergence report is required, or the run must explicitly remain provisional.

The three core final-channel targets are multiresolution angle/frequency energy, within-panel normalized covariance and panel plus energy-weighted per-beam frequency-correlation magnitude. Relative FIT-referenced CSI power is auxiliary. The prediction reference is derived exclusively from the predicted FIT-donor carrier. No query CSI alignment or fake UE ports are used.

Current correctness checks do not establish convergence, superiority, geometry contribution or absolute complex-channel recovery. See the Chinese implementation report and acceptance.json under analysis/results/hall_south_structured_v2. All earlier data, code, checkpoints and results are retained; north remains confirmation-only.
