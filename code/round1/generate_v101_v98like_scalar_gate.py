#!/usr/bin/env python3
"""Generate v101: v98-like LOO trained component scalar gate.

This wrapper intentionally reuses the v99 implementation while changing the
training base/residual arguments at run time.  The actual v101 submission was
generated with:

    --base artifacts/version_cache/v098/v98_Round1_Test_Channel.npy
    --train-base-ad artifacts/shared_features/train_v98like_loo_base_ad.npy
    --train-residual-ad artifacts/shared_features/train_v98like_loo_residual_ad.npy

The main difference from v99 is therefore the training-label distribution:
v99 learned scalar corrections from a v36-like LOO base, while v101 learns from
a v98-like LOO base that better matches the v98 test-time base.
"""

from generate_v99_learned_component_scalar_gate import main


if __name__ == "__main__":
    main()
