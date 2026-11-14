"""Physical units, immutable stage scope, and versioned paths."""
from dataclasses import dataclass,asdict
from project_paths import PATHS
REPORT=PATHS.analysis/"results/hall_south_structured_v2"
WORK=PATHS.artifacts/"measured/hall_sw_structured_v2"
SOURCE=PATHS.code_root/"measured/hall_structured_v2"
LAGS=(1,2,4,8,16,32,64,128,256,512)
ALLOWED_GPUS=("GPU-6b53c3c2-3071-5dfb-f306-2f2f142f8009","GPU-9a72820b-a8c2-0a94-2297-4f65afc67eb0","GPU-70b6fceb-c7c4-9467-3c81-097c9284c160")
@dataclass(frozen=True)
class Config:
    subbands:int=16
    reserve_candidates:int=16
    geometry_candidates:int=128
    horizontal_bound_bins:float=.5
    vertical_bound_bins:float=.25
    relative_delay_bound_s:float=20e-9
    spectral_log_power_bound:float=.7
    receive_log_amplitude_bound:float=.25
    power_bound_fit_std:float=3.
    geometry_gate_cap:float=.6
    projection_cycles:int=4
    context_width:int=64
    use_geometry:bool=True
    use_sequence:bool=True
    use_relative_delay:bool=True
    use_receive_energy:bool=True
    power_loss_weight:float=.1
    residual_loss_weight:float=1e-4
    def to_dict(self):return asdict(self)
