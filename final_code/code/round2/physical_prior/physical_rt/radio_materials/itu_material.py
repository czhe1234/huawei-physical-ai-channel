"""Frequency-dependent ITU-R P.2040 radio materials."""

import mitsuba as mi

from .itu import ITU_MATERIALS_PROPERTIES, itu_material
from .radio_material import RadioMaterial


class ITURadioMaterial(RadioMaterial):
    """Radio material whose permittivity follows an ITU material family."""

    def __init__(
        self,
        name: str | None = None,
        itu_type: str | None = None,
        thickness: float | mi.Float | None = None,
        scattering_coefficient: float | mi.Float = 0.0,
        xpd_coefficient: float | mi.Float = 0.0,
        scattering_pattern=None,
        props: mi.Properties | None = None,
        **kwargs,
    ) -> None:
        has_props = props is not None
        if has_props:
            direct_args_are_defaults = (
                name is None
                and itu_type is None
                and thickness is None
                and scattering_coefficient == 0.0
                and xpd_coefficient == 0.0
            )
            if not direct_args_are_defaults:
                raise ValueError(
                    "direct material parameters cannot be combined with props"
                )
            if "type" not in props:
                raise ValueError("missing ITU material type")
            itu_type = props["type"]
            del props["type"]

        if itu_type not in ITU_MATERIALS_PROPERTIES:
            raise ValueError(f"invalid ITU material type '{itu_type}'")
        self._itu_type = itu_type

        def update_from_frequency(frequency: float):
            return itu_material(itu_type, frequency)

        if has_props:
            super().__init__(
                scattering_pattern=scattering_pattern,
                frequency_update_callback=update_from_frequency,
                props=props,
                **kwargs,
            )
        else:
            super().__init__(
                name=name,
                thickness=thickness,
                scattering_coefficient=scattering_coefficient,
                xpd_coefficient=xpd_coefficient,
                scattering_pattern=scattering_pattern,
                frequency_update_callback=update_from_frequency,
                **kwargs,
            )

    @property
    def itu_type(self) -> str:
        return self._itu_type

    def to_string(self) -> str:
        return (
            f"ITURadioMaterial type={self._itu_type}\n"
            f"                 eta_r={self._eta_r[0]:.3f}\n"
            f"                 sigma={self._sigma[0]:.3f}\n"
            f"                 thickness={self._d[0]:.3f}\n"
            f"                 scattering_coefficient={self._s[0]:.3f}\n"
            f"                 xpd_coefficient={self._kx[0]:.3f}"
        )


mi.register_bsdf(
    "itu-radio-material",
    lambda props: ITURadioMaterial(props=props),
)
