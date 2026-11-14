"""Scene ownership shared by radio-material implementations."""

import weakref

import mitsuba as mi

from round2.physical_prior.geometry_prior_engine import scene as scene_module


class RadioMaterialBase(mi.BSDF):
    """Base type used to identify radio materials in a scene."""

    def __init__(self, props: mi.Properties) -> None:
        super().__init__(props)
        self._scene: weakref.ReferenceType | None = None
        for name in ("color", "reflectance", "base_color"):
            if name in props:
                del props[name]

    @property
    def scene(self):
        return None if self._scene is None else self._scene()

    @scene.setter
    def scene(self, scene: mi.Scene) -> None:
        if not isinstance(scene, scene_module.Scene):
            raise ValueError("scene must be an instance of Scene")
        current = self.scene
        if current is not None and current is not scene:
            raise ValueError(
                f"radio material '{self.name}' is already used by another scene"
            )
        self._scene = weakref.ref(scene)

    @property
    def name(self) -> str:
        name = self.id()
        return name[4:] if name.startswith("mat-") else name
