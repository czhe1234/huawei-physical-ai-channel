from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import drjit as dr
import mitsuba as mi
from scipy.constants import speed_of_light

from .array_model import ArrayModel
from .constants import DEFAULT_FREQUENCY
from .endpoints import SceneEndpoint, Source, Target
from .scene_response import SceneInteractionResponse
from .utils.meshes import remove_mesh_duplicate_vertices


def _merge_scene_geometry(xml_string: str) -> str:
    root = ET.fromstring(xml_string)
    merge_node = ET.Element("shape", {"type": "merge", "id": "merged-shapes"})
    for shape in root.findall("shape"):
        root.remove(shape)
        merge_node.append(shape)
    root.append(merge_node)
    ET.indent(root, space="    ")
    return ET.tostring(root).decode("utf-8")


class Scene:
    def __init__(self, mi_scene: mi.Scene):
        if not isinstance(mi_scene, mi.Scene):
            raise TypeError("mi_scene must be a Mitsuba Scene")
        self._scene = mi_scene
        self._tx_array: ArrayModel | None = None
        self._rx_array: ArrayModel | None = None
        self._sources: dict[str, Source] = {}
        self._targets: dict[str, Target] = {}
        self._interaction_responses: dict[str, SceneInteractionResponse] = {}
        self._objects: dict[str, mi.Mesh] = {}
        self._frequency = mi.Float(DEFAULT_FREQUENCY)
        self._load_scene_objects()

    @property
    def mi_scene(self) -> mi.Scene:
        return self._scene

    @property
    def frequency(self):
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        if value <= 0.0:
            raise ValueError("Frequency must be positive")
        self._frequency = mi.Float(value)
        for response in self._interaction_responses.values():
            response.frequency_update()

    @property
    def wavelength(self):
        return speed_of_light / self.frequency

    @property
    def angular_frequency(self):
        return dr.two_pi * self.frequency

    @property
    def tx_array(self):
        return self._tx_array

    @tx_array.setter
    def tx_array(self, value):
        if not isinstance(value, ArrayModel):
            raise TypeError("tx_array must be an ArrayModel")
        self._tx_array = value

    @property
    def rx_array(self):
        return self._rx_array

    @rx_array.setter
    def rx_array(self, value):
        if not isinstance(value, ArrayModel):
            raise TypeError("rx_array must be an ArrayModel")
        self._rx_array = value

    @property
    def source_count(self) -> int:
        return len(self._sources)

    @property
    def target_count(self) -> int:
        return len(self._targets)

    @property
    def interaction_responses(self):
        return dict(self._interaction_responses)

    def get(self, name: str):
        return (
            self._objects.get(name)
            or self._sources.get(name)
            or self._targets.get(name)
        )

    def add(self, item: SceneEndpoint) -> None:
        name = item.name
        current = self.get(name)
        if current is not None:
            if current is not item:
                raise ValueError(f"Name '{name }' is already used by another item")
            return
        if isinstance(item, Source):
            self._sources[name] = item
        elif isinstance(item, Target):
            self._targets[name] = item
        else:
            raise ValueError(f"Unsupported scene item: {type (item )}")

    def remove(self, name: str) -> None:
        self._sources.pop(name, None)
        self._targets.pop(name, None)

    def validate(self) -> None:
        if self.tx_array is None:
            raise ValueError("Transmitter array not set")
        if not self._sources:
            raise ValueError("Scene has no source")
        if self.rx_array is None:
            raise ValueError("Receiver array not set")
        if not self._targets:
            raise ValueError("Scene has no targets")

    def sources(self):
        return self._endpoints(
            self._sources.values(),
            self.tx_array,
        )

    def targets(self):
        return self._endpoints(
            self._targets.values(),
            self.rx_array,
        )

    def _load_scene_objects(self) -> None:
        for shape in self._scene.shapes():
            if not isinstance(shape, mi.Mesh):
                raise TypeError("Only triangle meshes are supported")
            response = shape.bsdf()
            if not isinstance(response, SceneInteractionResponse):
                raise ValueError(
                    f"Shape {shape .id ()} does not use a scene interaction response"
                )
            mi.set_log_level(mi.LogLevel.Error)
            shape.build_directed_edges()
            mi.set_log_level(mi.LogLevel.Warn)
            remove_mesh_duplicate_vertices(shape)
            name = shape.id()
            if name.startswith("mesh-"):
                name = name[5:]
            self._objects[name] = shape
            if response.name not in self._interaction_responses:
                response.scene = self
                self._interaction_responses[response.name] = response

    def _endpoints(
        self,
        endpoints,
        array: ArrayModel,
    ):
        endpoints = list(endpoints)
        endpoint_count = len(endpoints)
        positions = dr.zeros(mi.Point3f, endpoint_count)
        orientations = dr.zeros(mi.Point3f, endpoint_count)
        relative_positions = dr.zeros(mi.Point3f, endpoint_count * array.array_size)
        relative_indices = dr.arange(mi.UInt, array.array_size)
        for index, endpoint in enumerate(endpoints):
            position = endpoint.position
            orientation = endpoint.orientation
            antenna_positions = array.rotate(self.wavelength, orientation)
            indices = relative_indices + index * array.array_size
            dr.scatter(relative_positions.x, antenna_positions.x, indices)
            dr.scatter(relative_positions.y, antenna_positions.y, indices)
            dr.scatter(relative_positions.z, antenna_positions.z, indices)
            dr.scatter(positions.x, position.x, index)
            dr.scatter(positions.y, position.y, index)
            dr.scatter(positions.z, position.z, index)
            dr.scatter(orientations.x, orientation.x, index)
            dr.scatter(orientations.y, orientation.y, index)
            dr.scatter(orientations.z, orientation.z, index)
        return positions, orientations, relative_positions


def load_scene(filename: str) -> Scene:
    with open(filename, "r", encoding="utf-8") as handle:
        xml_string = handle.read()
    old_resolver = mi.file_resolver()
    resolver = mi.FileResolver(old_resolver)
    resolver.append(os.path.dirname(filename))
    try:
        mi.set_file_resolver(resolver)
        processed = _merge_scene_geometry(xml_string)
        return Scene(mi.load_string(processed, optimize=False))
    finally:
        mi.set_file_resolver(old_resolver)
