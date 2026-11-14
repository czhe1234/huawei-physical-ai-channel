"""Minimal scene wrapper for the source-aligned propagation solver.

This keeps the required Sionna RT 2.0.1 scene behavior while omitting
rendering, preview, radio-map, and scene-editing APIs.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import drjit as dr
import mitsuba as mi
from scipy.constants import speed_of_light

from .antenna_array import AntennaArray
from .constants import DEFAULT_FREQUENCY, DEFAULT_THICKNESS
from .radio_devices import RadioDevice, Receiver, Transmitter
from .radio_materials import RadioMaterialBase
from .radio_materials.itu import ITU_MATERIALS_PROPERTIES
from .utils.meshes import remove_mesh_duplicate_vertices


def process_xml(
    xml_string: str,
    merge_shapes: bool = True,
    merge_shapes_exclude_regex: str | None = None,
    default_thickness: float = DEFAULT_THICKNESS,
) -> str:
    """Apply the required XML material conversion and shape merge."""

    regex = (
        re.compile(merge_shapes_exclude_regex)
        if merge_shapes_exclude_regex is not None
        else None
    )
    root = ET.fromstring(xml_string)

    for bsdf in root.findall("./bsdf") + root.findall(".//shape/bsdf"):
        bsdf_type = bsdf.attrib.get("type")
        mat_id = bsdf.attrib.get("id")
        name = mat_id
        if name is None:
            bsdf_string = ET.tostring(bsdf, encoding="unicode")
            raise ValueError(
                "Error when loading scene: found BSDF element without 'name'"
                " or 'id' attribute:\n" + bsdf_string
            )
        if name.startswith("mat-"):
            name = name[4:]
        if name.startswith("itu_") or name.startswith("itu-"):
            bsdf_type = "itu-radio-material"
            props: dict[str, tuple[str, object]] = {}
            thickness = default_thickness
            thickness_prop = bsdf.find("float[@name='thickness']")
            if thickness_prop is not None:
                thickness = float(thickness_prop.get("value", thickness))
            for prop_name in ("color", "reflectance", "base_color"):
                color_prop = bsdf.find(f".//rgb[@name='{prop_name}']")
                if color_prop is not None:
                    props["color"] = ("rgb", color_prop.get("value"))
                    break
            itu_type = name[4:]
            type_prop = bsdf.find("string[@name='type']")
            if type_prop is not None:
                itu_type = type_prop.get("value")
            props["type"] = ("string", itu_type)
            props["thickness"] = ("float", thickness)
            bsdf.clear()
            bsdf.attrib["type"] = bsdf_type
            if mat_id is not None:
                bsdf.attrib["id"] = mat_id
            for key, (kind, value) in props.items():
                bsdf.append(ET.Element(kind, {"name": key, "value": str(value)}))
        elif bsdf_type != "itu-radio-material" and name in ITU_MATERIALS_PROPERTIES:
            raise ValueError(
                f'Found material with name "{mat_id}". ITU material names must '
                'start with "itu_", e.g. "mat-itu_concrete".'
            )

    merge_node = ET.Element("shape", {"type": "merge", "id": "merged-shapes"})
    merge_node_empty = True
    for shape in root.findall("shape"):
        if merge_shapes and (
            merge_shapes_exclude_regex is None
            or not regex.search(shape.attrib.get("id", ""))
        ):
            root.remove(shape)
            merge_node.append(shape)
            merge_node_empty = False
    if not merge_node_empty:
        root.append(merge_node)

    ET.indent(root, space="    ")
    return ET.tostring(root).decode("utf-8")


class Scene:
    """Subset of the original RT scene required by ``PathSolver``."""

    def __init__(self, mi_scene: mi.Scene, remove_duplicate_vertices: bool = False):
        if not isinstance(mi_scene, mi.Scene):
            raise TypeError("mi_scene must be a Mitsuba Scene")
        self._scene = mi_scene
        self._tx_array: AntennaArray | None = None
        self._rx_array: AntennaArray | None = None
        self._transmitters: dict[str, Transmitter] = {}
        self._receivers: dict[str, Receiver] = {}
        self._radio_materials: dict[str, RadioMaterialBase] = {}
        self._objects: dict[str, mi.Mesh] = {}
        self._frequency = mi.Float(DEFAULT_FREQUENCY)
        self._load_scene_objects(remove_duplicate_vertices)

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
        for material in self._radio_materials.values():
            material.frequency_update()

    @property
    def wavelength(self):
        return speed_of_light / self.frequency

    @property
    def angular_frequency(self):
        return dr.two_pi * self.frequency

    @property
    def wavenumber(self):
        return dr.two_pi / self.wavelength

    @property
    def tx_array(self):
        return self._tx_array

    @tx_array.setter
    def tx_array(self, value):
        if not isinstance(value, AntennaArray):
            raise TypeError("tx_array must be an AntennaArray")
        self._tx_array = value

    @property
    def rx_array(self):
        return self._rx_array

    @rx_array.setter
    def rx_array(self, value):
        if not isinstance(value, AntennaArray):
            raise TypeError("rx_array must be an AntennaArray")
        self._rx_array = value

    @property
    def transmitters(self):
        return dict(self._transmitters)

    @property
    def receivers(self):
        return dict(self._receivers)

    @property
    def radio_materials(self):
        return dict(self._radio_materials)

    @property
    def objects(self):
        return dict(self._objects)

    def get(self, name: str):
        return (
            self._radio_materials.get(name)
            or self._objects.get(name)
            or self._transmitters.get(name)
            or self._receivers.get(name)
        )

    def add(self, item: RadioDevice | RadioMaterialBase) -> None:
        name = item.name
        current = self.get(name)
        if current is not None:
            if current is not item:
                raise ValueError(f"Name '{name}' is already used by another item")
            return
        if isinstance(item, RadioMaterialBase):
            item.scene = self
            self._radio_materials[name] = item
        elif isinstance(item, Transmitter):
            self._transmitters[name] = item
        elif isinstance(item, Receiver):
            self._receivers[name] = item
        else:
            raise ValueError(f"Unsupported scene item: {type(item)}")

    def remove(self, name: str) -> None:
        self._transmitters.pop(name, None)
        self._receivers.pop(name, None)

    def all_set(self, radio_map: bool = False) -> None:
        if self.tx_array is None:
            raise ValueError("Transmitter array not set")
        if not self._transmitters:
            raise ValueError("Scene has no transmitters")
        if not radio_map:
            if self.rx_array is None:
                raise ValueError("Receiver array not set")
            if not self._receivers:
                raise ValueError("Scene has no receivers")

    def sources(self, synthetic_array: bool, return_velocities: bool):
        return self._endpoints(
            self._transmitters.values(),
            self.tx_array,
            synthetic_array,
            return_velocities,
        )

    def targets(self, synthetic_array: bool, return_velocities: bool):
        return self._endpoints(
            self._receivers.values(), self.rx_array, synthetic_array, return_velocities
        )

    def _load_scene_objects(self, remove_duplicate_vertices: bool) -> None:
        for shape in self._scene.shapes():
            if not isinstance(shape, mi.Mesh):
                raise TypeError("Only triangle meshes are supported")
            material = shape.bsdf()
            if not isinstance(material, RadioMaterialBase):
                raise ValueError(f"Shape {shape.id()} does not use a radio material")
            mi.set_log_level(mi.LogLevel.Error)
            shape.build_directed_edges()
            mi.set_log_level(mi.LogLevel.Warn)
            if remove_duplicate_vertices:
                remove_mesh_duplicate_vertices(shape)
            name = shape.id()
            if name.startswith("mesh-"):
                name = name[5:]
            self._objects[name] = shape
            material.add_object()
            if material.name not in self._radio_materials:
                material.scene = self
                self._radio_materials[material.name] = material

    def _endpoints(
        self,
        radio_devices,
        array: AntennaArray,
        synthetic_array: bool,
        return_velocities: bool,
    ):
        radio_devices = list(radio_devices)
        n_devices = len(radio_devices)
        if synthetic_array or array is None:
            n_endpoints = n_devices
            effective_array_size = 1
        else:
            n_endpoints = n_devices * array.array_size
            effective_array_size = array.array_size

        positions = dr.zeros(mi.Point3f, n_endpoints)
        orientations = dr.zeros(mi.Point3f, n_endpoints)
        if synthetic_array:
            relative_positions = dr.zeros(mi.Point3f, n_endpoints * array.array_size)
            relative_indices = dr.arange(mi.UInt, array.array_size)
        else:
            relative_positions = None
        velocities = dr.zeros(mi.Vector3f, n_devices) if return_velocities else None

        endpoint_indices = dr.arange(mi.UInt, effective_array_size)
        for index, device in enumerate(radio_devices):
            position = device.position
            orientation = device.orientation
            if synthetic_array:
                antenna_positions = array.rotate(self.wavelength, orientation)
                indices = relative_indices + index * array.array_size
                dr.scatter(relative_positions.x, antenna_positions.x, indices)
                dr.scatter(relative_positions.y, antenna_positions.y, indices)
                dr.scatter(relative_positions.z, antenna_positions.z, indices)
            else:
                position = array.rotate(self.wavelength, orientation) + position
            indices = endpoint_indices + index * effective_array_size
            dr.scatter(positions.x, position.x, indices)
            dr.scatter(positions.y, position.y, indices)
            dr.scatter(positions.z, position.z, indices)
            dr.scatter(orientations.x, orientation.x, indices)
            dr.scatter(orientations.y, orientation.y, indices)
            dr.scatter(orientations.z, orientation.z, indices)
            if return_velocities:
                dr.scatter(velocities.x, device.velocity.x, index)
                dr.scatter(velocities.y, device.velocity.y, index)
                dr.scatter(velocities.z, device.velocity.z, index)
        return positions, orientations, relative_positions, velocities


def load_scene(
    filename: str,
    merge_shapes: bool = True,
    merge_shapes_exclude_regex: str | None = None,
    remove_duplicate_vertices: bool = False,
) -> Scene:
    with open(filename, "r", encoding="utf-8") as handle:
        xml_string = handle.read()
    old_resolver = mi.file_resolver()
    resolver = mi.FileResolver(old_resolver)
    resolver.append(os.path.dirname(filename))
    try:
        mi.set_file_resolver(resolver)
        return load_scene_from_string(
            xml_string,
            merge_shapes=merge_shapes,
            merge_shapes_exclude_regex=merge_shapes_exclude_regex,
            remove_duplicate_vertices=remove_duplicate_vertices,
        )
    finally:
        mi.set_file_resolver(old_resolver)


def load_scene_from_string(
    xml_string: str,
    merge_shapes: bool = True,
    merge_shapes_exclude_regex: str | None = None,
    remove_duplicate_vertices: bool = False,
) -> Scene:
    processed = process_xml(
        xml_string,
        merge_shapes=merge_shapes,
        merge_shapes_exclude_regex=merge_shapes_exclude_regex,
    )
    mi_scene = mi.load_string(processed, optimize=False)
    return Scene(mi_scene, remove_duplicate_vertices=remove_duplicate_vertices)
