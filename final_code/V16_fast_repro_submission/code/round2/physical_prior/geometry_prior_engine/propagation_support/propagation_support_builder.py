
"""Build scene-conditioned propagation supports for the offline prior."""

import drjit as dr

from .multibounce_candidate_generator import MultiBounceCandidateGenerator
from .specular_path_constructor import SpecularPathConstructor
from .field_calculator import FieldCalculator
from .paths import Paths
from round2.physical_prior.geometry_prior_engine.scene import Scene


class PropagationSupportBuilder:
    # pylint: disable=line-too-long
    r"""
    Construct propagation supports between scene transmitters and receivers.

    The builder computes propagation paths between the antennas of
    all transmitters and receivers in the a scene.
    For each propagation path :math:`i`, the corresponding channel coefficient
    :math:`a_i` and delay :math:`\tau_i`, the
    angles of departure :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
    and arrival :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`, as
    well as the Doppler shifts :math:`f_{\Delta, i}` are computed.
    For more detail, see :eq:`H_final`.
    The builder supports line-of-sight, specular and diffuse
    reflection, as well as refraction. Paths can consist of any of these interaction
    types, in any order.
    Different propagation phenomena can be individually enabled/disabled.

    This solver assumes that materials are thin enough that their effect on
    transmitted rays (i.e., rays that traverse the materials through double
    refraction) is negligible. Rays are traced without angular deflection, and
    objects like walls should be modeled as single flat surfaces having an
    attached radio material that accounts for its thickness.
    This approach may be inaccurate for very thick objects.
    The figure below illustrates this model, where :math:`E_i` is the
    incident electric field, :math:`E_r` is the reflected field and :math:`E_t` is
    the transmitted field. The Jones matrices, :math:`\mathbf{R}(d)` and
    :math:`\mathbf{T}(d)`, represent the effects of reflection and transmission,
    respectively, and depend on the slab thickness, :math:`d`.

    .. figure:: ../figures/transmission_model.png
        :width: 80%
        :align: center

    Transmitters and receivers are traced from their device positions. The
    responses for individual array elements are then computed using planar-wave
    phase shifts, which keeps large production arrays tractable.
    """

    def __init__(self):

        # Instantiate the multi-bounce candidate generator.
        self._candidate_generator = MultiBounceCandidateGenerator()
        # Refine specular candidates with mirrored-source geometry.
        self._specular_constructor = SpecularPathConstructor()
        # Instantiate the field calculator.
        self._field_calculator = FieldCalculator()

    @property
    def loop_mode(self):
        # pylint: disable=line-too-long
        r"""Get/set the Dr.Jit mode used to evaluate the loops that implement
        the solver. Should be one of "evaluated" or "symbolic". Symbolic mode
        (default) is the fastest one but does not support automatic
        differentiation.
        For more details, see the `corresponding Dr.Jit documentation <https://drjit.readthedocs.io/en/latest/cflow.html#sym-eval>`_.

        :type: "evaluated" | "symbolic"
        """
        return self._field_calculator.loop_mode

    @loop_mode.setter
    def loop_mode(self, mode):
        if mode not in ("evaluated", "symbolic"):
            raise ValueError("Invalid loop mode. Must be either 'evaluated'"
                             " or 'symbolic'")
        self._specular_constructor.loop_mode = mode
        self._field_calculator.loop_mode = mode

    def __call__(self,
                 scene: Scene,
                 max_depth: int = 3,
                 max_num_paths_per_src: int = 1000000,
                 samples_per_src: int = 1000000,
                 los: bool = True,
                 specular_reflection: bool = True,
                 diffuse_reflection: bool = False,
                 refraction: bool = True,
                 seed: int = 42) -> Paths:
        # pylint: disable=line-too-long
        r"""
        Build propagation supports for the configured scene.

        :param scene: Scene for which to compute paths
        :param max_depth: Maximum depth
        :param max_num_paths_per_src: Maximum number of paths per source
        :param samples_per_src: Number of samples per source
        :param los: Enable line-of-sight paths
        :param specular_reflection: Enables specular reflection
        :param diffuse_reflection: Enables diffuse reflection
        :param refraction: Enables refraction
        :param seed: Seed

        :return: Computed paths
        """

        # Check that the scene is all set for simulations
        scene.all_set(radio_map=False)

        # Generates sources positions and orientations
        src_positions, src_orientations, rel_ant_positions_tx = \
            scene.sources()
        tgt_positions, tgt_orientations, rel_ant_positions_rx = \
            scene.targets()

        # Trace paths and compute channel impulse responses
        src_antenna_patterns = scene.tx_array.antenna_pattern.patterns
        tgt_antenna_patterns = scene.rx_array.antenna_pattern.patterns
        dr.make_opaque(src_positions, tgt_positions, src_orientations,
                       tgt_orientations)

        # Generate candidates
        paths_buffer = self._candidate_generator(
            mi_scene=scene.mi_scene,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            samples_per_src=samples_per_src,
            max_num_paths_per_src=max_num_paths_per_src,
            max_depth=max_depth,
            los=los,
            specular_reflection=specular_reflection,
            diffuse_reflection=diffuse_reflection,
            refraction=refraction,
            seed=seed
        )

        paths_buffer.schedule()
        dr.eval()

        # Shrink the paths buffer to fit the number of paths effectively found
        paths_buffer.shrink()

        # Detach the paths geometry to avoid differentiation through the
        # candidate generator
        paths_buffer.detach_geometry()

        # Solve specular chains and suffixes
        paths_buffer = self._specular_constructor(
            scene=scene.mi_scene,
            paths=paths_buffer,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
        )

        # Compute channel coefficients and delays
        paths_buffer = self._field_calculator(
            wavelength=scene.wavelength,
            paths=paths_buffer,
            samples_per_src=samples_per_src,
            src_positions=src_positions,
            tgt_positions=tgt_positions,
            src_orientations=src_orientations,
            tgt_orientations=tgt_orientations,
            src_antenna_patterns=src_antenna_patterns,
            tgt_antenna_patterns=tgt_antenna_patterns
        )

        # Discard invalid paths
        paths_buffer.discard_invalid()

        # Build the path object
        paths = Paths(scene, src_positions, tgt_positions, paths_buffer,
                      rel_ant_positions_tx, rel_ant_positions_rx)

        return paths
