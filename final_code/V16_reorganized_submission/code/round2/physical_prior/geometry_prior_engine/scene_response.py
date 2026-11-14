import weakref

import drjit as dr
import mitsuba as mi
from scipy.constants import speed_of_light

from round2.physical_prior.geometry_prior_engine.constants import (
    DEFAULT_FREQUENCY,
    DEFAULT_THICKNESS,
    NO_JONES_MATRIX,
    InteractionType,
)
from round2.physical_prior.geometry_prior_engine.utils import (
    complex_dielectric_response,
    finite_layer_coefficients,
    jones_matrix_to_world_implicit,
)


class _CosineAngularResponse:
    def __call__(
        self,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
    ) -> mi.Float:
        del ki_local
        return dr.abs(ko_local.z) / dr.pi


class SceneInteractionResponse(mi.BSDF):
    def __init__(self, props: mi.Properties) -> None:
        self._eta_r = mi.Float(1.0)
        self._sigma = mi.Float(0.0)
        self._d = mi.Float(
            props["thickness"] if "thickness" in props else DEFAULT_THICKNESS
        )
        if "thickness" in props:
            del props["thickness"]
        self._s = mi.Float(
            props["scattering_coefficient"]
            if "scattering_coefficient" in props
            else 0.0
        )
        if "scattering_coefficient" in props:
            del props["scattering_coefficient"]
        for name in ("color", "reflectance", "base_color"):
            if name in props:
                del props[name]
        super().__init__(props)
        self._scene: weakref.ReferenceType | None = None
        self._diffuse_response = _CosineAngularResponse()
        self._diffuse_polarization_response = mi.Matrix4f(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

    @property
    def scene(self):
        return None if self._scene is None else self._scene()

    @scene.setter
    def scene(self, scene) -> None:
        from round2.physical_prior.geometry_prior_engine.scene import Scene

        if not isinstance(scene, Scene):
            raise ValueError("scene must be an instance of Scene")
        current = self.scene
        if current is not None and current is not scene:
            raise ValueError(
                f"interaction response '{self.name}' is already used by another scene"
            )
        self._scene = weakref.ref(scene)
        self.frequency_update()

    @property
    def name(self) -> str:
        name = self.id()
        return name[4:] if name.startswith("mat-") else name

    @property
    def scattering_coefficient(self):
        return self._s

    @scattering_coefficient.setter
    def scattering_coefficient(self, value) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError("scattering coefficient must be in [0, 1]")
        self._s = mi.Float(value)

    def frequency_update(self) -> None:
        if self.scene is None:
            return
        frequency_ghz = self.scene.frequency / 1e9
        if not 1.0 <= frequency_ghz <= 100.0:
            raise ValueError("scene surface parameters are undefined for this frequency")
        self._eta_r = mi.Float(5.24 * dr.power(frequency_ghz, 0.0))
        self._sigma = mi.Float(0.0462 * dr.power(frequency_ghz, 0.7822))

    def sample(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        sample1: mi.Float,
        sample2: mi.Point2f,
        active: bool | mi.Bool = True,
    ) -> tuple[mi.BSDFSample3f, mi.Spectrum]:
        ki_local = si.wi
        cos_theta_i = -ki_local.z
        compute_jones_matrix = ctx.component & NO_JONES_MATRIX == 0
        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T
        eta = complex_dielectric_response(self._eta_r, self._sigma, angular_frequency)
        r_te, r_tm, t_te, t_tm = finite_layer_coefficients(
            cos_theta_i, eta, self._d, wavelength
        )
        sampled_event, probs, specular, diffuse, _ = self._sample_event_type(
            sample1, r_te, r_tm, t_te, t_tm, loc_en_inter
        )
        reflection = specular | diffuse
        ko_spec_trans_local = self._specular_reflection_transmission_direction(
            ki_local, reflection
        )
        ko_diffuse_local = self._diffuse_reflection_direction(sample2)
        if compute_jones_matrix:
            spec_trans_mat = self._specular_reflection_transmission_matrix(
                to_world,
                ki_local,
                ko_spec_trans_local,
                reflection,
                r_te,
                r_tm,
                t_te,
                t_tm,
            )
            diff_mat = self._diffuse_reflection_matrix(
                si, ki_local, ko_diffuse_local, spec_trans_mat
            )
            jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
            s = dr.select(
                reflection, dr.sqrt(1.0 - dr.square(self._s)), 1.0
            )
            s = dr.select(diffuse, self._s, s)
            jones_mat *= s * dr.detach(dr.rsqrt(probs))
            jones_mat = mi.Spectrum(jones_mat)
        else:
            jones_mat = mi.Spectrum(0.0)
        ko_local = dr.select(diffuse, ko_diffuse_local, ko_spec_trans_local)
        bs = mi.BSDFSample3f()
        bs.sampled_component = sampled_event
        bs.wo = si.to_world(ko_local)
        bs.pdf = probs
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.DeltaReflection)
        bs.eta = 1.0
        return bs, jones_mat

    def eval(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True,
    ) -> mi.Spectrum:
        ki_local = si.wi
        cos_theta_i = -ki_local.z
        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
        ko_local = si.to_local(wo)
        to_world = mi.Matrix3f(si.sh_frame.s, si.sh_frame.t, si.sh_frame.n).T
        eta = complex_dielectric_response(self._eta_r, self._sigma, angular_frequency)
        r_te, r_tm, t_te, t_tm = finite_layer_coefficients(
            cos_theta_i, eta, self._d, wavelength
        )
        sampled_event = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        reflection = (sampled_event == InteractionType.SPECULAR) | (
            sampled_event == InteractionType.DIFFUSE
        )
        diffuse = sampled_event == InteractionType.DIFFUSE
        ko_spec_trans_local = self._specular_reflection_transmission_direction(
            ki_local, reflection
        )
        spec_trans_mat = self._specular_reflection_transmission_matrix(
            to_world,
            ki_local,
            ko_spec_trans_local,
            reflection,
            r_te,
            r_tm,
            t_te,
            t_tm,
        )
        diff_mat = self._diffuse_reflection_matrix(
            si, ki_local, ko_local, spec_trans_mat
        )
        jones_mat = dr.select(diffuse, diff_mat, spec_trans_mat)
        s = dr.select(reflection, dr.sqrt(1.0 - dr.square(self._s)), 1.0)
        s = dr.select(diffuse, self._s, s)
        jones_mat *= s
        return mi.Spectrum(jones_mat)

    def pdf(
        self,
        ctx: mi.BSDFContext,
        si: mi.SurfaceInteraction3f,
        wo: mi.Vector3f,
        active: bool | mi.Bool = True,
    ) -> mi.Float:
        ki_local = si.wi
        cos_theta_i = -ki_local.z
        if self.scene is None:
            angular_frequency = dr.two_pi * DEFAULT_FREQUENCY
            wavelength = speed_of_light / DEFAULT_FREQUENCY
        else:
            angular_frequency = self.scene.angular_frequency
            wavelength = self.scene.wavelength
        loc_en_inter = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        eta = complex_dielectric_response(self._eta_r, self._sigma, angular_frequency)
        r_te, r_tm, t_te, t_tm = finite_layer_coefficients(
            cos_theta_i, eta, self._d, wavelength
        )
        sampled_event = dr.reinterpret_array(mi.UInt, si.dp_du.z)
        specular = sampled_event == InteractionType.SPECULAR
        transmission = sampled_event == InteractionType.REFRACTION
        prs, prd, pt, _ = self._event_probabilities(
            r_te, r_tm, t_te, t_tm, loc_en_inter
        )
        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt
        return probs

    def traverse(self, callback: mi.TraversalCallback) -> None:
        callback.put("eta_r", self._eta_r, mi.ParamFlags.Differentiable)
        callback.put("sigma", self._sigma, mi.ParamFlags.Differentiable)
        callback.put("d", self._d, mi.ParamFlags.Differentiable)
        callback.put("s", self._s, mi.ParamFlags.Differentiable)

    def to_string(self) -> str:
        return "SceneInteractionResponse"

    def _event_probabilities(
        self,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt,
    ) -> tuple[mi.Float, mi.Float, mi.Float, mi.Bool]:
        r = dr.square(dr.abs(r_te)) + dr.square(dr.abs(r_tm))
        t = dr.square(dr.abs(t_te)) + dr.square(dr.abs(t_tm))
        s = dr.square(self._s)
        specular_enabled = enabled_interactions & InteractionType.SPECULAR > 0
        diffuse_enabled = enabled_interactions & InteractionType.DIFFUSE > 0
        refraction_enabled = enabled_interactions & InteractionType.REFRACTION > 0
        prs = dr.select(specular_enabled, r * (1.0 - s), mi.Float(0.0))
        prd = dr.select(diffuse_enabled, r * s, mi.Float(0.0))
        pt = dr.select(refraction_enabled, t, mi.Float(0.0))
        sum_probs = prs + prd + pt
        no_interaction = sum_probs <= 0.0
        norm_factor = dr.select(no_interaction, 1.0, dr.rcp(sum_probs))
        prs *= norm_factor
        prd *= norm_factor
        pt *= norm_factor
        return prs, prd, pt, no_interaction

    def _sample_event_type(
        self,
        sample1: mi.Float,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
        enabled_interactions: mi.UInt,
    ) -> tuple[mi.UInt, mi.Float, mi.Bool, mi.Bool, mi.Bool]:
        prs, prd, pt, no_interaction = self._event_probabilities(
            r_te, r_tm, t_te, t_tm, enabled_interactions
        )
        specular = sample1 < prs
        diffuse = (prs <= sample1) & (sample1 < prd + prs)
        transmission = sample1 >= prd + prs
        sampled_event = dr.select(
            specular, InteractionType.SPECULAR, InteractionType.DIFFUSE
        )
        sampled_event[transmission] = InteractionType.REFRACTION
        sampled_event[no_interaction] = InteractionType.NONE
        probs = dr.select(specular, prs, prd)
        probs[transmission] = pt
        output = (sampled_event, probs, specular, diffuse, transmission)
        return output

    def _specular_reflection_transmission_direction(
        self,
        ki_local: mi.Vector3f,
        reflection: mi.Bool,
    ) -> mi.Vector3f:
        ko_local_spec_refl = mi.reflect(-ki_local)
        ko_local_trans = ki_local
        ko_local = dr.select(reflection, ko_local_spec_refl, ko_local_trans)
        return ko_local

    def _specular_reflection_transmission_matrix(
        self,
        to_world: mi.Matrix3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        reflection: mi.Bool,
        r_te: mi.Complex2f,
        r_tm: mi.Complex2f,
        t_te: mi.Complex2f,
        t_tm: mi.Complex2f,
    ) -> mi.Matrix4f:
        c1 = dr.select(reflection, r_te, t_te)
        c2 = dr.select(reflection, r_tm, t_tm)
        jones_mat = jones_matrix_to_world_implicit(
            c1, c2, to_world, ki_local, ko_local
        )
        return jones_mat

    def _diffuse_reflection_direction(self, sample2: mi.Point2f) -> mi.Vector3f:
        ko_local = mi.warp.square_to_uniform_hemisphere(sample2)
        ko_local.z = dr.abs(ko_local.z)
        return ko_local

    def _diffuse_reflection_matrix(
        self,
        si: mi.SurfaceInteraction3f,
        ki_local: mi.Vector3f,
        ko_local: mi.Vector3f,
        specular_reflection_mat: mi.Matrix4f,
    ) -> mi.Matrix4f:
        ei = mi.Vector4f(
            si.duv_dx.x, si.duv_dx.y, si.duv_dy.x, si.duv_dy.y
        )
        solid_angle = si.t
        er_spec = specular_reflection_mat @ ei
        er_spec_norm = dr.norm(er_spec)
        ei_norm = dr.norm(ei)
        gamma = dr.select(
            ei_norm > 0.0,
            er_spec_norm * dr.rcp(ei_norm),
            mi.Float(0.0),
        )
        fs = self._diffuse_response(ki_local, ko_local)
        jones_mat = (
            dr.sqrt(fs * solid_angle)
            * gamma
            * self._diffuse_polarization_response
        )
        return jones_mat


mi.register_bsdf(
    "scene-interaction-response",
    lambda props: SceneInteractionResponse(props),
)
