"""Staged beam physics for a modular optomechanical laser simulator.

All public functions use SI units internally:
- lengths in meters
- power in watts
- angles in radians
- attenuation coefficients in inverse meters
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


REFERENCE_WAVELENGTH_M = 0.55e-6
METERS_PER_KILOMETER = 1_000.0


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}")


def calculate_beam_waist(
    aperture_diameter_m: float,
    beam_quality_m2: float,
) -> float:
    """Calculate the source beam waist from aperture diameter and M^2."""
    _require_positive("aperture_diameter_m", aperture_diameter_m)
    _require_positive("beam_quality_m2", beam_quality_m2)

    return aperture_diameter_m / (np.pi * beam_quality_m2)


def calculate_wavenumber(wavelength_m: float) -> float:
    """Calculate optical wavenumber k = 2*pi/lambda."""
    _require_positive("wavelength_m", wavelength_m)
    return float(2.0 * np.pi / wavelength_m)


def calculate_far_field_divergence(
    wavelength_m: float,
    beam_waist_m: float,
    beam_quality_m2: float,
) -> float:
    """Calculate far-field divergence angle for a non-ideal Gaussian beam."""
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("beam_waist_m", beam_waist_m)
    _require_positive("beam_quality_m2", beam_quality_m2)
    return float(beam_quality_m2 * wavelength_m / (np.pi * beam_waist_m))


def calculate_kruse_q(visibility_m: float) -> float:
    """Calculate the Kruse scattering exponent from visibility.

    The Kruse model thresholds are conventionally stated in kilometers. This
    function accepts SI input and converts to kilometers internally.
    """
    _require_positive("visibility_m", visibility_m)

    visibility_km = visibility_m / METERS_PER_KILOMETER
    if visibility_km > 50.0:
        return 1.6
    if visibility_km > 6.0:
        return 1.3
    return 0.585 * visibility_km ** (1.0 / 3.0)


def calculate_spot_size(
    distance_m: float,
    wavelength_m: float,
    beam_waist_m: float,
    beam_quality_m2: float,
) -> float:
    """Calculate diffraction-driven beam radius at range z."""
    if distance_m < 0:
        raise ValueError(f"distance_m must be non-negative; got {distance_m!r}")
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("beam_waist_m", beam_waist_m)
    _require_positive("beam_quality_m2", beam_quality_m2)

    divergence_term = beam_quality_m2 * wavelength_m * distance_m
    divergence_term /= np.pi * beam_waist_m
    return float(np.sqrt(beam_waist_m**2 + divergence_term**2))


def calculate_aperture_for_target_spot(
    target_spot_radius_m: float,
    distance_m: float,
    wavelength_m: float,
    beam_quality_m2: float,
    branch: str = "large_aperture",
) -> float:
    """Solve for aperture diameter that produces a target w(z).

    The Gaussian beam radius ``w`` is the 1/e^2 irradiance radius. A "10 cm
    beam diameter" therefore maps to ``target_spot_radius_m = 0.05``.

    The quadratic has two physical roots when the requested target spot is
    above the diffraction-limited minimum. The large-aperture branch is usually
    the more relevant high-energy-laser operating point because it is less
    divergence dominated.
    """
    _require_positive("target_spot_radius_m", target_spot_radius_m)
    if distance_m < 0:
        raise ValueError(f"distance_m must be non-negative; got {distance_m!r}")
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("beam_quality_m2", beam_quality_m2)

    propagation_factor = beam_quality_m2 * wavelength_m * distance_m / np.pi
    discriminant = target_spot_radius_m**4 - 4.0 * propagation_factor**2
    if discriminant < 0:
        minimum_radius = np.sqrt(2.0 * propagation_factor)
        raise ValueError(
            "target_spot_radius_m is below the diffraction-limited minimum "
            f"of {minimum_radius:.6g} m for the selected range, wavelength, and M^2"
        )

    sqrt_discriminant = np.sqrt(discriminant)
    large_waist_squared = 0.5 * (target_spot_radius_m**2 + sqrt_discriminant)
    small_waist_squared = 0.5 * (target_spot_radius_m**2 - sqrt_discriminant)
    if branch == "large_aperture":
        beam_waist_m = np.sqrt(large_waist_squared)
    elif branch == "small_aperture":
        beam_waist_m = np.sqrt(small_waist_squared)
    else:
        raise ValueError("branch must be 'large_aperture' or 'small_aperture'")

    return float(np.pi * beam_quality_m2 * beam_waist_m)


def calculate_attenuation_coefficient(
    wavelength_m: float,
    visibility_m: float,
) -> float:
    """Calculate Kruse atmospheric attenuation coefficient in 1/m."""
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("visibility_m", visibility_m)

    q = calculate_kruse_q(visibility_m)
    visibility_km = visibility_m / METERS_PER_KILOMETER
    alpha_per_km = (3.91 / visibility_km) * (wavelength_m / REFERENCE_WAVELENGTH_M) ** (-q)
    return alpha_per_km / METERS_PER_KILOMETER


def calculate_fried_parameter(
    distance_m: float,
    wavelength_m: float,
    cn2_m_minus_2_over_3: float,
) -> float:
    """Calculate Fried coherence diameter r0 from the primer model."""
    if distance_m < 0:
        raise ValueError(f"distance_m must be non-negative; got {distance_m!r}")
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("cn2_m_minus_2_over_3", cn2_m_minus_2_over_3)

    wavenumber = calculate_wavenumber(wavelength_m)
    return float((0.423 * wavenumber**2 * cn2_m_minus_2_over_3 * distance_m) ** (-3.0 / 5.0))


def calculate_turbulence_spread(
    distance_m: float,
    wavelength_m: float,
    cn2_m_minus_2_over_3: float,
) -> tuple[float, float]:
    """Calculate turbulence-induced beam broadening and Fried parameter."""
    fried_parameter_m = calculate_fried_parameter(
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        cn2_m_minus_2_over_3=cn2_m_minus_2_over_3,
    )
    turbulence_radius_m = wavelength_m * distance_m / fried_parameter_m
    return float(turbulence_radius_m), float(fried_parameter_m)


def calculate_beam_wander_variance(
    distance_m: float,
    wavelength_m: float,
    aperture_diameter_m: float,
    cn2_m_minus_2_over_3: float,
) -> float:
    """Calculate the primer's approximate beam-wander variance in m^2."""
    if distance_m < 0:
        raise ValueError(f"distance_m must be non-negative; got {distance_m!r}")
    _require_positive("wavelength_m", wavelength_m)
    _require_positive("aperture_diameter_m", aperture_diameter_m)
    _require_positive("cn2_m_minus_2_over_3", cn2_m_minus_2_over_3)

    wavenumber = calculate_wavenumber(wavelength_m)
    return float(2.42 * cn2_m_minus_2_over_3 * distance_m**3 * (wavenumber * aperture_diameter_m) ** (-1.0 / 3.0))


def sample_beam_wander_shift(
    beam_wander_variance_m2: float,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Sample a stochastic beam-center shift from beam wander variance."""
    if beam_wander_variance_m2 < 0:
        raise ValueError(f"beam_wander_variance_m2 must be non-negative; got {beam_wander_variance_m2!r}")
    generator = rng if rng is not None else np.random.default_rng()
    sigma_m = float(np.sqrt(beam_wander_variance_m2))
    shift = generator.normal(0.0, sigma_m, size=2)
    return float(shift[0]), float(shift[1])


def beam_direction_unit_vector(pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Return the beam unit vector implied by the pitch/yaw projection model."""
    tan_phi = np.tan(yaw_rad)
    tan_theta = np.tan(pitch_rad)
    vector = np.array([tan_phi, tan_theta, 1.0], dtype=float)
    return vector / np.linalg.norm(vector)


def calculate_incidence_angle(pitch_rad: float, yaw_rad: float) -> float:
    """Return the incidence angle relative to the surface normal."""
    unit_vector = beam_direction_unit_vector(pitch_rad, yaw_rad)
    return float(np.arccos(np.clip(unit_vector[2], -1.0, 1.0)))


@dataclass(frozen=True)
class SensorPose3D:
    """A photodiode on a cuboid face."""

    position_m: tuple[float, float, float]
    normal_m: tuple[float, float, float]
    face_name: str = ""


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("vector must be non-zero")
    return vector / norm


def _as_face_vector(face_index: int) -> tuple[str, np.ndarray]:
    faces = (
        ("+x", np.array([1.0, 0.0, 0.0], dtype=float)),
        ("-x", np.array([-1.0, 0.0, 0.0], dtype=float)),
        ("+y", np.array([0.0, 1.0, 0.0], dtype=float)),
        ("-y", np.array([0.0, -1.0, 0.0], dtype=float)),
        ("+z", np.array([0.0, 0.0, 1.0], dtype=float)),
        ("-z", np.array([0.0, 0.0, -1.0], dtype=float)),
    )
    return faces[int(face_index) % len(faces)]


def cuboid_face_pose(
    face_index: int,
    u: float,
    v: float,
    lx_m: float,
    ly_m: float,
    lz_m: float,
) -> SensorPose3D:
    """Map normalized face coordinates to a cuboid face pose."""
    face_name, normal = _as_face_vector(face_index)
    half_x = lx_m / 2.0
    half_y = ly_m / 2.0
    half_z = lz_m / 2.0
    if face_name == "+x":
        position = np.array([half_x, u * half_y, v * half_z], dtype=float)
    elif face_name == "-x":
        position = np.array([-half_x, u * half_y, v * half_z], dtype=float)
    elif face_name == "+y":
        position = np.array([u * half_x, half_y, v * half_z], dtype=float)
    elif face_name == "-y":
        position = np.array([u * half_x, -half_y, v * half_z], dtype=float)
    elif face_name == "+z":
        position = np.array([u * half_x, v * half_y, half_z], dtype=float)
    else:
        position = np.array([u * half_x, v * half_y, -half_z], dtype=float)
    return SensorPose3D(
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        normal_m=(float(normal[0]), float(normal[1]), float(normal[2])),
        face_name=face_name,
    )


def beam_axis_perpendicular_distance(
    sensor_position_m: np.ndarray,
    beam_origin_m: np.ndarray,
    beam_unit_vector: np.ndarray,
) -> np.ndarray:
    """Return perpendicular distance from points to a 3D beam axis."""
    axis = _normalize_vector(np.asarray(beam_unit_vector, dtype=float))
    relative = np.asarray(sensor_position_m, dtype=float) - np.asarray(beam_origin_m, dtype=float)
    cross = np.cross(relative, axis)
    return np.linalg.norm(cross, axis=-1)


def calculate_cuboid_sensor_irradiance(
    sensor_positions_m: np.ndarray,
    sensor_normals_m: np.ndarray,
    beam_origin_m: np.ndarray,
    beam_unit_vector: np.ndarray,
    peak_irradiance_w_per_m2: float,
    effective_beam_radius_m: float,
) -> np.ndarray:
    """Evaluate the 3D beam footprint on a cuboid surface."""
    if peak_irradiance_w_per_m2 < 0.0:
        raise ValueError("peak_irradiance_w_per_m2 must be non-negative")
    _require_positive("effective_beam_radius_m", effective_beam_radius_m)
    positions = np.asarray(sensor_positions_m, dtype=float)
    normals = np.asarray(sensor_normals_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("sensor_positions_m must have shape (N, 3)")
    if normals.shape != positions.shape:
        raise ValueError("sensor_normals_m must match sensor_positions_m")

    axis = _normalize_vector(np.asarray(beam_unit_vector, dtype=float))
    rho_m = beam_axis_perpendicular_distance(positions, np.asarray(beam_origin_m, dtype=float), axis)
    cos_gamma = np.einsum("ij,j->i", normals, axis)
    valid = cos_gamma > 0.0
    irradiance = np.zeros(positions.shape[0], dtype=float)
    irradiance[valid] = (
        peak_irradiance_w_per_m2
        * cos_gamma[valid]
        * np.exp((-2.0 / effective_beam_radius_m**2) * rho_m[valid] ** 2)
    )
    return irradiance


@dataclass
class KinematicEngagement:
    """Simple beam-center and angle kinematics for temporal fly-by simulation."""

    initial_beam_center_x_m: float = 0.0
    initial_beam_center_y_m: float = 0.0
    initial_pitch_rad: float = 0.0
    initial_yaw_rad: float = 0.0
    vx_m_per_s: float = 0.0
    vy_m_per_s: float = 0.0
    omega_pitch_rad_per_s: float = 0.0
    omega_yaw_rad_per_s: float = 0.0
    dt_s: float = 0.05
    beam_wander_variance_m2: float = 0.0
    rng: np.random.Generator | None = None
    time_s: float = field(init=False, default=0.0)
    beam_center_x_m: float = field(init=False, default=0.0)
    beam_center_y_m: float = field(init=False, default=0.0)
    pitch_rad: float = field(init=False, default=0.0)
    yaw_rad: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive; got {self.dt_s!r}")
        self.time_s = 0.0
        self.beam_center_x_m = float(self.initial_beam_center_x_m)
        self.beam_center_y_m = float(self.initial_beam_center_y_m)
        self.pitch_rad = float(self.initial_pitch_rad)
        self.yaw_rad = float(self.initial_yaw_rad)

    def step(self) -> dict[str, float]:
        """Advance the engagement by one time step and inject beam wander."""
        self.time_s += self.dt_s
        self.beam_center_x_m += self.vx_m_per_s * self.dt_s
        self.beam_center_y_m += self.vy_m_per_s * self.dt_s
        self.pitch_rad += self.omega_pitch_rad_per_s * self.dt_s
        self.yaw_rad += self.omega_yaw_rad_per_s * self.dt_s

        if self.beam_wander_variance_m2 > 0.0:
            generator = self.rng if self.rng is not None else np.random.default_rng()
            wander_std_m = float(np.sqrt(self.beam_wander_variance_m2 * self.dt_s))
            wander_x_m, wander_y_m = generator.normal(0.0, wander_std_m, size=2)
            self.beam_center_x_m += float(wander_x_m)
            self.beam_center_y_m += float(wander_y_m)

        return {
            "time_s": float(self.time_s),
            "beam_center_x_m": float(self.beam_center_x_m),
            "beam_center_y_m": float(self.beam_center_y_m),
            "pitch_rad": float(self.pitch_rad),
            "yaw_rad": float(self.yaw_rad),
        }


def combine_effective_spot_size(
    base_spot_radius_m: float,
    turbulence_radius_m: float = 0.0,
) -> float:
    """Combine diffraction and turbulence beam radii into w_eff."""
    _require_positive("base_spot_radius_m", base_spot_radius_m)
    if turbulence_radius_m < 0:
        raise ValueError(f"turbulence_radius_m must be non-negative; got {turbulence_radius_m!r}")
    return float(np.sqrt(base_spot_radius_m**2 + turbulence_radius_m**2))


def propagate_beam(
    initial_power_w: float,
    distance_m: float,
    wavelength_m: float,
    visibility_m: float,
    aperture_diameter_m: float,
    beam_waist_m: float,
    beam_quality_m2: float,
    cn2_m_minus_2_over_3: float | None = None,
) -> dict[str, float | None]:
    """Propagate a Gaussian beam through attenuating atmosphere.

    Returns a dictionary containing base spot size, turbulence contribution,
    effective spot size, transmitted power, and attenuation coefficient.
    """
    if initial_power_w < 0:
        raise ValueError(f"initial_power_w must be non-negative; got {initial_power_w!r}")

    spot_radius_m = calculate_spot_size(
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        beam_waist_m=beam_waist_m,
        beam_quality_m2=beam_quality_m2,
    )
    attenuation_coefficient = calculate_attenuation_coefficient(
        wavelength_m=wavelength_m,
        visibility_m=visibility_m,
    )
    transmitted_power_w = initial_power_w * np.exp(-attenuation_coefficient * distance_m)

    turbulence_radius_m = 0.0
    fried_parameter_m: float | None = None
    beam_wander_variance_m2: float | None = None
    if cn2_m_minus_2_over_3 is not None and cn2_m_minus_2_over_3 > 0.0:
        turbulence_radius_m, fried_parameter_m = calculate_turbulence_spread(
            distance_m=distance_m,
            wavelength_m=wavelength_m,
            cn2_m_minus_2_over_3=cn2_m_minus_2_over_3,
        )
        beam_wander_variance_m2 = calculate_beam_wander_variance(
            distance_m=distance_m,
            wavelength_m=wavelength_m,
            aperture_diameter_m=aperture_diameter_m,
            cn2_m_minus_2_over_3=cn2_m_minus_2_over_3,
        )
    effective_radius_m = combine_effective_spot_size(
        base_spot_radius_m=spot_radius_m,
        turbulence_radius_m=turbulence_radius_m,
    )

    return {
        "base_spot_radius_m": float(spot_radius_m),
        "turbulence_radius_m": float(turbulence_radius_m),
        "fried_parameter_m": fried_parameter_m,
        "beam_wander_variance_m2": beam_wander_variance_m2,
        "effective_radius_m": float(effective_radius_m),
        "transmitted_power_w": float(transmitted_power_w),
        "attenuation_coefficient_1_per_m": float(attenuation_coefficient),
    }


def calculate_irradiance_map(
    transmitted_power_w: float,
    effective_beam_radius_m: float,
    pitch_rad: float,
    yaw_rad: float,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    beam_center_x_m: float = 0.0,
    beam_center_y_m: float = 0.0,
) -> np.ndarray:
    """Calculate projected 2D irradiance on a flat drone-facing plane."""
    if transmitted_power_w < 0:
        raise ValueError(f"transmitted_power_w must be non-negative; got {transmitted_power_w!r}")
    _require_positive("effective_beam_radius_m", effective_beam_radius_m)
    if x_grid_m.shape != y_grid_m.shape:
        raise ValueError("x_grid_m and y_grid_m must have matching shapes")

    tan_phi = np.tan(yaw_rad)
    tan_theta = np.tan(pitch_rad)
    projection_denominator = 1.0 + tan_phi**2 + tan_theta**2
    dx = x_grid_m - beam_center_x_m
    dy = y_grid_m - beam_center_y_m

    geometric_scale = 1.0 / np.sqrt(projection_denominator)
    gaussian_argument = dx**2 + dy**2
    gaussian_argument -= ((dx * tan_phi + dy * tan_theta) ** 2) / projection_denominator

    peak_scale = 2.0 * transmitted_power_w / (np.pi * effective_beam_radius_m**2)
    irradiance = peak_scale * geometric_scale
    irradiance *= np.exp((-2.0 / effective_beam_radius_m**2) * gaussian_argument)

    return irradiance


def run_continuous_beam_pipeline(
    aperture_diameter_m: float,
    wavelength_m: float,
    beam_quality_m2: float,
    initial_power_w: float,
    distance_m: float,
    visibility_m: float,
    pitch_rad: float,
    yaw_rad: float,
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    beam_center_x_m: float = 0.0,
    beam_center_y_m: float = 0.0,
    cn2_m_minus_2_over_3: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Run source, propagation, and projection stages as a readable pipeline."""
    beam_waist_m = calculate_beam_waist(
        aperture_diameter_m=aperture_diameter_m,
        beam_quality_m2=beam_quality_m2,
    )
    propagation = propagate_beam(
        initial_power_w=initial_power_w,
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        visibility_m=visibility_m,
        aperture_diameter_m=aperture_diameter_m,
        beam_waist_m=beam_waist_m,
        beam_quality_m2=beam_quality_m2,
        cn2_m_minus_2_over_3=cn2_m_minus_2_over_3,
    )
    irradiance_w_per_m2 = calculate_irradiance_map(
        transmitted_power_w=float(propagation["transmitted_power_w"]),
        effective_beam_radius_m=float(propagation["effective_radius_m"]),
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        x_grid_m=x_grid_m,
        y_grid_m=y_grid_m,
        beam_center_x_m=beam_center_x_m,
        beam_center_y_m=beam_center_y_m,
    )

    return {
        "beam_waist_m": beam_waist_m,
        "beam_divergence_rad": calculate_far_field_divergence(
            wavelength_m=wavelength_m,
            beam_waist_m=beam_waist_m,
            beam_quality_m2=beam_quality_m2,
        ),
        "wavenumber_rad_per_m": calculate_wavenumber(wavelength_m),
        "base_spot_radius_m": propagation["base_spot_radius_m"],
        "turbulence_radius_m": propagation["turbulence_radius_m"],
        "fried_parameter_m": propagation["fried_parameter_m"],
        "beam_wander_variance_m2": propagation["beam_wander_variance_m2"],
        "effective_radius_m": propagation["effective_radius_m"],
        "transmitted_power_w": propagation["transmitted_power_w"],
        "attenuation_coefficient_1_per_m": propagation["attenuation_coefficient_1_per_m"],
        "irradiance_w_per_m2": irradiance_w_per_m2,
    }
