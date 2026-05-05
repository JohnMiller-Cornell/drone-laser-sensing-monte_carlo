"""Monte Carlo scoring and optimization for laser photodiode layouts.

The implementation follows the math in ``laser_physics-3.tex``:

- sensor readings are modeled as irradiance plus additive Gaussian noise
- Fisher information is built from derivatives of irradiance, not log irradiance
- direction information is isolated with a Schur complement over nuisance terms
- layout quality is the expected log determinant over sampled engagement states
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from physics import (
    SensorPose3D,
    calculate_attenuation_coefficient,
    calculate_spot_size,
    calculate_turbulence_spread,
    cuboid_face_pose,
)
from sensors import (
    DEFAULT_LAYOUTS_PATH,
    PhotodiodeArray,
    get_layout,
    load_layouts_catalog,
    save_layouts_catalog,
)


EPSILON = 1e-12
DEFAULT_WAVELENGTH_M = 1064e-9
DEFAULT_VISIBILITY_M = 20_000.0
DEFAULT_INITIAL_POWER_W = 10_000.0
DEFAULT_BEAM_WAIST_M = 0.05
DEFAULT_BEAM_QUALITY_M2 = 1.2
MISSING = object()


@dataclass(frozen=True)
class BeamConfig:
    """Nominal beam and propagation parameters used per Monte Carlo scenario."""

    wavelength_m: float = DEFAULT_WAVELENGTH_M
    beam_quality_m2: float = DEFAULT_BEAM_QUALITY_M2
    beam_waist_m: float = DEFAULT_BEAM_WAIST_M
    initial_power_w: float = DEFAULT_INITIAL_POWER_W
    visibility_m: float = DEFAULT_VISIBILITY_M
    aperture_diameter_m: float | None = None
    cn2_m_minus_2_over_3: float | None = None

    @property
    def effective_aperture_diameter_m(self) -> float:
        if self.aperture_diameter_m is not None:
            return self.aperture_diameter_m
        return float(np.pi * self.beam_quality_m2 * self.beam_waist_m)


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Linear sensor model for Fisher scoring."""

    noise_floor_w_per_m2: float = 1.0
    relative_noise_fraction: float = 0.0
    optical_density: float = 0.0
    saturation_limit_w_per_m2: float = np.inf
    require_above_noise_floor: bool = False

    @property
    def transmission(self) -> float:
        return float(10.0 ** (-self.optical_density))


@dataclass(frozen=True)
class ScenarioDistribution:
    """Operational domain for Monte Carlo scenario sampling."""

    range_min_m: float = 500.0
    range_max_m: float = 3_000.0
    max_incidence_angle_rad: float = np.deg2rad(60.0)
    target_half_extents_m: tuple[float, float, float] = (0.15, 0.10, 0.0)
    uniform_solid_angle: bool = True


@dataclass(frozen=True)
class MonteCarloConfig:
    """Top-level Monte Carlo settings."""

    sample_count: int = 256
    seed: int = 0
    minimum_active_sensors: int = 6
    logdet_jitter: float = 1e-18
    invalid_score: float = -1e12


@dataclass(frozen=True)
class BeamScenario:
    """One sampled beam state."""

    origin_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    direction_unit: tuple[float, float, float]
    range_m: float
    theta_rad: float
    phi_rad: float
    effective_radius_m: float
    peak_irradiance_w_per_m2: float
    transmitted_power_w: float


@dataclass(frozen=True)
class ScenarioEvaluation:
    """Fisher scoring diagnostics for one scenario."""

    logdet_direction: float
    active_sensor_count: int
    saturated_sensor_count: int
    below_noise_sensor_count: int
    condition_number: float
    valid: bool


@dataclass(frozen=True)
class MonteCarloScore:
    """Aggregate score for one fixed layout."""

    mean_logdet: float
    std_logdet: float
    mean_valid_logdet: float
    std_valid_logdet: float
    valid_fraction: float
    mean_active_sensor_count: float
    mean_saturated_sensor_count: float
    mean_below_noise_sensor_count: float
    scenario_count: int
    evaluations: tuple[ScenarioEvaluation, ...]


@dataclass(frozen=True)
class LayoutOptimizationResult:
    """Result returned by the differential-evolution layout optimizer."""

    photodiodes: PhotodiodeArray
    score: MonteCarloScore
    objective_value: float
    optimizer_success: bool
    optimizer_message: str
    iterations: int


def beam_direction_from_angles(theta_rad: float, phi_rad: float) -> np.ndarray:
    """Convert polar angle from +z and azimuth into a unit direction vector."""
    return np.array(
        [
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad),
        ],
        dtype=float,
    )


def angles_from_beam_direction(direction: np.ndarray) -> tuple[float, float]:
    """Convert a unit direction vector into polar angle and azimuth."""
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    theta = float(np.arccos(np.clip(unit[2], -1.0, 1.0)))
    phi = float(np.arctan2(unit[1], unit[0]))
    return theta, phi


def propagate_nominal_beam(range_m: float, beam: BeamConfig) -> tuple[float, float, float]:
    """Return effective radius, transmitted power, and peak irradiance."""
    base_radius_m = calculate_spot_size(
        distance_m=range_m,
        wavelength_m=beam.wavelength_m,
        beam_waist_m=beam.beam_waist_m,
        beam_quality_m2=beam.beam_quality_m2,
    )
    turbulence_radius_m = 0.0
    if beam.cn2_m_minus_2_over_3 is not None and beam.cn2_m_minus_2_over_3 > 0.0:
        turbulence_radius_m, _ = calculate_turbulence_spread(
            distance_m=range_m,
            wavelength_m=beam.wavelength_m,
            cn2_m_minus_2_over_3=beam.cn2_m_minus_2_over_3,
        )
    effective_radius_m = float(np.sqrt(base_radius_m**2 + turbulence_radius_m**2))
    attenuation = calculate_attenuation_coefficient(
        wavelength_m=beam.wavelength_m,
        visibility_m=beam.visibility_m,
    )
    transmitted_power_w = float(beam.initial_power_w * np.exp(-attenuation * range_m))
    peak_irradiance_w_per_m2 = float(2.0 * transmitted_power_w / (np.pi * effective_radius_m**2))
    return effective_radius_m, transmitted_power_w, peak_irradiance_w_per_m2


def sample_scenarios(
    distribution: ScenarioDistribution,
    beam: BeamConfig,
    monte_carlo: MonteCarloConfig,
) -> tuple[BeamScenario, ...]:
    """Sample independent beam states from the configured operational domain."""
    if monte_carlo.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if distribution.range_min_m <= 0.0 or distribution.range_max_m <= distribution.range_min_m:
        raise ValueError("range bounds must be positive and increasing")

    rng = np.random.default_rng(monte_carlo.seed)
    half_extents = np.asarray(distribution.target_half_extents_m, dtype=float)
    scenarios: list[BeamScenario] = []
    for _ in range(monte_carlo.sample_count):
        range_m = float(rng.uniform(distribution.range_min_m, distribution.range_max_m))
        if distribution.uniform_solid_angle:
            cos_theta = float(rng.uniform(np.cos(distribution.max_incidence_angle_rad), 1.0))
            theta_rad = float(np.arccos(cos_theta))
        else:
            theta_rad = float(rng.uniform(0.0, distribution.max_incidence_angle_rad))
        phi_rad = float(rng.uniform(-np.pi, np.pi))
        direction = beam_direction_from_angles(theta_rad, phi_rad)
        target = rng.uniform(-half_extents, half_extents)
        origin = target - range_m * direction
        effective_radius_m, transmitted_power_w, peak_irradiance_w_per_m2 = propagate_nominal_beam(range_m, beam)
        scenarios.append(
            BeamScenario(
                origin_m=(float(origin[0]), float(origin[1]), float(origin[2])),
                target_m=(float(target[0]), float(target[1]), float(target[2])),
                direction_unit=(float(direction[0]), float(direction[1]), float(direction[2])),
                range_m=range_m,
                theta_rad=theta_rad,
                phi_rad=phi_rad,
                effective_radius_m=effective_radius_m,
                peak_irradiance_w_per_m2=peak_irradiance_w_per_m2,
                transmitted_power_w=transmitted_power_w,
            )
        )
    return tuple(scenarios)


def sensor_irradiance_from_parameters(
    params: np.ndarray,
    sensor_positions_m: np.ndarray,
    sensor_normals_m: np.ndarray,
    sensor_noise: SensorNoiseConfig,
    beam_point_z_m: float,
) -> np.ndarray:
    """Evaluate expected measured irradiance for the Fisher parameter vector.

    Parameter order:
        x0, y0, log_peak, log_effective_radius, theta, phi
    """
    beam_point = np.array([float(params[0]), float(params[1]), float(beam_point_z_m)], dtype=float)
    peak = float(np.exp(params[2]))
    radius = float(np.exp(params[3]))
    direction = beam_direction_from_angles(float(params[4]), float(params[5]))

    relative = sensor_positions_m - beam_point
    rho_m = np.linalg.norm(np.cross(relative, direction), axis=1)
    cos_gamma = np.einsum("ij,j->i", sensor_normals_m, direction)
    valid = cos_gamma > 0.0
    irradiance = np.zeros(sensor_positions_m.shape[0], dtype=float)
    irradiance[valid] = (
        peak
        * cos_gamma[valid]
        * np.exp((-2.0 / radius**2) * rho_m[valid] ** 2)
    )
    return sensor_noise.transmission * irradiance


def finite_difference_jacobian(
    params: np.ndarray,
    sensor_positions_m: np.ndarray,
    sensor_normals_m: np.ndarray,
    sensor_noise: SensorNoiseConfig,
    beam_point_z_m: float,
) -> np.ndarray:
    """Central finite-difference Jacobian of measured irradiance."""
    jacobian = np.empty((sensor_positions_m.shape[0], params.shape[0]), dtype=float)
    for index, value in enumerate(params):
        step = 1e-6 * max(1.0, abs(float(value)))
        forward = params.copy()
        backward = params.copy()
        forward[index] += step
        backward[index] -= step
        jacobian[:, index] = (
            sensor_irradiance_from_parameters(
                forward,
                sensor_positions_m,
                sensor_normals_m,
                sensor_noise,
                beam_point_z_m,
            )
            - sensor_irradiance_from_parameters(
                backward,
                sensor_positions_m,
                sensor_normals_m,
                sensor_noise,
                beam_point_z_m,
            )
        ) / (2.0 * step)
    return jacobian


def _sensor_sigmas(measured_irradiance_w_per_m2: np.ndarray, sensor_noise: SensorNoiseConfig) -> np.ndarray:
    floor = max(float(sensor_noise.noise_floor_w_per_m2), EPSILON)
    relative = np.maximum(float(sensor_noise.relative_noise_fraction), 0.0) * measured_irradiance_w_per_m2
    return np.sqrt(floor**2 + relative**2)


def evaluate_scenario_information(
    photodiodes: PhotodiodeArray,
    scenario: BeamScenario,
    sensor_noise: SensorNoiseConfig,
    monte_carlo: MonteCarloConfig,
) -> ScenarioEvaluation:
    """Compute Schur-complement direction information for one layout/scenario."""
    positions = np.asarray(photodiodes.positions_m, dtype=float)
    normals = np.asarray(photodiodes.normals_m, dtype=float)
    params = np.array(
        [
            scenario.target_m[0],
            scenario.target_m[1],
            np.log(max(scenario.peak_irradiance_w_per_m2, EPSILON)),
            np.log(max(scenario.effective_radius_m, EPSILON)),
            scenario.theta_rad,
            scenario.phi_rad,
        ],
        dtype=float,
    )

    measured = sensor_irradiance_from_parameters(
        params,
        positions,
        normals,
        sensor_noise,
        beam_point_z_m=scenario.target_m[2],
    )
    saturated = measured >= sensor_noise.saturation_limit_w_per_m2
    below_noise = measured < sensor_noise.noise_floor_w_per_m2
    active = np.isfinite(measured) & (measured > 0.0) & ~saturated
    if sensor_noise.require_above_noise_floor:
        active &= ~below_noise
    active_count = int(np.count_nonzero(active))
    if active_count < monte_carlo.minimum_active_sensors:
        return ScenarioEvaluation(
            logdet_direction=monte_carlo.invalid_score,
            active_sensor_count=active_count,
            saturated_sensor_count=int(np.count_nonzero(saturated)),
            below_noise_sensor_count=int(np.count_nonzero(below_noise)),
            condition_number=np.inf,
            valid=False,
        )

    jacobian = finite_difference_jacobian(
        params,
        positions,
        normals,
        sensor_noise,
        beam_point_z_m=scenario.target_m[2],
    )
    sigmas = _sensor_sigmas(measured, sensor_noise)
    weighted_jacobian = jacobian[active] / sigmas[active, None]
    fisher = weighted_jacobian.T @ weighted_jacobian

    direction_indices = [4, 5]
    nuisance_indices = [0, 1, 2, 3]
    faa = fisher[np.ix_(direction_indices, direction_indices)]
    fab = fisher[np.ix_(direction_indices, nuisance_indices)]
    fbb = fisher[np.ix_(nuisance_indices, nuisance_indices)]
    directional = faa - fab @ np.linalg.pinv(fbb) @ fab.T
    directional = 0.5 * (directional + directional.T)
    sign, logdet = np.linalg.slogdet(directional + monte_carlo.logdet_jitter * np.eye(2))
    valid = bool(sign > 0 and np.isfinite(logdet))
    condition_number = float(np.linalg.cond(directional + monte_carlo.logdet_jitter * np.eye(2)))
    return ScenarioEvaluation(
        logdet_direction=float(logdet if valid else monte_carlo.invalid_score),
        active_sensor_count=active_count,
        saturated_sensor_count=int(np.count_nonzero(saturated)),
        below_noise_sensor_count=int(np.count_nonzero(below_noise)),
        condition_number=condition_number,
        valid=valid,
    )


def score_layout(
    photodiodes: PhotodiodeArray,
    scenarios: Iterable[BeamScenario],
    sensor_noise: SensorNoiseConfig,
    monte_carlo: MonteCarloConfig,
) -> MonteCarloScore:
    """Score a fixed photodiode layout over pre-sampled scenarios."""
    evaluations = tuple(
        evaluate_scenario_information(
            photodiodes=photodiodes,
            scenario=scenario,
            sensor_noise=sensor_noise,
            monte_carlo=monte_carlo,
        )
        for scenario in scenarios
    )
    if not evaluations:
        raise ValueError("at least one scenario is required")

    logdets = np.asarray([evaluation.logdet_direction for evaluation in evaluations], dtype=float)
    valid = np.asarray([evaluation.valid for evaluation in evaluations], dtype=bool)
    active_counts = np.asarray([evaluation.active_sensor_count for evaluation in evaluations], dtype=float)
    saturated_counts = np.asarray([evaluation.saturated_sensor_count for evaluation in evaluations], dtype=float)
    below_noise_counts = np.asarray([evaluation.below_noise_sensor_count for evaluation in evaluations], dtype=float)
    valid_logdets = logdets[valid]
    mean_valid_logdet = float(np.mean(valid_logdets)) if valid_logdets.size else float("nan")
    std_valid_logdet = float(np.std(valid_logdets)) if valid_logdets.size else float("nan")

    return MonteCarloScore(
        mean_logdet=float(np.mean(logdets)),
        std_logdet=float(np.std(logdets)),
        mean_valid_logdet=mean_valid_logdet,
        std_valid_logdet=std_valid_logdet,
        valid_fraction=float(np.mean(valid)),
        mean_active_sensor_count=float(np.mean(active_counts)),
        mean_saturated_sensor_count=float(np.mean(saturated_counts)),
        mean_below_noise_sensor_count=float(np.mean(below_noise_counts)),
        scenario_count=len(evaluations),
        evaluations=evaluations,
    )


def spacing_penalty(photodiodes: PhotodiodeArray, minimum_spacing_m: float, scale: float = 1e6) -> float:
    """Quadratic penalty for layouts violating minimum sensor spacing."""
    if minimum_spacing_m <= 0.0:
        return 0.0
    coordinates = np.asarray(photodiodes.positions_m, dtype=float)
    if coordinates.shape[0] < 2:
        return 0.0
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.sqrt(np.sum(deltas**2, axis=-1) + np.eye(coordinates.shape[0]))
    upper = np.triu(np.ones_like(distances, dtype=bool), k=1)
    violations = np.maximum(minimum_spacing_m - distances[upper], 0.0)
    return float(scale * np.sum(violations**2))


def poses_from_optimization_vector(
    vector: np.ndarray,
    sensor_count: int,
    lx_m: float,
    ly_m: float,
    lz_m: float,
) -> PhotodiodeArray:
    """Decode optimizer variables into cuboid-face photodiode poses."""
    poses: list[SensorPose3D] = []
    for index in range(sensor_count):
        face_selector, u_raw, v_raw = vector[index * 3 : index * 3 + 3]
        face_index = int(np.floor(face_selector)) % 6
        u = 2.0 * (float(u_raw) - 0.5)
        v = 2.0 * (float(v_raw) - 0.5)
        poses.append(cuboid_face_pose(face_index, u, v, lx_m, ly_m, lz_m))
    return PhotodiodeArray.from_iterable(poses)


def optimize_layout(
    sensor_count: int,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    minimum_spacing_m: float,
    scenarios: Iterable[BeamScenario],
    sensor_noise: SensorNoiseConfig,
    monte_carlo: MonteCarloConfig,
    generations: int = 40,
    popsize: int = 10,
) -> LayoutOptimizationResult:
    """Optimize a cuboid photodiode layout using Monte Carlo Fisher scoring."""
    try:
        from scipy.optimize import differential_evolution
    except ImportError as exc:
        raise RuntimeError("scipy is required for layout optimization") from exc

    scenario_tuple = tuple(scenarios)
    bounds = [(0.0, 6.0), (0.0, 1.0), (0.0, 1.0)] * sensor_count

    def objective(vector: np.ndarray) -> float:
        photodiodes = poses_from_optimization_vector(vector, sensor_count, lx_m, ly_m, lz_m)
        score = score_layout(photodiodes, scenario_tuple, sensor_noise, monte_carlo)
        return -score.mean_logdet + spacing_penalty(photodiodes, minimum_spacing_m)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=generations,
        popsize=popsize,
        seed=monte_carlo.seed,
        polish=False,
        updating="deferred",
    )
    optimized = poses_from_optimization_vector(result.x, sensor_count, lx_m, ly_m, lz_m)
    score = score_layout(optimized, scenario_tuple, sensor_noise, monte_carlo)
    return LayoutOptimizationResult(
        photodiodes=optimized,
        score=score,
        objective_value=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
    )


def _print_score(score: MonteCarloScore) -> None:
    print(f"mean_logdet={score.mean_logdet:.6e}")
    print(f"std_logdet={score.std_logdet:.6e}")
    print(f"mean_valid_logdet={score.mean_valid_logdet:.6e}")
    print(f"std_valid_logdet={score.std_valid_logdet:.6e}")
    print(f"valid_fraction={score.valid_fraction:.6f}")
    print(f"mean_active_sensor_count={score.mean_active_sensor_count:.3f}")
    print(f"mean_saturated_sensor_count={score.mean_saturated_sensor_count:.3f}")
    print(f"mean_below_noise_sensor_count={score.mean_below_noise_sensor_count:.3f}")
    print(f"scenario_count={score.scenario_count}")


def load_config_file(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON config, or YAML when PyYAML is installed."""
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("YAML configs require PyYAML; use JSON or install pyyaml") from exc
            loaded = yaml.safe_load(handle)
        else:
            loaded = json.load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError("Monte Carlo config must be a JSON/YAML object")
    return loaded


def _lookup(config: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return MISSING
        current = current[key]
    return current


def _value(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    attr: str,
    default: Any,
    *paths: tuple[str, ...],
) -> Any:
    arg_value = getattr(args, attr, None)
    if arg_value is not None:
        return arg_value
    for path in paths:
        config_value = _lookup(config, path)
        if config_value is not MISSING:
            return config_value
    return default


def _required_value(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    attr: str,
    *paths: tuple[str, ...],
) -> Any:
    value = _value(args, config, attr, MISSING, *paths)
    if value is MISSING:
        names = ", ".join(".".join(path) for path in paths)
        raise ValueError(f"Missing required value for {attr}; provide CLI flag or config key: {names}")
    return value


def _float_or_inf(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(value)


def _target_half_extents(args: argparse.Namespace, config: Mapping[str, Any]) -> tuple[float, float, float]:
    configured = _lookup(config, ("distribution", "target_half_extents_m"))
    if configured is MISSING:
        configured = _lookup(config, ("target_half_extents_m",))
    if configured is not MISSING:
        values = tuple(float(value) for value in configured)
        if len(values) != 3:
            raise ValueError("target_half_extents_m must contain exactly three values")
        defaults = values
    else:
        defaults = (0.15, 0.10, 0.0)
    return (
        float(_value(args, config, "target_half_x", defaults[0], ("distribution", "target_half_x"), ("target_half_x",))),
        float(_value(args, config, "target_half_y", defaults[1], ("distribution", "target_half_y"), ("target_half_y",))),
        float(_value(args, config, "target_half_z", defaults[2], ("distribution", "target_half_z"), ("target_half_z",))),
    )


def _uniform_solid_angle(args: argparse.Namespace, config: Mapping[str, Any]) -> bool:
    if getattr(args, "theta_uniform", None) is not None:
        return not bool(args.theta_uniform)
    configured = _lookup(config, ("distribution", "uniform_solid_angle"))
    if configured is not MISSING:
        return bool(configured)
    configured = _lookup(config, ("uniform_solid_angle",))
    if configured is not MISSING:
        return bool(configured)
    configured = _lookup(config, ("distribution", "theta_uniform"))
    if configured is not MISSING:
        return not bool(configured)
    configured = _lookup(config, ("theta_uniform",))
    if configured is not MISSING:
        return not bool(configured)
    return True


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="JSON config, or YAML if PyYAML is installed")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--range-min", type=float, default=None)
    parser.add_argument("--range-max", type=float, default=None)
    parser.add_argument("--angle-max-deg", type=float, default=None)
    parser.add_argument("--target-half-x", type=float, default=None)
    parser.add_argument("--target-half-y", type=float, default=None)
    parser.add_argument("--target-half-z", type=float, default=None)
    parser.add_argument("--theta-uniform", action="store_true", default=None)
    parser.add_argument("--wavelength-nm", type=float, default=None)
    parser.add_argument("--m2", type=float, default=None)
    parser.add_argument("--beam-waist", type=float, default=None)
    parser.add_argument("--power", type=float, default=None)
    parser.add_argument("--visibility-km", type=float, default=None)
    parser.add_argument("--cn2", type=float, default=None)
    parser.add_argument("--aperture", type=float, default=None)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument("--relative-noise", type=float, default=None)
    parser.add_argument("--od", type=float, default=None)
    parser.add_argument("--saturation", default=None)
    parser.add_argument("--require-above-noise", action="store_true", default=None)
    parser.add_argument("--min-active", type=int, default=None)


def _configs_from_args(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> tuple[BeamConfig, SensorNoiseConfig, ScenarioDistribution, MonteCarloConfig]:
    cn2 = _value(args, config, "cn2", 0.0, ("beam", "cn2"), ("beam", "cn2_m_minus_2_over_3"), ("cn2",))
    beam = BeamConfig(
        wavelength_m=float(_value(args, config, "wavelength_nm", 1064.0, ("beam", "wavelength_nm"), ("wavelength_nm",))) * 1e-9,
        beam_quality_m2=float(_value(args, config, "m2", DEFAULT_BEAM_QUALITY_M2, ("beam", "m2"), ("beam", "beam_quality_m2"), ("m2",))),
        beam_waist_m=float(_value(args, config, "beam_waist", DEFAULT_BEAM_WAIST_M, ("beam", "beam_waist"), ("beam", "beam_waist_m"), ("beam_waist",))),
        initial_power_w=float(_value(args, config, "power", DEFAULT_INITIAL_POWER_W, ("beam", "power"), ("beam", "initial_power_w"), ("power",))),
        visibility_m=float(_value(args, config, "visibility_km", 20.0, ("beam", "visibility_km"), ("visibility_km",))) * 1_000.0,
        aperture_diameter_m=_value(args, config, "aperture", None, ("beam", "aperture"), ("beam", "aperture_diameter_m"), ("aperture",)),
        cn2_m_minus_2_over_3=float(cn2) if float(cn2) > 0.0 else None,
    )
    saturation = _value(args, config, "saturation", np.inf, ("sensor", "saturation"), ("sensor", "saturation_limit_w_per_m2"), ("saturation",))
    sensor_noise = SensorNoiseConfig(
        noise_floor_w_per_m2=float(_value(args, config, "noise_floor", 1.0, ("sensor", "noise_floor"), ("sensor", "noise_floor_w_per_m2"), ("noise_floor",))),
        relative_noise_fraction=float(_value(args, config, "relative_noise", 0.0, ("sensor", "relative_noise"), ("sensor", "relative_noise_fraction"), ("relative_noise",))),
        optical_density=float(_value(args, config, "od", 0.0, ("sensor", "od"), ("sensor", "optical_density"), ("od",))),
        saturation_limit_w_per_m2=_float_or_inf(saturation),
        require_above_noise_floor=bool(_value(args, config, "require_above_noise", False, ("sensor", "require_above_noise"), ("sensor", "require_above_noise_floor"), ("require_above_noise",))),
    )
    target_half_extents = _target_half_extents(args, config)
    distribution = ScenarioDistribution(
        range_min_m=float(_value(args, config, "range_min", 500.0, ("distribution", "range_min"), ("distribution", "range_min_m"), ("range_min",))),
        range_max_m=float(_value(args, config, "range_max", 3000.0, ("distribution", "range_max"), ("distribution", "range_max_m"), ("range_max",))),
        max_incidence_angle_rad=float(np.deg2rad(_value(args, config, "angle_max_deg", 60.0, ("distribution", "angle_max_deg"), ("angle_max_deg",)))),
        target_half_extents_m=target_half_extents,
        uniform_solid_angle=_uniform_solid_angle(args, config),
    )
    monte_carlo = MonteCarloConfig(
        sample_count=int(_value(args, config, "samples", 256, ("monte_carlo", "samples"), ("monte_carlo", "sample_count"), ("samples",))),
        seed=int(_value(args, config, "seed", 0, ("monte_carlo", "seed"), ("seed",))),
        minimum_active_sensors=int(_value(args, config, "min_active", 6, ("monte_carlo", "min_active"), ("monte_carlo", "minimum_active_sensors"), ("min_active",))),
    )
    return beam, sensor_noise, distribution, monte_carlo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="global_config",
        type=Path,
        default=None,
        help="JSON config, or YAML if PyYAML is installed",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score an existing saved layout")
    _common_parser(score_parser)
    score_parser.add_argument("--layouts-path", type=Path, default=None)
    score_parser.add_argument("--layout-family", type=str, default=None)
    score_parser.add_argument("--sensor-count", type=int, default=None)

    optimize_parser = subparsers.add_parser("optimize", help="optimize and optionally save a cuboid layout")
    _common_parser(optimize_parser)
    optimize_parser.add_argument("--n", type=int, default=None, help="sensor count")
    optimize_parser.add_argument("--lx", type=float, default=None, help="cuboid length in x, meters")
    optimize_parser.add_argument("--ly", type=float, default=None, help="cuboid length in y, meters")
    optimize_parser.add_argument("--lz", type=float, default=None, help="cuboid length in z, meters")
    optimize_parser.add_argument("--dmin", type=float, default=None, help="minimum spacing in meters")
    optimize_parser.add_argument("--generations", type=int, default=None)
    optimize_parser.add_argument("--popsize", type=int, default=None)
    optimize_parser.add_argument("--layouts-path", type=Path, default=None)
    optimize_parser.add_argument("--family", type=str, default=None)
    optimize_parser.add_argument("--save", action="store_true", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = getattr(args, "config", None) or getattr(args, "global_config", None)
    config = load_config_file(config_path)
    beam, sensor_noise, distribution, monte_carlo = _configs_from_args(args, config)
    scenarios = sample_scenarios(distribution, beam, monte_carlo)

    if args.command == "score":
        layout_family = str(_value(args, config, "layout_family", "default", ("score", "layout_family"), ("layout", "family"), ("layout_family",)))
        sensor_count = int(_required_value(args, config, "sensor_count", ("score", "sensor_count"), ("layout", "sensor_count"), ("sensor_count",)))
        layouts_path = Path(_value(args, config, "layouts_path", DEFAULT_LAYOUTS_PATH, ("score", "layouts_path"), ("layout", "layouts_path"), ("layouts_path",)))
        photodiodes = get_layout(layout_family, sensor_count, layouts_path)
        score = score_layout(photodiodes, scenarios, sensor_noise, monte_carlo)
        _print_score(score)
        return

    if args.command == "optimize":
        sensor_count = int(_required_value(args, config, "n", ("optimize", "n"), ("optimize", "sensor_count"), ("n",)))
        lx_m = float(_required_value(args, config, "lx", ("optimize", "lx"), ("optimize", "lx_m"), ("lx",)))
        ly_m = float(_required_value(args, config, "ly", ("optimize", "ly"), ("optimize", "ly_m"), ("ly",)))
        lz_m = float(_required_value(args, config, "lz", ("optimize", "lz"), ("optimize", "lz_m"), ("lz",)))
        minimum_spacing_m = float(_value(args, config, "dmin", 0.0, ("optimize", "dmin"), ("optimize", "minimum_spacing_m"), ("dmin",)))
        generations = int(_value(args, config, "generations", 40, ("optimize", "generations"), ("generations",)))
        popsize = int(_value(args, config, "popsize", 10, ("optimize", "popsize"), ("popsize",)))
        result = optimize_layout(
            sensor_count=sensor_count,
            lx_m=lx_m,
            ly_m=ly_m,
            lz_m=lz_m,
            minimum_spacing_m=minimum_spacing_m,
            scenarios=scenarios,
            sensor_noise=sensor_noise,
            monte_carlo=monte_carlo,
            generations=generations,
            popsize=popsize,
        )
        _print_score(result.score)
        print(f"objective_value={result.objective_value:.6e}")
        print(f"optimizer_success={result.optimizer_success}")
        print(f"optimizer_message={result.optimizer_message}")
        print(f"iterations={result.iterations}")
        for index, pose in enumerate(result.photodiodes.poses_m, start=1):
            print(
                f"sensor_{index}: face={pose.face_name} "
                f"position={tuple(round(value, 6) for value in pose.position_m)} "
                f"normal={tuple(round(value, 6) for value in pose.normal_m)}"
            )
        save = bool(_value(args, config, "save", False, ("optimize", "save"), ("save",)))
        if save:
            family = str(_value(args, config, "family", "optimized_mc", ("optimize", "family"), ("family",)))
            layouts_path = Path(_value(args, config, "layouts_path", DEFAULT_LAYOUTS_PATH, ("optimize", "layouts_path"), ("layout", "layouts_path"), ("layouts_path",)))
            catalog = load_layouts_catalog(layouts_path)
            catalog.setdefault(family, {})[sensor_count] = result.photodiodes
            save_layouts_catalog(catalog, layouts_path)
            print(f"saved_layout={family}:{sensor_count} path={layouts_path}")
        return

    raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
