"""Run single-face and multi-face Monte Carlo layout trials.

This script is intentionally backend-only. It optimizes sensor placements over
explicit 20 cm x 30 cm face domains, evaluates the resulting layouts on held-out
Monte Carlo scenarios, and writes an Excel workbook plus a concise markdown
report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from monte_carlo import (
    BeamConfig,
    MonteCarloConfig,
    MonteCarloScore,
    ScenarioDistribution,
    SensorNoiseConfig,
    score_layout,
    sample_scenarios,
    spacing_penalty,
)
from physics import SensorPose3D
from sensors import PhotodiodeArray


DEFAULT_COUNTS = (6, 8, 10, 12, 15, 20)


@dataclass(frozen=True)
class FaceSpec:
    name: str
    center_m: tuple[float, float, float]
    normal_m: tuple[float, float, float]
    u_axis_m: tuple[float, float, float]
    v_axis_m: tuple[float, float, float]
    width_u_m: float
    width_v_m: float


@dataclass(frozen=True)
class TrialResult:
    trial_name: str
    domain_name: str
    sensor_count: int
    train_score: MonteCarloScore
    eval_score: MonteCarloScore
    objective_value: float
    optimizer_success: bool
    optimizer_message: str
    iterations: int
    photodiodes: PhotodiodeArray
    minimum_spacing_m: float


def _unit(vector: tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("face axis vectors must be non-zero")
    return values / norm


def pose_on_face(face: FaceSpec, u_normalized: float, v_normalized: float) -> SensorPose3D:
    center = np.asarray(face.center_m, dtype=float)
    u_axis = _unit(face.u_axis_m)
    v_axis = _unit(face.v_axis_m)
    normal = _unit(face.normal_m)
    position = center + u_normalized * face.width_u_m * 0.5 * u_axis + v_normalized * face.width_v_m * 0.5 * v_axis
    return SensorPose3D(
        position_m=(float(position[0]), float(position[1]), float(position[2])),
        normal_m=(float(normal[0]), float(normal[1]), float(normal[2])),
        face_name=face.name,
    )


def single_face_domain(width_m: float, height_m: float) -> tuple[FaceSpec, ...]:
    return (
        FaceSpec(
            name="+z_panel",
            center_m=(0.0, 0.0, 0.0),
            normal_m=(0.0, 0.0, 1.0),
            u_axis_m=(1.0, 0.0, 0.0),
            v_axis_m=(0.0, 1.0, 0.0),
            width_u_m=width_m,
            width_v_m=height_m,
        ),
    )


def multi_face_domain(width_m: float, height_m: float, offset_m: float) -> tuple[FaceSpec, ...]:
    return (
        FaceSpec("+z_panel", (0.0, 0.0, offset_m), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), width_m, height_m),
        FaceSpec("+x_panel", (offset_m, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), width_m, height_m),
        FaceSpec("-x_panel", (-offset_m, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), width_m, height_m),
        FaceSpec("+y_panel", (0.0, offset_m, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), width_m, height_m),
        FaceSpec("-y_panel", (0.0, -offset_m, 0.0), (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), width_m, height_m),
    )


def layout_from_vector_fixed_faces(
    vector: np.ndarray,
    sensors_per_face: int,
    faces: tuple[FaceSpec, ...],
) -> PhotodiodeArray:
    """Decode optimizer variables into a fixed-per-face layout.

    This enforces exactly ``sensors_per_face`` sensors on each face in ``faces``.
    Total sensors = sensors_per_face * len(faces).

    Vector layout: (u_raw, v_raw) repeated per sensor, ordered by face then sensor index.
    """
    poses: list[SensorPose3D] = []
    expected = len(faces) * sensors_per_face * 2
    if vector.shape[0] != expected:
        raise ValueError(f"expected vector length {expected}, got {vector.shape[0]}")
    cursor = 0
    for face in faces:
        for _ in range(sensors_per_face):
            u_raw = float(vector[cursor])
            v_raw = float(vector[cursor + 1])
            cursor += 2
            poses.append(pose_on_face(face, 2.0 * (u_raw - 0.5), 2.0 * (v_raw - 0.5)))
    return PhotodiodeArray.from_iterable(poses)


def _grid_uv(count: int) -> list[tuple[float, float]]:
    side = int(np.ceil(np.sqrt(count)))
    if side <= 1:
        return [(0.5, 0.5)]
    coords = np.linspace(0.05, 0.95, side)
    uv = [(float(u), float(v)) for v in coords for u in coords]
    return uv[:count]


def _ring_uv(count: int, radii: tuple[float, ...] = (0.25, 0.45)) -> list[tuple[float, float]]:
    if count <= 1:
        return [(0.5, 0.5)]
    points: list[tuple[float, float]] = [(0.5, 0.5)]
    remaining = count - 1
    per_ring = max(4, remaining // max(len(radii), 1))
    for radius in radii:
        if remaining <= 0:
            break
        k = min(per_ring, remaining)
        angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        for angle in angles:
            points.append((0.5 + radius * float(np.cos(angle)), 0.5 + radius * float(np.sin(angle))))
        remaining = count - len(points)
    while len(points) < count:
        angle = 2.0 * np.pi * (len(points) / max(1, count))
        points.append((0.5 + 0.45 * float(np.cos(angle)), 0.5 + 0.45 * float(np.sin(angle))))
    return points[:count]


def _rect_ring_uv(count: int) -> list[tuple[float, float]]:
    if count <= 1:
        return [(0.5, 0.5)]
    perimeter = []
    steps = max(4, int(np.ceil(count / 4)))
    t = np.linspace(0.1, 0.9, steps)
    for u in t:
        perimeter.append((float(u), 0.1))
    for v in t[1:]:
        perimeter.append((0.9, float(v)))
    for u in t[-2::-1]:
        perimeter.append((float(u), 0.9))
    for v in t[-2:0:-1]:
        perimeter.append((0.1, float(v)))
    points = [(0.5, 0.5)] + perimeter
    return points[:count]


def heuristic_layout_fixed_faces(
    sensors_per_face: int,
    faces: tuple[FaceSpec, ...],
    pattern: str,
    seed: int,
) -> PhotodiodeArray:
    rng = np.random.default_rng(seed)
    poses: list[SensorPose3D] = []
    for face_index, face in enumerate(faces):
        if pattern == "grid":
            uv = _grid_uv(sensors_per_face)
        elif pattern == "rings":
            uv = _ring_uv(sensors_per_face)
        elif pattern == "rect":
            uv = _rect_ring_uv(sensors_per_face)
        elif pattern == "random":
            uv = [(float(rng.uniform(0.05, 0.95)), float(rng.uniform(0.05, 0.95))) for _ in range(sensors_per_face)]
        else:
            raise ValueError(f"unknown pattern {pattern!r}")
        for u_raw, v_raw in uv:
            poses.append(pose_on_face(face, 2.0 * (u_raw - 0.5), 2.0 * (v_raw - 0.5)))
    return PhotodiodeArray.from_iterable(poses)


def layout_from_vector(vector: np.ndarray, sensor_count: int, faces: tuple[FaceSpec, ...]) -> PhotodiodeArray:
    poses: list[SensorPose3D] = []
    for index in range(sensor_count):
        if len(faces) == 1:
            u_raw, v_raw = vector[index * 2 : index * 2 + 2]
            face = faces[0]
        else:
            face_raw, u_raw, v_raw = vector[index * 3 : index * 3 + 3]
            face = faces[int(np.floor(face_raw)) % len(faces)]
        poses.append(pose_on_face(face, 2.0 * (float(u_raw) - 0.5), 2.0 * (float(v_raw) - 0.5)))
    return PhotodiodeArray.from_iterable(poses)


def optimize_face_layout(
    trial_name: str,
    domain_name: str,
    sensor_count: int,
    faces: tuple[FaceSpec, ...],
    train_scenarios: tuple,
    eval_scenarios: tuple,
    sensor_noise: SensorNoiseConfig,
    train_mc: MonteCarloConfig,
    eval_mc: MonteCarloConfig,
    minimum_spacing_m: float,
    generations: int,
    popsize: int,
    seed: int,
) -> TrialResult:
    bounds = [(0.0, 1.0), (0.0, 1.0)] * sensor_count
    if len(faces) > 1:
        bounds = [(0.0, float(len(faces))), (0.0, 1.0), (0.0, 1.0)] * sensor_count

    def objective(vector: np.ndarray) -> float:
        photodiodes = layout_from_vector(vector, sensor_count, faces)
        score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
        return -score.mean_logdet + spacing_penalty(photodiodes, minimum_spacing_m, scale=1e8)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=generations,
        popsize=popsize,
        seed=seed,
        polish=False,
        updating="deferred",
        workers=1,
    )
    photodiodes = layout_from_vector(result.x, sensor_count, faces)
    train_score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
    eval_score = score_layout(photodiodes, eval_scenarios, sensor_noise, eval_mc)
    return TrialResult(
        trial_name=trial_name,
        domain_name=domain_name,
        sensor_count=sensor_count,
        train_score=train_score,
        eval_score=eval_score,
        objective_value=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        photodiodes=photodiodes,
        minimum_spacing_m=minimum_spacing_m,
    )


def optimize_multi_face_per_face_layout(
    trial_name: str,
    sensors_per_face: int,
    faces: tuple[FaceSpec, ...],
    train_scenarios: tuple,
    eval_scenarios: tuple,
    sensor_noise: SensorNoiseConfig,
    train_mc: MonteCarloConfig,
    eval_mc: MonteCarloConfig,
    minimum_spacing_m: float,
    generations: int,
    popsize: int,
    seed: int,
) -> TrialResult:
    total_sensors = sensors_per_face * len(faces)
    bounds = [(0.0, 1.0), (0.0, 1.0)] * total_sensors

    def objective(vector: np.ndarray) -> float:
        photodiodes = layout_from_vector_fixed_faces(vector, sensors_per_face, faces)
        score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
        return -score.mean_logdet + spacing_penalty(photodiodes, minimum_spacing_m, scale=1e8)

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=generations,
        popsize=popsize,
        seed=seed,
        polish=False,
        updating="deferred",
        workers=1,
    )
    photodiodes = layout_from_vector_fixed_faces(result.x, sensors_per_face, faces)
    train_score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
    eval_score = score_layout(photodiodes, eval_scenarios, sensor_noise, eval_mc)
    return TrialResult(
        trial_name=trial_name,
        domain_name="multi_face_per_face",
        sensor_count=total_sensors,
        train_score=train_score,
        eval_score=eval_score,
        objective_value=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        iterations=int(result.nit),
        photodiodes=photodiodes,
        minimum_spacing_m=minimum_spacing_m,
    )


def score_to_record(prefix: str, score: MonteCarloScore) -> dict[str, float | int]:
    return {
        f"{prefix}_mean_logdet": score.mean_logdet,
        f"{prefix}_std_logdet": score.std_logdet,
        f"{prefix}_mean_valid_logdet": score.mean_valid_logdet,
        f"{prefix}_std_valid_logdet": score.std_valid_logdet,
        f"{prefix}_valid_fraction": score.valid_fraction,
        f"{prefix}_mean_active_sensors": score.mean_active_sensor_count,
        f"{prefix}_mean_saturated_sensors": score.mean_saturated_sensor_count,
        f"{prefix}_mean_below_noise_sensors": score.mean_below_noise_sensor_count,
        f"{prefix}_scenario_count": score.scenario_count,
    }


def result_summary_record(result: TrialResult) -> dict[str, object]:
    record: dict[str, object] = {
        "trial": result.trial_name,
        "domain": result.domain_name,
        "sensor_count": result.sensor_count,
        "minimum_spacing_m": result.minimum_spacing_m,
        "objective_value": result.objective_value,
        "optimizer_success": result.optimizer_success,
        "optimizer_message": result.optimizer_message,
        "iterations": result.iterations,
    }
    record.update(score_to_record("train", result.train_score))
    record.update(score_to_record("eval", result.eval_score))
    return record


def sensor_records(result: TrialResult) -> list[dict[str, object]]:
    records = []
    for index, pose in enumerate(result.photodiodes.poses_m, start=1):
        records.append(
            {
                "sensor": index,
                "face": pose.face_name,
                "x_m": pose.position_m[0],
                "y_m": pose.position_m[1],
                "z_m": pose.position_m[2],
                "normal_x": pose.normal_m[0],
                "normal_y": pose.normal_m[1],
                "normal_z": pose.normal_m[2],
            }
        )
    return records


def evaluation_records(result: TrialResult, score: MonteCarloScore, split: str) -> list[dict[str, object]]:
    return [
        {
            "split": split,
            "scenario": index,
            "logdet_direction": evaluation.logdet_direction,
            "active_sensor_count": evaluation.active_sensor_count,
            "saturated_sensor_count": evaluation.saturated_sensor_count,
            "below_noise_sensor_count": evaluation.below_noise_sensor_count,
            "condition_number": evaluation.condition_number,
            "valid": evaluation.valid,
        }
        for index, evaluation in enumerate(score.evaluations, start=1)
    ]


def write_excel(results: list[TrialResult], output_path: Path) -> None:
    summary = pd.DataFrame([result_summary_record(result) for result in results])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        for result in results:
            sheet = result.trial_name[:31]
            rows = []
            rows.extend({"section": "summary", **result_summary_record(result)} for _ in range(1))
            sensors = pd.DataFrame(sensor_records(result))
            evals = pd.DataFrame(
                evaluation_records(result, result.train_score, "train")
                + evaluation_records(result, result.eval_score, "eval")
            )
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False, startrow=0)
            sensors.to_excel(writer, sheet_name=sheet, index=False, startrow=4)
            evals.to_excel(writer, sheet_name=sheet, index=False, startrow=8 + len(sensors))


def write_markdown(
    results: list[TrialResult],
    output_path: Path,
    args: argparse.Namespace,
    beam: BeamConfig,
    distribution: ScenarioDistribution,
    sensor_noise: SensorNoiseConfig,
) -> None:
    best_by_domain: dict[str, TrialResult] = {}
    for result in results:
        current = best_by_domain.get(result.domain_name)
        if current is None or result.eval_score.mean_logdet > current.eval_score.mean_logdet:
            best_by_domain[result.domain_name] = result

    lines = [
        "# Monte Carlo Face Layout Trials",
        "",
        "## Setup",
        f"- Domains: single 20x30 cm +z panel; multi-face 20x30 cm panel set (+z, +/-x, +/-y).",
        f"- Sensor counts: {', '.join(str(count) for count in args.counts)}; maximum tested: {max(args.counts)}.",
        f"- Minimum spacing: {args.minimum_spacing_m:.3f} m.",
        f"- Train/eval scenarios: {args.train_samples}/{args.eval_samples}; seeds: {args.seed}/{args.seed + args.eval_seed_offset}.",
        f"- Range: {distribution.range_min_m:.0f}-{distribution.range_max_m:.0f} m; max incidence: {np.rad2deg(distribution.max_incidence_angle_rad):.1f} deg.",
        f"- Beam: wavelength={beam.wavelength_m * 1e9:.1f} nm, M2={beam.beam_quality_m2:.3g}, waist={beam.beam_waist_m:.3g} m, power={beam.initial_power_w:.3g} W, visibility={beam.visibility_m / 1000:.3g} km.",
        f"- Sensor model: noise floor={sensor_noise.noise_floor_w_per_m2:.3g} W/m2, relative noise={sensor_noise.relative_noise_fraction:.3g}, OD={sensor_noise.optical_density:.3g}, saturation={sensor_noise.saturation_limit_w_per_m2:.3g} W/m2.",
        "",
        "## Math",
        "- Each candidate layout is scored by Monte Carlo expectation of log det(F_direction).",
        "- Sensor mean: I_i = I_peak cos(gamma_i) exp(-2 rho_i^2 / w_eff^2), with OD transmission applied.",
        "- Fisher matrix: F = J^T Sigma^-1 J using finite-difference derivatives of irradiance with respect to x0, y0, log(I_peak), log(w_eff), theta, phi.",
        "- Direction information is the Schur complement over theta/phi after marginalizing x0, y0, peak, and width.",
        "- Invalid scenarios receive the configured invalid score when fewer than the required active sensors remain.",
        "",
        "## Results",
        "| domain | N | eval mean logdet | eval valid frac | eval active sensors | below noise | optimizer status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.domain_name} | {result.sensor_count} | {result.eval_score.mean_logdet:.3e} | "
            f"{result.eval_score.valid_fraction:.3f} | {result.eval_score.mean_active_sensor_count:.2f} | "
            f"{result.eval_score.mean_below_noise_sensor_count:.2f} | {result.optimizer_message} |"
        )
    lines.extend(["", "## Best Held-Out Layouts"])
    for domain, result in best_by_domain.items():
        lines.append(
            f"- {domain}: N={result.sensor_count}, eval mean logdet={result.eval_score.mean_logdet:.3e}, "
            f"valid fraction={result.eval_score.valid_fraction:.3f}, sheet={result.trial_name}."
        )
    lines.extend(
        [
            "",
            "## Assumptions",
            "- The multi-face domain uses five separate 20x30 cm panels: one top-facing panel and four side-facing panels. It is a panel-set approximation, not a closed cuboid with physically matching edge dimensions.",
            "- Optimization uses a train scenario set and reports held-out evaluation results from a separate scenario set.",
            "- These runs optimize information geometry, not reconstruction error from a full estimator.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not counts:
        raise ValueError("at least one sensor count is required")
    if min(counts) < 1 or max(counts) > 20:
        raise ValueError("sensor counts must be between 1 and 20")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("monte_carlo_results"))
    parser.add_argument("--counts", type=parse_counts, default=DEFAULT_COUNTS)
    parser.add_argument("--width-m", type=float, default=0.30)
    parser.add_argument("--height-m", type=float, default=0.20)
    parser.add_argument("--multi-face-offset-m", type=float, default=0.10)
    parser.add_argument("--minimum-spacing-m", type=float, default=0.01)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-seed-offset", type=int, default=10_000)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=4)
    parser.add_argument("--range-min-m", type=float, default=500.0)
    parser.add_argument("--range-max-m", type=float, default=3000.0)
    parser.add_argument("--angle-max-deg", type=float, default=60.0)
    parser.add_argument("--wavelength-nm", type=float, default=1064.0)
    parser.add_argument("--m2", type=float, default=1.2)
    parser.add_argument("--beam-waist-m", type=float, default=0.05)
    parser.add_argument("--power-w", type=float, default=10_000.0)
    parser.add_argument("--visibility-km", type=float, default=20.0)
    parser.add_argument("--cn2", type=float, default=0.0)
    parser.add_argument("--noise-floor", type=float, default=1.0)
    parser.add_argument("--relative-noise", type=float, default=0.0)
    parser.add_argument("--od", type=float, default=0.0)
    parser.add_argument("--saturation", type=float, default=np.inf)
    parser.add_argument("--min-active", type=int, default=6)
    parser.add_argument("--multi-face-per-face", action="store_true", help="Interpret --counts as sensors per face for multi-face trials.")
    parser.add_argument("--patterns", type=str, default="grid,rings,rect,random", help="Comma-separated heuristic patterns to score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    beam = BeamConfig(
        wavelength_m=args.wavelength_nm * 1e-9,
        beam_quality_m2=args.m2,
        beam_waist_m=args.beam_waist_m,
        initial_power_w=args.power_w,
        visibility_m=args.visibility_km * 1000.0,
        cn2_m_minus_2_over_3=args.cn2 if args.cn2 > 0.0 else None,
    )
    distribution = ScenarioDistribution(
        range_min_m=args.range_min_m,
        range_max_m=args.range_max_m,
        max_incidence_angle_rad=np.deg2rad(args.angle_max_deg),
        target_half_extents_m=(args.width_m / 2.0, args.height_m / 2.0, 0.0),
        uniform_solid_angle=True,
    )
    sensor_noise = SensorNoiseConfig(
        noise_floor_w_per_m2=args.noise_floor,
        relative_noise_fraction=args.relative_noise,
        optical_density=args.od,
        saturation_limit_w_per_m2=args.saturation,
    )
    train_mc = MonteCarloConfig(sample_count=args.train_samples, seed=args.seed, minimum_active_sensors=args.min_active)
    eval_mc = MonteCarloConfig(sample_count=args.eval_samples, seed=args.seed + args.eval_seed_offset, minimum_active_sensors=args.min_active)
    train_scenarios = sample_scenarios(distribution, beam, train_mc)
    eval_scenarios = sample_scenarios(distribution, beam, eval_mc)
    domains = {
        "single_face": single_face_domain(args.width_m, args.height_m),
        "multi_face": multi_face_domain(args.width_m, args.height_m, args.multi_face_offset_m),
    }

    results: list[TrialResult] = []
    for domain_name, faces in domains.items():
        if domain_name == "multi_face" and args.multi_face_per_face:
            for per_face in args.counts:
                trial_name = f"multi_face_per_face_n{per_face:02d}"
                print(f"running {trial_name}")
                results.append(
                    optimize_multi_face_per_face_layout(
                        trial_name=trial_name,
                        sensors_per_face=int(per_face),
                        faces=faces,
                        train_scenarios=train_scenarios,
                        eval_scenarios=eval_scenarios,
                        sensor_noise=sensor_noise,
                        train_mc=train_mc,
                        eval_mc=eval_mc,
                        minimum_spacing_m=args.minimum_spacing_m,
                        generations=args.generations,
                        popsize=args.popsize,
                        seed=args.seed + len(results),
                    )
                )
                # Heuristic baselines for the same per-face count.
                patterns = [part.strip() for part in str(args.patterns).split(",") if part.strip()]
                for pattern in patterns:
                    layout = heuristic_layout_fixed_faces(int(per_face), faces, pattern, seed=args.seed + 1000 + len(results))
                    train_score = score_layout(layout, train_scenarios, sensor_noise, train_mc)
                    eval_score = score_layout(layout, eval_scenarios, sensor_noise, eval_mc)
                    results.append(
                        TrialResult(
                            trial_name=f"{trial_name}_{pattern}",
                            domain_name="multi_face_per_face",
                            sensor_count=int(per_face) * len(faces),
                            train_score=train_score,
                            eval_score=eval_score,
                            objective_value=float("nan"),
                            optimizer_success=True,
                            optimizer_message=f"heuristic:{pattern}",
                            iterations=0,
                            photodiodes=layout,
                            minimum_spacing_m=args.minimum_spacing_m,
                        )
                    )
            continue
        for sensor_count in args.counts:
            trial_name = f"{domain_name}_n{sensor_count:02d}"
            print(f"running {trial_name}")
            results.append(
                optimize_face_layout(
                    trial_name=trial_name,
                    domain_name=domain_name,
                    sensor_count=sensor_count,
                    faces=faces,
                    train_scenarios=train_scenarios,
                    eval_scenarios=eval_scenarios,
                    sensor_noise=sensor_noise,
                    train_mc=train_mc,
                    eval_mc=eval_mc,
                    minimum_spacing_m=args.minimum_spacing_m,
                    generations=args.generations,
                    popsize=args.popsize,
                    seed=args.seed + len(results),
                )
            )

    workbook_path = args.output_dir / "monte_carlo_face_trials.xlsx"
    readme_path = args.output_dir / "README.md"
    write_excel(results, workbook_path)
    write_markdown(results, readme_path, args, beam, distribution, sensor_noise)
    print(f"wrote {workbook_path}")
    print(f"wrote {readme_path}")


if __name__ == "__main__":
    main()
