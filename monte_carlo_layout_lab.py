"""Streamlit app for comparing sensor arrangements with Monte Carlo scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

from monte_carlo import (
    BeamConfig,
    MonteCarloConfig,
    ScenarioDistribution,
    SensorNoiseConfig,
    sample_scenarios,
    score_layout,
    sensor_irradiance_from_parameters,
    spacing_penalty,
)
from reconstruction import fit_log_quadratic_ellipse
from run_face_monte_carlo import (
    TrialResult,
    heuristic_layout_fixed_faces,
    multi_face_domain,
    optimize_face_layout,
    optimize_multi_face_per_face_layout,
    single_face_domain,
)
from sensors import PhotodiodeArray


PATTERN_LABELS = {
    "grid": "Grid",
    "rings": "Rings",
    "rect": "Rect Ring",
    "random": "Random",
    "optimized": "Optimized",
}

OPTIMIZATION_TARGET_LABELS = {
    "logdet": "Log-det score",
    "validity": "Valid fraction",
    "error": "Valid gamma error",
}

DEFAULT_APERTURE_DIAMETER_M = 0.10
DEFAULT_M2 = 1.5
DEFAULT_BEAM_WAIST_M = DEFAULT_APERTURE_DIAMETER_M / (np.pi * DEFAULT_M2)
DEFAULT_POWER_W = 5_000.0
DEFAULT_VISIBILITY_KM = 10.0
DEFAULT_CN2 = 1e-14
DEFAULT_COUNTS_TEXT = "6,8,10,12"
DEFAULT_PATTERNS = ("grid", "rings", "rect", "random")
LAST_RUN_STATE_PATH = Path(__file__).with_name("monte_carlo_layout_lab_last_run.json")
LAST_RUN_RESULTS_PATH = Path(__file__).with_name("monte_carlo_layout_lab_last_results.json")


@dataclass(frozen=True)
class FaceGeometry:
    name: str
    center_m: tuple[float, float, float]
    u_axis_m: tuple[float, float, float]
    v_axis_m: tuple[float, float, float]
    width_u_m: float
    width_v_m: float


METRIC_SPECS = (
    {
        "column": "eval_mean_logdet",
        "delta_column": "mean_score_delta",
        "ci_low_column": "score_delta_ci_low",
        "ci_high_column": "score_delta_ci_high",
        "improves_fraction_column": "score_improves_fraction",
        "p_value_column": "score_p_value",
        "significant_column": "score_significant_at_0_05",
        "higher_is_better": True,
    },
    {
        "column": "eval_mean_valid_logdet",
        "delta_column": "mean_valid_delta",
        "ci_low_column": "mean_valid_ci_low",
        "ci_high_column": "mean_valid_ci_high",
        "improves_fraction_column": "mean_valid_improves_fraction",
        "p_value_column": "mean_valid_p_value",
        "significant_column": "mean_valid_significant_at_0_05",
        "higher_is_better": True,
    },
    {
        "column": "eval_any_active_fraction",
        "delta_column": "mean_any_active_fraction_delta",
        "ci_low_column": "any_active_ci_low",
        "ci_high_column": "any_active_ci_high",
        "improves_fraction_column": "any_active_improves_fraction",
        "p_value_column": "any_active_p_value",
        "significant_column": "any_active_significant_at_0_05",
        "higher_is_better": True,
    },
    {
        "column": "eval_mean_gamma_error_deg",
        "delta_column": "mean_error_reduction_deg",
        "ci_low_column": "mean_error_reduction_ci_low",
        "ci_high_column": "mean_error_reduction_ci_high",
        "improves_fraction_column": "mean_error_improves_fraction",
        "p_value_column": "mean_error_p_value",
        "significant_column": "mean_error_significant_at_0_05",
        "higher_is_better": False,
    },
)


def parse_counts(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Enter at least one sensor count.")
    if min(values) < 1:
        raise ValueError("Sensor counts must be positive integers.")
    return values


def default_experiment_settings() -> dict[str, object]:
    return {
        "domain_mode": "single_face",
        "counts_text": DEFAULT_COUNTS_TEXT,
        "patterns": list(DEFAULT_PATTERNS),
        "include_optimized": True,
        "optimization_target": "logdet",
        "width_m": 0.30,
        "height_m": 0.20,
        "multi_face_offset_m": 0.10,
        "minimum_spacing_m": 0.01,
        "train_samples": 64,
        "eval_samples": 256,
        "min_active": 6,
        "seed": 7,
        "eval_seed_offset": 10000,
        "generations": 8,
        "popsize": 4,
        "range_min_m": 500.0,
        "range_max_m": 3000.0,
        "angle_max_deg": 60,
        "wavelength_nm": 1064.0,
        "m2": DEFAULT_M2,
        "beam_waist_m": DEFAULT_BEAM_WAIST_M,
        "power_w": DEFAULT_POWER_W,
        "visibility_km": DEFAULT_VISIBILITY_KM,
        "cn2": DEFAULT_CN2,
        "uniform_solid_angle": True,
        "noise_floor": 1.0,
        "relative_noise": 0.0,
        "optical_density": 0.0,
        "saturation_limit": 1e12,
        "require_above_noise": False,
    }


def default_layout_lab_state() -> dict[str, object]:
    return {
        "experiment_settings": default_experiment_settings(),
        "enable_multi_seed": False,
        "significance_trials": 10,
        "seed_step": 1,
        "reuse_optimized_layouts_across_seeds": False,
    }


def load_layout_lab_state() -> dict[str, object]:
    state = default_layout_lab_state()
    if not LAST_RUN_STATE_PATH.exists():
        return state
    try:
        payload = json.loads(LAST_RUN_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state
    if not isinstance(payload, dict):
        return state

    raw_settings = payload.get("experiment_settings", payload.get("settings", {}))
    if isinstance(raw_settings, dict):
        state["experiment_settings"].update(raw_settings)
    for key in ("enable_multi_seed", "significance_trials", "seed_step", "reuse_optimized_layouts_across_seeds"):
        if key in payload:
            state[key] = payload[key]
    return state


def save_layout_lab_state(
    experiment_settings: dict[str, object],
    enable_multi_seed: bool,
    significance_trials: int,
    seed_step: int,
    reuse_optimized_layouts_across_seeds: bool,
) -> None:
    payload = {
        "experiment_settings": experiment_settings,
        "enable_multi_seed": bool(enable_multi_seed),
        "significance_trials": int(significance_trials),
        "seed_step": int(seed_step),
        "reuse_optimized_layouts_across_seeds": bool(reuse_optimized_layouts_across_seeds),
    }
    LAST_RUN_STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _serialize_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    return frame.to_dict(orient="records")


def _deserialize_frame(records: object) -> pd.DataFrame:
    if not isinstance(records, list):
        return pd.DataFrame()
    return pd.DataFrame(records)


def serialize_experiment_bundle(bundle: dict[str, object]) -> dict[str, object]:
    return {
        "summary_records": _serialize_frame(bundle["summary_df"]),
        "winner_records": _serialize_frame(bundle["winner_df"]),
        "trials": bundle["trials"],
        "counts": list(bundle["counts"]),
        "settings": bundle["settings"],
    }


def deserialize_experiment_bundle(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return None
    summary_df = _deserialize_frame(payload.get("summary_records"))
    winner_df = _deserialize_frame(payload.get("winner_records"))
    counts = tuple(int(value) for value in payload.get("counts", []))
    settings = payload.get("settings", {})
    if not isinstance(settings, dict):
        settings = {}
    return {
        "summary_df": summary_df,
        "winner_df": winner_df,
        "trial_lookup": {str(trial["trial_key"]): trial for trial in trials},
        "trials": trials,
        "counts": counts,
        "settings": settings,
    }


def serialize_multi_seed_bundle(bundle: dict[str, object]) -> dict[str, object]:
    return {
        "counts": list(bundle["counts"]),
        "seed_values": list(bundle["seed_values"]),
        "reuse_optimized_layouts_across_seeds": bool(bundle.get("reuse_optimized_layouts_across_seeds", False)),
        "optimized_reference_seed": bundle.get("optimized_reference_seed"),
        "per_seed_records": _serialize_frame(bundle["per_seed_df"]),
        "aggregate_records": _serialize_frame(bundle["aggregate_df"]),
        "stability_records": _serialize_frame(bundle["stability_df"]),
        "significance_records": _serialize_frame(bundle["significance_df"]),
        "best_progression_records": _serialize_frame(bundle["best_progression_df"]),
        "method_progression_records": _serialize_frame(bundle["method_progression_df"]),
    }


def deserialize_multi_seed_bundle(payload: object) -> dict[str, object] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "counts": tuple(int(value) for value in payload.get("counts", [])),
        "seed_values": [int(value) for value in payload.get("seed_values", [])],
        "reuse_optimized_layouts_across_seeds": bool(payload.get("reuse_optimized_layouts_across_seeds", False)),
        "optimized_reference_seed": (
            int(payload["optimized_reference_seed"])
            if payload.get("optimized_reference_seed") is not None
            else None
        ),
        "per_seed_df": _deserialize_frame(payload.get("per_seed_records")),
        "aggregate_df": _deserialize_frame(payload.get("aggregate_records")),
        "stability_df": _deserialize_frame(payload.get("stability_records")),
        "significance_df": _deserialize_frame(payload.get("significance_records")),
        "best_progression_df": _deserialize_frame(payload.get("best_progression_records")),
        "method_progression_df": _deserialize_frame(payload.get("method_progression_records")),
    }


def save_layout_lab_results(
    experiment_bundle: dict[str, object] | None,
    multi_seed_bundle: dict[str, object] | None,
) -> None:
    payload = {
        "experiment_bundle": serialize_experiment_bundle(experiment_bundle) if experiment_bundle is not None else None,
        "multi_seed_bundle": serialize_multi_seed_bundle(multi_seed_bundle) if multi_seed_bundle is not None else None,
    }
    LAST_RUN_RESULTS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_layout_lab_results() -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if not LAST_RUN_RESULTS_PATH.exists():
        return None, None
    try:
        payload = json.loads(LAST_RUN_RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    return (
        deserialize_experiment_bundle(payload.get("experiment_bundle")),
        deserialize_multi_seed_bundle(payload.get("multi_seed_bundle")),
    )


def method_sort_key(method: str) -> int:
    order = ["Optimized", "Rings", "Grid", "Rect Ring", "Random"]
    try:
        return order.index(method)
    except ValueError:
        return len(order)


def unit(vector: tuple[float, float, float]) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise ValueError("face axis vectors must be non-zero")
    return values / norm


def normalize_face_geometries(faces: tuple) -> dict[str, FaceGeometry]:
    return {
        face.name: FaceGeometry(
            name=face.name,
            center_m=tuple(float(value) for value in face.center_m),
            u_axis_m=tuple(float(value) for value in face.u_axis_m),
            v_axis_m=tuple(float(value) for value in face.v_axis_m),
            width_u_m=float(face.width_u_m),
            width_v_m=float(face.width_v_m),
        )
        for face in faces
    }


def serialize_layout(photodiodes: PhotodiodeArray, face_map: dict[str, FaceGeometry]) -> list[dict[str, float | int | str]]:
    records: list[dict[str, float | int | str]] = []
    for sensor_index, pose in enumerate(photodiodes.poses_m, start=1):
        position = np.asarray(pose.position_m, dtype=float)
        face = face_map.get(pose.face_name)
        if face is not None:
            delta = position - np.asarray(face.center_m, dtype=float)
            local_u_m = float(delta @ unit(face.u_axis_m))
            local_v_m = float(delta @ unit(face.v_axis_m))
            width_u_m = face.width_u_m
            width_v_m = face.width_v_m
        else:
            local_u_m = float(position[0])
            local_v_m = float(position[1])
            width_u_m = 0.0
            width_v_m = 0.0
        records.append(
            {
                "sensor": sensor_index,
                "face": str(pose.face_name),
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
                "normal_x": float(pose.normal_m[0]),
                "normal_y": float(pose.normal_m[1]),
                "normal_z": float(pose.normal_m[2]),
                "local_u_m": local_u_m,
                "local_v_m": local_v_m,
                "local_u_cm": local_u_m * 100.0,
                "local_v_cm": local_v_m * 100.0,
                "face_width_cm": width_u_m * 100.0,
                "face_height_cm": width_v_m * 100.0,
            }
        )
    return records


def build_trial_result(
    trial_name: str,
    domain_name: str,
    sensor_count: int,
    minimum_spacing_m: float,
    method_key: str,
    photodiodes: PhotodiodeArray,
    train_score,
    eval_score,
) -> TrialResult:
    return TrialResult(
        trial_name=trial_name,
        domain_name=domain_name,
        sensor_count=sensor_count,
        train_score=train_score,
        eval_score=eval_score,
        objective_value=float("nan"),
        optimizer_success=True,
        optimizer_message=f"heuristic:{method_key}",
        iterations=0,
        photodiodes=photodiodes,
        minimum_spacing_m=minimum_spacing_m,
    )


def evaluate_gamma_error_metrics(
    photodiodes: PhotodiodeArray,
    scenarios: tuple,
    sensor_noise: SensorNoiseConfig,
    monte_carlo: MonteCarloConfig,
    domain_name: str,
) -> dict[str, float]:
    if domain_name != "single_face":
        return {
            "mean_gamma_error_deg": float("nan"),
            "gamma_error_valid_fraction": float("nan"),
        }

    positions = np.asarray(photodiodes.positions_m, dtype=float)
    normals = np.asarray(photodiodes.normals_m, dtype=float)
    coordinates = positions[:, :2]
    errors_deg: list[float] = []
    reconstruction_count = 0
    minimum_required = max(6, int(monte_carlo.minimum_active_sensors))
    sigma_noise = max(float(sensor_noise.noise_floor_w_per_m2), 1e-12)

    for scenario in scenarios:
        params = np.array(
            [
                scenario.target_m[0],
                scenario.target_m[1],
                np.log(max(scenario.peak_irradiance_w_per_m2, 1e-12)),
                np.log(max(scenario.effective_radius_m, 1e-12)),
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
        if int(np.count_nonzero(active)) < minimum_required:
            continue
        try:
            ellipse = fit_log_quadratic_ellipse(
                coordinates[active],
                measured[active],
                sigma_noise_w_per_m2=sigma_noise,
            )
        except ValueError:
            continue
        actual_gamma_deg = float(np.rad2deg(np.arccos(np.clip(float(scenario.direction_unit[2]), -1.0, 1.0))))
        estimated_gamma_deg = float(np.rad2deg(ellipse.incidence_angle_rad))
        errors_deg.append(abs(estimated_gamma_deg - actual_gamma_deg))
        reconstruction_count += 1

    scenario_total = len(scenarios)
    return {
        "mean_gamma_error_deg": float(np.mean(errors_deg)) if errors_deg else float("nan"),
        "gamma_error_valid_fraction": float(reconstruction_count / scenario_total) if scenario_total else float("nan"),
    }


def build_experiment_context(settings: dict[str, object]) -> dict[str, object]:
    counts = parse_counts(str(settings["counts_text"]))
    beam = BeamConfig(
        wavelength_m=float(settings["wavelength_nm"]) * 1e-9,
        beam_quality_m2=float(settings["m2"]),
        beam_waist_m=float(settings["beam_waist_m"]),
        initial_power_w=float(settings["power_w"]),
        visibility_m=float(settings["visibility_km"]) * 1000.0,
        cn2_m_minus_2_over_3=float(settings["cn2"]) if float(settings["cn2"]) > 0.0 else None,
    )
    distribution = ScenarioDistribution(
        range_min_m=float(settings["range_min_m"]),
        range_max_m=float(settings["range_max_m"]),
        max_incidence_angle_rad=float(np.deg2rad(float(settings["angle_max_deg"]))),
        target_half_extents_m=(float(settings["width_m"]) / 2.0, float(settings["height_m"]) / 2.0, 0.0),
        uniform_solid_angle=bool(settings["uniform_solid_angle"]),
    )
    sensor_noise = SensorNoiseConfig(
        noise_floor_w_per_m2=float(settings["noise_floor"]),
        relative_noise_fraction=float(settings["relative_noise"]),
        optical_density=float(settings["optical_density"]),
        saturation_limit_w_per_m2=float(settings["saturation_limit"]),
        require_above_noise_floor=bool(settings["require_above_noise"]),
    )
    train_mc = MonteCarloConfig(
        sample_count=int(settings["train_samples"]),
        seed=int(settings["seed"]),
        minimum_active_sensors=int(settings["min_active"]),
    )
    eval_mc = MonteCarloConfig(
        sample_count=int(settings["eval_samples"]),
        seed=int(settings["seed"]) + int(settings["eval_seed_offset"]),
        minimum_active_sensors=int(settings["min_active"]),
    )
    train_scenarios = sample_scenarios(distribution, beam, train_mc)
    eval_scenarios = sample_scenarios(distribution, beam, eval_mc)

    if str(settings["domain_mode"]) == "single_face":
        domain_name = "single_face"
        faces = single_face_domain(float(settings["width_m"]), float(settings["height_m"]))
    else:
        domain_name = "multi_face_per_face"
        faces = multi_face_domain(
            float(settings["width_m"]),
            float(settings["height_m"]),
            float(settings["multi_face_offset_m"]),
        )
    face_map = normalize_face_geometries(faces)
    return {
        "counts": counts,
        "sensor_noise": sensor_noise,
        "train_mc": train_mc,
        "eval_mc": eval_mc,
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "domain_name": domain_name,
        "faces": faces,
        "face_map": face_map,
    }


def build_optimization_objective(
    settings: dict[str, object],
    domain_name: str,
    train_scenarios: tuple,
    sensor_noise: SensorNoiseConfig,
    train_mc: MonteCarloConfig,
) -> tuple[Callable[[PhotodiodeArray], float], str]:
    target = str(settings.get("optimization_target", "logdet"))

    if target == "validity":
        def validity_objective(photodiodes: PhotodiodeArray) -> float:
            score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
            return -score.valid_fraction

        return validity_objective, target

    if target == "error":
        if domain_name != "single_face":
            raise ValueError("The gamma error objective is only available in Single face mode.")

        def error_objective(photodiodes: PhotodiodeArray) -> float:
            gamma_metrics = evaluate_gamma_error_metrics(
                photodiodes,
                train_scenarios,
                sensor_noise,
                train_mc,
                domain_name,
            )
            mean_error = float(gamma_metrics["mean_gamma_error_deg"])
            if not np.isfinite(mean_error):
                return 1e6
            return mean_error

        return error_objective, target

    def logdet_objective(photodiodes: PhotodiodeArray) -> float:
        score = score_layout(photodiodes, train_scenarios, sensor_noise, train_mc)
        return -score.mean_logdet

    return logdet_objective, "logdet"


def run_optimizer_with_objective_support(
    optimizer: Callable[..., TrialResult],
    kwargs: dict[str, object],
) -> TrialResult:
    try:
        return optimizer(**kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'objective_fn'" not in str(exc):
            raise
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("objective_fn", None)
        fallback_kwargs.pop("objective_label", None)
        return optimizer(**fallback_kwargs)


def optimize_trial_for_count(
    settings: dict[str, object],
    domain_name: str,
    faces: tuple,
    train_scenarios: tuple,
    eval_scenarios: tuple,
    sensor_noise: SensorNoiseConfig,
    train_mc: MonteCarloConfig,
    eval_mc: MonteCarloConfig,
    count: int,
    seed_base: int,
) -> TrialResult:
    objective_fn, objective_label = build_optimization_objective(
        settings=settings,
        domain_name=domain_name,
        train_scenarios=train_scenarios,
        sensor_noise=sensor_noise,
        train_mc=train_mc,
    )
    if domain_name == "single_face":
        return run_optimizer_with_objective_support(
            optimize_face_layout,
            {
                "trial_name": f"{domain_name}_n{count:02d}_optimized",
                "domain_name": domain_name,
                "sensor_count": int(count),
                "faces": faces,
                "train_scenarios": train_scenarios,
                "eval_scenarios": eval_scenarios,
                "sensor_noise": sensor_noise,
                "train_mc": train_mc,
                "eval_mc": eval_mc,
                "minimum_spacing_m": float(settings["minimum_spacing_m"]),
                "generations": int(settings["generations"]),
                "popsize": int(settings["popsize"]),
                "seed": seed_base,
                "objective_fn": objective_fn,
                "objective_label": objective_label,
            },
        )
    return run_optimizer_with_objective_support(
        optimize_multi_face_per_face_layout,
        {
            "trial_name": f"{domain_name}_n{count:02d}_optimized",
            "sensors_per_face": int(count),
            "faces": faces,
            "train_scenarios": train_scenarios,
            "eval_scenarios": eval_scenarios,
            "sensor_noise": sensor_noise,
            "train_mc": train_mc,
            "eval_mc": eval_mc,
            "minimum_spacing_m": float(settings["minimum_spacing_m"]),
            "generations": int(settings["generations"]),
            "popsize": int(settings["popsize"]),
            "seed": seed_base,
            "objective_fn": objective_fn,
            "objective_label": objective_label,
        },
    )


def build_reference_optimized_layouts(settings: dict[str, object]) -> dict[int, PhotodiodeArray]:
    context = build_experiment_context(settings)
    layouts: dict[int, PhotodiodeArray] = {}
    for offset, count in enumerate(context["counts"]):
        seed_base = int(settings["seed"]) + offset * 100
        optimized = optimize_trial_for_count(
            settings=settings,
            domain_name=str(context["domain_name"]),
            faces=tuple(context["faces"]),
            train_scenarios=tuple(context["train_scenarios"]),
            eval_scenarios=tuple(context["eval_scenarios"]),
            sensor_noise=context["sensor_noise"],
            train_mc=context["train_mc"],
            eval_mc=context["eval_mc"],
            count=int(count),
            seed_base=seed_base,
        )
        layouts[int(count)] = optimized.photodiodes
    return layouts


def serialize_trial(
    result: TrialResult,
    method_key: str,
    method_kind: str,
    count_input: int,
    face_map: dict[str, FaceGeometry],
    eval_gamma_metrics: dict[str, float],
    include_layout: bool = True,
) -> dict[str, object]:
    method_label = PATTERN_LABELS[method_key]
    trial = {
        "trial_key": result.trial_name,
        "trial_name": result.trial_name,
        "domain_name": result.domain_name,
        "sensor_count": int(result.sensor_count),
        "method_key": method_key,
        "method": method_label,
        "method_kind": method_kind,
        "objective_value": float(result.objective_value),
        "optimizer_success": bool(result.optimizer_success),
        "optimizer_message": str(result.optimizer_message),
        "minimum_spacing_m": float(result.minimum_spacing_m),
        "train_mean_logdet": float(result.train_score.mean_logdet),
        "train_mean_valid_logdet": float(result.train_score.mean_valid_logdet),
        "train_valid_fraction": float(result.train_score.valid_fraction),
        "train_any_active_fraction": float(result.train_score.any_active_fraction),
        "train_mean_active_sensors": float(result.train_score.mean_active_sensor_count),
        "eval_mean_logdet": float(result.eval_score.mean_logdet),
        "eval_mean_valid_logdet": float(result.eval_score.mean_valid_logdet),
        "eval_valid_fraction": float(result.eval_score.valid_fraction),
        "eval_any_active_fraction": float(result.eval_score.any_active_fraction),
        "eval_mean_active_sensors": float(result.eval_score.mean_active_sensor_count),
        "eval_mean_below_noise_sensors": float(result.eval_score.mean_below_noise_sensor_count),
        "eval_mean_saturated_sensors": float(result.eval_score.mean_saturated_sensor_count),
        "eval_mean_gamma_error_deg": float(eval_gamma_metrics["mean_gamma_error_deg"]),
        "eval_gamma_error_valid_fraction": float(eval_gamma_metrics["gamma_error_valid_fraction"]),
    }
    if include_layout:
        trial["layout"] = serialize_layout(result.photodiodes, face_map)
    return trial


def make_trial_dataframe(trials: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "trial_key": [trial["trial_key"] for trial in trials],
            "method": [trial["method"] for trial in trials],
            "method_kind": [trial["method_kind"] for trial in trials],
            "sensor_count": [trial["sensor_count"] for trial in trials],
            "eval_mean_logdet": [trial["eval_mean_logdet"] for trial in trials],
            "eval_mean_valid_logdet": [trial["eval_mean_valid_logdet"] for trial in trials],
            "eval_valid_fraction": [trial["eval_valid_fraction"] for trial in trials],
            "eval_any_active_fraction": [trial["eval_any_active_fraction"] for trial in trials],
            "eval_mean_gamma_error_deg": [trial["eval_mean_gamma_error_deg"] for trial in trials],
            "eval_gamma_error_valid_fraction": [trial["eval_gamma_error_valid_fraction"] for trial in trials],
            "eval_mean_active_sensors": [trial["eval_mean_active_sensors"] for trial in trials],
            "eval_mean_below_noise_sensors": [trial["eval_mean_below_noise_sensors"] for trial in trials],
            "optimizer_message": [trial["optimizer_message"] for trial in trials],
        }
    )
    frame = frame.sort_values(
        by=["sensor_count", "method"],
        key=lambda series: series.map(method_sort_key) if series.name == "method" else series,
    ).reset_index(drop=True)
    return frame


def make_winner_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_count, group in summary_df.groupby("sensor_count", sort=True):
        ranked = group.sort_values("eval_mean_logdet", ascending=False).reset_index(drop=True)
        winner = ranked.iloc[0]
        runner_up = ranked.iloc[1] if len(ranked) > 1 else None
        row = {
            "sensor_count": int(sensor_count),
            "winner": str(winner["method"]),
            "winner_score": float(winner["eval_mean_logdet"]),
            "winner_valid_only_score": float(winner["eval_mean_valid_logdet"]),
            "winner_valid_fraction": float(winner["eval_valid_fraction"]),
            "winner_any_active_fraction": float(winner["eval_any_active_fraction"]),
            "mean_gamma_error_deg": float(winner["eval_mean_gamma_error_deg"]),
            "runner_up": str(runner_up["method"]) if runner_up is not None else "",
            "score_margin": float(winner["eval_mean_logdet"] - runner_up["eval_mean_logdet"]) if runner_up is not None else np.nan,
        }
        optimized = group[group["method"] == "Optimized"]
        heuristics = group[group["method_kind"] == "heuristic"]
        if not optimized.empty and not heuristics.empty:
            best_heuristic = heuristics.sort_values("eval_mean_logdet", ascending=False).iloc[0]
            optimized_row = optimized.iloc[0]
            row["best_heuristic"] = str(best_heuristic["method"])
            row["optimized_minus_best_heuristic"] = float(
                optimized_row["eval_mean_logdet"] - best_heuristic["eval_mean_logdet"]
            )
            row["optimized_valid_fraction_delta"] = float(
                optimized_row["eval_valid_fraction"] - best_heuristic["eval_valid_fraction"]
            )
        else:
            row["best_heuristic"] = ""
            row["optimized_minus_best_heuristic"] = np.nan
            row["optimized_valid_fraction_delta"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def make_trial_summary_display_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    frame = summary_df.copy()
    if "eval_mean_valid_logdet" in frame.columns:
        frame["eval_logdet_valid_only"] = frame["eval_mean_valid_logdet"]
    frame = frame.rename(
        columns={
            "sensor_count": "total_sensors",
            "eval_mean_gamma_error_deg": "eval_mean_valid_gamma_error_deg",
        }
    )
    ordered_columns = [
        "total_sensors",
        "method",
        "trial_key",
        "method_kind",
        "eval_mean_logdet",
        "eval_logdet_valid_only",
        "eval_valid_fraction",
        "eval_any_active_fraction",
        "eval_mean_valid_gamma_error_deg",
        "eval_mean_active_sensors",
        "eval_mean_below_noise_sensors",
        "optimizer_message",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def make_winner_display_frame(winner_df: pd.DataFrame) -> pd.DataFrame:
    frame = winner_df.copy()
    frame = frame.rename(
        columns={
            "sensor_count": "total_sensors",
            "mean_gamma_error_deg": "mean_valid_gamma_error_deg",
        }
    )
    ordered_columns = [
        "total_sensors",
        "winner",
        "winner_score",
        "winner_valid_only_score",
        "winner_valid_fraction",
        "winner_any_active_fraction",
        "mean_valid_gamma_error_deg",
        "runner_up",
        "score_margin",
        "best_heuristic",
        "optimized_minus_best_heuristic",
        "optimized_valid_fraction_delta",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


MULTI_SEED_DISPLAY_RENAMES = {
    "sensor_count": "total_sensors",
    "lower_sensor_count": "lower_total_sensors",
    "higher_sensor_count": "higher_total_sensors",
    "mean_eval_mean_logdet": "mean_eval_logdet",
    "mean_eval_mean_valid_logdet": "mean_eval_logdet_valid_only",
    "mean_gamma_error_deg": "mean_valid_gamma_error_deg",
    "mean_gamma_error_valid_fraction": "mean_gamma_estimate_fraction",
    "error_valid_ci_low": "gamma_estimate_fraction_ci_low",
    "error_valid_ci_high": "gamma_estimate_fraction_ci_high",
    "mean_valid_delta": "mean_valid_only_logdet_delta",
    "mean_valid_ci_low": "valid_only_logdet_ci_low",
    "mean_valid_ci_high": "valid_only_logdet_ci_high",
    "mean_valid_improves_fraction": "valid_only_logdet_improves_fraction",
    "mean_valid_p_value": "valid_only_logdet_p_value",
    "mean_valid_significant_at_0_05": "valid_only_logdet_significant_at_0_05",
    "mean_error_reduction_deg": "mean_valid_gamma_error_reduction_deg",
    "mean_error_reduction_ci_low": "valid_gamma_error_reduction_ci_low",
    "mean_error_reduction_ci_high": "valid_gamma_error_reduction_ci_high",
    "mean_error_improves_fraction": "valid_gamma_error_improves_fraction",
    "mean_error_p_value": "valid_gamma_error_p_value",
    "mean_error_significant_at_0_05": "valid_gamma_error_significant_at_0_05",
    "error_ci_low": "valid_gamma_error_ci_low",
    "error_ci_high": "valid_gamma_error_ci_high",
}


def make_multi_seed_stability_display_frame(stability_df: pd.DataFrame) -> pd.DataFrame:
    frame = stability_df.rename(columns=MULTI_SEED_DISPLAY_RENAMES).copy()
    ordered_columns = [
        "total_sensors",
        "most_frequent_winner",
        "winner_fraction",
        "winner_counts",
        "trial_count",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def make_multi_seed_aggregate_display_frame(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    frame = aggregate_df.rename(columns=MULTI_SEED_DISPLAY_RENAMES).copy()
    ordered_columns = [
        "total_sensors",
        "method",
        "trial_count",
        "mean_eval_logdet",
        "score_ci_low",
        "score_ci_high",
        "mean_eval_logdet_valid_only",
        "valid_only_logdet_ci_low",
        "valid_only_logdet_ci_high",
        "mean_valid_fraction",
        "valid_ci_low",
        "valid_ci_high",
        "mean_any_active_fraction",
        "any_active_fraction_ci_low",
        "any_active_fraction_ci_high",
        "mean_valid_gamma_error_deg",
        "valid_gamma_error_ci_low",
        "valid_gamma_error_ci_high",
        "mean_gamma_estimate_fraction",
        "gamma_estimate_fraction_ci_low",
        "gamma_estimate_fraction_ci_high",
        "mean_active_sensors",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def make_multi_seed_significance_display_frame(significance_df: pd.DataFrame) -> pd.DataFrame:
    frame = significance_df.rename(columns=MULTI_SEED_DISPLAY_RENAMES).copy()
    ordered_columns = [
        "total_sensors",
        "winner_by_mean_score",
        "runner_up",
        "trial_count",
        "mean_score_delta",
        "score_delta_ci_low",
        "score_delta_ci_high",
        "score_improves_fraction",
        "score_p_value",
        "score_significant_at_0_05",
        "score_ci_excludes_zero",
        "mean_valid_only_logdet_delta",
        "valid_only_logdet_ci_low",
        "valid_only_logdet_ci_high",
        "valid_only_logdet_improves_fraction",
        "valid_only_logdet_p_value",
        "valid_only_logdet_significant_at_0_05",
        "mean_valid_fraction_delta",
        "mean_any_active_fraction_delta",
        "any_active_ci_low",
        "any_active_ci_high",
        "any_active_improves_fraction",
        "any_active_p_value",
        "any_active_significant_at_0_05",
        "mean_valid_gamma_error_reduction_deg",
        "valid_gamma_error_reduction_ci_low",
        "valid_gamma_error_reduction_ci_high",
        "valid_gamma_error_improves_fraction",
        "valid_gamma_error_p_value",
        "valid_gamma_error_significant_at_0_05",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def make_best_progression_display_frame(best_progression_df: pd.DataFrame) -> pd.DataFrame:
    frame = best_progression_df.rename(columns=MULTI_SEED_DISPLAY_RENAMES).copy()
    ordered_columns = [
        "lower_total_sensors",
        "higher_total_sensors",
        "lower_most_frequent_winner",
        "higher_most_frequent_winner",
        "trial_count",
        "mean_score_delta",
        "score_delta_ci_low",
        "score_delta_ci_high",
        "score_improves_fraction",
        "score_p_value",
        "score_significant_at_0_05",
        "score_ci_excludes_zero",
        "mean_valid_only_logdet_delta",
        "valid_only_logdet_ci_low",
        "valid_only_logdet_ci_high",
        "valid_only_logdet_improves_fraction",
        "valid_only_logdet_p_value",
        "valid_only_logdet_significant_at_0_05",
        "mean_valid_fraction_delta",
        "valid_fraction_ci_low",
        "valid_fraction_ci_high",
        "valid_fraction_improves_fraction",
        "valid_fraction_p_value",
        "valid_fraction_significant_at_0_05",
        "mean_any_active_fraction_delta",
        "any_active_ci_low",
        "any_active_ci_high",
        "any_active_improves_fraction",
        "any_active_p_value",
        "any_active_significant_at_0_05",
        "mean_valid_gamma_error_reduction_deg",
        "valid_gamma_error_reduction_ci_low",
        "valid_gamma_error_reduction_ci_high",
        "valid_gamma_error_improves_fraction",
        "valid_gamma_error_p_value",
        "valid_gamma_error_significant_at_0_05",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def make_method_progression_display_frame(method_progression_df: pd.DataFrame) -> pd.DataFrame:
    frame = method_progression_df.rename(columns=MULTI_SEED_DISPLAY_RENAMES).copy()
    ordered_columns = [
        "method",
        "lower_total_sensors",
        "higher_total_sensors",
        "trial_count",
        "mean_score_delta",
        "score_delta_ci_low",
        "score_delta_ci_high",
        "score_improves_fraction",
        "score_p_value",
        "score_significant_at_0_05",
        "score_ci_excludes_zero",
        "mean_valid_only_logdet_delta",
        "valid_only_logdet_ci_low",
        "valid_only_logdet_ci_high",
        "valid_only_logdet_improves_fraction",
        "valid_only_logdet_p_value",
        "valid_only_logdet_significant_at_0_05",
        "mean_valid_fraction_delta",
        "valid_fraction_ci_low",
        "valid_fraction_ci_high",
        "valid_fraction_improves_fraction",
        "valid_fraction_p_value",
        "valid_fraction_significant_at_0_05",
        "mean_any_active_fraction_delta",
        "any_active_ci_low",
        "any_active_ci_high",
        "any_active_improves_fraction",
        "any_active_p_value",
        "any_active_significant_at_0_05",
        "mean_valid_gamma_error_reduction_deg",
        "valid_gamma_error_reduction_ci_low",
        "valid_gamma_error_reduction_ci_high",
        "valid_gamma_error_improves_fraction",
        "valid_gamma_error_p_value",
        "valid_gamma_error_significant_at_0_05",
    ]
    return frame[[column for column in ordered_columns if column in frame.columns]]


def generate_trials(
    settings: dict[str, object],
    include_layout: bool = True,
    frozen_optimized_layouts: dict[int, PhotodiodeArray] | None = None,
    frozen_optimized_seed: int | None = None,
) -> list[dict[str, object]]:
    context = build_experiment_context(settings)
    counts = context["counts"]
    sensor_noise = context["sensor_noise"]
    train_mc = context["train_mc"]
    eval_mc = context["eval_mc"]
    train_scenarios = tuple(context["train_scenarios"])
    eval_scenarios = tuple(context["eval_scenarios"])
    domain_name = str(context["domain_name"])
    faces = tuple(context["faces"])
    face_map = context["face_map"]
    trials: list[dict[str, object]] = []
    patterns = [str(pattern) for pattern in settings["patterns"]]
    for offset, count in enumerate(counts):
        seed_base = int(settings["seed"]) + offset * 100
        if bool(settings["include_optimized"]):
            if frozen_optimized_layouts is not None and int(count) in frozen_optimized_layouts:
                optimized_layout = frozen_optimized_layouts[int(count)]
                train_score = score_layout(optimized_layout, train_scenarios, sensor_noise, train_mc)
                eval_score = score_layout(optimized_layout, eval_scenarios, sensor_noise, eval_mc)
                total_sensors = int(count if domain_name == "single_face" else count * len(faces))
                reference_seed_label = f"seed_{int(frozen_optimized_seed)}" if frozen_optimized_seed is not None else "shared"
                optimized = TrialResult(
                    trial_name=f"{domain_name}_n{count:02d}_optimized",
                    domain_name=domain_name,
                    sensor_count=total_sensors,
                    train_score=train_score,
                    eval_score=eval_score,
                    objective_value=float("nan"),
                    optimizer_success=True,
                    optimizer_message=f"optimized:frozen_layouts_from_{reference_seed_label}",
                    iterations=0,
                    photodiodes=optimized_layout,
                    minimum_spacing_m=float(settings["minimum_spacing_m"]),
                )
            else:
                optimized = optimize_trial_for_count(
                    settings=settings,
                    domain_name=domain_name,
                    faces=faces,
                    train_scenarios=train_scenarios,
                    eval_scenarios=eval_scenarios,
                    sensor_noise=sensor_noise,
                    train_mc=train_mc,
                    eval_mc=eval_mc,
                    count=int(count),
                    seed_base=seed_base,
                )
            optimized_gamma_metrics = evaluate_gamma_error_metrics(
                optimized.photodiodes,
                eval_scenarios,
                sensor_noise,
                eval_mc,
                domain_name,
            )
            trials.append(
                serialize_trial(
                    optimized,
                    "optimized",
                    "optimized",
                    int(count),
                    face_map,
                    optimized_gamma_metrics,
                    include_layout=include_layout,
                )
            )

        for pattern_index, pattern in enumerate(patterns):
            layout = heuristic_layout_fixed_faces(
                int(count),
                faces,
                pattern,
                seed=seed_base + 1000 + pattern_index,
            )
            train_score = score_layout(layout, train_scenarios, sensor_noise, train_mc)
            eval_score = score_layout(layout, eval_scenarios, sensor_noise, eval_mc)
            heuristic = build_trial_result(
                trial_name=f"{domain_name}_n{count:02d}_{pattern}",
                domain_name=domain_name,
                sensor_count=int(count if domain_name == "single_face" else count * len(faces)),
                minimum_spacing_m=float(settings["minimum_spacing_m"]),
                method_key=pattern,
                photodiodes=layout,
                train_score=train_score,
                eval_score=eval_score,
            )
            heuristic_gamma_metrics = evaluate_gamma_error_metrics(
                heuristic.photodiodes,
                eval_scenarios,
                sensor_noise,
                eval_mc,
                domain_name,
            )
            trials.append(
                serialize_trial(
                    heuristic,
                    pattern,
                    "heuristic",
                    int(count),
                    face_map,
                    heuristic_gamma_metrics,
                    include_layout=include_layout,
                )
            )

    return trials


@st.cache_data(show_spinner=False)
def run_experiment(settings_json: str) -> dict[str, object]:
    settings = json.loads(settings_json)
    counts = parse_counts(str(settings["counts_text"]))
    trials = generate_trials(settings, include_layout=True)

    summary_df = make_trial_dataframe(trials)
    winner_df = make_winner_table(summary_df)
    return {
        "summary_df": summary_df,
        "winner_df": winner_df,
        "trial_lookup": {str(trial["trial_key"]): trial for trial in trials},
        "trials": trials,
        "counts": counts,
        "settings": settings,
    }


def confidence_interval(
    series: pd.Series,
    confidence: float = 0.95,
    clamp: tuple[float, float] | None = None,
) -> tuple[float, float]:
    values = series.dropna().astype(float).to_numpy()
    count = values.size
    if count == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    if count == 1:
        lower = mean
        upper = mean
    else:
        sem = float(stats.sem(values))
        if not np.isfinite(sem) or np.isclose(sem, 0.0):
            lower = mean
            upper = mean
        else:
            interval = stats.t.interval(confidence, df=count - 1, loc=mean, scale=sem)
            lower = float(interval[0])
            upper = float(interval[1])
    if clamp is not None:
        lower = float(np.clip(lower, clamp[0], clamp[1]))
        upper = float(np.clip(upper, clamp[0], clamp[1]))
    return lower, upper


def summarize_delta_series(deltas: pd.Series) -> dict[str, float | bool | int]:
    values = deltas.dropna().astype(float)
    if values.empty:
        return {
            "trial_count": 0,
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "ci_excludes_zero": False,
            "win_fraction": float("nan"),
            "p_value": float("nan"),
            "significant_at_0_05": False,
        }

    mean_delta = float(values.mean())
    ci_low, ci_high = confidence_interval(values)
    if len(values) > 1:
        array = values.to_numpy(dtype=float)
        if np.allclose(array, array[0]):
            p_value = 0.0 if not np.isclose(array[0], 0.0) else 1.0
        else:
            p_value = float(stats.ttest_1samp(array, popmean=0.0).pvalue)
    else:
        p_value = float("nan")
    return {
        "trial_count": int(values.shape[0]),
        "mean_delta": mean_delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
        "win_fraction": float((values > 0.0).mean()),
        "p_value": p_value,
        "significant_at_0_05": bool(np.isfinite(p_value) and p_value < 0.05),
    }


def add_metric_comparison(
    row: dict[str, object],
    left_values: pd.Series,
    right_values: pd.Series,
    metric_spec: dict[str, object],
) -> None:
    if bool(metric_spec["higher_is_better"]):
        deltas = left_values - right_values
    else:
        deltas = right_values - left_values
    summary = summarize_delta_series(deltas)
    row[str(metric_spec["delta_column"])] = float(summary["mean_delta"])
    row[str(metric_spec["ci_low_column"])] = float(summary["ci_low"])
    row[str(metric_spec["ci_high_column"])] = float(summary["ci_high"])
    row[str(metric_spec["improves_fraction_column"])] = float(summary["win_fraction"])
    row[str(metric_spec["p_value_column"])] = float(summary["p_value"])
    row[str(metric_spec["significant_column"])] = bool(summary["significant_at_0_05"])


def make_multi_seed_aggregate_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = per_seed_df.groupby(["sensor_count", "method"], sort=True)
    for (sensor_count, method), group in grouped:
        score_low, score_high = confidence_interval(group["eval_mean_logdet"])
        mean_valid_low, mean_valid_high = confidence_interval(group["eval_mean_valid_logdet"])
        valid_low, valid_high = confidence_interval(group["eval_valid_fraction"], clamp=(0.0, 1.0))
        any_active_low, any_active_high = confidence_interval(group["eval_any_active_fraction"], clamp=(0.0, 1.0))
        error_low, error_high = confidence_interval(group["eval_mean_gamma_error_deg"], clamp=(0.0, float("inf")))
        error_valid_low, error_valid_high = confidence_interval(group["eval_gamma_error_valid_fraction"], clamp=(0.0, 1.0))
        rows.append(
            {
                "sensor_count": int(sensor_count),
                "method": str(method),
                "trial_count": int(group["seed"].nunique()),
                "mean_eval_mean_logdet": float(group["eval_mean_logdet"].mean()),
                "score_ci_low": score_low,
                "score_ci_high": score_high,
                "mean_eval_mean_valid_logdet": float(group["eval_mean_valid_logdet"].mean()),
                "mean_valid_ci_low": mean_valid_low,
                "mean_valid_ci_high": mean_valid_high,
                "mean_valid_fraction": float(group["eval_valid_fraction"].mean()),
                "valid_ci_low": valid_low,
                "valid_ci_high": valid_high,
                "mean_any_active_fraction": float(group["eval_any_active_fraction"].mean()),
                "any_active_fraction_ci_low": any_active_low,
                "any_active_fraction_ci_high": any_active_high,
                "mean_gamma_error_deg": float(group["eval_mean_gamma_error_deg"].mean()),
                "error_ci_low": error_low,
                "error_ci_high": error_high,
                "mean_gamma_error_valid_fraction": float(group["eval_gamma_error_valid_fraction"].mean()),
                "error_valid_ci_low": error_valid_low,
                "error_valid_ci_high": error_valid_high,
                "mean_active_sensors": float(group["eval_mean_active_sensors"].mean()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        by=["sensor_count", "method"],
        key=lambda series: series.map(method_sort_key) if series.name == "method" else series,
    ).reset_index(drop=True)


def make_winner_stability_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        per_seed_df.sort_values(
            by=["seed", "sensor_count", "eval_mean_logdet", "eval_valid_fraction", "eval_mean_active_sensors"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["seed", "sensor_count"], as_index=False, sort=True)
        .first()
    )
    rows: list[dict[str, object]] = []
    for sensor_count, group in ranked.groupby("sensor_count", sort=True):
        winner_counts = group["method"].value_counts()
        if winner_counts.empty:
            continue
        most_frequent_winner = str(winner_counts.index[0])
        win_fraction = float(winner_counts.iloc[0] / len(group))
        counts_text = ", ".join(f"{method}: {int(count)}" for method, count in winner_counts.items())
        rows.append(
            {
                "sensor_count": int(sensor_count),
                "most_frequent_winner": most_frequent_winner,
                "winner_fraction": win_fraction,
                "winner_counts": counts_text,
                "trial_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values("sensor_count").reset_index(drop=True)


def make_significance_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sensor_count, group in per_seed_df.groupby("sensor_count", sort=True):
        method_means = group.groupby("method")["eval_mean_logdet"].mean().sort_values(ascending=False)
        if len(method_means) < 2:
            continue
        winner = str(method_means.index[0])
        runner_up = str(method_means.index[1])
        valid_pivot = group.pivot_table(index="seed", columns="method", values="eval_valid_fraction")
        metric_pivots = {
            str(metric_spec["column"]): group.pivot_table(index="seed", columns="method", values=str(metric_spec["column"]))
            for metric_spec in METRIC_SPECS
        }
        if any(winner not in pivot or runner_up not in pivot for pivot in metric_pivots.values()):
            continue
        valid_deltas = (valid_pivot[winner] - valid_pivot[runner_up]).dropna()
        summary = summarize_delta_series(metric_pivots["eval_mean_logdet"][winner] - metric_pivots["eval_mean_logdet"][runner_up])
        if summary["trial_count"] == 0:
            continue
        row = {
            "sensor_count": int(sensor_count),
            "winner_by_mean_score": winner,
            "runner_up": runner_up,
            "trial_count": int(summary["trial_count"]),
            "score_ci_excludes_zero": bool(summary["ci_excludes_zero"]),
            "mean_valid_fraction_delta": float(valid_deltas.mean()) if not valid_deltas.empty else float("nan"),
        }
        for metric_spec in METRIC_SPECS:
            add_metric_comparison(
                row,
                metric_pivots[str(metric_spec["column"])][winner],
                metric_pivots[str(metric_spec["column"])][runner_up],
                metric_spec,
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sensor_count").reset_index(drop=True)


def make_best_count_progression_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    ranked = (
        per_seed_df.sort_values(
            by=["seed", "sensor_count", "eval_mean_logdet", "eval_valid_fraction", "eval_mean_active_sensors"],
            ascending=[True, True, False, False, False],
        )
        .groupby(["seed", "sensor_count"], as_index=False, sort=True)
        .first()
    )
    counts = sorted(int(value) for value in ranked["sensor_count"].unique())
    rows: list[dict[str, object]] = []
    for lower_count, higher_count in zip(counts, counts[1:]):
        lower_rows = ranked[ranked["sensor_count"] == lower_count].set_index("seed")
        higher_rows = ranked[ranked["sensor_count"] == higher_count].set_index("seed")
        common_seeds = lower_rows.index.intersection(higher_rows.index)
        if common_seeds.empty:
            continue
        lower_comp = lower_rows.loc[common_seeds].sort_index()
        higher_comp = higher_rows.loc[common_seeds].sort_index()
        summary = summarize_delta_series(higher_comp["eval_mean_logdet"] - lower_comp["eval_mean_logdet"])
        if summary["trial_count"] == 0:
            continue
        valid_deltas = higher_comp["eval_valid_fraction"] - lower_comp["eval_valid_fraction"]
        valid_summary = summarize_delta_series(valid_deltas)
        lower_winners = lower_comp["method"].value_counts()
        higher_winners = higher_comp["method"].value_counts()
        row = {
            "lower_sensor_count": int(lower_count),
            "higher_sensor_count": int(higher_count),
            "lower_most_frequent_winner": str(lower_winners.index[0]) if not lower_winners.empty else "",
            "higher_most_frequent_winner": str(higher_winners.index[0]) if not higher_winners.empty else "",
            "trial_count": int(summary["trial_count"]),
            "score_ci_excludes_zero": bool(summary["ci_excludes_zero"]),
            "mean_valid_fraction_delta": float(valid_deltas.mean()) if not valid_deltas.empty else float("nan"),
            "valid_fraction_ci_low": float(valid_summary["ci_low"]),
            "valid_fraction_ci_high": float(valid_summary["ci_high"]),
            "valid_fraction_improves_fraction": float(valid_summary["win_fraction"]),
            "valid_fraction_p_value": float(valid_summary["p_value"]),
            "valid_fraction_significant_at_0_05": bool(valid_summary["significant_at_0_05"]),
        }
        for metric_spec in METRIC_SPECS:
            add_metric_comparison(row, higher_comp[str(metric_spec["column"])], lower_comp[str(metric_spec["column"])], metric_spec)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["lower_sensor_count", "higher_sensor_count"]).reset_index(drop=True)


def make_method_count_progression_table(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, group in per_seed_df.groupby("method", sort=True):
        counts = sorted(int(value) for value in group["sensor_count"].unique())
        valid_pivot = group.pivot_table(index="seed", columns="sensor_count", values="eval_valid_fraction")
        metric_pivots = {
            str(metric_spec["column"]): group.pivot_table(index="seed", columns="sensor_count", values=str(metric_spec["column"]))
            for metric_spec in METRIC_SPECS
        }
        for lower_count, higher_count in zip(counts, counts[1:]):
            if any(lower_count not in pivot or higher_count not in pivot for pivot in metric_pivots.values()):
                continue
            summary = summarize_delta_series(metric_pivots["eval_mean_logdet"][higher_count] - metric_pivots["eval_mean_logdet"][lower_count])
            if summary["trial_count"] == 0:
                continue
            valid_deltas = (valid_pivot[higher_count] - valid_pivot[lower_count]).dropna()
            valid_summary = summarize_delta_series(valid_deltas)
            row = {
                "method": str(method),
                "lower_sensor_count": int(lower_count),
                "higher_sensor_count": int(higher_count),
                "trial_count": int(summary["trial_count"]),
                "score_ci_excludes_zero": bool(summary["ci_excludes_zero"]),
                "mean_valid_fraction_delta": float(valid_deltas.mean()) if not valid_deltas.empty else float("nan"),
                "valid_fraction_ci_low": float(valid_summary["ci_low"]),
                "valid_fraction_ci_high": float(valid_summary["ci_high"]),
                "valid_fraction_improves_fraction": float(valid_summary["win_fraction"]),
                "valid_fraction_p_value": float(valid_summary["p_value"]),
                "valid_fraction_significant_at_0_05": bool(valid_summary["significant_at_0_05"]),
            }
            for metric_spec in METRIC_SPECS:
                add_metric_comparison(
                    row,
                    metric_pivots[str(metric_spec["column"])][higher_count],
                    metric_pivots[str(metric_spec["column"])][lower_count],
                    metric_spec,
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        by=["method", "lower_sensor_count", "higher_sensor_count"],
        key=lambda series: series.map(method_sort_key) if series.name == "method" else series,
    ).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_multi_seed_experiment(
    settings_json: str,
    trial_count: int,
    seed_step: int,
    reuse_optimized_layouts_across_seeds: bool = False,
) -> dict[str, object]:
    settings = json.loads(settings_json)
    counts = parse_counts(str(settings["counts_text"]))
    per_seed_frames: list[pd.DataFrame] = []
    seed_values: list[int] = []
    frozen_optimized_layouts: dict[int, PhotodiodeArray] | None = None
    optimized_reference_seed: int | None = None
    if bool(reuse_optimized_layouts_across_seeds) and bool(settings.get("include_optimized", False)):
        optimized_reference_seed = int(settings["seed"])
        frozen_optimized_layouts = build_reference_optimized_layouts(settings)
    for trial_index in range(trial_count):
        seed_value = int(settings["seed"]) + trial_index * seed_step
        seed_values.append(seed_value)
        trial_settings = dict(settings)
        trial_settings["seed"] = seed_value
        trials = generate_trials(
            trial_settings,
            include_layout=False,
            frozen_optimized_layouts=frozen_optimized_layouts,
            frozen_optimized_seed=optimized_reference_seed,
        )
        summary_df = make_trial_dataframe(trials)
        summary_df["seed"] = seed_value
        summary_df["trial_index"] = trial_index + 1
        per_seed_frames.append(summary_df)

    per_seed_df = pd.concat(per_seed_frames, ignore_index=True) if per_seed_frames else pd.DataFrame()
    aggregate_df = make_multi_seed_aggregate_table(per_seed_df) if not per_seed_df.empty else pd.DataFrame()
    stability_df = make_winner_stability_table(per_seed_df) if not per_seed_df.empty else pd.DataFrame()
    significance_df = make_significance_table(per_seed_df) if not per_seed_df.empty else pd.DataFrame()
    best_progression_df = make_best_count_progression_table(per_seed_df) if not per_seed_df.empty else pd.DataFrame()
    method_progression_df = make_method_count_progression_table(per_seed_df) if not per_seed_df.empty else pd.DataFrame()
    return {
        "counts": counts,
        "seed_values": seed_values,
        "reuse_optimized_layouts_across_seeds": bool(reuse_optimized_layouts_across_seeds),
        "optimized_reference_seed": optimized_reference_seed,
        "per_seed_df": per_seed_df,
        "aggregate_df": aggregate_df,
        "stability_df": stability_df,
        "significance_df": significance_df,
        "best_progression_df": best_progression_df,
        "method_progression_df": method_progression_df,
    }


def make_score_chart(summary_df: pd.DataFrame) -> go.Figure:
    figure = px.line(
        summary_df,
        x="sensor_count",
        y="eval_mean_valid_logdet",
        color="method",
        markers=True,
        symbol="method",
        hover_data=["eval_mean_logdet", "eval_valid_fraction", "eval_mean_gamma_error_deg", "eval_mean_active_sensors", "trial_key"],
    )
    figure.update_layout(
        title="Held-out valid-only direction information by arrangement",
        xaxis_title="Total sensors",
        yaxis_title="Eval mean valid log det(F_direction)",
        legend_title="Arrangement",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def make_validity_chart(summary_df: pd.DataFrame) -> go.Figure:
    figure = px.line(
        summary_df,
        x="sensor_count",
        y="eval_valid_fraction",
        color="method",
        markers=True,
        symbol="method",
        hover_data=["eval_mean_logdet", "eval_mean_gamma_error_deg", "eval_mean_active_sensors", "trial_key"],
    )
    figure.update_layout(
        title="Held-out valid-scenario fraction",
        xaxis_title="Total sensors",
        yaxis_title="Eval valid fraction",
        legend_title="Arrangement",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    figure.update_yaxes(range=[0.0, 1.02])
    return figure


def make_gamma_error_chart(summary_df: pd.DataFrame) -> go.Figure:
    figure = px.line(
        summary_df,
        x="sensor_count",
        y="eval_mean_gamma_error_deg",
        color="method",
        markers=True,
        symbol="method",
        hover_data=["eval_mean_logdet", "eval_valid_fraction", "eval_mean_active_sensors", "trial_key"],
    )
    figure.update_layout(
        title="Held-out mean reconstructed gamma error on valid estimates",
        xaxis_title="Total sensors",
        yaxis_title="Mean valid gamma error (deg)",
        legend_title="Arrangement",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def make_any_active_chart(summary_df: pd.DataFrame) -> go.Figure:
    figure = px.line(
        summary_df,
        x="sensor_count",
        y="eval_any_active_fraction",
        color="method",
        markers=True,
        symbol="method",
        hover_data=["eval_mean_logdet", "eval_valid_fraction", "eval_mean_active_sensors", "trial_key"],
    )
    figure.update_layout(
        title="Held-out any-sensor-active fraction",
        xaxis_title="Total sensors",
        yaxis_title="Eval any-active fraction",
        legend_title="Arrangement",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    figure.update_yaxes(range=[0.0, 1.02])
    return figure


def make_layout_plot(layout_records: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(layout_records)
    if frame["face"].nunique() == 1:
        width_cm = float(frame["face_width_cm"].iloc[0])
        height_cm = float(frame["face_height_cm"].iloc[0])
        figure = px.scatter(
            frame,
            x="local_u_cm",
            y="local_v_cm",
            text="sensor",
            color="face",
            hover_data=["x_m", "y_m", "z_m"],
        )
        figure.update_traces(marker=dict(size=14), textposition="top center")
        figure.add_shape(
            type="rect",
            x0=-width_cm / 2.0,
            x1=width_cm / 2.0,
            y0=-height_cm / 2.0,
            y1=height_cm / 2.0,
            line=dict(color="#4b5563", width=2),
        )
        figure.update_layout(
            title="Sensor arrangement on the face",
            xaxis_title="Local u (cm)",
            yaxis_title="Local v (cm)",
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        figure.update_yaxes(scaleanchor="x", scaleratio=1.0)
        return figure

    figure = px.scatter_3d(
        frame,
        x="x_m",
        y="y_m",
        z="z_m",
        color="face",
        text="sensor",
        hover_data=["local_u_cm", "local_v_cm", "normal_x", "normal_y", "normal_z"],
    )
    figure.update_traces(marker=dict(size=6), textposition="top center")
    figure.update_layout(
        title="Sensor arrangement in 3D",
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return figure


def render_layout_summary(trial: dict[str, object]) -> None:
    st.markdown(
        (
            f"**{trial['method']}**  "
            f"`total sensors={trial['sensor_count']}`  "
            f"`held-out score={float(trial['eval_mean_logdet']):.3e}`  "
            f"`valid fraction={float(trial['eval_valid_fraction']):.3f}`"
        )
    )
    st.plotly_chart(
        make_layout_plot(list(trial["layout"])),
        use_container_width=True,
        key=f"winner_layout_plot_{trial['trial_key']}",
    )
    if st.button("Inspect this layout", key=f"inspect_{trial['trial_key']}"):
        st.session_state["selected_trial_key"] = str(trial["trial_key"])


def render_winner_gallery(summary_df: pd.DataFrame, trial_lookup: dict[str, dict[str, object]]) -> None:
    winner_rows = (
        summary_df.sort_values(
            by=["sensor_count", "eval_mean_logdet", "eval_valid_fraction", "eval_mean_active_sensors"],
            ascending=[True, False, False, False],
        )
        .groupby("sensor_count", sort=True, as_index=False)
        .first()
        .sort_values("sensor_count")
        .reset_index(drop=True)
    )
    st.subheader("Winning Layout For Each Sensor Count")
    if winner_rows.empty:
        st.info("Run a comparison to see the winning arrangement at each sensor count.")
        return

    for start in range(0, len(winner_rows), 2):
        row = winner_rows.iloc[start : start + 2]
        columns = st.columns(len(row))
        for column, (_, trial_row) in zip(columns, row.iterrows(), strict=True):
            trial = trial_lookup[str(trial_row["trial_key"])]
            with column:
                render_layout_summary(trial)


def render_multi_seed_results(bundle: dict[str, object], seed_step: int) -> None:
    aggregate_df = bundle["aggregate_df"]
    stability_df = bundle["stability_df"]
    significance_df = bundle["significance_df"]
    best_progression_df = bundle["best_progression_df"]
    method_progression_df = bundle["method_progression_df"]
    per_seed_df = bundle["per_seed_df"]
    seed_values = bundle["seed_values"]
    reuse_optimized_layouts_across_seeds = bool(bundle.get("reuse_optimized_layouts_across_seeds", False))
    optimized_reference_seed = bundle.get("optimized_reference_seed")
    aggregate_display_df = make_multi_seed_aggregate_display_frame(aggregate_df)
    stability_display_df = make_multi_seed_stability_display_frame(stability_df)
    significance_display_df = make_multi_seed_significance_display_frame(significance_df)
    best_progression_display_df = make_best_progression_display_frame(best_progression_df)
    method_progression_display_df = make_method_progression_display_frame(method_progression_df)

    st.subheader("Multi-Seed Significance")
    caption = (
        f"Trials use seeds {seed_values[0]} through {seed_values[-1]} with a step of {seed_step}. "
        "Each trial reruns the full layout comparison on a new train/eval scenario draw."
    )
    has_optimized_trials = not per_seed_df.empty and "Optimized" in set(per_seed_df["method"].dropna().astype(str))
    if has_optimized_trials:
        if reuse_optimized_layouts_across_seeds:
            reference_text = (
                f"reference seed {int(optimized_reference_seed)}"
                if optimized_reference_seed is not None
                else "one shared reference run"
            )
            caption += f" Optimized layouts were frozen from {reference_text} and reused across all seeds."
        else:
            caption += " Optimized layouts were re-optimized independently for each seed."
    st.caption(caption)

    if not stability_display_df.empty:
        st.dataframe(stability_display_df, use_container_width=True, hide_index=True)
    if not significance_display_df.empty:
        st.dataframe(significance_display_df, use_container_width=True, hide_index=True)
    if not best_progression_display_df.empty:
        st.markdown("**Best-Available Progression By Sensor Count**")
        st.caption(
            "Compares the best-performing layout available at each adjacent sensor count. "
            "Positive score delta means the higher sensor count performed better."
        )
        st.dataframe(best_progression_display_df, use_container_width=True, hide_index=True)
    if not method_progression_display_df.empty:
        st.markdown("**Per-Method Progression By Sensor Count**")
        st.caption(
            "Compares adjacent sensor counts within the same layout family, using the same seed-by-seed reruns."
        )
        st.dataframe(method_progression_display_df, use_container_width=True, hide_index=True)
    if not aggregate_display_df.empty:
        st.dataframe(aggregate_display_df, use_container_width=True, hide_index=True)

    if not per_seed_df.empty:
        st.download_button(
            "Download multi-seed CSV",
            data=per_seed_df.to_csv(index=False).encode("utf-8"),
            file_name="monte_carlo_multi_seed_results.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(
        page_title="Monte Carlo Layout Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    restored_state = load_layout_lab_state()
    restored_settings = dict(restored_state["experiment_settings"])
    restored_experiment_bundle, restored_multi_seed_bundle = load_layout_lab_results()
    if "mc_layout_lab_settings" not in st.session_state:
        st.session_state["mc_layout_lab_settings"] = json.dumps(restored_settings, sort_keys=True)
    if "mc_layout_lab_bundle" not in st.session_state:
        st.session_state["mc_layout_lab_bundle"] = restored_experiment_bundle
    if "mc_layout_lab_multi_seed_bundle" not in st.session_state:
        st.session_state["mc_layout_lab_multi_seed_bundle"] = restored_multi_seed_bundle
    if "mc_layout_lab_restored" not in st.session_state:
        st.session_state["mc_layout_lab_restored"] = LAST_RUN_RESULTS_PATH.exists() or LAST_RUN_STATE_PATH.exists()
    if "mc_layout_lab_multi_seed_config" not in st.session_state:
        st.session_state["mc_layout_lab_multi_seed_config"] = {
            "enable_multi_seed": bool(restored_state.get("enable_multi_seed", False)),
            "significance_trials": int(restored_state.get("significance_trials", 10)),
            "seed_step": int(restored_state.get("seed_step", 1)),
            "reuse_optimized_layouts_across_seeds": bool(restored_state.get("reuse_optimized_layouts_across_seeds", False)),
        }
    try:
        committed_settings = json.loads(st.session_state["mc_layout_lab_settings"])
    except json.JSONDecodeError:
        committed_settings = restored_settings
    committed_settings = dict(committed_settings)
    committed_multi_seed_defaults = dict(st.session_state["mc_layout_lab_multi_seed_config"])

    st.title("Monte Carlo Layout Lab")
    st.caption(
        "Compare sensor arrangements under the repo's Fisher-information scoring model. "
        "Higher held-out log det(F_direction) is better, and valid fraction shows how often a layout remains observable."
    )
    if st.session_state.pop("mc_layout_lab_restored", False):
        if st.session_state.get("mc_layout_lab_bundle") is not None:
            st.info("Restored the last Monte Carlo experiment after the session reset.")
        else:
            st.info("Restored the last Monte Carlo settings after the session reset. Press Run to execute them again.")

    st.sidebar.header("Experiment")
    domain_options = ["single_face", "multi_face_per_face"]
    domain_default = str(committed_settings.get("domain_mode", "single_face"))
    if domain_default not in domain_options:
        domain_default = "single_face"
    domain_mode = st.sidebar.selectbox(
        "Domain",
        options=domain_options,
        index=domain_options.index(domain_default),
        format_func=lambda value: "Single face" if value == "single_face" else "Five-panel set (counts are per face)",
    )
    counts_label = "Sensor counts" if domain_mode == "single_face" else "Sensors per face"
    counts_help = (
        "Comma-separated total sensor counts for one face."
        if domain_mode == "single_face"
        else "Comma-separated counts per face. Total sensors = count x 5."
    )
    counts_text = st.sidebar.text_input(counts_label, value=str(committed_settings.get("counts_text", DEFAULT_COUNTS_TEXT)), help=counts_help)
    pattern_defaults = [pattern for pattern in committed_settings.get("patterns", list(DEFAULT_PATTERNS)) if pattern in DEFAULT_PATTERNS]
    if not pattern_defaults:
        pattern_defaults = list(DEFAULT_PATTERNS)
    patterns = st.sidebar.multiselect(
        "Heuristic baselines",
        options=["grid", "rings", "rect", "random"],
        default=pattern_defaults,
        format_func=lambda value: PATTERN_LABELS[value],
    )
    include_optimized = st.sidebar.checkbox("Include optimized layout", value=bool(committed_settings.get("include_optimized", True)))

    with st.sidebar.expander("Geometry", expanded=True):
        width_m = st.number_input("Face width (m)", min_value=0.05, max_value=1.0, value=float(committed_settings.get("width_m", 0.30)), step=0.01)
        height_m = st.number_input("Face height (m)", min_value=0.05, max_value=1.0, value=float(committed_settings.get("height_m", 0.20)), step=0.01)
        multi_face_offset_m = st.number_input("Panel offset (m)", min_value=0.01, max_value=0.5, value=float(committed_settings.get("multi_face_offset_m", 0.10)), step=0.01)
        minimum_spacing_m = st.number_input("Minimum spacing (m)", min_value=0.0, max_value=0.2, value=float(committed_settings.get("minimum_spacing_m", 0.01)), step=0.005)

    with st.sidebar.expander("Monte Carlo budget", expanded=True):
        train_samples = st.number_input("Train samples", min_value=16, max_value=2000, value=int(committed_settings.get("train_samples", 64)), step=16)
        eval_samples = st.number_input("Eval samples", min_value=32, max_value=5000, value=int(committed_settings.get("eval_samples", 256)), step=32)
        min_active = st.number_input("Min active sensors", min_value=3, max_value=50, value=int(committed_settings.get("min_active", 6)), step=1)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=int(committed_settings.get("seed", 7)), step=1)
        eval_seed_offset = st.number_input("Eval seed offset", min_value=1, max_value=1_000_000, value=int(committed_settings.get("eval_seed_offset", 10000)), step=1)
        generations = st.number_input("Optimizer generations", min_value=1, max_value=100, value=int(committed_settings.get("generations", 8)), step=1)
        popsize = st.number_input("Optimizer population size", min_value=2, max_value=30, value=int(committed_settings.get("popsize", 4)), step=1)
        optimization_target = st.selectbox(
            "Optimize for",
            options=list(OPTIMIZATION_TARGET_LABELS.keys()),
            index=list(OPTIMIZATION_TARGET_LABELS.keys()).index(str(committed_settings.get("optimization_target", "logdet")) if str(committed_settings.get("optimization_target", "logdet")) in OPTIMIZATION_TARGET_LABELS else "logdet"),
            format_func=lambda value: OPTIMIZATION_TARGET_LABELS[value],
            help="Choose the training objective used when generating the Optimized layout.",
        )

    with st.sidebar.expander("Multi-seed significance", expanded=False):
        enable_multi_seed = st.checkbox("Enable multi-seed analysis", value=bool(committed_multi_seed_defaults.get("enable_multi_seed", False)))
        significance_trials = st.number_input("Trials (seeds)", min_value=2, max_value=100, value=int(committed_multi_seed_defaults.get("significance_trials", 10)), step=1)
        seed_step = st.number_input("Seed step", min_value=1, max_value=100000, value=int(committed_multi_seed_defaults.get("seed_step", 1)), step=1)
        reuse_optimized_layouts_across_seeds = st.checkbox(
            "Reuse one optimized layout set across all seeds",
            value=bool(committed_multi_seed_defaults.get("reuse_optimized_layouts_across_seeds", False)),
            help=(
                "When enabled, the app optimizes once using the reference Seed and then evaluates that same optimized "
                "layout set against every multi-seed rerun."
            ),
        )

    with st.sidebar.expander("Beam and scenario", expanded=False):
        range_min_m = st.number_input("Range min (m)", min_value=10.0, max_value=10000.0, value=float(committed_settings.get("range_min_m", 500.0)), step=50.0)
        range_max_m = st.number_input("Range max (m)", min_value=20.0, max_value=20000.0, value=float(committed_settings.get("range_max_m", 3000.0)), step=50.0)
        angle_max_deg = st.slider("Max incidence angle (deg)", min_value=5, max_value=85, value=int(committed_settings.get("angle_max_deg", 60)), step=1)
        wavelength_nm = st.number_input("Wavelength (nm)", min_value=355.0, max_value=2000.0, value=float(committed_settings.get("wavelength_nm", 1064.0)), step=1.0)
        m2 = st.number_input("Beam quality M2", min_value=1.0, max_value=5.0, value=float(committed_settings.get("m2", DEFAULT_M2)), step=0.05)
        beam_waist_m = st.number_input("Beam waist (m)", min_value=0.001, max_value=0.50, value=float(committed_settings.get("beam_waist_m", DEFAULT_BEAM_WAIST_M)), step=0.001, format="%.3f")
        power_w = st.number_input("Power (W)", min_value=1.0, max_value=1000000.0, value=float(committed_settings.get("power_w", DEFAULT_POWER_W)), step=100.0)
        visibility_km = st.number_input("Visibility (km)", min_value=0.1, max_value=100.0, value=float(committed_settings.get("visibility_km", DEFAULT_VISIBILITY_KM)), step=0.5)
        cn2 = st.number_input("Cn2", min_value=0.0, max_value=1e-10, value=float(committed_settings.get("cn2", DEFAULT_CN2)), step=1e-15, format="%.2e")
        uniform_solid_angle = st.checkbox("Sample uniformly in solid angle", value=bool(committed_settings.get("uniform_solid_angle", True)))

    with st.sidebar.expander("Sensor model", expanded=False):
        noise_floor = st.number_input("Noise floor (W/m^2)", min_value=0.0, max_value=1e6, value=float(committed_settings.get("noise_floor", 1.0)), step=0.1)
        relative_noise = st.number_input("Relative noise fraction", min_value=0.0, max_value=1.0, value=float(committed_settings.get("relative_noise", 0.0)), step=0.01)
        optical_density = st.number_input("Optical density", min_value=0.0, max_value=10.0, value=float(committed_settings.get("optical_density", 0.0)), step=0.1)
        saturation_limit = st.number_input(
            "Saturation limit (W/m^2)",
            min_value=1.0,
            max_value=1e12,
            value=float(committed_settings.get("saturation_limit", 1e12)),
            step=1000.0,
            format="%.3e",
        )
        require_above_noise = st.checkbox("Require above-noise sensors", value=bool(committed_settings.get("require_above_noise", False)))

    current_settings = {
        "domain_mode": domain_mode,
        "counts_text": counts_text,
        "patterns": patterns,
        "include_optimized": include_optimized,
        "width_m": width_m,
        "height_m": height_m,
        "multi_face_offset_m": multi_face_offset_m,
        "minimum_spacing_m": minimum_spacing_m,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "min_active": min_active,
        "seed": seed,
        "eval_seed_offset": eval_seed_offset,
        "generations": generations,
        "popsize": popsize,
        "optimization_target": optimization_target,
        "range_min_m": range_min_m,
        "range_max_m": range_max_m,
        "angle_max_deg": angle_max_deg,
        "wavelength_nm": wavelength_nm,
        "m2": m2,
        "beam_waist_m": beam_waist_m,
        "power_w": power_w,
        "visibility_km": visibility_km,
        "cn2": cn2,
        "uniform_solid_angle": uniform_solid_angle,
        "noise_floor": noise_floor,
        "relative_noise": relative_noise,
        "optical_density": optical_density,
        "saturation_limit": saturation_limit,
        "require_above_noise": require_above_noise,
    }
    current_multi_seed_config = {
        "enable_multi_seed": bool(enable_multi_seed),
        "significance_trials": int(significance_trials),
        "seed_step": int(seed_step),
        "reuse_optimized_layouts_across_seeds": bool(reuse_optimized_layouts_across_seeds),
    }

    run_clicked = st.sidebar.button("Run Monte Carlo comparison", type="primary")

    if run_clicked:
        if not patterns and not include_optimized:
            st.sidebar.error("Choose at least one heuristic baseline or include the optimized layout.")
        elif include_optimized and str(current_settings["optimization_target"]) == "error" and str(current_settings["domain_mode"]) != "single_face":
            st.sidebar.error("The valid gamma error optimization target is only available in Single face mode.")
        else:
            settings_json = json.dumps(current_settings, sort_keys=True)
            try:
                with st.spinner("Running Monte Carlo layout comparison..."):
                    bundle = run_experiment(settings_json)
                    multi_seed_bundle = None
                    if bool(current_multi_seed_config["enable_multi_seed"]):
                        multi_seed_bundle = run_multi_seed_experiment(
                            settings_json,
                            int(current_multi_seed_config["significance_trials"]),
                            int(current_multi_seed_config["seed_step"]),
                            bool(current_multi_seed_config["reuse_optimized_layouts_across_seeds"]),
                        )
            except Exception as exc:
                st.error(f"Monte Carlo run failed: {exc}")
                return
            st.session_state["mc_layout_lab_settings"] = settings_json
            st.session_state["mc_layout_lab_multi_seed_config"] = current_multi_seed_config
            st.session_state["mc_layout_lab_bundle"] = bundle
            st.session_state["mc_layout_lab_multi_seed_bundle"] = multi_seed_bundle
            save_layout_lab_state(
                current_settings,
                bool(current_multi_seed_config["enable_multi_seed"]),
                int(current_multi_seed_config["significance_trials"]),
                int(current_multi_seed_config["seed_step"]),
                bool(current_multi_seed_config["reuse_optimized_layouts_across_seeds"]),
            )
            save_layout_lab_results(bundle, multi_seed_bundle)

    settings_json = st.session_state.get("mc_layout_lab_settings")
    committed_multi_seed = dict(
        st.session_state.get(
            "mc_layout_lab_multi_seed_config",
            {
                "enable_multi_seed": False,
                "significance_trials": 10,
                "seed_step": 1,
                "reuse_optimized_layouts_across_seeds": False,
            },
        )
    )
    bundle = st.session_state.get("mc_layout_lab_bundle")
    multi_seed_bundle = st.session_state.get("mc_layout_lab_multi_seed_bundle")
    if bundle is None:
        st.info("Choose a domain and counts in the sidebar, then run the comparison.")
        return

    pending_settings = settings_json != json.dumps(current_settings, sort_keys=True)
    pending_multi_seed = committed_multi_seed != current_multi_seed_config
    if pending_settings or pending_multi_seed:
        st.warning("Sidebar changes are pending. Results below still reflect the last time you pressed Run.")

    summary_df = bundle["summary_df"]
    winner_df = bundle["winner_df"]
    trial_lookup = bundle["trial_lookup"]
    trial_summary_display_df = make_trial_summary_display_frame(summary_df)
    winner_display_df = make_winner_display_frame(winner_df)

    best_row = summary_df.sort_values("eval_mean_logdet", ascending=False).iloc[0]
    average_valid_fraction = float(summary_df["eval_valid_fraction"].mean())
    metric_columns = st.columns(3)
    metric_columns[0].metric("Best arrangement", str(best_row["method"]))
    metric_columns[1].metric("Best held-out score", f"{float(best_row['eval_mean_logdet']):.3e}")
    metric_columns[2].metric("Average valid fraction", f"{average_valid_fraction:.3f}")

    chart_columns = st.columns(4)
    chart_columns[0].plotly_chart(
        make_score_chart(summary_df),
        use_container_width=True,
        key="score_chart",
    )
    chart_columns[1].plotly_chart(
        make_validity_chart(summary_df),
        use_container_width=True,
        key="validity_chart",
    )
    chart_columns[2].plotly_chart(
        make_gamma_error_chart(summary_df),
        use_container_width=True,
        key="gamma_error_chart",
    )
    chart_columns[3].plotly_chart(
        make_any_active_chart(summary_df),
        use_container_width=True,
        key="any_active_chart",
    )

    st.subheader("Per-count winners")
    st.dataframe(winner_display_df, use_container_width=True, hide_index=True)

    st.subheader("Trial summary")
    st.dataframe(trial_summary_display_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download summary CSV",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="monte_carlo_layout_summary.csv",
        mime="text/csv",
    )

    render_winner_gallery(summary_df, trial_lookup)

    if bool(committed_multi_seed["enable_multi_seed"]) and multi_seed_bundle is not None:
        render_multi_seed_results(multi_seed_bundle, int(committed_multi_seed["seed_step"]))

    trial_options = summary_df["trial_key"].tolist()
    default_trial_key = str(best_row["trial_key"])
    if st.session_state.get("selected_trial_key") not in trial_options:
        st.session_state["selected_trial_key"] = default_trial_key
    selected_trial_key = st.selectbox(
        "Inspect a layout",
        options=trial_options,
        key="selected_trial_key",
        format_func=lambda key: (
            f"{trial_lookup[key]['method']} | total sensors={trial_lookup[key]['sensor_count']} | {key}"
        ),
    )
    selected_trial = trial_lookup[selected_trial_key]
    layout_records = list(selected_trial["layout"])
    st.plotly_chart(
        make_layout_plot(layout_records),
        use_container_width=True,
        key=f"inspect_layout_plot_{selected_trial_key}",
    )

    layout_df = pd.DataFrame(layout_records)
    st.dataframe(
        layout_df[
            [
                "sensor",
                "face",
                "local_u_cm",
                "local_v_cm",
                "x_m",
                "y_m",
                "z_m",
                "normal_x",
                "normal_y",
                "normal_z",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("How to read these results"):
        st.markdown(
            """
- `Eval mean log det(F_direction)` is the held-out Monte Carlo objective. Higher is better.
- `Eval mean valid log det(F_direction)` is the same score averaged only over valid held-out scenarios. Higher is better.
- `Eval valid fraction` is the share of held-out scenarios that still have enough active sensors to score.
- `Eval mean valid gamma error (deg)` is the mean reconstructed gamma error over the held-out scenarios where a valid single-face gamma estimate was available. Lower is better.
- Compare methods at the same total sensor count to isolate arrangement effects.
- In the multi-seed section, positive `score` and `mean_valid` deltas mean the first item improved, while positive `mean_error_reduction_deg` means it lowered angular error.
- In the five-panel mode, the count input is per face, so the plotted total sensors are `count x 5`.
- Angle sampling spans the full azimuth range and is converted into signed pitch/yaw tilts, so both positive and negative tilt directions are included.
            """.strip()
        )


if __name__ == "__main__":
    main()
