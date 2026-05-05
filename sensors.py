"""Photodiode layout loading and irradiance sampling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from physics import (
    SensorPose3D,
    calculate_cuboid_sensor_irradiance,
    calculate_incidence_angle,
)


SensorCoordinate = tuple[float, float]
DEFAULT_LAYOUTS_PATH = Path(__file__).with_name("layouts.json")
DEFAULT_SENSOR_NORMAL = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PhotodiodeArray:
    """Discrete photodiode poses on a cuboid body."""

    poses_m: tuple[SensorPose3D, ...]

    @classmethod
    def from_iterable(cls, coordinates: Iterable[object]) -> "PhotodiodeArray":
        return cls(tuple(_coerce_pose(item) for item in coordinates))

    @property
    def coordinates_m(self) -> tuple[SensorCoordinate, ...]:
        return tuple((pose.position_m[0], pose.position_m[1]) for pose in self.poses_m)

    @property
    def positions_m(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(pose.position_m for pose in self.poses_m)

    @property
    def normals_m(self) -> tuple[tuple[float, float, float], ...]:
        return tuple(pose.normal_m for pose in self.poses_m)


def _coerce_pose(item: object) -> SensorPose3D:
    if isinstance(item, SensorPose3D):
        return item
    if isinstance(item, dict):
        position = item.get("position", item.get("position_m"))
        normal = item.get("normal", item.get("normal_m", DEFAULT_SENSOR_NORMAL))
        face_name = str(item.get("face", item.get("face_name", "")))
        if position is None:
            raise ValueError("pose dictionary must include a position")
        position_tuple = tuple(float(value) for value in position)
        if len(position_tuple) == 2:
            position_tuple = (position_tuple[0], position_tuple[1], 0.0)
        if len(position_tuple) != 3:
            raise ValueError("pose position must contain 2 or 3 values")
        normal_tuple = tuple(float(value) for value in normal)
        if len(normal_tuple) != 3:
            raise ValueError("pose normal must contain 3 values")
        return SensorPose3D(position_m=position_tuple, normal_m=normal_tuple, face_name=face_name)
    values = tuple(float(value) for value in item)  # type: ignore[arg-type]
    if len(values) == 2:
        return SensorPose3D(position_m=(values[0], values[1], 0.0), normal_m=DEFAULT_SENSOR_NORMAL, face_name="+z")
    if len(values) == 3:
        return SensorPose3D(position_m=(values[0], values[1], values[2]), normal_m=DEFAULT_SENSOR_NORMAL, face_name="+z")
    raise ValueError("sensor pose must contain 2, 3, or structured values")


def _pose_to_record(pose: SensorPose3D) -> dict[str, object]:
    return {
        "position": [float(value) for value in pose.position_m],
        "normal": [float(value) for value in pose.normal_m],
        "face": pose.face_name,
    }


def _normalize_layout_catalog(raw_catalog: dict) -> dict[str, dict[int, PhotodiodeArray]]:
    catalog: dict[str, dict[int, PhotodiodeArray]] = {}
    for layout_name, layouts_by_count in raw_catalog.items():
        if not isinstance(layouts_by_count, dict):
            continue
        catalog[layout_name] = {}
        for count_key, coordinates in layouts_by_count.items():
            try:
                sensor_count = int(count_key)
            except (TypeError, ValueError):
                continue
            if not isinstance(coordinates, list):
                continue
            catalog[layout_name][sensor_count] = PhotodiodeArray.from_iterable(coordinates)
    return catalog


def load_layouts_catalog(layouts_path: str | Path = DEFAULT_LAYOUTS_PATH) -> dict[str, dict[int, PhotodiodeArray]]:
    """Load named sensor layouts from layouts.json."""
    path = Path(layouts_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        raw_catalog = json.load(handle)
    if not isinstance(raw_catalog, dict):
        return {}
    return _normalize_layout_catalog(raw_catalog)


def save_layouts_catalog(
    catalog: dict[str, dict[int, PhotodiodeArray]],
    layouts_path: str | Path = DEFAULT_LAYOUTS_PATH,
) -> None:
    """Persist a layout catalog to layouts.json."""
    path = Path(layouts_path)
    serializable: dict[str, dict[str, list[dict[str, object]]]] = {}
    for layout_name, layouts_by_count in catalog.items():
        serializable[layout_name] = {}
        for sensor_count, photodiodes in layouts_by_count.items():
            serializable[layout_name][str(int(sensor_count))] = [_pose_to_record(pose) for pose in photodiodes.poses_m]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def available_layouts(layouts_path: str | Path = DEFAULT_LAYOUTS_PATH) -> list[str]:
    """Return layout families present in layouts.json."""
    return sorted(load_layouts_catalog(layouts_path))


def available_layout_counts(layout_name: str, layouts_path: str | Path = DEFAULT_LAYOUTS_PATH) -> list[int]:
    """Return available sensor counts for a layout family."""
    catalog = load_layouts_catalog(layouts_path)
    return sorted(catalog.get(layout_name, {}))


def get_layout(
    layout_name: str,
    sensor_count: int,
    layouts_path: str | Path = DEFAULT_LAYOUTS_PATH,
) -> PhotodiodeArray:
    """Load a specific layout family/count pair from layouts.json."""
    catalog = load_layouts_catalog(layouts_path)
    layouts_by_count = catalog.get(layout_name, {})
    if sensor_count not in layouts_by_count:
        valid_counts = ", ".join(str(count) for count in sorted(layouts_by_count))
        raise KeyError(
            f"Layout {layout_name!r} does not contain sensor_count={sensor_count}. "
            f"Available counts: {valid_counts or 'none'}"
        )
    return layouts_by_count[sensor_count]


def sample_irradiance_nearest(
    photodiodes: PhotodiodeArray | Iterable[SensorCoordinate],
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    irradiance_w_per_m2: np.ndarray,
) -> dict[str, float]:
    """Sample irradiance at photodiode locations using nearest mesh nodes."""
    coordinates = photodiodes.coordinates_m if isinstance(photodiodes, PhotodiodeArray) else tuple(
        (float(x), float(y)) for x, y in photodiodes
    )
    if x_grid_m.shape != y_grid_m.shape or x_grid_m.shape != irradiance_w_per_m2.shape:
        raise ValueError("x_grid_m, y_grid_m, and irradiance_w_per_m2 must have matching shapes")

    readings: dict[str, float] = {}
    for index, (sensor_x_m, sensor_y_m) in enumerate(coordinates, start=1):
        squared_distance = (x_grid_m - sensor_x_m) ** 2 + (y_grid_m - sensor_y_m) ** 2
        nearest_index = np.unravel_index(int(np.argmin(squared_distance)), x_grid_m.shape)
        readings[f"Sensor {index} ({sensor_x_m:.3f} m, {sensor_y_m:.3f} m)"] = float(
            irradiance_w_per_m2[nearest_index]
        )
    return readings


def apply_sensor_hardware(
    irradiance_w_per_m2: Iterable[float],
    optical_density: float = 0.0,
    saturation_limit_w_per_m2: float = np.inf,
    noise_floor_w_per_m2: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Apply OD filtering, saturation, and additive Gaussian noise."""
    readings = np.asarray(tuple(irradiance_w_per_m2), dtype=float)
    generator = rng if rng is not None else np.random.default_rng()
    filtered = readings * (10.0 ** (-optical_density))
    clamped = np.minimum(filtered, saturation_limit_w_per_m2)
    if noise_floor_w_per_m2 > 0.0:
        noise = generator.normal(0.0, noise_floor_w_per_m2, size=clamped.shape)
        noisy = clamped + noise
    else:
        noisy = clamped
    noisy = np.clip(noisy, 0.0, None)
    return noisy, {
        "any_saturated": bool(np.any(filtered >= saturation_limit_w_per_m2)),
        "any_below_noise": bool(np.any(filtered < noise_floor_w_per_m2)),
    }


def sample_sensor_readings(
    photodiodes: PhotodiodeArray | Iterable[SensorCoordinate],
    x_grid_m: np.ndarray | None = None,
    y_grid_m: np.ndarray | None = None,
    irradiance_w_per_m2: np.ndarray | None = None,
    pitch_rad: float | None = None,
    yaw_rad: float | None = None,
    theta_fov_rad: float = np.deg2rad(60.0),
    beam_origin_m: np.ndarray | None = None,
    beam_unit_vector: np.ndarray | None = None,
    peak_irradiance_w_per_m2: float | None = None,
    effective_beam_radius_m: float | None = None,
    optical_density: float = 0.0,
    saturation_limit_w_per_m2: float = np.inf,
    noise_floor_w_per_m2: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[dict[str, float], dict[str, bool], dict[str, float]]:
    """Sample a layout and apply FOV, OD, saturation, and noise.

    If 3D beam geometry is provided, the sensor pose is evaluated analytically.
    Otherwise the legacy 2D nearest-grid sampling path is used.
    """
    if isinstance(photodiodes, PhotodiodeArray):
        poses = photodiodes.poses_m
    else:
        poses = tuple(_coerce_pose(item) for item in photodiodes)
        photodiodes = PhotodiodeArray(poses)

    if (
        beam_origin_m is not None
        and beam_unit_vector is not None
        and peak_irradiance_w_per_m2 is not None
        and effective_beam_radius_m is not None
    ):
        positions = np.asarray(photodiodes.positions_m, dtype=float)
        normals = np.asarray(photodiodes.normals_m, dtype=float)
        base_irradiance = calculate_cuboid_sensor_irradiance(
            sensor_positions_m=positions,
            sensor_normals_m=normals,
            beam_origin_m=np.asarray(beam_origin_m, dtype=float),
            beam_unit_vector=np.asarray(beam_unit_vector, dtype=float),
            peak_irradiance_w_per_m2=float(peak_irradiance_w_per_m2),
            effective_beam_radius_m=float(effective_beam_radius_m),
        )
        axis = np.asarray(beam_unit_vector, dtype=float)
        axis = axis / np.linalg.norm(axis)
        cos_gamma = np.clip(np.einsum("ij,j->i", normals, axis), -1.0, 1.0)
        incidence_angles = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        within_fov = incidence_angles <= theta_fov_rad
        measured = np.where(within_fov, base_irradiance, 0.0)
        noisy, flags = apply_sensor_hardware(
            measured,
            optical_density=optical_density,
            saturation_limit_w_per_m2=saturation_limit_w_per_m2,
            noise_floor_w_per_m2=noise_floor_w_per_m2,
            rng=rng,
        )
        readings = {}
        for index, (pose, value) in enumerate(zip(poses, noisy, strict=True), start=1):
            position = pose.position_m
            label = (
                f"Sensor {index} [{pose.face_name}] "
                f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
            )
            readings[label] = float(value)
        flags.update(
            {
                "within_fov": bool(np.all(within_fov)),
                "incidence_angle_rad": float(np.max(incidence_angles) if incidence_angles.size else 0.0),
            }
        )
        nearest = {
            key: float(value)
            for key, value in zip(readings, base_irradiance, strict=True)
        }
        return readings, flags, nearest

    if x_grid_m is None or y_grid_m is None or irradiance_w_per_m2 is None:
        raise ValueError("2D sampling requires x_grid_m, y_grid_m, and irradiance_w_per_m2")
    if pitch_rad is None or yaw_rad is None:
        raise ValueError("2D sampling requires pitch_rad and yaw_rad")

    nearest = sample_irradiance_nearest(photodiodes, x_grid_m, y_grid_m, irradiance_w_per_m2)
    gamma_rad = calculate_incidence_angle(pitch_rad, yaw_rad)
    fov_scale = 0.0 if gamma_rad > theta_fov_rad else float(np.cos(gamma_rad))
    measured = np.asarray(tuple(nearest.values()), dtype=float) * fov_scale
    noisy, flags = apply_sensor_hardware(
        measured,
        optical_density=optical_density,
        saturation_limit_w_per_m2=saturation_limit_w_per_m2,
        noise_floor_w_per_m2=noise_floor_w_per_m2,
        rng=rng,
    )
    readings = dict(zip(nearest.keys(), noisy, strict=True))
    flags.update({"within_fov": bool(gamma_rad <= theta_fov_rad), "incidence_angle_rad": float(gamma_rad)})
    return readings, flags, nearest
