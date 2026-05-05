"""Beam footprint and direction reconstruction from photodiode readings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


SensorCoordinate = tuple[float, float]


@dataclass(frozen=True)
class LogLinearEstimate:
    """Minimum viable estimate available from three or more sensors."""

    centroid_x_m: float
    centroid_y_m: float
    gradient_x_1_per_m: float
    gradient_y_1_per_m: float


@dataclass(frozen=True)
class EllipseReconstruction:
    """Full log-quadratic Gaussian ellipse reconstruction."""

    center_x_m: float
    center_y_m: float
    semi_major_axis_m: float
    semi_minor_axis_m: float
    ellipse_angle_rad: float
    incidence_angle_rad: float
    pitch_rad: float
    yaw_rad: float
    unit_vector: tuple[float, float, float]
    condition_number: float
    sigma_pitch_rad: float = 0.0
    sigma_yaw_rad: float = 0.0


def _as_arrays(
    coordinates_m: Iterable[SensorCoordinate],
    irradiance_w_per_m2: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.array(tuple(coordinates_m), dtype=float)
    readings = np.array(tuple(irradiance_w_per_m2), dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates_m must contain (x, y) pairs")
    if readings.ndim != 1 or readings.shape[0] != coordinates.shape[0]:
        raise ValueError("irradiance_w_per_m2 must match coordinate count")
    if coordinates.shape[0] < 3:
        raise ValueError("At least 3 sensors are required")
    return coordinates, readings


def log_gaussian_parameters(
    center_x_m: float,
    center_y_m: float,
    log_peak: float,
    log_base_radius: float,
    gamma_rad: float,
    psi_rad: float,
    coordinates_m: np.ndarray,
) -> np.ndarray:
    """Evaluate the compact physical log-Gaussian beam model at sensor points."""
    dx = coordinates_m[:, 0] - center_x_m
    dy = coordinates_m[:, 1] - center_y_m
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    cos_gamma = np.cos(gamma_rad)
    u = cos_psi * dx + sin_psi * dy
    v = -sin_psi * dx + cos_psi * dy
    base_radius_m = np.exp(log_base_radius)
    shape = (cos_gamma**2) * u**2 + v**2
    return log_peak - 2.0 * shape / (base_radius_m**2)


def log_gaussian_jacobian(
    params: np.ndarray,
    coordinates_m: np.ndarray,
) -> np.ndarray:
    """Analytic Jacobian of the physical log-Gaussian beam model."""
    x0, y0, log_peak, log_base_radius, gamma_rad, psi_rad = params
    dx = coordinates_m[:, 0] - x0
    dy = coordinates_m[:, 1] - y0
    cos_psi = np.cos(psi_rad)
    sin_psi = np.sin(psi_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    base_radius_m = np.exp(log_base_radius)
    inv_r2 = 1.0 / (base_radius_m**2)

    u = cos_psi * dx + sin_psi * dy
    v = -sin_psi * dx + cos_psi * dy
    shape = (cos_gamma**2) * u**2 + v**2

    jacobian = np.empty((coordinates_m.shape[0], 6), dtype=float)
    jacobian[:, 0] = 4.0 * inv_r2 * ((cos_gamma**2) * cos_psi * u - sin_psi * v)
    jacobian[:, 1] = 4.0 * inv_r2 * ((cos_gamma**2) * sin_psi * u + cos_psi * v)
    jacobian[:, 2] = 1.0
    jacobian[:, 3] = 4.0 * shape * inv_r2
    jacobian[:, 4] = 4.0 * cos_gamma * sin_gamma * u**2 * inv_r2
    jacobian[:, 5] = 4.0 * (sin_gamma**2) * u * v * inv_r2
    return jacobian


def infer_angle_uncertainty(
    reconstruction: EllipseReconstruction,
    coordinates_m: np.ndarray,
    irradiance_w_per_m2: np.ndarray,
    sigma_noise_w_per_m2: float,
) -> tuple[float, float, np.ndarray]:
    """Compute CRLB-style angle uncertainty from the localized Fisher matrix."""
    if coordinates_m.shape[0] < 6:
        return 0.0, 0.0, np.zeros((2, 2), dtype=float)
    positive = np.clip(np.asarray(irradiance_w_per_m2, dtype=float), 1e-300, None)
    if sigma_noise_w_per_m2 <= 0.0:
        sigma_noise_w_per_m2 = float(np.std(positive) if np.std(positive) > 0 else 1.0)

    params = np.array(
        [
            reconstruction.center_x_m,
            reconstruction.center_y_m,
            float(np.log(np.max(positive))),
            float(np.log(max(reconstruction.semi_minor_axis_m, 1e-12))),
            float(reconstruction.incidence_angle_rad),
            float(reconstruction.ellipse_angle_rad),
        ],
        dtype=float,
    )
    jacobian = log_gaussian_jacobian(params, coordinates_m)
    weights = np.full(coordinates_m.shape[0], 1.0 / (sigma_noise_w_per_m2**2), dtype=float)
    fisher = jacobian.T @ (weights[:, None] * jacobian)
    covariance = np.linalg.pinv(fisher)

    gamma = float(reconstruction.incidence_angle_rad)
    psi = float(reconstruction.ellipse_angle_rad)
    tan_gamma = float(np.tan(gamma))
    sec2_gamma = 1.0 / (np.cos(gamma) ** 2)
    denom_pitch = 1.0 + (tan_gamma * np.sin(psi)) ** 2
    denom_yaw = 1.0 + (tan_gamma * np.cos(psi)) ** 2
    transform = np.array(
        [
            [sec2_gamma * np.sin(psi) / denom_pitch, tan_gamma * np.cos(psi) / denom_pitch],
            [sec2_gamma * np.cos(psi) / denom_yaw, -tan_gamma * np.sin(psi) / denom_yaw],
        ],
        dtype=float,
    )
    angle_cov = transform @ covariance[np.ix_([4, 5], [4, 5])] @ transform.T
    sigma_pitch = float(np.sqrt(max(angle_cov[0, 0], 0.0)))
    sigma_yaw = float(np.sqrt(max(angle_cov[1, 1], 0.0)))
    return sigma_pitch, sigma_yaw, angle_cov


def estimate_log_linear_gradient(
    coordinates_m: Iterable[SensorCoordinate],
    irradiance_w_per_m2: Iterable[float],
    intensity_floor: float = 1e-300,
) -> LogLinearEstimate:
    """Estimate beam centroid and local log-intensity gradient for N >= 3.

    This is intentionally not a full incidence-angle reconstruction. It is a
    stable fallback when there are too few sensors for the primer's ellipse fit.
    """
    coordinates, readings = _as_arrays(coordinates_m, irradiance_w_per_m2)
    positive = np.clip(readings, intensity_floor, None)
    weights = positive / np.sum(positive)
    centroid = weights @ coordinates

    design = np.column_stack([np.ones(coordinates.shape[0]), coordinates[:, 0], coordinates[:, 1]])
    coeffs, *_ = np.linalg.lstsq(design, np.log(positive), rcond=None)
    return LogLinearEstimate(
        centroid_x_m=float(centroid[0]),
        centroid_y_m=float(centroid[1]),
        gradient_x_1_per_m=float(coeffs[1]),
        gradient_y_1_per_m=float(coeffs[2]),
    )


def fit_log_quadratic_ellipse(
    coordinates_m: Iterable[SensorCoordinate],
    irradiance_w_per_m2: Iterable[float],
    intensity_floor: float = 1e-300,
    sigma_noise_w_per_m2: float | None = None,
) -> EllipseReconstruction:
    """Fit the primer's log-quadratic Gaussian model and recover direction.

    Model:
        ln I = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2

    This unconstrained linear fit has six coefficients, so it requires at least
    six independent positive readings. With fewer sensors the inverse problem is
    underdetermined unless additional physical constraints are imposed.
    """
    coordinates, readings = _as_arrays(coordinates_m, irradiance_w_per_m2)
    if coordinates.shape[0] < 6:
        raise ValueError("Full ellipse reconstruction requires at least 6 sensors")

    positive = np.clip(readings, intensity_floor, None)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    design = np.column_stack([np.ones_like(x), x, y, x**2, x * y, y**2])
    coeffs, *_ = np.linalg.lstsq(design, np.log(positive), rcond=None)
    _, c1, c2, c3, c4, c5 = coeffs

    quadratic_matrix = np.array([[-2.0 * c3, -c4], [-c4, -2.0 * c5]], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(quadratic_matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Fitted footprint is not a valid positive-definite Gaussian ellipse")

    linear = np.array([c1, c2], dtype=float)
    center = np.linalg.solve(quadratic_matrix, linear)

    lambda_min_index = int(np.argmin(eigenvalues))
    lambda_max_index = int(np.argmax(eigenvalues))
    lambda_min = float(eigenvalues[lambda_min_index])
    lambda_max = float(eigenvalues[lambda_max_index])
    semi_major_axis = 1.0 / np.sqrt(lambda_min)
    semi_minor_axis = 1.0 / np.sqrt(lambda_max)

    major_vector = eigenvectors[:, lambda_min_index]
    if major_vector[0] < 0:
        major_vector = -major_vector
    ellipse_angle = float(np.arctan2(major_vector[1], major_vector[0]))

    aspect_ratio = float(np.clip(semi_minor_axis / semi_major_axis, 0.0, 1.0))
    incidence_angle = float(np.arccos(aspect_ratio))
    tan_gamma = float(np.tan(incidence_angle))
    yaw_rad = float(np.arctan(tan_gamma * np.cos(ellipse_angle)))
    pitch_rad = float(np.arctan(tan_gamma * np.sin(ellipse_angle)))
    unit_vector = (
        float(np.sqrt(max(0.0, 1.0 - aspect_ratio**2)) * np.cos(ellipse_angle)),
        float(np.sqrt(max(0.0, 1.0 - aspect_ratio**2)) * np.sin(ellipse_angle)),
        aspect_ratio,
    )

    sigma_pitch_rad = 0.0
    sigma_yaw_rad = 0.0
    if sigma_noise_w_per_m2 is not None and sigma_noise_w_per_m2 > 0.0:
        sigma_pitch_rad, sigma_yaw_rad, _ = infer_angle_uncertainty(
            reconstruction=EllipseReconstruction(
                center_x_m=float(center[0]),
                center_y_m=float(center[1]),
                semi_major_axis_m=float(semi_major_axis),
                semi_minor_axis_m=float(semi_minor_axis),
                ellipse_angle_rad=ellipse_angle,
                incidence_angle_rad=incidence_angle,
                pitch_rad=pitch_rad,
                yaw_rad=yaw_rad,
                unit_vector=unit_vector,
                condition_number=float(np.linalg.cond(design)),
            ),
            coordinates_m=coordinates,
            irradiance_w_per_m2=positive,
            sigma_noise_w_per_m2=sigma_noise_w_per_m2,
        )

    return EllipseReconstruction(
        center_x_m=float(center[0]),
        center_y_m=float(center[1]),
        semi_major_axis_m=float(semi_major_axis),
        semi_minor_axis_m=float(semi_minor_axis),
        ellipse_angle_rad=ellipse_angle,
        incidence_angle_rad=incidence_angle,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        unit_vector=unit_vector,
        condition_number=float(np.linalg.cond(design)),
        sigma_pitch_rad=sigma_pitch_rad,
        sigma_yaw_rad=sigma_yaw_rad,
    )
