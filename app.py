"""Streamlit dashboard for laser beam sensing and reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from physics import (
    KinematicEngagement,
    calculate_aperture_for_target_spot,
    calculate_beam_wander_variance,
    calculate_turbulence_spread,
    run_continuous_beam_pipeline,
)
from reconstruction import estimate_log_linear_gradient, fit_log_quadratic_ellipse
from sensors import available_layout_counts, get_layout, load_layouts_catalog, sample_sensor_readings


LAYOUTS_PATH = Path(__file__).with_name("layouts.json")
DEFAULT_SURFACE_SIZE_M = 0.6
DEFAULT_TARGET_DIAMETER_M = 0.10
DEFAULT_NOISE_FLOOR_W_PER_M2 = 1e-3
DEFAULT_SATURATION_LIMIT_W_PER_M2 = 1e7


def apply_css() -> None:
    st.markdown(
        """
        <style>
            header, footer, #MainMenu,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            .stDeployButton {display: none !important;}
            .stApp {background: #f4f7fb;}
            * {border-radius: 0 !important; letter-spacing: 0 !important;}
            .block-container {
                max-width: 1520px;
                padding: 0.65rem 1rem 0.8rem 1rem;
            }
            h1 {font-size: 1.25rem !important; margin: 0 !important;}
            h2, h3 {font-size: 0.92rem !important; margin: 0.15rem 0 0.25rem 0 !important;}
            p, label, .stMarkdown, .stCaption {font-size: 0.78rem !important;}
            div[data-testid="stVerticalBlock"] {gap: 0.38rem;}
            div[data-testid="column"] {padding: 0 0.35rem;}
            div[data-testid="stPlotlyChart"] {
                background: white;
                border: 1px solid #d7dde7;
                padding: 0.35rem;
            }
            div[data-testid="stDataFrame"] {
                border: 1px solid #d7dde7;
                overflow: hidden;
                font-size: 0.74rem;
            }
            div[data-testid="stSidebar"] {
                background: #eef3f9;
                border-right: 1px solid #d7dde7;
            }
            div[data-testid="stSidebar"] h2,
            div[data-testid="stSidebar"] h3 {
                font-size: 0.86rem !important;
            }
            div[data-testid="stNumberInput"] input {
                min-height: 1.8rem !important;
                height: 1.8rem !important;
                font-size: 0.78rem !important;
            }
            div[data-testid="stSelectbox"] div[data-baseweb="select"] {
                min-height: 1.8rem !important;
                font-size: 0.78rem !important;
            }
            div[data-testid="stSlider"] {padding-top: 0 !important;}
            .status-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.45rem;
                margin: 0.35rem 0 0.45rem 0;
            }
            .status-cell {
                border: 1px solid #d7dde7;
                background: white;
                padding: 0.38rem 0.5rem;
                min-width: 0;
            }
            .status-label {
                color: #64748b;
                font-size: 0.66rem;
                font-weight: 700;
                text-transform: uppercase;
            }
            .status-value {
                color: #1f2937;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 0.86rem;
                font-weight: 700;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_grid(surface_size_m: float, grid_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    half_width = surface_size_m / 2.0
    x_axis_m = np.linspace(-half_width, half_width, grid_points)
    y_axis_m = np.linspace(-half_width, half_width, grid_points)
    x_grid_m, y_grid_m = np.meshgrid(x_axis_m, y_axis_m)
    return x_axis_m, y_axis_m, x_grid_m, y_grid_m


def sci(value: float, unit: str = "") -> str:
    return f"{value:.3e}{f' {unit}' if unit else ''}"


def status_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"<div class='status-cell'><div class='status-label'>{label}</div><div class='status-value'>{value}</div></div>"
        for label, value in items
    )
    st.markdown(f"<div class='status-strip'>{cells}</div>", unsafe_allow_html=True)


def table_from_pairs(items: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(items, columns=["Metric", "Value"])


def make_readings_table(
    coordinates_m: tuple[tuple[float, float], ...],
    sensor_readings: dict[str, float],
    peak_irradiance_w_per_m2: float,
) -> pd.DataFrame:
    values = list(sensor_readings.values())
    return pd.DataFrame(
        {
            "S": range(1, len(values) + 1),
            "x": [f"{x:.3f}" for x, _ in coordinates_m],
            "y": [f"{y:.3f}" for _, y in coordinates_m],
            "I": [f"{value:.3e}" for value in values],
            "rel": [f"{value / peak_irradiance_w_per_m2:.3e}" for value in values],
        }
    )


def reconstructed_irradiance_map(
    coordinates_m: tuple[tuple[float, float], ...],
    sensor_values: list[float],
    x_grid_m: np.ndarray,
    y_grid_m: np.ndarray,
    fallback_center_m: tuple[float, float],
    fallback_radius_m: float,
    ellipse: object | None,
) -> np.ndarray:
    """Evaluate the fitted log-quadratic footprint on the display grid."""
    coordinates = np.asarray(coordinates_m, dtype=float)
    values = np.clip(np.asarray(sensor_values, dtype=float), 1e-300, None)
    if coordinates.shape[0] >= 6 and ellipse is not None:
        design = np.column_stack(
            [
                np.ones(coordinates.shape[0]),
                coordinates[:, 0],
                coordinates[:, 1],
                coordinates[:, 0] ** 2,
                coordinates[:, 0] * coordinates[:, 1],
                coordinates[:, 1] ** 2,
            ]
        )
        coeffs, *_ = np.linalg.lstsq(design, np.log(values), rcond=None)
        log_map = (
            coeffs[0]
            + coeffs[1] * x_grid_m
            + coeffs[2] * y_grid_m
            + coeffs[3] * x_grid_m**2
            + coeffs[4] * x_grid_m * y_grid_m
            + coeffs[5] * y_grid_m**2
        )
        return np.exp(np.clip(log_map, -700.0, 700.0))

    dx = x_grid_m - fallback_center_m[0]
    dy = y_grid_m - fallback_center_m[1]
    peak = float(np.max(values))
    radius = max(float(fallback_radius_m), 1e-6)
    return peak * np.exp((-2.0 / radius**2) * (dx**2 + dy**2))


def _ellipse_trace(ellipse: object | None) -> tuple[list[float], list[float]]:
    if ellipse is None:
        return [], []
    angles = np.linspace(0.0, 2.0 * np.pi, 181)
    cos_a = np.cos(float(ellipse.ellipse_angle_rad))
    sin_a = np.sin(float(ellipse.ellipse_angle_rad))
    u = float(ellipse.semi_major_axis_m) * np.cos(angles)
    v = float(ellipse.semi_minor_axis_m) * np.sin(angles)
    x = float(ellipse.center_x_m) + cos_a * u - sin_a * v
    y = float(ellipse.center_y_m) + sin_a * u + cos_a * v
    return x.tolist(), y.tolist()


def make_beam_comparison_figure(
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
    truth_w_per_m2: np.ndarray,
    reconstructed_w_per_m2: np.ndarray,
    coordinates_m: tuple[tuple[float, float], ...],
    sensor_readings: dict[str, float],
    actual_center_m: tuple[float, float],
    reconstructed_center_m: tuple[float, float],
    ellipse: object | None,
) -> go.Figure:
    sensor_x = [x for x, _ in coordinates_m]
    sensor_y = [y for _, y in coordinates_m]
    sensor_kw = np.asarray(list(sensor_readings.values())) / 1_000.0
    half_width = max(abs(float(x_axis_m[0])), abs(float(x_axis_m[-1])))

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.05,
        subplot_titles=("Original beam and sensors", "Reconstructed ellipse"),
    )
    fig.add_trace(
        go.Heatmap(
            x=x_axis_m,
            y=y_axis_m,
            z=truth_w_per_m2 / 1_000.0,
            coloraxis="coloraxis",
            zsmooth=False,
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>I=%{z:.3e} kW/m2<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sensor_x,
            y=sensor_y,
            mode="markers+text" if len(sensor_x) <= 12 else "markers",
            text=[str(i) for i in range(1, len(sensor_x) + 1)],
            textposition="middle center",
            marker={
                "size": 15,
                "color": sensor_kw,
                "coloraxis": "coloraxis",
                "line": {"color": "#222", "width": 1.2},
            },
            textfont={"size": 9, "color": "#111827"},
            name="Sensors",
            hovertemplate="Sensor %{text}<br>x=%{x:.3f} m<br>y=%{y:.3f} m<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[actual_center_m[0]],
            y=[actual_center_m[1]],
            mode="markers",
            marker={"size": 14, "symbol": "x", "color": "#dc2626", "line": {"width": 2}},
            name="True center",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(
            x=x_axis_m,
            y=y_axis_m,
            z=reconstructed_w_per_m2 / 1_000.0,
            coloraxis="coloraxis",
            zsmooth=False,
            hovertemplate="x=%{x:.3f} m<br>y=%{y:.3f} m<br>I_recon=%{z:.3e} kW/m2<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[reconstructed_center_m[0]],
            y=[reconstructed_center_m[1]],
            mode="markers",
            marker={"size": 11, "symbol": "diamond", "color": "#2563eb", "line": {"color": "white", "width": 1}},
            name="Reconstructed center",
        ),
        row=1,
        col=2,
    )
    ellipse_x, ellipse_y = _ellipse_trace(ellipse)
    if ellipse_x:
        fig.add_trace(
            go.Scatter(
                x=ellipse_x,
                y=ellipse_y,
                mode="lines",
                line={"color": "#2563eb", "width": 2.0},
                name="Fitted ellipse",
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        height=500,
        margin={"l": 30, "r": 72, "t": 32, "b": 32},
        coloraxis={
            "colorscale": [
                [0.0, "#f6f4ed"],
                [0.18, "#d8d9d2"],
                [0.38, "#efc057"],
                [0.58, "#e47b30"],
                [0.78, "#c52f32"],
                [1.0, "#fff8f8"],
            ],
            "cmin": 0.0,
            "cmax": max(float(np.max(truth_w_per_m2 / 1_000.0)), float(np.max(reconstructed_w_per_m2 / 1_000.0)), 1e-12),
            "colorbar": {"title": "kW/m2", "thickness": 16, "len": 0.64, "tickformat": ".2e", "tickfont": {"size": 9}},
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        legend={"orientation": "h", "x": 0.01, "y": 1.08, "font": {"size": 9}},
        font={"family": "Avenir, Helvetica Neue, sans-serif", "size": 10, "color": "#1f2937"},
    )
    axis_style = {
            "range": [-half_width, half_width],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
            "tickformat": ".2f",
            "linecolor": "#c7c7c7",
            "mirror": True,
            "title": {"text": "x (m)", "font": {"size": 10}},
            "tickfont": {"size": 9},
        }
    y_axis_style = {
            "range": [-half_width, half_width],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
            "tickformat": ".2f",
            "linecolor": "#c7c7c7",
            "mirror": True,
            "title": {"text": "y (m)", "font": {"size": 10}},
            "tickfont": {"size": 9},
        }
    fig.update_xaxes(**axis_style, scaleanchor="y", row=1, col=1)
    fig.update_xaxes(**axis_style, scaleanchor="y2", row=1, col=2)
    fig.update_yaxes(**y_axis_style, row=1, col=1)
    fig.update_yaxes(**y_axis_style, row=1, col=2)
    return fig


def make_path_figure(times_s: np.ndarray, true_path: np.ndarray, recon_path: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=times_s, y=true_path[:, 0], name="true x", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times_s, y=recon_path[:, 0], name="recon x", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=times_s, y=true_path[:, 1], name="true y", mode="lines"), row=2, col=1)
    fig.add_trace(go.Scatter(x=times_s, y=recon_path[:, 1], name="recon y", mode="lines"), row=2, col=1)
    fig.update_layout(
        height=540,
        margin={"l": 34, "r": 18, "t": 10, "b": 36},
        showlegend=True,
        legend={"orientation": "h", "x": 0.02, "y": 1.04, "font": {"size": 10}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Avenir, Helvetica Neue, sans-serif", "size": 10},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont={"size": 9}, title_font={"size": 10})
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont={"size": 9}, title_font={"size": 10})
    return fig


def reconstruct_position(
    sensor_coordinates_m: tuple[tuple[float, float], ...],
    sensor_values: list[float],
    noise_floor_w_per_m2: float = 0.0,
) -> tuple[float, float, str, object | None]:
    fallback = estimate_log_linear_gradient(sensor_coordinates_m, sensor_values)
    if len(sensor_values) < 6:
        return fallback.centroid_x_m, fallback.centroid_y_m, "centroid", None
    try:
        ellipse = fit_log_quadratic_ellipse(
            sensor_coordinates_m,
            sensor_values,
            sigma_noise_w_per_m2=noise_floor_w_per_m2,
        )
        return ellipse.center_x_m, ellipse.center_y_m, "ellipse", ellipse
    except ValueError:
        return fallback.centroid_x_m, fallback.centroid_y_m, "centroid", None


def layout_options() -> tuple[list[str], dict[str, list[int]]]:
    catalog = load_layouts_catalog(LAYOUTS_PATH)
    families = [name for name, counts in catalog.items() if counts]
    counts = {family: available_layout_counts(family, LAYOUTS_PATH) for family in families}
    return families, counts


def sidebar_static_inputs() -> dict[str, object]:
    families, counts_by_family = layout_options()
    if not families:
        st.sidebar.error(f"No layouts found in {LAYOUTS_PATH.name}.")
        st.stop()

    st.sidebar.header("Single Frame")
    with st.sidebar.expander("Laser", expanded=True):
        initial_power_w = st.number_input("P0 (W)", min_value=0.0, value=10_000.0, step=500.0, format="%.3e")
        wavelength_nm = st.number_input("lambda (nm)", min_value=100.0, value=1064.0, step=10.0, format="%.3f")
        beam_quality_m2 = st.number_input("M2", min_value=1.0, value=1.2, step=0.1, format="%.3f")
        lock_target_spot = st.checkbox("Solve aperture from spot", value=True)
        if lock_target_spot:
            target_spot_diameter_cm = st.number_input("Target diameter (cm)", min_value=0.1, value=10.0, step=0.5, format="%.3f")
            aperture_diameter_m = 0.0
        else:
            target_spot_diameter_cm = DEFAULT_TARGET_DIAMETER_M * 100.0
            aperture_diameter_m = st.number_input("Aperture D (m)", min_value=1e-4, value=0.10, step=0.01, format="%.3e")

    with st.sidebar.expander("Atmosphere", expanded=True):
        distance_m = st.slider("Range z (m)", 1.0, 10_000.0, 1_000.0, 10.0)
        visibility_km = st.slider("Visibility V (km)", 0.1, 80.0, 20.0, 0.1)
        enable_turbulence = st.checkbox("Turbulence", value=False)
        cn2_value = st.number_input("Cn2 (m^-2/3)", min_value=0.0, value=1e-14, step=1e-15, format="%.3e")

    with st.sidebar.expander("Geometry", expanded=True):
        pitch_deg = st.slider("Pitch theta (deg)", 0.0, 75.0, 20.0, 1.0)
        yaw_deg = st.slider("Yaw phi (deg)", 0.0, 75.0, 10.0, 1.0)
        beam_center_x_m = st.slider("Beam x0 (m)", -0.08, 0.08, 0.0, 0.005)
        beam_center_y_m = st.slider("Beam y0 (m)", -0.08, 0.08, 0.0, 0.005)

    with st.sidebar.expander("Sensors", expanded=True):
        layout_family = st.selectbox("Layout", families, index=0)
        sensor_counts = counts_by_family[layout_family]
        sensor_count = st.selectbox("Sensor count", sensor_counts, index=max(len(sensor_counts) - 1, 0))
        od_filter = st.slider("OD", 0.0, 5.0, 0.0, 0.1)
        theta_fov_deg = st.slider("FOV (deg)", 5.0, 90.0, 60.0, 1.0)
        saturation_limit_w_per_m2 = st.number_input("I_sat (W/m2)", min_value=1.0, value=DEFAULT_SATURATION_LIMIT_W_PER_M2, step=1e5, format="%.3e")
        noise_floor_w_per_m2 = st.number_input("Noise floor (W/m2)", min_value=0.0, value=DEFAULT_NOISE_FLOOR_W_PER_M2, step=1e-4, format="%.3e")

    with st.sidebar.expander("Numerics", expanded=False):
        grid_points = st.slider("Grid N", 101, 401, 221, 20)
        surface_size_m = st.slider("Surface width (m)", 0.2, 1.0, DEFAULT_SURFACE_SIZE_M, 0.05)

    return locals()


def solve_aperture(inputs: dict[str, object], wavelength_m: float, cn2_for_model: float | None) -> float:
    if not inputs["lock_target_spot"]:
        return float(inputs["aperture_diameter_m"])

    distance_m = float(inputs["distance_m"])
    beam_quality_m2 = float(inputs["beam_quality_m2"])
    target_radius_m = float(inputs["target_spot_diameter_cm"]) / 100.0 / 2.0
    base_radius_m = target_radius_m
    if cn2_for_model is not None and cn2_for_model > 0.0:
        turbulence_radius_m, _ = calculate_turbulence_spread(distance_m, wavelength_m, cn2_for_model)
        if turbulence_radius_m >= target_radius_m:
            raise ValueError("Target spot is smaller than turbulence broadening.")
        base_radius_m = float(np.sqrt(target_radius_m**2 - turbulence_radius_m**2))
    return calculate_aperture_for_target_spot(
        target_spot_radius_m=base_radius_m,
        distance_m=distance_m,
        wavelength_m=wavelength_m,
        beam_quality_m2=beam_quality_m2,
    )


def single_frame_view() -> None:
    inputs = sidebar_static_inputs()
    wavelength_m = float(inputs["wavelength_nm"]) * 1e-9
    visibility_m = float(inputs["visibility_km"]) * 1_000.0
    pitch_rad = np.deg2rad(float(inputs["pitch_deg"]))
    yaw_rad = np.deg2rad(float(inputs["yaw_deg"]))
    cn2_for_model = float(inputs["cn2_value"]) if inputs["enable_turbulence"] else None

    try:
        aperture_diameter_m = solve_aperture(inputs, wavelength_m, cn2_for_model)
    except ValueError as exc:
        st.error(str(exc))
        return

    beam_shift_x_m = 0.0
    beam_shift_y_m = 0.0
    if cn2_for_model is not None and cn2_for_model > 0.0:
        beam_wander_variance_m2 = calculate_beam_wander_variance(
            distance_m=float(inputs["distance_m"]),
            wavelength_m=wavelength_m,
            aperture_diameter_m=aperture_diameter_m,
            cn2_m_minus_2_over_3=cn2_for_model,
        )
        beam_shift_x_m, beam_shift_y_m = np.random.default_rng().normal(0.0, np.sqrt(beam_wander_variance_m2), size=2)

    x_axis_m, y_axis_m, x_grid_m, y_grid_m = build_grid(float(inputs["surface_size_m"]), int(inputs["grid_points"]))
    true_center = (
        float(inputs["beam_center_x_m"]) + float(beam_shift_x_m),
        float(inputs["beam_center_y_m"]) + float(beam_shift_y_m),
    )
    simulation = run_continuous_beam_pipeline(
        aperture_diameter_m=aperture_diameter_m,
        wavelength_m=wavelength_m,
        beam_quality_m2=float(inputs["beam_quality_m2"]),
        initial_power_w=float(inputs["initial_power_w"]),
        distance_m=float(inputs["distance_m"]),
        visibility_m=visibility_m,
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        x_grid_m=x_grid_m,
        y_grid_m=y_grid_m,
        beam_center_x_m=true_center[0],
        beam_center_y_m=true_center[1],
        cn2_m_minus_2_over_3=cn2_for_model,
    )

    photodiodes = get_layout(str(inputs["layout_family"]), int(inputs["sensor_count"]), LAYOUTS_PATH)
    sensor_readings, hardware_flags, _ = sample_sensor_readings(
        photodiodes=photodiodes,
        x_grid_m=x_grid_m,
        y_grid_m=y_grid_m,
        irradiance_w_per_m2=simulation["irradiance_w_per_m2"],
        pitch_rad=pitch_rad,
        yaw_rad=yaw_rad,
        theta_fov_rad=np.deg2rad(float(inputs["theta_fov_deg"])),
        optical_density=float(inputs["od_filter"]),
        saturation_limit_w_per_m2=float(inputs["saturation_limit_w_per_m2"]),
        noise_floor_w_per_m2=float(inputs["noise_floor_w_per_m2"]),
        rng=np.random.default_rng(),
    )

    peak = float(np.max(simulation["irradiance_w_per_m2"]))
    recon_x, recon_y, recon_mode, ellipse = reconstruct_position(
        photodiodes.coordinates_m,
        list(sensor_readings.values()),
        float(inputs["noise_floor_w_per_m2"]),
    )
    recon_error = float(np.hypot(recon_x - true_center[0], recon_y - true_center[1]))
    reconstructed_map = reconstructed_irradiance_map(
        coordinates_m=photodiodes.coordinates_m,
        sensor_values=list(sensor_readings.values()),
        x_grid_m=x_grid_m,
        y_grid_m=y_grid_m,
        fallback_center_m=(recon_x, recon_y),
        fallback_radius_m=float(simulation["effective_radius_m"]),
        ellipse=ellipse,
    )

    st.title("Laser Irradiance Sensor Simulator")
    status_strip(
        [
            ("Peak", sci(peak, "W/m2")),
            ("Spot diameter", sci(2.0 * float(simulation["effective_radius_m"]) * 100.0, "cm")),
            ("Recon error", sci(recon_error, "m")),
            ("Mode", recon_mode),
        ]
    )

    st.plotly_chart(
        make_beam_comparison_figure(
            x_axis_m,
            y_axis_m,
            simulation["irradiance_w_per_m2"],
            reconstructed_map,
            photodiodes.coordinates_m,
            sensor_readings,
            true_center,
            (recon_x, recon_y),
            ellipse,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    recon_rows = [
        ("x true", sci(true_center[0], "m")),
        ("y true", sci(true_center[1], "m")),
        ("x recon", sci(recon_x, "m")),
        ("y recon", sci(recon_y, "m")),
        ("error", sci(recon_error, "m")),
        ("mode", recon_mode),
        ("within FOV", "yes" if hardware_flags["within_fov"] else "no"),
        ("sensor flags", ", ".join(flag for flag, value in [("saturated", hardware_flags["any_saturated"]), ("below noise", hardware_flags["any_below_noise"])] if value) or "none"),
    ]
    if ellipse is not None:
        recon_rows.extend(
            [
                ("pitch est", f"{np.rad2deg(ellipse.pitch_rad):.3e} +/- {np.rad2deg(ellipse.sigma_pitch_rad):.3e} deg"),
                ("yaw est", f"{np.rad2deg(ellipse.yaw_rad):.3e} +/- {np.rad2deg(ellipse.sigma_yaw_rad):.3e} deg"),
            ]
        )

    recon_col, sensor_col = st.columns([0.34, 0.66], gap="medium")
    with recon_col:
        st.subheader("Reconstruction")
        st.dataframe(table_from_pairs(recon_rows), width="stretch", hide_index=True, height=220)
    with sensor_col:
        st.subheader("Sensors")
        st.dataframe(
            make_readings_table(photodiodes.coordinates_m, sensor_readings, peak),
            width="stretch",
            hide_index=True,
            height=220,
        )


def temporal_sidebar_inputs() -> dict[str, object]:
    families, counts_by_family = layout_options()
    if not families:
        st.sidebar.error(f"No layouts found in {LAYOUTS_PATH.name}.")
        st.stop()

    st.sidebar.header("Temporal Fly-By")
    with st.sidebar.expander("Beam", expanded=True):
        initial_power_w = st.number_input("P0 (W)", min_value=0.0, value=10_000.0, step=500.0, format="%.3e", key="temp_p0")
        wavelength_nm = st.number_input("lambda (nm)", min_value=100.0, value=1064.0, step=10.0, format="%.3f", key="temp_lambda")
        beam_quality_m2 = st.number_input("M2", min_value=1.0, value=1.2, step=0.1, format="%.3f", key="temp_m2")
        distance_m = st.slider("Range z (m)", 1.0, 10_000.0, 1_000.0, 10.0, key="temp_z")
        visibility_km = st.slider("Visibility V (km)", 0.1, 80.0, 20.0, 0.1, key="temp_v")

    with st.sidebar.expander("Layout", expanded=True):
        layout_family = st.selectbox("Layout", families, index=0, key="temp_layout")
        sensor_count = st.selectbox("Sensor count", counts_by_family[layout_family], index=max(len(counts_by_family[layout_family]) - 1, 0), key="temp_n")
        od_filter = st.slider("OD", 0.0, 5.0, 0.0, 0.1, key="temp_od")
        theta_fov_deg = st.slider("FOV (deg)", 5.0, 90.0, 60.0, 1.0, key="temp_fov")
        noise_floor_w_per_m2 = st.number_input("Noise floor (W/m2)", min_value=0.0, value=DEFAULT_NOISE_FLOOR_W_PER_M2, step=1e-4, format="%.3e", key="temp_noise")

    with st.sidebar.expander("Motion", expanded=True):
        vx_m_per_s = st.number_input("v_x (m/s)", value=0.02, step=0.01, format="%.3f", key="temp_vx")
        vy_m_per_s = st.number_input("v_y (m/s)", value=0.0, step=0.01, format="%.3f", key="temp_vy")
        omega_pitch_deg = st.number_input("omega pitch (deg/s)", value=0.0, step=0.5, format="%.3f", key="temp_op")
        omega_yaw_deg = st.number_input("omega yaw (deg/s)", value=0.0, step=0.5, format="%.3f", key="temp_oy")
        sim_time_s = st.number_input("Simulation T (s)", min_value=0.1, value=12.0, step=0.5, format="%.3f", key="temp_T")
        dt_s = st.number_input("dt (s)", min_value=0.01, value=0.1, step=0.01, format="%.3f", key="temp_dt")
        run_button = st.button("Run simulation", width="stretch")

    with st.sidebar.expander("Numerics", expanded=False):
        grid_points = st.slider("Grid N", 101, 401, 201, 20, key="temp_grid")
        surface_size_m = st.slider("Surface width (m)", 0.2, 1.0, DEFAULT_SURFACE_SIZE_M, 0.05, key="temp_surface")

    return locals()


def temporal_view() -> None:
    inputs = temporal_sidebar_inputs()
    st.title("Temporal Fly-By")
    if not inputs["run_button"]:
        st.info("Set temporal parameters in the sidebar and run the simulation.")
        return

    wavelength_m = float(inputs["wavelength_nm"]) * 1e-9
    visibility_m = float(inputs["visibility_km"]) * 1_000.0
    pitch_rad = 0.0
    yaw_rad = 0.0
    cn2_for_model = 1e-14
    x_axis_m, y_axis_m, x_grid_m, y_grid_m = build_grid(float(inputs["surface_size_m"]), int(inputs["grid_points"]))
    photodiodes = get_layout(str(inputs["layout_family"]), int(inputs["sensor_count"]), LAYOUTS_PATH)

    try:
        aperture_diameter_m = calculate_aperture_for_target_spot(
            target_spot_radius_m=DEFAULT_TARGET_DIAMETER_M / 2.0,
            distance_m=float(inputs["distance_m"]),
            wavelength_m=wavelength_m,
            beam_quality_m2=float(inputs["beam_quality_m2"]),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    wander_variance = calculate_beam_wander_variance(
        distance_m=float(inputs["distance_m"]),
        wavelength_m=wavelength_m,
        aperture_diameter_m=aperture_diameter_m,
        cn2_m_minus_2_over_3=cn2_for_model,
    )
    engagement = KinematicEngagement(
        vx_m_per_s=float(inputs["vx_m_per_s"]),
        vy_m_per_s=float(inputs["vy_m_per_s"]),
        omega_pitch_rad_per_s=np.deg2rad(float(inputs["omega_pitch_deg"])),
        omega_yaw_rad_per_s=np.deg2rad(float(inputs["omega_yaw_deg"])),
        dt_s=float(inputs["dt_s"]),
        beam_wander_variance_m2=wander_variance,
        rng=np.random.default_rng(0),
    )

    steps = max(2, int(np.ceil(float(inputs["sim_time_s"]) / float(inputs["dt_s"]))) + 1)
    times = np.zeros(steps)
    true_path = np.zeros((steps, 2))
    recon_path = np.zeros((steps, 2))
    progress = st.progress(0.0)

    for index in range(steps):
        state = {
            "time_s": engagement.time_s,
            "beam_center_x_m": engagement.beam_center_x_m,
            "beam_center_y_m": engagement.beam_center_y_m,
            "pitch_rad": engagement.pitch_rad,
            "yaw_rad": engagement.yaw_rad,
        }
        simulation = run_continuous_beam_pipeline(
            aperture_diameter_m=aperture_diameter_m,
            wavelength_m=wavelength_m,
            beam_quality_m2=float(inputs["beam_quality_m2"]),
            initial_power_w=float(inputs["initial_power_w"]),
            distance_m=float(inputs["distance_m"]),
            visibility_m=visibility_m,
            pitch_rad=state["pitch_rad"],
            yaw_rad=state["yaw_rad"],
            x_grid_m=x_grid_m,
            y_grid_m=y_grid_m,
            beam_center_x_m=state["beam_center_x_m"],
            beam_center_y_m=state["beam_center_y_m"],
            cn2_m_minus_2_over_3=cn2_for_model,
        )
        readings, _, _ = sample_sensor_readings(
            photodiodes=photodiodes,
            x_grid_m=x_grid_m,
            y_grid_m=y_grid_m,
            irradiance_w_per_m2=simulation["irradiance_w_per_m2"],
            pitch_rad=state["pitch_rad"],
            yaw_rad=state["yaw_rad"],
            theta_fov_rad=np.deg2rad(float(inputs["theta_fov_deg"])),
            optical_density=float(inputs["od_filter"]),
            saturation_limit_w_per_m2=DEFAULT_SATURATION_LIMIT_W_PER_M2,
            noise_floor_w_per_m2=float(inputs["noise_floor_w_per_m2"]),
            rng=np.random.default_rng(index),
        )
        recon_x, recon_y, _, _ = reconstruct_position(
            photodiodes.coordinates_m,
            list(readings.values()),
            float(inputs["noise_floor_w_per_m2"]),
        )
        times[index] = state["time_s"]
        true_path[index] = (state["beam_center_x_m"], state["beam_center_y_m"])
        recon_path[index] = (recon_x, recon_y)
        progress.progress((index + 1) / steps)
        if index < steps - 1:
            engagement.step()

    rmse = float(np.sqrt(np.mean(np.sum((true_path - recon_path) ** 2, axis=1))))
    status_strip(
        [
            ("Samples", str(steps)),
            ("RMSE", sci(rmse, "m")),
            ("Duration", sci(float(inputs["sim_time_s"]), "s")),
            ("dt", sci(float(inputs["dt_s"]), "s")),
        ]
    )
    st.plotly_chart(make_path_figure(times, true_path, recon_path), width="stretch", config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Laser Irradiance Sensor Simulator", layout="wide", initial_sidebar_state="expanded")
    apply_css()

    view = st.sidebar.radio("View", ["Single frame", "Temporal fly-by"], horizontal=True)
    if view == "Single frame":
        single_frame_view()
    else:
        temporal_view()


if __name__ == "__main__":
    main()
