"""Compatibility CLI for Monte Carlo photodiode layout optimization.

The real Monte Carlo machinery lives in ``monte_carlo.py``. This wrapper keeps
the older command shape available while avoiding a second, divergent optimizer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from monte_carlo import (
    DEFAULT_BEAM_QUALITY_M2,
    DEFAULT_WAVELENGTH_M,
    BeamConfig,
    MonteCarloConfig,
    ScenarioDistribution,
    SensorNoiseConfig,
    optimize_layout as optimize_monte_carlo_layout,
    sample_scenarios,
)
from sensors import PhotodiodeArray, load_layouts_catalog, save_layouts_catalog


def optimize_layout(
    sensor_count: int,
    lx_m: float,
    ly_m: float,
    lz_m: float,
    minimum_spacing_m: float,
    mc_samples: int,
    generations: int,
    seed: int,
    beam_radius_m: float = 0.05,
    wavelength_m: float = DEFAULT_WAVELENGTH_M,
    beam_quality_m2: float = DEFAULT_BEAM_QUALITY_M2,
    noise_floor_w_per_m2: float = 1.0,
    cn2_m_minus_2_over_3: float | None = None,
) -> PhotodiodeArray:
    """Run the canonical Monte Carlo optimizer and return sensor poses."""
    beam = BeamConfig(
        wavelength_m=wavelength_m,
        beam_quality_m2=beam_quality_m2,
        beam_waist_m=beam_radius_m,
        cn2_m_minus_2_over_3=cn2_m_minus_2_over_3,
    )
    distribution = ScenarioDistribution(
        target_half_extents_m=(lx_m / 2.0, ly_m / 2.0, lz_m / 2.0),
    )
    monte_carlo = MonteCarloConfig(sample_count=mc_samples, seed=seed)
    sensor_noise = SensorNoiseConfig(noise_floor_w_per_m2=noise_floor_w_per_m2)
    scenarios = sample_scenarios(distribution, beam, monte_carlo)
    result = optimize_monte_carlo_layout(
        sensor_count=sensor_count,
        lx_m=lx_m,
        ly_m=ly_m,
        lz_m=lz_m,
        minimum_spacing_m=minimum_spacing_m,
        scenarios=scenarios,
        sensor_noise=sensor_noise,
        monte_carlo=monte_carlo,
        generations=generations,
    )
    return result.photodiodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="Sensor count")
    parser.add_argument("--dmin", type=float, required=True, help="Minimum spacing in meters")
    parser.add_argument("--lx", type=float, required=True, help="Cuboid length in x")
    parser.add_argument("--ly", type=float, required=True, help="Cuboid length in y")
    parser.add_argument("--lz", type=float, required=True, help="Cuboid length in z")
    parser.add_argument("--layouts-path", type=Path, default=Path(__file__).with_name("layouts.json"))
    parser.add_argument("--mc-samples", type=int, default=24)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beam-radius", type=float, default=0.05)
    parser.add_argument("--noise-floor", type=float, default=1.0)
    parser.add_argument("--family", type=str, default="optimized_mc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimized = optimize_layout(
        sensor_count=args.n,
        lx_m=args.lx,
        ly_m=args.ly,
        lz_m=args.lz,
        minimum_spacing_m=args.dmin,
        mc_samples=args.mc_samples,
        generations=args.generations,
        seed=args.seed,
        beam_radius_m=args.beam_radius,
        noise_floor_w_per_m2=args.noise_floor,
    )
    catalog = load_layouts_catalog(args.layouts_path)
    catalog.setdefault(args.family, {})[args.n] = optimized
    save_layouts_catalog(catalog, args.layouts_path)
    coords = [
        {
            "position": [round(value, 6) for value in pose.position_m],
            "normal": [round(value, 6) for value in pose.normal_m],
            "face": pose.face_name,
        }
        for pose in optimized.poses_m
    ]
    print(f"Saved {args.family!r} layout for N={args.n} to {args.layouts_path}")
    print(coords)


if __name__ == "__main__":
    main()
