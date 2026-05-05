# Drone Laser Sensing

Backend tools for modeling a laser footprint on photodiodes and optimizing sensor placement for beam-direction observability.

The useful parts of the project are the physics model, photodiode layout representation, Monte Carlo Fisher scoring, and backend layout optimization. The Streamlit app is only a visualization layer around the same primitives.

## Core Model

The Monte Carlo optimizer scores a fixed photodiode layout over sampled laser engagement states:

1. Sample range, beam direction, and target hit point from an operational domain.
2. Propagate a Gaussian-like beam through atmospheric attenuation and optional turbulence broadening.
3. Evaluate the expected irradiance at each photodiode pose.
4. Build a Fisher information matrix from finite-difference derivatives of sensor readings.
5. Marginalize nuisance terms and maximize the direction-only `log det(F_direction)` score.

The current photodiode model is intentionally simple:

- Each diode is a point irradiance sampler at `position_m` with a surface `normal_m`.
- Angular response is a projected-area term, `cos(gamma) = normal . beam_direction`.
- Optical density is a scalar transmission, `10^-OD`.
- Saturated sensors are excluded from Fisher scoring.
- Noise is Gaussian in irradiance units: `sqrt(noise_floor^2 + (relative_noise * irradiance)^2)`.

The model does not yet include finite active area integration, wavelength-dependent responsivity, dark current, shot noise, package angular response, or amplifier saturation as electrical current/voltage. Those are the next changes if the goal is a physically calibrated photodiode model rather than layout geometry ranking.

## Important Files

- `physics.py` contains beam propagation, atmospheric attenuation, turbulence spread, 3D cuboid surface geometry, and irradiance evaluation.
- `sensors.py` loads photodiode layouts, applies OD/saturation/noise for sampled readings, and stores sensor poses.
- `monte_carlo.py` scores or optimizes layouts using Monte Carlo Fisher information.
- `run_face_monte_carlo.py` runs backend-only single-face and multi-face panel trials and writes result summaries.
- `reconstruction.py` fits a log-quadratic Gaussian footprint to readings for direction reconstruction experiments.
- `layouts.json` stores named fixed layouts.
- `monte_carlo_config.example.json` shows the configurable backend parameters.

## Parameters That Matter

Beam and propagation:

- `wavelength_nm`
- `m2`
- `beam_waist_m`
- `power_w` / `initial_power_w`
- `visibility_km`
- `cn2`

Scenario distribution:

- `range_min_m`, `range_max_m`
- `angle_max_deg`
- `target_half_extents_m`
- `uniform_solid_angle`
- train/eval sample counts and seeds

Sensor and layout:

- diode `position_m`
- diode `normal_m`
- face/domain dimensions
- `minimum_spacing_m`
- `noise_floor_w_per_m2`
- `relative_noise_fraction`
- `optical_density`
- `saturation_limit_w_per_m2`
- `minimum_active_sensors`

Layout search:

- free continuous placement through differential evolution
- heuristic baselines: grid, circular rings, rectangular rings, and random placement
- single face or five-panel multi-face domains

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Score A Saved Layout

```bash
python3 monte_carlo.py score \
  --config monte_carlo_config.example.json \
  --layout-family default \
  --sensor-count 9
```

## Optimize A Cuboid Layout

```bash
python3 monte_carlo.py optimize \
  --config monte_carlo_config.example.json \
  --n 12 \
  --lx 0.30 \
  --ly 0.20 \
  --lz 0.08 \
  --dmin 0.02
```

Add `--save` to write the optimized layout into `layouts.json`.

## Run Face Trials

```bash
python3 run_face_monte_carlo.py \
  --output-dir monte_carlo_results \
  --counts 6,8,10,12,15,20 \
  --patterns grid,rings,rect,random
```

For multi-face trials with a fixed number of sensors per face:

```bash
python3 run_face_monte_carlo.py \
  --output-dir monte_carlo_results_per_face \
  --counts 4,6,8 \
  --multi-face-per-face
```

Generated Excel workbooks and result folders are ignored by git.

## Development Notes

- Keep calculations in SI units internally.
- Keep Monte Carlo scoring backend-only and deterministic under explicit seeds.
- Treat generated workbooks, caches, and virtual environments as local artifacts.
- When adding physical photodiode realism, update the forward measurement model and Fisher noise weighting together.
