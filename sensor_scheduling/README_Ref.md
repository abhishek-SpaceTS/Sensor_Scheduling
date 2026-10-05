# Space Domain Awareness (SDA) Sensor Scheduling

## Overview
This project aims to build an AI-driven Space-Based Sensor Scheduling system, modeled after the methodology in the AMOS 2024 paper *"Machine Learning for Space Domain Awareness Sensor Scheduling" (Dhingra et al.)*. 

Unlike traditional ground-based tracking, this project specifically models the **Orienteering Problem** for a short-range, space-based "Inspector" satellite (**M4V**). Because both the sensor and the targets are moving at extreme orbital speeds (~7.5 km/s), the scheduler must dynamically optimize which targets to observe while adhering to physical constraints like slew rates, battery power, and lighting.

## Progress & Record of Work

### 1. Data Acquisition & Filtering (`src/data/celestrak_api.py`)
- Pivoted from API JSON downloads to directly parsing the massive, raw `.tle` catalogs from CelesTrak.
- Implemented orbital math (Kepler's Third Law) to filter the full 15,000+ active satellite catalog down to exactly **1,624 target satellites** that fall into the specific 500km–600km altitude band.

### 2. Space-Based Sensor Definition (`config/sensor_config.yaml`)
- Designed the **M4V Spacecraft**, our primary space-based sensor, orbiting in a 550km Sun-Synchronous Orbit (SSO).
- Defined its physical limits: a wide 35° Field of View but a strict **50 km maximum tracking range**.
- Built a dynamic Python parser (`src/sensor/sensor_model.py`) that instantly absorbs new payload, EPS (battery), and ADCS (slew rate) variables from the YAML without requiring hardcoded changes.

### 3. SGP4 Physics Propagator (`src/orbit/sgp4_propagator.py`)
- Integrated the `skyfield` library to handle complex Earth-Centered Inertial (ECI) coordinate geometry.
- Programmed a mathematical mock-TLE generator to physically spawn the M4V sensor into the 3D physics engine using exact Keplerian elements (97.590° inclination, near-circular).

### 4. Visibility Engine & Target Density Experiments (`src/sensor/visibility.py`)
- Built the geometry engine to calculate precise 3D distances between M4V and targets.
- **The "Two Bullets" Experiment**: To test how rare a 50km space-to-space intercept is, we ran 3 different scale experiments in a simulated 24-hour orbital window:
  - **20 Targets**: Produced **0** intercepts. Space is simply too large.
  - **200 Targets**: Produced **2** intercepts.
  - **1,624 Targets (Full Catalog)**: Produced **72 total visibility ticks** spanning **39 unique targets**. 
- The script algorithmically consolidated those 72 raw 30-second ticks into **37 continuous "Visibility Windows"** (representing the exact continuous duration a target remained in range).
- Output is routed into the standard placeholder architecture: `data/processed/visibility_windows/`.

## The AI Training Pipeline

To build the scheduler, the project follows this supervised learning pipeline:

### Phase 1: The Teacher (`src/scheduling/ortools_scheduler.py`) - [COMPLETE]
- Uses Google OR-Tools (CP-SAT) to ingest the raw visibility windows and brute-force the mathematically perfect scheduling sequence.
- Enforces strict Slew Rate physics (e.g., M4V needs 60 seconds to turn its camera between targets).
- Outputs the optimal "Answer Key" labels (1 for Scheduled, 0 for Ignored) and routes them to `data/processed/scheduling_dataset/`.

### Phase 2: The Student (`src/scheduling/ai_scheduler.py`) - [READY]
- A PyTorch Neural Network designed to learn from the Teacher's labels.
- Normalizes features (Priority, Distance, Duration) and attempts to predict the Teacher's schedule probabilities using Binary Cross-Entropy Loss.

### Phase 3: The Graduation (Reinforcement Learning) - [UPCOMING]
- Unleash the trained Neural Network onto the full 1,624 target catalog, bypassing the Teacher, where it will use its trained weights to instantly schedule observations in milliseconds.
