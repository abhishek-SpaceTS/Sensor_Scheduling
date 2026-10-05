# Space Domain Awareness (SDA) Sensor Scheduling 🛰️

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end framework for training AI to autonomously schedule observations for Space-Based Space Domain Awareness (SDA) Inspector satellites. 

This project replicates and extends the methodologies presented in the AMOS 2024 paper *"Machine Learning for Space Domain Awareness Sensor Scheduling"* by Dhingra, DeJac, and McGuire.

## 🚀 Overview

Unlike traditional ground-based tracking, this project models the **Orienteering Problem** for a short-range (50km), space-based "Inspector" satellite in Low Earth Orbit (LEO). Because both the sensor and the targets are moving at extreme orbital velocities (~7.5 km/s), the scheduler must dynamically optimize which targets to observe while adhering to physical constraints like slew rates and limited time windows.

This repository provides a complete pipeline:
1. **Physics Engine**: Downloads real TLEs from CelesTrak and propagates them in a 3D environment to detect <50km proximity intercepts.
2. **The Teacher (Google OR-Tools)**: Brute-forces the optimal observation schedule using Constraint Programming (CP-SAT), accounting for slew speeds and target priorities.
3. **The Student (PyTorch)**: A neural network that learns from the Teacher's mathematically perfect schedules, enabling millisecond-speed AI decision-making.

---

## 🛠️ Features

* **Real Orbital Data**: Directly parses live CelesTrak `leo.txt` catalogs.
* **SGP4 Propagation**: Built on `skyfield` for accurate Earth-Centered Inertial (ECI) coordinate geometry.
* **Contiguous Window Grouping**: Algorithmically compresses thousands of 30-second physics ticks into clean, continuous observation windows.
* **CP-SAT Optimizer**: A classic Operations Research solver that generates perfect "Answer Key" labels for training.
* **Modular Configuration**: Dynamically loads payload, ADCS, and EPS parameters from simple YAML files.

---

## 📂 Project Structure

```text
├── config/                  # YAML configurations (sensor_config, mission_config)
├── data/
│   ├── raw/celestrak/       # Raw TLE downloads
│   └── processed/           # Consolidated visibility windows & AI Labels
├── src/
│   ├── data/                # CelesTrak API fetchers and TLE parsers
│   ├── orbit/               # SGP4 Propagators and 3D math
│   ├── sensor/              # Visibility geometry engine
│   └── scheduling/          # OR-Tools (Teacher) & PyTorch (Student) Models
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sda-sensor-scheduling.git
cd sda-sensor-scheduling
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Requires `skyfield`, `ortools`, `torch`, and `pyyaml`)*

---

## 💻 Usage Pipeline

Follow this pipeline to generate data and train the AI:

**1. Fetch the Target Catalog**
Downloads the latest active LEO catalog from CelesTrak and filters it to a specific altitude band (e.g., 500-600km).
```bash
python src/data/celestrak_api.py
```

**2. Run the Physics Engine (Visibility Windows)**
Propagates the targets against the M4V inspector satellite to find physical intercepts. 
```bash
python src/sensor/visibility.py
```

**3. Generate the Answer Keys (The Teacher)**
Uses Google OR-Tools to solve the Orienteering Problem and find the optimal schedule.
```bash
python src/scheduling/ortools_scheduler.py
```

**4. Train the AI (The Student)**
Trains a PyTorch Neural Network to mimic the Teacher's scheduling logic.
```bash
python src/scheduling/ai_scheduler.py
```

---

## 📚 Acknowledgements
This project's architecture is heavily inspired by the work of Dhingra, DeJac, and McGuire in their 2024 AMOS paper on Machine Learning for SDA Sensor Scheduling (Project Heimdall). 

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
