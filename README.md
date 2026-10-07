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
* **Continuous-Time CP-SAT Optimizer**: A mathematically perfect OR-Tools solver utilizing `IntervalVar` logic to dynamically stretch observation times and calculate physical ADCS slew gaps to generate perfect "Answer Key" labels for training.
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
├── main.py                  # The Master Pipeline Orchestrator
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/abhishek-SpaceTS/Sensor_Scheduling.git
cd Sensor_Scheduling
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
*(Requires `skyfield`, `ortools`, `torch`, and `pyyaml`)*

---

## 💻 Usage Pipeline

Follow this pipeline to run the project. We have provided a central orchestrator script (`main.py`) that handles the complex scheduling steps automatically.

**1. Fetch Targets & Run Physics Engine**
*(Use `src/sensor/visibility.py` or synthetic generators to build raw visibility windows)*

**2. Run the Master Orchestrator**
This single command automatically runs **Phase 1** (The OR-Tools Math Teacher) to generate the perfect mathematical schedules based on continuous-time interval logic. It then immediately proceeds to **Phase 2** (The PyTorch AI Student) to train the Neural Network on the newly generated Answer Keys.
```bash
python main.py
```

*(Note: If you want to train the AI manually or test individual scripts, they are still accessible inside `src/scheduling/`)*

---

## 📚 Acknowledgements
This project's architecture is heavily inspired by the work of Dhingra, DeJac, and McGuire in their 2024 AMOS paper on Machine Learning for SDA Sensor Scheduling (Project Heimdall). 

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
