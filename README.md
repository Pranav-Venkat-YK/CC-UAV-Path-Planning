# 🚁 Energy Consumption Optimization of UAV-Assisted Traffic Monitoring

A Python-based simulation and analysis framework implementing **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** and its optimized variant **Tiny-MAUP** for UAV energy-efficient path planning in highway traffic monitoring scenarios.

---

## 📋 Overview

This project simulates UAV (drone)-based traffic monitoring and evaluates **energy consumption optimization** using **multi-agent reinforcement learning (MARL)**.  
It is based on the research paper *“Energy Consumption Optimization of UAV-Assisted Traffic Monitoring Scheme With Tiny Reinforcement Learning” (IEEE IoT Journal, 2024)*.

The implementation models:

- Highway traffic monitoring with multiple UAVs  
- **Flight and hover energy consumption** using physical models  
- **Multi-agent coordination** via MADDPG  
- **Lightweight model pruning** in Tiny-MAUP for deployment efficiency  
- **Comparative analysis** with baseline algorithms (DDPG and Random Policy)

---

## 🚀 Features

- **Realistic UAV energy model** (hover + flight + communication)
- **MADDPG-based cooperative path planning**
- **Tiny-MAUP variant** with reduced hidden layer neurons and fine-grained pruning
- **Configurable environment parameters** (UAV count, area size, time slots)
- **Visualization of training progress** and energy convergence
- **Performance comparison** between algorithms:
  - Total Energy Consumption
  - Convergence Time
  - Model Complexity and Memory Usage

---

## 🔬 Algorithms Implemented

### 🧠 MAUP (Multi-Agent UAV Path Planning)
- Based on **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**
- Centralized training, decentralized execution
- UAVs cooperatively learn energy-efficient flight strategies
- Each agent optimizes:
  - Flight direction (ω)
  - Flight distance (d)
- Reward inversely proportional to energy consumption
- 
 $$
  R_i = \begin{cases} P_1, & \text{If UAV flies away from the task area} \\
  \frac{1}{E_{j(t)}}, & \text{Otherwise.} \end{cases} \
$$

---

### ⚡ Tiny-MAUP (Optimized Lightweight Variant)
- Reduces neurons in hidden layers (from 128 → 16)
- Applies **L1 norm pruning** for low-weight connections
- Achieves significant reduction in:
  - Computation cost  
  - Memory usage  
  - Energy consumption during model inference  
- Retains near-identical convergence performance compared to full MAUP

---

## ⚙️ System Model

- **Scenario**: UAVs monitor a fixed-length highway region  
- **Altitude**: 50 m (constant for all UAVs)  
- **Energy Components**:
  - Flight energy (proportional to velocity and drag)
  - Hover energy (constant)
  - Communication energy (transmission with base station)
- **Objective**:  
  Minimize total UAV energy consumption
  
   min $\sum_{j=1}^{M} E_j$
  
  s.t. Constraints

---


## 🖥️ Installation

### Prerequisites
- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- Matplotlib  

### Setup
```bash
git clone https://github.com/Pranav-Venkat-YK/UAV_Energy_Optimization.git
cd UAV_Energy_Optimization
pip install -r requirements.txt
```

---

## 📊 Key Insights

- **Tiny-MAUP** achieves **comparable accuracy** with **>60% lower energy and memory usage**.  
- Optimal learning rate **α = 0.0001** ensures smooth convergence.  
- Smaller soft-update coefficient (**τ = 0.01**) yields stable training.  
- UAVs learn to remain within mission zones after ~18k episodes.  
- Tiny-MAUP demonstrates strong potential for **onboard deployment** in small UAVs or edge devices.

---

