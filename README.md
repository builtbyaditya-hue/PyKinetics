# PyKinetics: Real-Time Reactor Kinetics Simulator

An interactive, event-driven web dashboard that simulates nuclear reactor point kinetics with thermal feedback. 

This project solves stiff systems of ordinary differential equations (ODEs) to model real-time transient behavior in a nuclear core, tracking prompt and delayed neutron populations alongside thermodynamic changes.

## 🚀 Features
* **Real-Time Physics Engine:** Solves the Point Kinetics Equations (6 delayed neutron groups) on a continuous, asynchronous 150ms loop.
* **Stiff ODE Integration:** Utilizes SciPy's Backward Differentiation Formula (BDF) method with high-precision tolerances (`rtol=1e-6`, `atol=1e-9`) to prevent numerical oscillation.
* **Doppler Temperature Feedback:** Dynamically models negative reactivity insertion as core temperatures rise.
* **Zero-Latency Dashboard:** Built with Plotly and Dash using `uirevision` state locking for buttery-smooth, high-frequency continuous data streaming.

## 🛠️ Tech Stack
* **Core Logic:** Python, NumPy, SciPy (`solve_ivp`)
* **Frontend UI:** Dash, Plotly Graph Objects

## ⚙️ Installation & Usage
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/PyKinetics.git](https://github.com/YOUR-USERNAME/PyKinetics.git)