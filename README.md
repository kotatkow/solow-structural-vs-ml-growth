# Structural vs ML Modeling of Long-Run Growth
### A Solow-Based, Data-Driven Study

This project compares structural economic modeling (Solow growth model) with modern machine learning approaches for predicting long-run economic growth.

The objective is to evaluate whether economic theory-based models outperform purely data-driven models in forecasting cross-country growth.

---

## Project Status

🟢 Stage 1 — Structural Solow Model (Completed)  
🟡 Stage 2 — Empirical Solow Test (Upcoming)  
⚪ Stage 3 — ML Comparison (Planned)

---

## Stage 1: Structural Solow Model

We implement the continuous-time Solow growth model in per-effective-worker form:

\[
\dot{k} = s k^\alpha - (n + g + \delta) k
\]

Steady-state capital:

\[
k^* = \left( \frac{s}{n + g + \delta} \right)^{\frac{1}{1-\alpha}}
\]

Steady-state output:

\[
y^* = (k^*)^\alpha
\]

### What Was Implemented

- Numerical simulation via Euler discretization
- Analytical steady-state solution
- Break-even investment diagram
- Parameter sensitivity analysis:
  - Savings rate (s)
  - Population growth (n)
  - Capital share (α)

### Key Insights

- Higher savings increases steady-state income.
- Higher population growth reduces steady-state capital.
- Capital share affects curvature and convergence dynamics.
- The model exhibits stable convergence toward equilibrium.

Figures are available in:
reports/figures/


---

## Repository Structure
solow-structural-vs-ml-growth/
├── src/simulation/solow.py
├── notebooks/01_solow_simulation.ipynb
├── reports/figures/
├── data/
└── environment.yml


---

## Reproducibility

```bash
conda env create -f environment.yml
conda activate solow-growth
jupyter notebook