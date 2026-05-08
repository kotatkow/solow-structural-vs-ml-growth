# Structural vs ML Modeling of Long-Run Growth  
### A Solow-Based, Data-Driven Study

This project evaluates whether classical economic theory (the Solow growth model) or modern machine learning methods better explain and predict long-run economic growth across countries.

The core idea is to:

> derive the model → test it empirically → challenge it with ML → compare outcomes

---

**Project status:** Completed v1

---

## 🔍 Research Question

- Can Solow-style variables explain cross-country income differences?
- Can they **predict future economic growth**?
- Does machine learning outperform theory-based models?

---

## ⚙️ Methodology

### Stage 1 — Structural Model

Implemented the Solow growth model in per-effective-worker form:

$$
\dot{k} = s k^{\alpha} - (n + g + \delta)k
$$

where:

- $k$ = capital per effective worker
- $s$ = savings / investment rate
- $n$ = population growth rate
- $g$ = technological growth rate
- $\delta$ = depreciation rate
- $\alpha$ = capital share of output

The steady-state level of capital per effective worker is:

$$
k^* = \left(\frac{s}{n + g + \delta}\right)^{\frac{1}{1-\alpha}}
$$

Steady-state output per effective worker is:

$$
y^* = (k^*)^{\alpha}
$$

Implemented and visualized:

- convergence paths
- break-even investment
- parameter sensitivity

---

### Stage 2 — Empirical Solow Test

Built a cross-country dataset using World Bank WDI data and transformed the raw data into a tidy panel.

Estimated the empirical Solow equation:

$$
\ln y = \beta_0 + \beta_1 \ln s - \beta_2 \ln(n + g + \delta) + \varepsilon
$$

#### Findings

- Investment is positively associated with income.
- Population growth is negatively associated with income.
- Human capital significantly improves explanatory power.

---

### Stage 3 — ML Forecasting (Core Contribution)

#### Task

Predict **10-year forward average GDP per capita growth**.

The prediction target is defined as:

$$
g_{i,t,t+10} = \frac{\ln y_{i,t+10} - \ln y_{i,t}}{10}
$$

#### Setup

- Features:
  - Theory-only: `ln_y0`, `ln_s`, `ln_ngd`
  - Extended: + `ln_h` (human capital)
- Target:
  - future 10-year average GDP per capita growth
- Split:
  - Train: 1990, 2000  
  - Test: 2010  
- No data leakage: strict time-based split

---

## 📊 Results

### 1. Theory-only models

| Model | RMSE | R² (OOS) |
|---|---:|---:|
| Ridge | ~0.0236 | negative |
| Linear | ~0.0237 | negative |

Models slightly outperform the mean baseline, but predictive power remains weak.

---

### 2. Adding Human Capital

- Improves **in-sample explanation** in Stage 2.
- Does **not** improve out-of-sample growth prediction.
- Same-sample comparison shows slightly worse performance.

Key insight:

> Human capital explains income levels, but not future growth dynamics.

---

### 3. Nonlinear ML Models

| Model | RMSE |
|---|---:|
| Random Forest | worse |
| Gradient Boosting | much worse |

Nonlinear ML models underperform simple linear models in this setup.

---

## 🧠 Key Insights

### 1. Theory vs Prediction

- Solow variables explain **income levels**.
- They have **limited predictive power for growth**.

---

### 2. Human Capital

- Strong explanatory variable for development.
- Weak predictor for short-to-medium-term growth.

---

### 3. ML vs Theory

ML does not outperform linear models in this setup because:

- the dataset is small,
- the predictive signal is weak,
- the relationships are approximately linear.

---

## 📌 Final Conclusion

> Economic theory helps explain long-run equilibrium differences, but both theory-based and machine learning models struggle to predict future economic growth.

---

## 📁 Repository Structure

```text
solow-structural-vs-ml-growth/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_solow_simulation.ipynb
│   ├── 02_empirical_solow.ipynb
│   └── 03_ml_growth_models.ipynb
├── src/
│   ├── simulation/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── reports/
│   ├── figures/
│   ├── tables/
│   └── final_results.csv
├── environment.yml
└── README.md
```

---

## 🔁 Reproducibility

```bash
conda env create -f environment.yml
conda activate solow-growth
jupyter notebook
```
