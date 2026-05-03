# Structural vs ML Modeling of Long-Run Growth  
### A Solow-Based, Data-Driven Study

This project evaluates whether classical economic theory (Solow growth model) or modern machine learning methods better explain and predict long-run economic growth across countries.

The core idea is to:

> derive the model → test it empirically → challenge it with ML → compare outcomes

---

## 🔍 Research Question

- Can Solow-style variables explain cross-country income differences?
- Can they **predict future economic growth**?
- Does machine learning outperform theory-based models?

---

## ⚙️ Methodology

### Stage 1 — Structural Model
- Implemented Solow growth model:
  \[
  \dot{k} = s k^\alpha - (n + g + \delta) k
  \]
- Simulated convergence dynamics and steady states
- Visualized:
  - convergence paths
  - break-even investment
  - parameter sensitivity

---

### Stage 2 — Empirical Solow Test
- Built cross-country dataset using World Bank WDI
- Transformed raw data → tidy panel
- Estimated:
  \[
  \ln y = \beta_0 + \beta_1 \ln s - \beta_2 \ln(n+g+\delta) + \varepsilon
  \]

#### Findings:
- Investment positively associated with income  
- Population growth negatively associated with income  
- Human capital significantly improves explanatory power  

---

### Stage 3 — ML Forecasting (Core Contribution)

#### Task:
Predict **10-year forward average GDP per capita growth**

#### Setup:
- Features:
  - Theory-only: `ln_y0`, `ln_s`, `ln_ngd`
  - Extended: + `ln_h` (human capital)
- Target:
  - future 10-year growth
- Split:
  - Train: 1990, 2000  
  - Test: 2010  
- No data leakage (strict time-based split)

---

## 📊 Results

### 1. Theory-only models

| Model | RMSE | R² (OOS) |
|------|------|----------|
| Ridge | ~0.0236 | negative |
| Linear | ~0.0237 | negative |

👉 Models slightly outperform mean baseline  
👉 Predictive power remains weak

---

### 2. Adding Human Capital

- Improves **in-sample explanation (Stage 2)**
- Does **NOT improve out-of-sample growth prediction**
- Same-sample comparison shows slightly worse performance

👉 Key insight:
> Human capital explains income levels, but not future growth dynamics

---

### 3. Nonlinear ML Models

| Model | RMSE |
|------|------|
| Random Forest | worse |
| Gradient Boosting | much worse |

👉 ML underperforms simple linear models

---

## 🧠 Key Insights

### 1. Theory vs Prediction
- Solow variables explain **income levels**
- They have **limited predictive power for growth**

---

### 2. Human Capital
- Strong explanatory variable for development
- Weak predictor for short-to-medium term growth

---

### 3. ML vs Theory
- ML does NOT outperform linear models in this setup
- Reason:
  - small dataset
  - weak signal
  - approximately linear relationships

---

## 📌 Final Conclusion

> Economic theory helps explain long-run equilibrium differences, but both theory-based and machine learning models struggle to predict future economic growth.

---

## 📁 Repository Structure

'''
text
solow-structural-vs-ml-growth/
├── data/
│ ├── raw/
│ └── processed/
├── notebooks/
│ ├── 01_solow_simulation.ipynb
│ ├── 02_empirical_solow.ipynb
│ └── 03_ml_growth_models.ipynb
├── src/
│ ├── simulation/
│ ├── data/
│ ├── features/
│ ├── models/
│ └── evaluation/
├── reports/
│ ├── figures/
│ ├── tables/
│ └── final_results.csv
├── environment.yml
└── README.md
'''

---

## 🔁 Reproducibility

```bash
conda env create -f environment.yml
conda activate solow-growth
jupyter notebook