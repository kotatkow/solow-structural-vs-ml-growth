# Structural vs ML Modeling of Long-Run Growth

This project compares structural economic modeling (Solow growth model) with machine learning approaches for predicting long-run economic growth.

## Objectives

- Derive and simulate the Solow growth model
- Empirically test steady-state predictions
- Compare structural models vs ML models
- Evaluate generalization performance

## Project Structure

- `src/simulation/` → Solow model implementation
- `notebooks/` → experiments and analysis
- `data/` → raw and processed datasets
- `reports/` → figures and final report

## Reproducibility

```bash
conda env create -f environment.yml
conda activate solow-growth