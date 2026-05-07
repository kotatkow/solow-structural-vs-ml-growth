# Structural vs Machine Learning Modeling of Long-Run Growth
## A Solow-Based, Data-Driven Study

**Project repository:** `solow-structural-vs-ml-growth`  
**Report version:** Final draft after aligned ML comparison  
**Author:** Seunghwan  

---

## Abstract

This report compares structural economic modeling and machine learning approaches in the study of long-run cross-country economic growth. The project begins with the Solow growth model, derives its core steady-state implications, and implements numerical simulations to illustrate convergence, break-even investment, and parameter sensitivity. It then tests the empirical implications of the Solow model using World Bank-style cross-country data, focusing on GDP per capita, investment rates, population growth, and a human-capital proxy. Finally, the project reframes the problem as a forecasting task: using information observed at year \(t\), can models predict average annual GDP per capita growth over the next ten years?

The main result is that structural explanation does not automatically imply forecasting power. The empirical Solow regressions support the qualitative predictions of the model: investment is positively associated with income per capita, and population growth is negatively associated with income through the capital dilution channel. Adding human capital substantially improves the explanatory power of the income-level regression. However, when the task changes from explaining income levels to predicting future growth, the same variables show weak out-of-sample predictive power. In the final aligned comparison, Ridge regression using the theory-only feature set achieves the lowest prediction error. Adding human capital does not improve same-sample out-of-sample growth prediction, and nonlinear models such as Random Forest and Gradient Boosting perform worse than the simpler Ridge baseline. The project therefore shows that flexible machine learning models do not automatically outperform theory-guided linear models when the dataset is small, the predictive signal is weak, and the underlying economic relationships are approximately linear.

---

## 1. Introduction

Long-run economic growth is one of the central questions in macroeconomics. Some countries remain persistently rich, while others grow slowly or remain trapped at lower income levels. At the same time, some initially poor countries experience rapid convergence, while others do not. Traditional economic theory approaches these questions through structural models, especially the Solow growth model, which emphasizes capital accumulation, population growth, technological progress, and convergence toward a steady state.

Modern machine learning approaches the same broad problem from a different angle. Instead of beginning with a fully specified structural model, machine learning focuses on predictive performance. Flexible algorithms can, in principle, capture nonlinear relationships, threshold effects, and interactions that are difficult to specify in a simple economic equation. This raises a natural question: can machine learning outperform theory-based models in forecasting long-run economic growth?

This project answers that question through a staged research design. The first stage builds the Solow model from theory and implements simulations to show the mechanics of convergence. The second stage tests whether Solow-style variables explain cross-country income differences using real data. The third stage converts the problem into a forecasting task and compares theory-based linear models with nonlinear machine learning models.

The project is not designed as a pure accuracy competition. Its goal is to understand the relationship between theory, explanation, and prediction. A model can explain income levels well without forecasting future growth well. Conversely, a flexible machine learning model can fail to generalize when the dataset is small or when the signal in the data is weak. The central contribution of this project is therefore methodological as well as empirical: it demonstrates how to compare structural economic models and machine learning models in a disciplined, no-leakage forecasting framework.

The final result is clear. Solow-style variables provide useful explanatory structure for income levels. Human capital improves the explanation of cross-country income differences. But future growth is much harder to predict. In the final same-sample benchmark, the best-performing model is a regularized linear model using theory-only features, while both human-capital-augmented models and nonlinear machine learning models perform worse.

---

## 2. Theoretical Framework: The Solow Growth Model

The theoretical baseline is the Solow growth model in per-effective-worker form. The model begins with a Cobb-Douglas production function:

\[
y = f(k) = k^\alpha,
\]

where \(y\) is output per effective worker, \(k\) is capital per effective worker, and \(\alpha\) is the capital share of output. The assumption \(0 < \alpha < 1\) implies diminishing marginal returns to capital. As capital accumulates, each additional unit of capital contributes less additional output than the previous one.

The law of motion for capital per effective worker is:

\[
\dot{k} = s k^\alpha - (n + g + \delta)k.
\]

Here, \(s\) is the savings or investment rate, \(n\) is population growth, \(g\) is technological growth, and \(\delta\) is the depreciation rate. The term \(s k^\alpha\) represents actual investment per effective worker. The term \((n+g+\delta)k\) represents break-even investment, which is the amount of investment required to keep capital per effective worker constant despite population growth, technological progress, and depreciation.

The steady state occurs when capital per effective worker no longer changes:

\[
\dot{k} = 0.
\]

Setting actual investment equal to break-even investment gives:

\[
s k^\alpha = (n+g+\delta)k.
\]

Solving for steady-state capital per effective worker gives:

\[
k^* = \left(\frac{s}{n+g+\delta}\right)^{\frac{1}{1-\alpha}}.
\]

Steady-state output per effective worker is:

\[
y^* = (k^*)^\alpha.
\]

These equations imply several testable and interpretable predictions. A higher investment rate raises steady-state capital and output. A higher population growth rate increases the amount of investment needed to maintain capital per effective worker, reducing steady-state capital and output. The capital share \(\alpha\) affects the curvature of production and the sensitivity of steady-state output to changes in investment and capital dilution.

This theoretical structure gives the project its first benchmark. If the Solow model is useful empirically, countries with higher investment rates should tend to have higher income per capita, while countries with higher population growth and capital dilution should tend to have lower income per capita.

---

## 3. Numerical Simulation and Structural Intuition

The first stage of the project implements a numerical Solow simulation. The continuous-time law of motion is approximated using Euler discretization:

\[
k_{t+1} = k_t + \Delta t \left[s k_t^\alpha - (n+g+\delta)k_t\right].
\]

This simulation makes the model dynamic and visual. Instead of only computing the steady state analytically, the code shows how capital per effective worker moves over time. If \(k_t < k^*\), investment exceeds break-even investment and capital rises. If \(k_t > k^*\), break-even investment exceeds actual investment and capital falls. In both cases, the economy moves toward the steady state.

### Figure 1. Convergence of capital per effective worker

**Source file:** `reports/figures/convergence_k.png`

This figure shows the simulated path of capital per effective worker, \(k(t)\), converging toward the analytical steady state, \(k^*\). The visual result confirms the basic stability property of the Solow model: given standard parameter values, the economy converges toward a steady-state level of capital per effective worker.

### Figure 2. Break-even investment and steady state

**Source file:** `reports/figures/break_even.png`

This figure plots actual investment, \(s f(k)\), against break-even investment, \((n+g+\delta)k\). The intersection of the two curves identifies the steady state. The diagram provides the core intuition of the model. When actual investment is above break-even investment, capital accumulates. When actual investment is below break-even investment, capital declines.

The simulation stage also includes parameter sweeps over the savings rate, population growth, and the capital share. These exercises show how the steady state responds to changes in key parameters. A higher savings rate shifts the investment curve upward and raises steady-state capital. Higher population growth shifts the break-even investment requirement upward and lowers steady-state capital per effective worker. Changes in \(\alpha\) affect both the curvature of the production function and the sensitivity of output to capital accumulation.

The simulation stage establishes that the code correctly reproduces the theoretical mechanics of the Solow model. It also produces the first set of research artifacts: simulation figures that can be used to explain the structural baseline before moving into empirical testing.

---

## 4. Data and Empirical Strategy

The empirical analysis uses World Bank-style cross-country macroeconomic data. The dataset is organized into decade snapshots: 1990, 2000, 2010, and 2020. This structure is useful because it supports both income-level regressions and ten-year forward growth prediction.

The core variables are:

| Variable | Empirical measure | Role in model |
|---|---|---|
| \(y\) | GDP per capita, constant USD | Income level |
| \(s\) | Gross capital formation as % of GDP | Investment/savings proxy |
| \(n\) | Population growth rate | Capital dilution component |
| \(h\) | Secondary school enrollment | Human-capital proxy |

The raw data initially appears in a wide format, with one row per country-indicator pair and separate year columns such as `YR1990`, `YR2000`, `YR2010`, and `YR2020`. The project reshapes this into a tidy panel where each row represents one country-year observation and each variable has its own column. This transformation is essential because both econometric regressions and machine learning models require one observation per row.

The baseline empirical Solow regression is:

\[
\ln y = \beta_0 + \beta_1 \ln s + \beta_2 \ln(n+g+\delta) + \varepsilon.
\]

The expected signs are:

\[
\beta_1 > 0, \qquad \beta_2 < 0.
\]

The project assumes:

\[
g = 0.02, \qquad \delta = 0.03,
\]

so that:

\[
g + \delta = 0.05.
\]

Population growth is converted from percentage form into decimal form before constructing \(n+g+\delta\). The transformed regression variables are:

\[
\ln y, \qquad \ln s, \qquad \ln(n+g+\delta).
\]

The augmented Solow regression adds human capital:

\[
\ln y = \beta_0 + \beta_1 \ln s + \beta_2 \ln(n+g+\delta) + \beta_3 \ln h + \varepsilon.
\]

The purpose of the empirical stage is explanatory. It asks whether the structural variables are associated with income levels in the direction predicted by the Solow model. This is different from the later machine learning stage, which asks whether the same variables can predict future growth.

---

## 5. Empirical Solow Results

The baseline empirical Solow regression supports the qualitative predictions of the model. The coefficient on investment is positive and statistically significant, while the coefficient on the capital dilution term is strongly negative. The baseline regression uses approximately 788 country-year observations and has an \(R^2\) of about 0.195.

### Table 1. Baseline empirical Solow regression summary

| Variable | Approximate coefficient | Interpretation |
|---|---:|---|
| \(\ln s\) | 0.296 | Higher investment is associated with higher income per capita |
| \(\ln(n+g+\delta)\) | -2.926 | Higher capital dilution is associated with lower income per capita |
| \(R^2\) | 0.195 | Basic Solow variables explain part, but not all, of income variation |
| Observations | 788 | Country-year snapshot observations |

The baseline result is meaningful because the signs match the theory. Countries with higher investment rates tend to have higher income per capita, and countries with higher population growth and capital dilution tend to have lower income per capita. However, the relatively modest \(R^2\) shows that the basic Solow model is incomplete. Many important determinants of income are omitted, including human capital, institutions, technology, geography, trade integration, and macroeconomic stability.

The augmented Solow model adds human capital through a school-enrollment proxy. This substantially increases explanatory power. The \(R^2\) rises to approximately 0.568, and the coefficient on human capital is strongly positive. This suggests that education and worker capability explain a large portion of cross-country income differences.

### Table 2. Augmented Solow regression summary

| Result | Interpretation |
|---|---|
| \(R^2\) rises from about 0.195 to about 0.568 | Human capital greatly improves income-level explanation |
| \(\ln h\) is strongly positive | Higher education is associated with higher income per capita |
| \(\ln s\) becomes weaker and changes sign | Investment alone explains less after education is controlled for |
| No severe multicollinearity detected | Coefficient changes reflect reallocation of explanatory power, not unstable collinearity |

A key diagnostic result is that the coefficient change is not mainly caused by severe multicollinearity. The pairwise correlation between \(\ln s\) and \(\ln h\) is low, and the VIF values are near 1 for the main regressors. The estimated VIF values are approximately:

| Variable | VIF |
|---|---:|
| \(\ln s\) | 1.04 |
| \(\ln(n+g+\delta)\) | 1.57 |
| \(\ln h\) | 1.62 |

These diagnostics indicate that the augmented model is not suffering from serious multicollinearity. Instead, the interpretation is structural: part of what investment seemed to explain in the baseline model was actually related to broader development conditions captured better by human capital. Once education is explicitly included, human capital becomes the dominant explanatory variable for income levels.

The main conclusion from the empirical stage is therefore:

> The Solow model provides useful qualitative structure for explaining income levels, but the basic model is incomplete. Human capital substantially improves the explanation of cross-country income differences.

However, this conclusion applies to income levels. It does not automatically mean the same variables will forecast future growth well. That distinction motivates the machine learning stage.

---

## 6. Machine Learning Forecasting Design

The machine learning stage changes the research question. Instead of asking which variables explain current income, it asks whether variables observed at year \(t\) can predict future economic growth over the next ten years.

The target variable is ten-year forward average annual GDP per capita growth:

\[
\text{growth}_{i,t,t+10} = \frac{\ln(y_{i,t+10}) - \ln(y_{i,t})}{10}.
\]

This target is constructed by sorting observations by country and year, then shifting GDP per capita one decade forward. Because the dataset uses decade snapshots, one row forward corresponds to a ten-year horizon. Observations from 2020 cannot be used as input rows for prediction because the dataset does not contain 2030 GDP per capita.

The theory-only feature set is:

\[
\{\ln y_0, \ln s, \ln(n+g+\delta)\}.
\]

Here, \(\ln y_0\) is initial income. It is included because convergence theory suggests that poorer countries may grow faster, conditional on other factors. The theory-plus-human-capital feature set adds:

\[
\ln h.
\]

The train-test split is time-based rather than random. Observations from 1990 and 2000 are used for training, while observations from 2010 are used for testing. This design prevents data leakage and better reflects the forecasting problem. A random split would mix earlier and later observations and could make the model appear more predictive than it really is.

The evaluated models are:

1. Mean predictor
2. Linear regression
3. Ridge regression
4. Lasso regression
5. Random Forest Regressor
6. Gradient Boosting Regressor

The evaluation metrics are:

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2},
\]

\[
MAE = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|,
\]

and out-of-sample \(R^2\):

\[
R^2_{OOS} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{test})^2}.
\]

Out-of-sample \(R^2\) is interpreted carefully. A negative value means that the model performs worse than predicting the test-set mean. However, RMSE and MAE are also important because they directly measure forecast error.

---

## 7. Initial ML Results and Sample Alignment Issue

The first theory-only benchmark shows weak but real predictive signal. Ridge regression performs slightly better than the mean predictor and slightly better than ordinary linear regression. However, out-of-sample \(R^2\) remains negative, showing that the improvement is small relative to the variation in future growth.

The initial theory-plus-human-capital comparison appeared to improve RMSE and MAE. In the non-aligned comparison, Ridge + H achieved lower RMSE than Ridge using theory-only features. However, this comparison was not fully valid because adding human capital reduced the test sample. Education data were missing for many countries, so the theory-plus-human-capital model was evaluated on a different set of countries than the theory-only model.

This creates a sample-selection problem. If two models are evaluated on different countries, their RMSE and \(R^2\) are not directly comparable. The apparent improvement could reflect a different test sample rather than a genuinely better model.

To correct this, the project performs a same-sample aligned comparison. Both theory-only and theory-plus-human-capital models are re-estimated and evaluated on the same country sample. This aligned comparison is the scientifically relevant benchmark.

---

## 8. Final Aligned Model Comparison

The final aligned model-comparison table is the central result of the project.

### Table 3. Final aligned model comparison

| Model | RMSE | MAE | Out-of-sample \(R^2\) | Feature set |
|---|---:|---:|---:|---|
| Ridge | 0.02365 | 0.01693 | -0.110 | Theory-only, aligned |
| Linear | 0.02367 | 0.01699 | -0.112 | Theory-only, aligned |
| Lasso | 0.02455 | 0.01819 | -0.196 | Theory-only, aligned |
| Mean | 0.02570 | 0.01953 | -0.311 | Theory-only, aligned |
| Ridge + H | 0.02431 | 0.01719 | -0.173 | Theory + human capital, aligned |
| Linear + H | 0.02435 | 0.01733 | -0.177 | Theory + human capital, aligned |
| Lasso + H | 0.02437 | 0.01718 | -0.180 | Theory + human capital, aligned |
| Mean + H | 0.02575 | 0.01958 | -0.317 | Theory + human capital, aligned |
| Random Forest | 0.02557 | 0.01763 | -0.298 | Nonlinear ML, aligned |
| Gradient Boosting | 0.02790 | 0.01949 | -0.545 | Nonlinear ML, aligned |

The best-performing model is Ridge regression using the theory-only aligned feature set. It achieves the lowest RMSE and MAE among all models. Ordinary linear regression is very close, while Lasso performs worse. The mean predictor performs worse than the theory-based linear models, which suggests that the features contain some predictive signal. However, all out-of-sample \(R^2\) values remain negative, meaning that none of the models strongly predicts future ten-year growth relative to the test-set mean benchmark.

The aligned comparison changes the interpretation of human capital. In the income-level regression, human capital strongly improves explanation. But in the aligned forecasting setup, adding human capital worsens out-of-sample performance. The best Ridge + H model has RMSE of 0.02431, compared with 0.02365 for theory-only Ridge. This means that human capital helps explain where countries are in terms of income levels, but it does not improve prediction of how fast they will grow over the next decade in this dataset.

The nonlinear models perform worse than the linear models. Random Forest has RMSE of 0.02557, and Gradient Boosting has RMSE of 0.02790. Both are worse than Ridge and even close to or worse than the mean benchmark. This suggests that model flexibility did not improve generalization.

The final machine learning conclusion is:

> Flexible nonlinear machine learning models do not outperform simple theory-guided linear models in this small-sample, weak-signal growth forecasting setting.

---

## 9. Interpretation: Why Did Simple Models Win?

The final result may seem surprising because Random Forest and Gradient Boosting are more flexible than linear regression. However, the result is economically and statistically reasonable.

First, the dataset is small. After alignment, the training sample contains only a few hundred observations. Tree-based machine learning models usually require more data to learn stable nonlinear patterns. With limited data, they can easily fit noise rather than generalizable structure.

Second, the predictive signal is weak. The Solow variables are theoretically meaningful, but future growth is noisy. Growth over a decade can be affected by financial crises, commodity shocks, political transitions, wars, institutional reforms, external demand, exchange-rate movements, and technological diffusion. These factors are not fully captured by the narrow Solow feature set.

Third, the relationships encoded by the theory variables are approximately linear in log form. The Solow model itself motivates a log-linear empirical structure. Ridge regression has the right inductive bias: it preserves the linear structure while stabilizing coefficient estimates through regularization.

Fourth, human capital is slow-moving. Education helps explain long-run income levels because it reflects accumulated productive capacity. But future growth depends not only on the level of human capital but also on changes, shocks, and the ability to deploy skills productively. A high level of education does not automatically imply high future growth.

Fifth, out-of-sample prediction is harder than in-sample explanation. Stage 2 shows that human capital improves the explanation of income levels. Stage 3 shows that this does not translate into better future growth forecasts. This distinction is one of the most important lessons of the project.

---

## 10. Synthesis: Structural Explanation Does Not Imply Forecasting Power

The project’s central intellectual result is the distinction between structural explanation and forecasting power.

The empirical Solow regressions show that theory-based variables explain part of cross-country income differences. Investment has the expected positive association with income, and the capital dilution term has the expected negative association. Human capital substantially improves the explanation of income levels.

However, the forecasting results show that these same variables have limited ability to predict future growth. The best model is a simple Ridge regression using theory-only features, but even that model has negative out-of-sample \(R^2\). Adding human capital does not improve same-sample forecasting. Nonlinear machine learning performs worse.

This does not mean the Solow model is useless, and it does not mean machine learning is useless. Instead, it means that each tool has limits. Structural theory is valuable for organizing mechanisms and explaining long-run equilibrium relationships. Machine learning is valuable for testing predictive performance and exploring richer patterns. But when data are limited and the signal is weak, a more flexible model may not help.

The project therefore supports a balanced approach:

1. Use economic theory to define meaningful variables and avoid arbitrary feature selection.
2. Use empirical regression to test whether theoretical relationships appear in real data.
3. Use machine learning to evaluate forecasting performance out of sample.
4. Interpret results through both statistical metrics and economic reasoning.

The strongest conclusion is:

> A model can explain long-run income differences without being able to forecast future growth accurately.

---

## 11. Limitations

This project has several important limitations.

First, the dataset is small. The decade-snapshot design is clean and avoids leakage, but it limits the number of usable observations. This is especially restrictive for nonlinear machine learning models.

Second, the feature set is narrow. The project focuses on Solow-style variables: initial income, investment, population growth, and human capital. Many important growth determinants are not included, such as institutions, trade openness, inflation, financial development, political stability, industrial structure, commodity dependence, and technology adoption.

Third, the human-capital proxy is imperfect. Secondary school enrollment is practical and accessible, but it does not fully capture education quality, skills, health, worker productivity, or institutional ability to use human capital effectively.

Fourth, the empirical regressions use pooled country-year observations. More advanced panel methods could account for country fixed effects, regional effects, or clustered standard errors.

Fifth, the forecasting evaluation uses one main test decade. A stronger design would use rolling-origin validation across more historical periods, if additional data are collected.

Sixth, the nonlinear models are relatively basic. More careful hyperparameter tuning, cross-validation within the training period, or richer feature engineering may improve performance. However, this should be done carefully to avoid overfitting.

---

## 12. Next Steps

The next stage should expand the project while preserving the disciplined no-leakage design.

A first improvement is to add broader macroeconomic variables, such as inflation, trade openness, government consumption, unemployment, external debt, financial depth, and GDP growth volatility. These variables may capture short- and medium-term growth dynamics better than slow-moving structural variables.

A second improvement is to add institutional indicators, such as rule of law, governance quality, corruption control, political stability, and regulatory quality. These may help explain why some countries convert investment and human capital into growth more effectively than others.

A third improvement is to expand the time dimension. Instead of using only decade snapshots, the project could use annual or five-year panels. This would increase sample size and give machine learning models more data, while still preserving proper forecasting horizons.

A fourth improvement is to use rolling-origin validation. For example, models could train on earlier periods and test on multiple later periods. This would produce a more robust estimate of forecasting performance.

A fifth improvement is to add interpretability tools. For linear models, standardized coefficients and coefficient stability can be reported. For tree models, permutation importance can show which variables matter most. If richer ML models are used later, SHAP values could be considered.

Finally, the project should be packaged into a polished research artifact: clean notebooks, saved figures, final tables, a complete README, and a PDF version of this report. This would make the repository suitable as a portfolio project demonstrating economic modeling, data engineering, econometrics, and machine learning evaluation.

---

## 13. Conclusion

This project compares structural economic modeling and machine learning in the analysis of long-run growth. The Solow model provides the theoretical foundation. Simulations show convergence toward a steady state and illustrate the roles of investment, population growth, depreciation, and capital share. Empirical regressions confirm that Solow-style variables explain part of cross-country income differences, and human capital substantially improves income-level explanation.

However, the forecasting results are more cautious. Predicting future ten-year GDP per capita growth is difficult. In the final aligned comparison, Ridge regression using theory-only features performs best. Adding human capital does not improve same-sample out-of-sample prediction. Random Forest and Gradient Boosting perform worse than the simpler Ridge model.

The main lesson is not that theory always beats machine learning, nor that machine learning is ineffective. The lesson is that model performance depends on data size, signal strength, feature quality, and the structure of the problem. In this project, the data are limited, the growth signal is weak, and the theory-guided linear structure is a better inductive bias than flexible nonlinear modeling.

The final conclusion is:

> Structural economic models help explain long-run income differences, but they do not automatically provide strong forecasts of future growth. Machine learning does not automatically solve this problem when the data are small and the signal is weak.

This makes the project valuable as both an economics study and a machine learning lesson. It shows how to move from theory to data, from explanation to prediction, and from model accuracy to careful interpretation.

---

## References

Solow, R. M. (1956). *A Contribution to the Theory of Economic Growth.* Quarterly Journal of Economics.

Mankiw, N. G., Romer, D., & Weil, D. N. (1992). *A Contribution to the Empirics of Economic Growth.* Quarterly Journal of Economics.

World Bank. World Development Indicators.

Project repository artifacts:

- `notebooks/01_solow_simulation.ipynb`
- `notebooks/02_empirical_solow.ipynb`
- `notebooks/03_ml_growth_models.ipynb`
- `reports/figures/convergence_k.png`
- `reports/figures/break_even.png`
- `reports/tables/model_comparison_extended.csv`
- `reports/tables/final_model_comparison.csv`

