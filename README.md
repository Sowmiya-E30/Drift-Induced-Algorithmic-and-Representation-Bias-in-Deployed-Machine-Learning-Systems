# Drift Induced Algorithmic and Representation Bias in Deployed Machine Learning Systems

## 📌 Project Overview

This project presents a research-oriented machine learning workflow for analyzing **model performance, algorithmic bias, representation bias, and data drift** using the COMPAS dataset.

The project goes beyond simply training a classification model by simulating a real-world deployment environment where a trained model is exposed to changing data distributions.

The workflow includes:

* Loading and preprocessing COMPAS CSV data
* Cleaning and encoding tabular features
* Training multiple machine learning models
* Saving and loading trained models for deployment simulation
* Introducing controlled data drift without retraining
* Detecting drift using **Kolmogorov–Smirnov (KS) Test** and **Population Stability Index (PSI)**
* Measuring representation bias
* Measuring algorithmic fairness
* Comparing model accuracy and fairness
* Performing feature importance and bias-proxy analysis
* Visualizing model, drift, and bias behavior

The complete workflow is implemented in a Jupyter Notebook.

---

## 🎯 Objectives

The main objectives of this project are:

1. Build machine learning models using COMPAS data.
2. Evaluate model performance before deployment drift.
3. Simulate changing data distributions after deployment.
4. Measure the severity of data drift.
5. Analyze how drift affects model performance.
6. Evaluate fairness across demographic groups.
7. Examine representation changes before and after drift.
8. Identify features that may require additional bias auditing.

---

## 📊 Dataset

The project uses the **COMPAS dataset**.

The notebook loads schema-compatible CSV files from a configured `compas` directory and uses the following primary features:

### Numerical Features

* `RawScore`
* `DecileScore`

### Categorical Features

* `Sex_Code_Text`
* `Ethnic_Code_Text`

### Target

* `RecSupervisionLevel`

The dataset contains **60,843 records** in the loaded COMPAS CSV. After preprocessing, the same 60,843 records are retained with the five required source columns.

For fairness analysis, the target is converted into a binary outcome using the median threshold when necessary. In this experiment, the threshold was **1.000**.

---

## 🏗️ Project Workflow

```text
COMPAS Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Cleaning & Preparation
      │
      ▼
Categorical Encoding
      │
      ▼
Binary Target Creation
      │
      ▼
Train / Test Split
      │
      ▼
Model Training
 ┌────┼───────────────┐
 ▼    ▼               ▼
RF   GradientBoosting XGBoost
 └────┼───────────────┘
      │
      ▼
Save Models
      │
      ▼
Deployment Simulation
      │
      ▼
Controlled Data Drift
      │
      ├──────────────► KS Test
      │
      ├──────────────► PSI
      │
      ├──────────────► Representation Bias
      │
      └──────────────► Fairness Metrics
                         │
                         ▼
                Model Comparison
```

---

## 🤖 Machine Learning Models

Three classification models are evaluated:

### 1. Random Forest

A Random Forest classifier with 300 estimators is used as one of the baseline models.

### 2. Gradient Boosting

A Gradient Boosting classifier is trained using the same training data.

### 3. XGBoost

XGBoost is included when the package is available in the environment.

## All trained models are saved using `joblib` and subsequently loaded to simulate deployed models.

## 🚀 Deployment Simulation

Instead of evaluating only freshly trained models, the project simulates a production deployment environment.

The trained models are:

1. Saved to disk
2. Loaded again
3. Used on the original test distribution
4. Evaluated against increasingly drifted datasets

## Importantly, **the deployed models are not retrained during the drift experiment**, allowing the effect of distribution changes on a fixed model to be observed.

## 📈 Data Drift Simulation

Controlled drift is introduced at six levels:

```text
1.0
1.1
1.2
1.3
1.4
1.5
```

The drift primarily modifies:

* `RawScore`
* `DecileScore`

The simulation also introduces representation changes by resampling rows from the selected sensitive group.

This allows the project to study how increasing distribution changes influence deployed model behavior.

---

## 🔍 Drift Detection

Two statistical approaches are used.

### Kolmogorov–Smirnov (KS) Test

The KS test compares the reference and drifted distributions.

A p-value below `0.05` is treated as a drift flag in the notebook.

### Population Stability Index (PSI)

PSI is used to quantify distributional changes.

The project classifies PSI as:

|             PSI | Severity |
| --------------: | -------- |
|        `< 0.10` | Low      |
| `0.10 – < 0.25` | Moderate |
|       `>= 0.25` | High     |

These thresholds are implemented directly in the notebook.

---

## ⚖️ Fairness and Bias Analysis

The project evaluates fairness using:

### Demographic Parity Difference

Measures the difference in positive prediction rates between the selected demographic groups.

### Equal Opportunity Difference

Measures the difference in true-positive rates between demographic groups.

The sensitive attribute selected for this experiment is:

```text
Ethnic_Code_Text_African-American
```

## The fairness metrics are implemented as absolute differences, where a lower value indicates a smaller measured disparity.

## 📊 Baseline Results

Before introducing drift, the models achieved:

| Model            |   Accuracy | Demographic Parity Difference | Equal Opportunity Difference |
| ---------------- | ---------: | ----------------------------: | ---------------------------: |
| **XGBoost**      | **84.80%** |                        0.3732 |                       0.2606 |
| GradientBoosting |     84.78% |                        0.3871 |                       0.2777 |
| RandomForest     |     84.45% |                    **0.3479** |                   **0.2268** |

XGBoost achieved the highest baseline accuracy, while Random Forest produced the lowest demographic parity and equal opportunity differences among the three models.

---

## 📉 Effect of Severe Drift

At the highest simulated drift level of **1.5**, model performance decreased:

| Model            | Baseline Accuracy | Accuracy at Drift 1.5 |
| ---------------- | ----------------: | --------------------: |
| XGBoost          |            84.80% |            **81.09%** |
| RandomForest     |            84.45% |                78.53% |
| GradientBoosting |            84.78% |                77.95% |

For example, XGBoost's accuracy decreased from **84.80% to 81.09%** under the highest simulated drift level.
The fairness-related metrics also changed under drift, demonstrating that distribution shifts can affect both **model performance and measured disparities**.

---

## 🔬 Drift Detection Results

At drift level **1.50**, the notebook reports an overall drift severity score of:

```text
0.947
```

The individual feature results were:

| Feature     | KS Statistic |    PSI | PSI Severity | KS Drift Flag |
| ----------- | -----------: | -----: | ------------ | ------------- |
| RawScore    |       0.1850 | 1.2705 | High         | True          |
| DecileScore |       0.2685 | 0.1970 | Moderate     | True          |

The strongest PSI-based drift was observed for `RawScore`.

---

## 👥 Representation Bias

The test distribution before drift contained approximately:

* **78.17% Male**
* **21.83% Female**
* **44.04% African-American**
* **35.65% Caucasian**
* **14.88% Hispanic**

After the highest simulated drift level:

* Male: **78.19%**
* Female: **21.81%**
* African-American: **55.73%**
* Caucasian: **28.20%**
* Hispanic: **11.77%**

The largest representation change was observed for the **African-American group**, increasing by approximately **11.70 percentage points**.

---

## 🌳 Feature Importance and Bias Proxy

The Random Forest feature importance analysis found:

| Feature                    | Importance |
| -------------------------- | ---------: |
| DecileScore                |     0.6043 |
| RawScore                   |     0.2919 |
| African-American indicator |     0.0667 |
| Caucasian indicator        |     0.0162 |
| Hispanic indicator         |     0.0095 |

`DecileScore` was the most important feature, followed by `RawScore`.

A simple **bias proxy** was also calculated by combining feature importance with the group mean gap.

The highest proxy values were:

```text
DecileScore   → 0.779937
RawScore      → 0.180934
African-American indicator → 0.066683
```

## This proxy is intended only to identify features that may deserve further auditing; it is **not a causal explanation of bias**.

## 🏆 Key Findings

The experiment demonstrates several important observations:

* **XGBoost achieved the highest baseline accuracy** at approximately 84.80%.
* **Random Forest showed the lowest measured demographic parity difference** among the evaluated models.
* **Random Forest also showed the lowest equal opportunity difference**.
* Increasing simulated drift reduced model accuracy.
* Drift was successfully detected using both **KS testing and PSI**.
* `RawScore` experienced **high PSI-based drift** at the highest drift level.
* `DecileScore` experienced **moderate PSI-based drift**.
* Representation of demographic groups changed substantially under the simulated drift.
* The most important Random Forest features were `DecileScore` and `RawScore`.
* The results demonstrate why deployed ML systems should be monitored for **both data drift and fairness**, rather than relying only on initial model accuracy.

---

## 🛠️ Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* SciPy
* Scikit-learn
* XGBoost
* Joblib

---

## 📁 Project Structure

```text
project/
│
├── compas/
│   └── compas-scores-raw.csv
│
├── saved_models/
│   ├── randomforest_model.joblib
│   ├── gradientboosting_model.joblib
│   └── xgboost_model.joblib
│
└── compas_drift_bias_corrected.ipynb
```
---

## ⚠️ Important Note

This project is an **experimental analysis of the COMPAS dataset**. The controlled drift is simulated rather than observed from a live production system.

The bias-proxy analysis should not be interpreted as proving causal bias. It is intended to highlight features and group differences that may warrant additional investigation.

---

## 📌 Conclusion

This project demonstrates a complete monitoring-oriented machine learning workflow that considers more than model accuracy.

By combining **model evaluation, deployment simulation, controlled drift, statistical drift detection, representation analysis, and fairness metrics**, the project illustrates how machine learning systems can change after deployment and why continuous monitoring is important.

The experiment shows that a model with strong baseline accuracy can experience performance degradation and changes in fairness-related measurements when the underlying data distribution shifts.

---

## 👩‍💻 Project

**Title:** Drift, Bias, and Deployment Analysis on COMPAS

**Notebook:** `compas_drift_bias_corrected.ipynb`

**Focus Areas:**
Machine Learning • Model Monitoring • Data Drift • Algorithmic Fairness • Bias Detection • COMPAS • Responsible AI

## 👨‍💻 Author

Sowmiya E

Btech CSE AI & DS
