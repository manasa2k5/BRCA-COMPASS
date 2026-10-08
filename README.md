\# BRCA-COMPASS



\### Explainable BRCA1 Variant Interpretation and VUS Prioritization Framework



BRCA-COMPASS is an explainable machine-learning framework for BRCA1 variant interpretation and prioritization of Variants of Uncertain Significance (VUS).



The framework combines curated variant annotations, machine-learning-based pathogenicity prediction, probability calibration, SHAP-based explainability, ACMG/AMP evidence interpretation, and VUS prioritization into a unified workflow.



\---



\## Overview



BRCA1 variants can be difficult to interpret because a large number of variants remain uncertain even after clinical and computational evaluation.



BRCA-COMPASS addresses this challenge through a multi-stage computational framework:



1\. BRCA1-specific variant curation and quality control

2\. Feature engineering using functional and population-level annotations

3\. Machine-learning-based pathogenicity prediction

4\. Probability calibration

5\. SHAP-based model explainability

6\. Clinical evidence interpretation

7\. VUS prioritization

8\. Patient-level multi-variant analysis

9\. Clinician-oriented reporting



\---



\## Framework



The BRCA-COMPASS workflow consists of the following major components:



\*\*Data Sources\*\*  

↓  

\*\*Data Curation and Quality Control\*\*  

↓  

\*\*Feature Engineering\*\*  

↓  

\*\*Explainable Pathogenicity Engine\*\*  

↓  

\*\*Calibration and Threshold Optimization\*\*  

↓  

\*\*ACMG/AMP Evidence Interpretation\*\*  

↓  

\*\*VUS Prioritization\*\*  

↓  

\*\*Patient-Level Interpretation\*\*  

↓  

\*\*Clinician-Ready Reports\*\*



\---



\## Features



The framework uses a curated set of variant-level features covering multiple categories, including:



\- Population frequency

\- Evolutionary conservation

\- Functional prediction scores

\- Variant effect information

\- Protein/domain annotations

\- Other functional and computational evidence



The final feature set contains \*\*48 curated features\*\*.



\---



\## Machine Learning



The primary pathogenicity prediction model is based on \*\*XGBoost\*\*.



The workflow also includes:



\- Cross-validation

\- Class-balanced evaluation

\- Probability calibration using Platt scaling

\- Threshold optimization

\- SHAP-based feature importance and individual prediction explanations



\---



\## VUS Prioritization



Variants classified as VUS are excluded from the supervised training split and evaluated separately through the VUS prioritization framework.



The framework assigns VUS variants to risk/confidence categories based on their predicted pathogenicity probability and interpretation criteria.



This allows potentially high-risk VUS variants to be prioritized for further investigation.



\---



\## Explainability



BRCA-COMPASS uses SHAP (SHapley Additive exPlanations) to provide interpretable explanations for model predictions.



The repository contains:



\- Global feature importance

\- SHAP summary plots

\- SHAP beeswarm plots

\- Individual prediction explanations

\- Pathogenic and benign variant explanations



\---



\## External Validation



The framework includes an external validation workflow using an independent BRCA1 functional dataset.



The repository contains the corresponding:



\- External validation scripts

\- Evaluation metrics

\- ROC/PR analyses

\- Calibration analyses

\- Validation figures



\---



\## Results



The repository contains generated results and figures for:



\- ROC curves

\- Precision-recall curves

\- Confusion matrices

\- Calibration

\- Feature importance

\- SHAP analysis

\- Ablation studies

\- VUS prioritization

\- External validation

\- Manuscript figures



\---



\## Repository Structure



```text

BRCA-COMPASS/

│

├── models/

│   ├── imputer.pkl

│   ├── platt\_scaler.pkl

│   ├── threshold.pkl

│   └── xgb\_base.pkl

│

├── src/

│   ├── data\_loader.py

│   ├── feature\_engineer.py

│   ├── external\_validation.py

│   ├── external\_validation\_v2.py

│   ├── ablation\_study.py

│   ├── audit\_study.py

│   ├── module1\_vus\_prioritization.py

│   ├── module2\_patient\_reports.py

│   └── ...

│

├── results/

│   ├── figures/

│   ├── metrics/

│   ├── reports/

│   └── supplementary/

│

├── combine\_outputs\_figure.py

├── combine\_results\_figure.py

├── plot\_ablation.py

├── requirements.txt

└── README.md

