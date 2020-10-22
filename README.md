# Measuring the Impact‌ ‌of‌ ‌Exposure‌ ‌to‌ ‌COVID-19‌ ‌Vaccine‌ ‌Misinformation‌ ‌on‌ Vaccine Intent
*Sahil Loomba, Alexandre de Figueiredo, Simon Piatek, Kristen de Graaf, Heidi Larson*


## Jupyter Notebooks
These notebooks are intended to aid importing, transforming, and analysing the survey data in this study.

1. `import_data.ipynb`: demo of reading and transforming survey data for use in any downstream statistical modeling
2. `statistical_analyses.ipynb`: demo statistical modeling and generation of figures and tables in the paper

## Directory Structure
1. `.dat/`: contains raw and processed survey data
2. `.doc/`: contains survey questionnaire
3. `.src/models.py`: contains functions to define and fit all Stan models described in the paper
4. `.src/utils.py`: contains helper functions to import and transform survey data, compute and plot posterior statistics
5. `.src/bayesoc.py`: defines python classes `Dim()`, `Outcome()`, `Society()` and `Model()` to implement general Bayesian socio-demographic models using `pystan`