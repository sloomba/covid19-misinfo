# Measuring the Impact‌ ‌of‌ ‌Exposure‌ ‌to‌ ‌COVID-19‌ ‌Vaccine‌ ‌Misinformation‌ ‌on‌ Vaccine Intent
*Sahil Loomba, Alexandre de Figueiredo, Simon Piatek, Kristen de Graaf, Heidi Larson*

## Jupyter Notebooks
These notebooks are intended to aid importing, transforming, and analysing the survey data in this study. You may use [Jupyter nbviewer](https://nbviewer.jupyter.org/) to view these notebooks, or view their static (.html) versions in `.doc/`, or view them on OSF.

1. `import_data.ipynb`: demo of reading and transforming survey data for use in any downstream statistical modeling; [view on OSF](https://osf.io/ej4c6/)
2. `statistical_analyses.ipynb`: demo statistical modeling and generation of figures and tables in the paper; [view on OSF](https://osf.io/b3qkc/)

## Directory Structure
1. `.dat/`: contains processed survey data; sufficient to run all statistical analyses in the paper
2. `.doc/`: contains full survey questionnaire and static (.html) versions of the Jupyter Notebooks
3. `.src/models.py`: contains functions to define and fit all Bayesian models described in the paper
4. `.src/utils.py`: contains helper functions to import and transform survey data, compute and plot posterior statistics
5. `.src/bayesoc.py`: defines python classes `Dim()`, `Outcome()`, `Society()` and `Model()` to implement general Bayesian socio-demographic models using `pystan`

## Citation
If you use this data, cite it as:

Loomba, S., de Figueiredo, A., Piatek, S., de Graaf, K., & Larson, H. (2020, October 23). Measuring the Impact‌ ‌of‌ ‌Exposure‌ ‌to‌ ‌COVID-19‌ ‌Vaccine‌ ‌Misinformation‌ ‌on‌ Vaccine Intent in the UK and US. Retrieved from osf.io/cxwvp

## Links
1. View project on [OSF](https://osf.io/cxwvp/)
2. View project on [GitHub](https://github.com/sloomba/covid19-misinfo/)