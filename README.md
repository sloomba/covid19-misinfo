# Measuring the impact of COVID-19 vaccine misinformation on vaccination intent in the UK and USA
*Sahil Loomba, Alexandre de Figueiredo, Simon Piatek, Kristen de Graaf, Heidi Larson*
This repository contains code and data for our paper in [*Nature Human Behaviour*](https://www.nature.com/articles/s41562-021-01056-1) on measuring the impact of exposure to COVID-19 vaccine misinformation on the intent to vaccinate in the UK and USA.

## Jupyter Notebooks
These notebooks are intended to aid importing, transforming, and analysing the survey data in this study. You may use [Jupyter nbviewer](https://nbviewer.jupyter.org/) to view these notebooks, or view their static (.html) versions in `.doc/`, or view them on OSF (note: OSF link currently hosts a preprint version of the manuscript).

1. `tables_figures.ipynb`: generates all figures and tables of the paper
2. `import_data.ipynb`: demo of reading and transforming survey data for use in any downstream statistical modeling; [~~view on OSF~~](https://osf.io/ej4c6/)
3. `statistical_analyses.ipynb`: demo statistical modeling and generation of figures and tables in the paper; [~~view on OSF~~](https://osf.io/b3qkc/)

## Directory Structure
1. `.dat/`: contains processed survey data; sufficient to run all statistical analyses in the paper
2. `.doc/`: contains full survey questionnaire and static (.html) versions of the Jupyter Notebooks
3. `.src/paper.py`: contains helper functions to generate all figures and tables of the paper
4. `.src/models.py`: contains functions to define and fit all Bayesian models described in the paper
5. `.src/utils.py`: contains helper functions to import and transform survey data, compute and plot posterior statistics
6. `.src/bayesoc.py`: defines python classes `Dim()`, `Outcome()`, `Society()` and `Model()` to implement general Bayesian socio-demographic models using [pystan](https://pystan.readthedocs.io/en/latest/)

## Links
1. View paper in [Nature Human Behaviour](https://www.nature.com/articles/s41562-021-01056-1)
2. View project on [GitHub](https://github.com/sloomba/covid19-misinfo/)
3. ~~View paper preprint on [medRxiv](https://www.medrxiv.org/content/10.1101/2020.10.22.20217513v1)~~
4. ~~View project on [OSF](https://osf.io/cxwvp/)~~

## Citation
Loomba, S., de Figueiredo, A., Piatek, S.J. et al. Measuring the impact of COVID-19 vaccine misinformation on vaccination intent in the UK and USA. *Nat Hum Behav* (2021). https://doi.org/10.1038/s41562-021-01056-1