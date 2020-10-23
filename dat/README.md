# Data Descriptions
## Processed Survey Data
Processed survey data are stored as tables, with survey field-names in columns and samples in rows.
Refer to `doc/orb_questionnaire.pdf` for the original questionnaire and `transform_data()` in `src/utils.py` to see details of how data have been cleaned and recoded.

1. `orb_uk.csv`: processed survey data for the UK as a .csv
2. `orb_us.csv`: processed survey data for the US as a .csv

## Data Dictionaries
Contain the mapping of numeric-codes to field-values of the processed data to aid analysis and interpretation of results. The .json files are human-readable, while the .pkl files can be read into python using `pickle` (as done by `import_transformed_data()` in `src/utils.py`).

1. `orb_uk.json`: data dictionary for the UK as a .json file
2. `orb_uk.pkl`: data dictionary for the UK as a .pkl file
3. `orb_us.json`: data dictionary for the US as a .json file
4. `orb_us.pkl`: data dictionary for the US as a .pkl file
