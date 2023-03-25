# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project trains two classification machine learning model (randomforest and logistic regression) to predict the likelihood of bank customers to churn. Other than training, the raw notebook ( `churn_notebook.ipynb`) is rewitten to PEP8 standard and scores a near 8.0/10 on pylint scale.

## Files and data description
1. The training and testing datasets are found in './data'
2. A quick intuition into the dataframe through plots can be found in './images/results'
3. The trained models are saved in ''./models'
4. A refactor and  modularized version of './churn_notebook.ipynb' is saved as './churn_library.py'

## Running Files
1. Create environment:
```bash
conda create --name churn_predict python=3.6 
conda activate churn_predict
```
2. Install packages:
```bash
conda install --file requirements.txt
```
3. Run churn prediction:
```bash
python churn_library.py
```
4. Test churn prediction:
```bash
python churn_script_logging_and_tests.py
```


