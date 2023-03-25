'''
This module will:
- import the data
- perform EDA
- perform feature engineering
- generate reports
- train and test a model

Author: "Kunle Oguntoye"
Date: 21 March 2023
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
# import libraries
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import joblib


cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]
eda_columns = ['Total_Trans_Ct', 'Marital_Status',\
               'Customer_Age', 'Churn', 'heatmap']
keep_cols = ['Customer_Age', 'Dependent_count', \
             'Months_on_book','Total_Relationship_Count',\
             'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            data: pandas dataframe
    '''
    data = pd.read_csv(pth)
    return data


def perform_eda(data, eda_columns, cat_columns):
    '''
    perform eda on df and save figures to images folder
    input:
            data: pandas dataframe
    output:
            None
    '''
    for items in eda_columns:
        plt.figure(figsize=(20,10))
        if items in cat_columns:
            data[items].value_counts('normalize')\
            .plot(kind='bar')
        elif items == 'heatmap':
            sns.heatmap(df.corr(), annot=False,\
                        cmap="Dark2_r", linewidths=2)
        else:
            data[items].hist()
        plt.title(f'{items}  distribution')
        plt.savefig(f'./images/eda/{items}.jpg')
        plt.close()


def encoder_helper(data, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            data: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that \
            could be used for naming variables or index y column]
    output:
            data: pandas dataframe with new columns for
    '''
    data['Churn'] = data['Attrition_Flag'].apply(lambda val:\
                            0 if val == "Existing Customer" else 1)
    for items in category_lst:
        group = data.groupby(items).mean()['Churn']
        data[f'{items}_Churn'] = data[items].apply\
        (lambda x: group.loc[x])
    return data


def perform_feature_engineering(data):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument \
              that could be used for naming variables or index y column]
    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    data_predictors = data[keep_cols]
    data_target = data['Churn']
    X_train, X_test, y_train, y_test = train_test_split\
    (data_predictors, data_target, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and
    testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    lists_pred = [y_train_preds_rf, y_train_preds_lr, \
                  y_test_preds_rf, y_test_preds_lr]
    lists_names = ['Random Forest Train', 'Logistic Regression Train',\
                   'Random Forest Test', 'Logistic Regression Test']

    for idx, item in enumerate([y_train, y_test]):
        lists_on = lists_pred[2*idx : 2*idx + 2]
        lists_on_name = lists_names[2*idx : 2*idx + 2]
        for pred, name in zip(lists_on, lists_on_name):
            plt.rc('figure', figsize=(5, 5))
            if name.endswith('Train'):
                plt.text(0.01, 1.25, str(f'{name}'), {'fontsize': 10}, \
                         fontproperties = 'monospace')
                plt.text(0.01, 0.7, str(classification_report(item, pred)), \
                         {'fontsize': 10}, fontproperties = 'monospace')
                plt.axis('off')
                plt.savefig(f'./images/results/{name}.jpg')
                plt.close()
            else:
                plt.text(0.01, 1.25, str(f'{name}'), {'fontsize': 10}, \
                         fontproperties = 'monospace')
                plt.text(0.01, 0.7, str(classification_report(item, pred)), \
                         {'fontsize': 10}, fontproperties = 'monospace')
                plt.axis('off')
                plt.savefig(f'./images/results/{name}.jpg')
                plt.close()


def feature_importance_plot(model, X_data, output_pth):

    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]
    plt.figure(figsize=(20,5))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)




def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    #model fitting
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    #predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    #plot results
    count  = 0
    for model in [lrc, cv_rfc.best_estimator_]:
        plt.figure(figsize=(15, 8))
        plot_roc_curve(model, X_test, y_test)
        plt.savefig(f'./images/eda/{count}.png')
        plt.close()
        count += 1


if __name__ == '__main__':
    # read file
    df = import_data("./data/bank_data.csv")
    # encode
    df = encoder_helper(df, cat_columns)
    #perfrom eda
    perform_eda(df, eda_columns, cat_columns)
    # make predictions
    predictors = perform_feature_engineering(df)
    # train
    train_models(*predictors)
    