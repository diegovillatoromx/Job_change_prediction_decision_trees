# Data Science Job Change Prediction Model using Decision Trees

In the realm of Big Data and Data Science, a company specializes in recruiting data scientists from those who successfully complete their training courses. With a large pool of enrolled individuals, the company aims to differentiate candidates who genuinely intend to join their workforce post-training, from those who are actively seeking new job opportunities. This distinction holds the key to reducing costs, enhancing training quality, and optimizing course planning. Leveraging demographic, educational, and experiential data gathered during candidate enrollment, the task at hand is to develop predictive models that ascertain the likelihood of a candidate either seeking alternative employment or committing to the company. This analysis not only informs strategic human resource decisions, but also provides insights into the factors influencing employee decisions concerning their future career paths.

## Table of Contents

- [Description](#description)
- [Architecture](#architecture)
- [Features](#features)
- [Modular_Code_Overview](#modular_code_overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## Description

A Decision Tree is a supervised learning method suitable for both classification and regression tasks, often used for solving classification problems. It's structured as a tree-like classifier, with nodes representing dataset features, branches as decision rules, and leaf nodes as outcomes.

This graphical representation offers potential solutions based on given conditions. Beginning from a root node, it grows branches and forms a tree structure. The tree poses questions and branches further based on Yes/No answers.

Our case study focuses on a churn dataset, where "churned customers" are those ending relationships with their current company. XYZ, a service provider, offers a one-year subscription plan and wants to predict customer renewal.

Previously, we explored logistic regression on this dataset. It's recommended to review the [Build a Logistic Regression Model](https://github.com/diegovillatoromx/Strategic_Workforce_Analysis_Predicting_Job_Transition) project. Now, we aim to apply the decision tree classifier to the same dataset.
## Architecture

![Architecture Diagram](https://github.com/diegovillatoromx/Job_change_prediction_decision_trees/blob/main/architecture_diagram_decision_trees.png)

## Data Description
The CSV consists of around 19,158 rows and 14 columns in the [dataset](https://github.com/diegovillatoromx/Job_change_prediction_decision_trees/blob/main/input/DS_Job_Change_Data.csv)
### Features:
- enrollee_id : Unique ID for candidate
- city: City code
- city_ development _index : Developement index of the city (scaled)
- gender: Gender of candidate
- relevent_experience: Relevant experience of candidate
- enrolled_university: Type of University course enrolled if any
- education_level: Education level of candidate
- major_discipline :Education major discipline of candidate
- experience: Candidate total experience in years
- company_size: No of employees in current employer's company
- company_type : Type of current employer
- last_new_job: Difference in years between previous job and current job
- training_hours: training hours completed
- target: 0 – Not looking for job change, 1 – Looking for a job change

## Modular_Code_Overview

```
  input
    |_DS_Job_Change_Data.csv

  ML_pipeline
    |_evaluate_metrics.py
    |_feature.py
    |_ml_model.py
    |_plot_model.py
    |_utils.py

  Tutorial
    |_decision_tree.ipynb

  output
    |_Decision_Tree_plot.png
    |_Feature_Importance.png
    |_model.pkl
```
1. Input - It contains all the data that we have for analysis. There is one csv
file in our case:
   - DS_Job_Change_Data.csv
2. ML_Pipeline
   - The ML_pipeline is a folder that contains all the functions put into different
      python files, which are appropriately named. These python functions are
      then called inside the engine.py file.

3. Output
   – The output folder contains the best-fitted models that we trained
for this data. These models can  be easily loaded and used for future use and
the user need not have to train all the models from the beginning.
Note: This model is built over a chunk of data. One can obtain the model for the
entire data by running engine.py by taking the entire data to train the models.

4. Tutorial - This is a reference folder. It contains the ipython notebook tutorial.

## Installation

Below are the steps required to set up the environment and run this Data Science project on your local machine. Make sure you have the following installed:
- Python 3.x: You can download it from [python.org](https://www.python.org/downloads/).
- Pip: The Python package manager. In most cases, it comes pre-installed with Python. If not, you can install it by following [this guide](https://pip.pypa.io/en/stable/installing/).

### Prerequisites

Install required packages using the requirements.txt file:
``` bash
pip install -r requirements.txt
```
### Installation Steps

1. **Clone the Repository:**

   Clone this repository to your local machine using Git:

   ```bash
   git clone https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model
   cd yourproject
   ```
## Usage

How to utilize and operate the Data Science project after completing the installation steps.
### Data Preparation
Before analysis, prepare data by loading and processing it:
1. ##### Import the required libraries
    ```terminal
    import pickle
    from ML_Pipeline.utils import read_data,inspection,null_values
    from ML_Pipeline.ml_model import prepare_model_smote,run_model
    from ML_Pipeline.evaluate_metrics import confusion_matrix,roc_curve
    from ML_Pipeline.feature_imp import plot_feature_importances
    from ML_Pipeline.plot_model import plot_model
    import matplotlib.pyplot as plt
    ```
2. ##### Data loading
    If data is in CSV format, load it using Pandas:
    ```terminal
    datapath = 'input/data_regression.csv'
    df = read_data(datapath)
    df.head(6)
    ```
    ![df_head](https://github.com/diegovillatoromx/Job_change_prediction_decision_trees/blob/main/images/dfhead_trees.png)
 
3. #### Inspection and cleaning the data
    ```terminal
    x = inspection(df)
    ```
    ![inspection](https://github.com/diegovillatoromx/Job_change_prediction_decision_trees/blob/main/images/inspection_trees.png)
4. #### Cleaning and Preprocessing:
   Clean data by handling missing values, normalization, etc.
    ```terminal
    column_names = df.columns.tolist() #list the column names from dataframe
    target = column_names[-1] #select the target column
    cols_to_exclude = column_names[0:4] #columns to exclude because are not relevant
    df = null_values(df)
    ```
### Training Model
Perform analysis and modeling on prepared data:

1. #### Model Selection
   Selecting only the numerical columns and excluding the columns we specified in the function
   ```terminal
    X_train, X_test, y_train, y_test = prepare_model_smote(df,target,
                                                 cols_to_exclude)
    ```
### Evaluation

1. #### Evaluation Metrics
   ```terminal
   model_dectree,y_pred = run_model(X_train,X_test,y_train,y_test)
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/run_model.png)


2. #### Performance metrics
   ```terminal
   conf_matrix = confusion_matrix(y_test,y_pred)
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/cof_matrix.png)

   ```terminal
   roc_val = roc_curve(model_dectree,X_test,y_test)
   ```
   ![ROC](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Log_ROC.png)

   ```terminal
   decision_tree_plot = plot_model(model_dectree,['not churn','churn'])
   plt.savefig("output/"+"Decision_Tree_plot.png")
   ```
   ![tree](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Decision_Tree_plot.png)

3. #### Feature Importance
   ```terminal
   fea_imp = plot_feature_importances(model_dectree)
   plt.savefig("output/"+"Feature_Importance.png")
   ```
   ![running_model](https://github.com/diegovillatoromx/Customer_Churn_Prediction_Model/blob/main/images/Feature_Importance.png)

# Contributing
  1. Focus changes on specific improvements.
  2. Follow project's coding style.
  3. Provide detailed descriptions in pull requests.
## Reporting Issues
  Use "Issues" to report bugs or suggest improvements.
# Contact
For questions or contact, [email](diegovillatormx@gmail.com) or [Twitter](https://twitter.com/diegovillatomx) .

