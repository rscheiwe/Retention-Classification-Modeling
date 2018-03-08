# Classification Modeling of Employee Retention Program

![image](https://user-images.githubusercontent.com/29715062/37113674-088950ee-2214-11e8-9041-b2b99d21633d.png)

## Getting Started

#### Project Overview

Testing classification algorithms on employee data from an anonymized employer in order to determine a proactive means of retaining employees before they quit or face disciplinary action due to underperformance. 

Primary issues of employee retention concern: 
* Evaluation `'last_evaluation'`
* Tenure `'tenure'`
* Satisfaction '`satisfaction'`
* Workload `'n_jobs'`

Dataset '`employee_data.csv'` contains 14,000+ records for training models, and `'unseen_raw_data.csv'` contains a sampling to test the optimal model for prediction.

#### Installation and Use

Data may be downloaded and parsed via the `'data/'` directory. 

The individual notebooks (.ipynb) contain and walk-through of the machine-learning workflow, whereas `'retention_model.py'` is a model class containing all the cleaning and engineering features applicable to the included dataset, `'unseen_raw_data.csv'`. The file, `'requirements.txt'` is a brief overview of the necessary imports. 

## Summary of Aims

Because there is no quantifiable 'win' condition, the best model possible determines the predictability of retaining an employee per the indicator issues. The executable model script may be augmented based on the target variable (`'status'`). 
