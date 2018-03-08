from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
import sklearn
import sys

class EmployeeRetentionModel:
    
    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)
    
    def predict_proba(self, X_new, clean=True, augment=True):
        if clean:
            X_new = self.clean_data(X_new)
        
        if augment:
            X_new = self.engineer_features(X_new)
            
        return X_new, self.model.predict_proba(X_new)
    
    def clean_data(self, df):
        df = df.drop_duplicates()

        df = df[df.department != 'temp']

        df['filed_complaint'] = df.filed_complaint.fillna(0)

        df['recently_promoted'] = df.recently_promoted.fillna(0)

        df.department.replace('information_technology', 'IT', inplace=True)

        df['department'].fillna('Missing', inplace=True)

        df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)

        df.last_evaluation.fillna(0, inplace=True)

        return df
    
    def engineer_features(self, df):
        df['underperformer'] = ((df.last_evaluation < 0.6) & 
                                (df.last_evaluation_missing == 0)).astype(int)

        df['unhappy'] = (df.satisfaction < 0.2).astype(int)

        df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)

        df = pd.get_dummies(df, columns=['department', 'salary'])

        return df


def main(data_location, output_location, model_location, clean=True, augment=True):
    df = pd.read_csv(data_location)

    retention_model = EmployeeRetentionModel(model_location)

    df, pred = retention_model.predict_proba(df)
    pred = [p[1] for p in pred]

    df['prediction'] = pred

    df.to_csv(output_location, index=None)

if __name__ == '__main__':
    main( *sys.argv[1:] )
