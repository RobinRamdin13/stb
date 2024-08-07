import re
import os 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

from os.path import join, isdir
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

random_seed = 12345

def rename_cols(col:str)->str:
    """Function to rename columns 

    Args:
        col (str): initial column name

    Returns:
        str: updated column name
    """    
    return re.sub(r'_-_', '_', re.sub(r'[\s]|[\/]', '_', col.lower()))

def main(data_path:str)->None:
    # load csv files 
    df = pd.read_csv(join(data_path, 'data.csv'), index_col=0)
    
    # rename df cols
    df.rename(columns=lambda x: rename_cols(x), inplace=True)

    # save summary statistics
    df.describe().to_csv(join(data_path,'summary_stat.csv'))

    # remove redundant columns 
    cols = [f for f in df.columns.tolist() if 'purpose_of_visit' in f]
    cols.extend(['month', 'country_of_residence'])
    df = df[cols] # take subset of columns only 
    index = df.index.to_list() # extract index to merge in df_output
    df.reset_index(inplace=True, drop=True) # remove the index from df 

    # institate x and y variables 
    X = df.drop(columns='main_purpose_of_visit')
    y = df['main_purpose_of_visit']

    # use the Ordinal Encoder to encode y 
    y_encoder, X_encoder = OrdinalEncoder(), OrdinalEncoder()
    y = pd.DataFrame(data=y_encoder.fit_transform(y.to_frame()), columns=[y.name])
    X = pd.DataFrame(data=X_encoder.fit_transform(X), columns=X.columns.tolist())

    # using ordinal encoder to encode the categorical into numerical data 
    imputer = KNNImputer()
    classfier_model = RandomForestClassifier(random_state=random_seed)
    scaler = StandardScaler()
    
    # instantiate the transformation pipeline 
    pipeline = Pipeline(steps=[
        ('imputer', imputer),
        ('scaler', scaler),
        ('classifier', classfier_model)
    ])

    # parameters to find the best k for imputation
    param_grid = {
        'imputer__n_neighbors': [2,3,4,5,6,7,8,9,10]
    }

    # use gridsearch to find the best k 
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv = 3, 
        scoring='accuracy'
    )
    grid_search.fit(X, np.ravel(y))
    best_k = grid_search.best_params_['imputer__n_neighbors']
    best_score = grid_search.best_score_

    print(f'best k {best_k} with best score {best_score}')

    # perform the imputations with the best performing model 
    imputer = KNNImputer(n_neighbors=best_k)
    X_imputation = imputer.fit_transform(X)

    # reconstruct original dataset
    df_output = pd.DataFrame(data=X_imputation, columns=X.columns)
    df_output['main_purpose_of_visit'] = y 

    # inverse the transformation from the encoders 
    df_output[X.columns] = X_encoder.inverse_transform(df_output[X.columns])
    df_output[y.columns] = y_encoder.inverse_transform(df_output[y.columns])
    df_output.index = index

    df_output.to_csv(join(data_path, 'qu1_data.csv'))
    return


if __name__ == '__main__':
    # create folder paths
    cwd = os.getcwd() 
    data_path = join(cwd, 'data')
    # run main logic
    main(data_path)