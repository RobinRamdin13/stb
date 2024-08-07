import os 
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from os.path import join, isdir

label_prop = {'weight':'bold', 'fontsize':12}
title_prop = {'weight':'bold', 'fontsize':14}

def rename_cols(col:str)->str:
    """Function to rename columns 

    Args:
        col (str): initial column name

    Returns:
        str: updated column name
    """    
    return re.sub(r'_-_', '_', re.sub(r'[\s]|[\/]', '_', col.lower()))

def get_lineplots(x:str, y:str, hue:str, data:DataFrame, x_label:str, y_label:str, title:str, plot_path:str)-> None:
    plt.figure(figsize=(15, 10))
    sns.lineplot(x=x, y=y, data=data, hue=hue, marker="o", markersize=10)
    plt.xlabel(x_label, **label_prop)
    plt.ylabel(y_label, **label_prop)
    plt.title(title, **title_prop)
    plt.legend(prop={'size':12})
    plt.grid()
    plt.tight_layout()
    name = re.sub(r'[\s]', '_', title.lower())
    plt.savefig(join(plot_path, name+'.jpeg'))
    plt.close()
    return

def main(data_path: str, plot_path:str)-> None: 
    def change_transport(x):
        if not math.isnan(x.air_terminal): 
            return 'air'
        elif type(x.sea_terminal)==str: 
            return 'sea'
        elif type(x.land_terminal)==str: 
            return 'land'
        return None
    # load the imputed data and original data
    df_impute = pd.read_csv(join(data_path, 'qu1_data.csv'), index_col=0)
    df_original = pd.read_csv(join(data_path, 'data.csv'), index_col=0)
    
    # process dataframes
    df_impute.index.name = df_original.index.name # set the same index for both dataframes
    df_original.rename(columns=lambda x: rename_cols(x), inplace=True) # rename original column names
    impute_cols = df_impute.columns.tolist()
    original_cols = [f for f in df_original.columns.tolist() if f not in impute_cols]

    # merge the imputation data with the original values 
    df = pd.DataFrame(index=df_original.index, columns=df_original.columns)
    df[impute_cols] = df_impute
    df[original_cols] = df_original[original_cols]

    # check total expendidute is correctly computed 
    exp_cols = [f for f in original_cols if 'tot' in f and f != 'tot.exp']
    df_exp_tot = df[exp_cols].sum(axis=1)
    try:
        df_exp_tot.equals(df['tot.exp'])
    except: 
        df['tot.exp'] = df_exp_tot
    
    # create new transport feature 
    # print(df.columns.tolist())
    df['transport'] = df.apply(lambda x: change_transport(x), axis=1)
    df.drop(columns=['air_terminal', 'sea_terminal', 'land_terminal'], inplace=True)
    
    # create groupby to indentify top 10 countries with more mean expenditure
    df_group_country = df[['country_of_residence', 'tot.exp']].groupby('country_of_residence').mean().reset_index()
    countries_most_spend = df_group_country.sort_values('tot.exp', ascending=False).head(10)['country_of_residence'].values.tolist()
    df_countries = df[df['country_of_residence'].isin(countries_most_spend)]
    get_lineplots(x='month', y='tot.exp', hue='country_of_residence', data=df_countries, x_label='Months',
              y_label='Expenditure', title= 'Countries with Most Total Expendidutre',plot_path=plot_path)
    
    df_countries_pie = df_countries[['country_of_residence', 'tot.exp']].groupby('country_of_residence').sum()
    plt.figure(figsize=(15,15), dpi=400)
    df_countries_pie.plot(kind='pie', y='tot.exp', autopct='%1.0f%%', legend=False)
    plt.ylabel('')
    plt.title('Countries with Most Total Expenditure', **title_prop)
    plt.tight_layout()
    plt.savefig(join(plot_path, 'countries_with_most_expenditure_pie'+'.jpeg'))
    plt.close()

    # create groupby for expenditure per month
    get_lineplots(x='month', y='tot.exp', data=df[['month', 'tot.exp']], hue=None, x_label='Months',
          y_label='Mean Expenditure', title='Mean Expenditure Over Time', plot_path=plot_path)
    get_lineplots(x='month', y='tot.exp', data=df[['month', 'tot.exp', 'purpose_of_visit']], hue='purpose_of_visit', x_label='Months',
          y_label='Mean Expenditure', title='Mean Expenditure Over Time By Purpose', plot_path=plot_path)

    # create groupby for business revenue 
    get_lineplots(x='month', y='totbiz', hue='transport', data=df[['month', 'totbiz', 'transport']], x_label='Months',
                  y_label='Business Expenditure', title='Mean Business Expenditure Over Time', plot_path=plot_path)

    return

if __name__ == '__main__':
    # create folder paths
    cwd = os.getcwd() 
    data_path = join(cwd, 'data')
    plot_path = join(cwd, 'plots')
    # create folders
    if not isdir(plot_path): os.mkdir(plot_path)
    # run main logic
    main(data_path, plot_path)