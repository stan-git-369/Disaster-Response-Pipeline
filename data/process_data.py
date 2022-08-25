import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import sys
from sklearn.model_selection import GridSearchCV



def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv-files, merge messages and categories datasets messages and categories datasets,
    convert categories from strings to numbers.
    
    INPUT:
    messages_filepath: string. Filepath for csv file containing messages dataset.
    categories_filepath: string. Filepath for csv file containing categories dataset.
       
    OUTPUT:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    categories: list of categories
    """
    
    # load and merge datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner', left_on='id', right_on='id')
    # extract names of categories and convert categories from strings to numbers
    n = categories['categories'].str.split(';', expand=True)
    categories = categories.merge(n, left_on=categories.index, right_on=n.index)
    row = categories['categories'][0].split(';')
    names = {}
    for i in range(36):
        names[i] = row[i][:-2]
    category_colnames = names
    # set categories as columns names in merged dataset
    categories.rename(columns=category_colnames, inplace=True)
    categories = categories.drop(columns=['key_0', 'categories', 'id'])
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    
    return df, categories


def clean_data(df, categories):
     """
     Clean dataframe by removing duplicates and nulls.
    
    INPUT:
    df: dataframe. Dataframe containing merged messages and categories datasets.
       
    OUTPUT:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """
    # drop duplicates    
    df = df.drop_duplicates()
    # drop nan's    
    categories.columns
    df = df.dropna(subset=categories.columns)
    return df
    

def save_data(df, database_filename):
    """
    Save dataframe into  SQLite database.
    
    INPUT:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    OUTPUT:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df, categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
