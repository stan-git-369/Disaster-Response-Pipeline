# import packages
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
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


def load_data(database_filepath):
    """
    Load and merge datasets
    
    INPUT:
         database filepath
    
    OUTPUT:
        X: dataframe with messages 
        y: dataframe with labels
        cols: category names
    """
    # read in file
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    # clean data
    df.drop_duplicates()

    # load to database
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False) 
    
    # define features and label arrays
    cols = df.columns.tolist()[4:]
    X = df['message']
    y = df[cols]
    X = X.values
    y = y.values
        
    return X, y, cols

	
def tokenize(text):
    """ 
    Normalize and tokenize unput text
    
    INPUT:
         text
    
    OUTPUT:
        clean_tokens: clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
   
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    return clean_tokens


def display_results(y_test, y_pred, cols):
    """ 
    display the results of trained model
    
    INPUT:
         true and predicted values, categories names
    
    OUTPUT:
        print precision, recall, fscore for input data
    """
    # output model test results
    for c in range(y_test.shape[1]):
        print(c)
        y_true = y_test[c]
        y_pr = y_pred[c]
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pr, average='weighted')
        print('\nReport for the column ({}):\n'.format(cols[c]))
        print('Precision: {}'.format(round(precision, 2)))
        print('Recall: {}'.format(round(recall, 2)))
        print('F-score: {}'.format(round(fscore, 2)))


def build_model():
    """ 
    Build of text processing, model and gridsearch pipeline
    
    INPUT:
         None
    
    OUTPUT:
        model_pipeline
    """
    # 

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
# define parameters for GridSearchCV
    parameters = {'vect__min_df': [1, 3],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [2, 4]
                 }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(model, param_grid=parameters)
    
    return model_pipeline
    
    
def train(X, y, model):
    """ 
    Split input data to train and test, fit the model and predict results
    
    INPUT:
         X: feature dataset
	 y: label dataset
	 model: model pipeline
    
    OUTPUT:
        model: trained model
	y_test: array of test labels
	y_pred: array of predicted labels
    """
    
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # train classifier
    model = build_model()
    model.fit(X_train, y_train)

    # predict on test data
    y_pred = model.predict(X_test)

    return model, y_test, y_pred
    
    
def export_model(model):
    """
    Export model as a pickle file
    
    INPUT:
         model: trained model
	 
    OUTPUT:
        None
    """
    # Export model as a pickle file
    # create an iterator object with write permission - model.pkl
    with open('model_pkl', 'wb') as files:
        pickle.dump(model, files)
        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n')
	X, y, cols = load_data(database_filepath)
        
        print('Building model...\n')
        model = build_model()
              
        print('Building test and train data, training model...\n')
        model, y_test, y_pred = train(X, y, model)
    
        print('Output results...\n')
        display_results(y_test, y_pred, cols)
    
        print('Saving model...\n')
        export_model(model)

        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
