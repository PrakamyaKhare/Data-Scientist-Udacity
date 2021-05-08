import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
              
            
import numpy as np
import pandas as pd
pd.set_option('display.max_columns',500)
import re
import os
from sqlalchemy import create_engine
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import classification_report ,accuracy_score
from sklearn.model_selection import GridSearchCV
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
nltk.download('punkt')
nltk.download('wordnet')

stop_words = stopwords.words("english")


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM disasterResponse',con=engine)
    df.related.replace(2,1,inplace=True)
    X = df['message'].values
    y = df.iloc[:,4:]
    categ_names = list(df.columns[4:])
    return X,y,categ_names



def tokenize(text) :
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls :
        text = text.replace(url,'placeholder')
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens_list = []
    for tok in tokens :
        if tok in stop_words :
            continue
        tok = PorterStemmer().stem(tok)
        tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens_list.append(tok)
    
    clean_tokens_list = [tok for tok in clean_tokens_list if tok.isalpha()]
    
    return clean_tokens_list


def build_model():
    pipeline = Pipeline([
      ('vect',CountVectorizer(tokenizer=tokenize)),
       ('tfidf', TfidfTransformer()),
       ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    params = { 'clf__estimator__n_estimators':[50],
              'vect__max_df':[0.75,1.0]
    }
    cv = GridSearchCV(pipeline,param_grid=params,verbose=10,n_jobs=4)
   
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    for i in range(len(category_names)) :
        print('Category:', category_names[i])
        print(classification_report(Y_test.iloc[i],y_pred[:,i]))
    print("Total Accuracy:- ",np.mean(np.mean(y_pred==y_test)))

def save_model(model, model_filepath):
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()