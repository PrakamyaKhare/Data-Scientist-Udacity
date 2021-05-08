import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    message = pd.read_csv(messages_filepath)
    catge = pd.read_csv(categories_filepath)
    
    df = pd.merge(message,catge,on='id')
    
    return df

def clean_data(df):
    categ = df['categories'].str.split(pat=';',expand=True)
   
    row = categ.iloc[0]
    
    categ_col = [val[:-2] for val in row]
    
    categ.columns = categ_col
    
    for column in categ :
        
        categ[column] = categ[column].str[-1]
        categ[column] = categ[column].astype(int)
        
    df.drop('categories',axis=1,inplace=True)
    
    df = pd.concat([df,categ],axis=1,sort = False)
    
    df.drop(df[df.duplicated(keep='first')].index,inplace=True)
        
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disasterResponse',engine,index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        
        df = clean_data(df)
        
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