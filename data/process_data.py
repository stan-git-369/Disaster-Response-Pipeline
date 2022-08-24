import sys
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'inner', left_on='id', right_on='id')
    n = categories['categories'].str.split(';', expand=True)
    categories = categories.merge(n, left_on=categories.index, right_on=n.index)
    n = categories['categories'].str.split(';', expand=True)
    categories = categories.merge(n, left_on=categories.index, right_on=n.index)
    row = categories['categories'][0].split(';')
    names = {}
    for i in range(36):
        names[i] = row[i][:-2]
    category_colnames = names
    categories.rename(columns=category_colnames, inplace=True)
    categories = categories.drop(columns=['key_0', 'categories', 'id'])
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    return df, categories


def clean_data(df, categories):
    df = df.drop_duplicates()
    categories.columns
    df = df.dropna(subset=categories.columns)
    return df
    

def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('disaster_', engine, index=False)  


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