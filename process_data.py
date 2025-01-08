import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('Message', engine, index=False, if_exists='replace')

def main():
    messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
    df = load_data(messages_filepath, categories_filepath)
    df = clean_data(df)
    save_data(df, database_filepath)
    print(f"Data saved to database: {database_filepath}")

if __name__ == '__main__':
    main()
