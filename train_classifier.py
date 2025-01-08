import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle
import sys

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Message', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y, Y.columns

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 4]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
    return cv

def main():
    database_filepath, model_filepath = sys.argv[1:]
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    model = build_model()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f"Category: {column}")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filepath}")

if __name__ == '__main__':
    main()
