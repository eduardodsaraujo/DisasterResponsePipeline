from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """ Load the filepath and return the data """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM MESSAGE", engine)
    df.drop('id', axis=1, inplace=True)
    X = df.loc[:,['message']].values.ravel()
    df.drop(['message','original','genre'], axis=1, inplace=True)
    Y= df.values
    category_names = list(df.columns)
    return X, Y, category_names


def tokenize(text):
    """ Tokenize and transform input text. Return cleaned text """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """ Return Grid Search model with pipeline and Classifier """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(verbose=1, n_jobs=-1)))
        ])
    
    parameters = {  'clf__estimator__max_features': ['auto', 'sqrt', 'log2'],
                    'clf__estimator__criterion' :['gini', 'entropy'],
                 }

    cv = GridSearchCV(pipeline, parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Print model results """
    Y_pred = model.predict(X_test)
    
    for i in range(0,len(Y_test[0])):
        cr_y = classification_report(Y_test[:,i],Y_pred[:,i])
        print('Category : ' + category_names[i])
        print(cr_y)

def save_model(model, model_filepath):
    """ Save model as pickle file """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
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