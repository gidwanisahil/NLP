import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['text'], df['sentiment']

def preprocess_data(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    texts, sentiments = load_data('data/sentiment_data.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(texts, sentiments, test_size=0.2, random_state=42)
    
    # Preprocess the data
    X_train_vectorized, vectorizer = preprocess_data(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Train the model
    model = train_model(X_train_vectorized, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test_vectorized, y_test)
