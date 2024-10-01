from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from .preprocess import preprocess_text
from .feature_extraction import extract_features

def train_model(file_path):
    data = pd.read_csv(file_path)
    data['text'] = data['text'].apply(preprocess_text)  # Assuming 'text' is the column with your text data
    features, vectorizer = extract_features(data['text'])
    
    X_train, X_test, y_train, y_test = train_test_split(features, data['label'], test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(f'Accuracy: {accuracy_score(y_test, predictions)}')

    return model, vectorizer
