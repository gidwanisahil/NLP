from sklearn.feature_extraction.text import CountVectorizer

def extract_features(data):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data)
    return features, vectorizer
