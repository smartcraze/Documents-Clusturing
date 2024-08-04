import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



# nltk.download('stopwords')
# nltk.download('wordnet')


documents = [
    "Text of document one.",
    "Text of document two.",
    # Add more documents
]

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

processed_docs = [preprocess(doc) for doc in documents]

# print(processed_docs)