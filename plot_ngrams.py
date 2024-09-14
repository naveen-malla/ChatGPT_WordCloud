import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import re
import matplotlib.pyplot as plt
import streamlit as st

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Preprocess text: lowercasing, punctuation removal, stopword removal, lemmatization
def preprocess_text(text):
    print(text)
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens

# Extract n-grams from preprocessed tokens
def get_ngrams(tokens, n):
    return list(ngrams(tokens, n))

# Plot top n-grams
def plot_ngrams(ngram_list, n, top_n=10):
    # Count the n-grams
    ngram_counts = Counter(ngram_list)

    # Get the most common n-grams
    common_ngrams = ngram_counts.most_common(top_n)

    # Prepare the n-grams for plotting
    ngrams, counts = zip(*common_ngrams)
    ngrams = [' '.join(gram) for gram in ngrams]  # Join n-grams into readable strings

    # Plot the n-grams
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ngrams, counts, color='skyblue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel(f'{n}-grams')
    ax.set_title(f'Top {top_n} {n}-grams')
    ax.invert_yaxis()  # Highest frequency at the top
    st.pyplot(fig)

# Full analysis function: preprocess, extract, and plot n-grams
def analyze_ngrams(text, n, top_n=10):
    tokens = preprocess_text(text)
    ngram_list = get_ngrams(tokens, n)
    plot_ngrams(ngram_list, n, top_n)
