import pandas as pd
import streamlit as st
from wordcloud import WordCloud, STOPWORDS
from plot_wordcloud import plot_cloud
from plot_ngrams import analyze_ngrams, plot_ngrams
import nltk
from nltk.corpus import stopwords

# Download NLTK data if not already available
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset once
chatgpt_df = pd.read_csv('chatgpt_text.csv')

# Streamlit app setup
st.title("ChatGPT Text Analysis by Topic")

# Importing necessary libraries
from wordcloud import WordCloud, STOPWORDS
import streamlit as st

# List of topics
topics = ['sports', 'business', 'world', 'yelp', 'imdb', 'ivy', 'medical', 'finance', 'history', 'science']

# Creating extra stopwords for each topic
extra_stopwords = {
    'sports': {'team', 'game', 'olympic', 'victory', 'performance', 'athlete', 'event', 'medal', 'tournament', 'season'},
    'business': {'company', 'market', 'business', 'profit', 'loss', 'growth', 'sales', 'financial', 'revenue', 'shareholder'},
    'world': {'global', 'country', 'government', 'international', 'nation', 'world', 'policy', 'leader', 'state', 'agreement'},
    'yelp': {'restaurant', 'service', 'food', 'staff', 'menu', 'review', 'waiter', 'experience', 'customer', 'dish'},
    'imdb': {'film', 'movie', 'director', 'actor', 'role', 'performance', 'screen', 'character', 'cinema', 'production'},
    'ivy': {'university', 'college', 'student', 'campus', 'degree', 'professor', 'research', 'admission', 'course', 'graduate'},
    'medical': {'patient', 'doctor', 'disease', 'treatment', 'hospital', 'medicine', 'health', 'diagnosis', 'therapy', 'surgery'},
    'finance': {'investment', 'stock', 'bank', 'loan', 'capital', 'financial', 'market', 'fund', 'interest', 'economy'},
    'history': {'war', 'king', 'empire', 'battle', 'dynasty', 'reign', 'historical', 'century', 'revolution', 'civilization'},
    'science': {'experiment', 'research', 'energy', 'scientist', 'study', 'data', 'theory', 'result', 'analysis', 'method'}
}

# Topic selection dropdown
topic = st.selectbox('Select a Topic:', topics)

# Combine all text for the selected topic
text_topic = ' '.join([str(text) for text in chatgpt_df[chatgpt_df['topic'] == topic]['text']])

# Show word cloud
if text_topic:
    st.header(f"Word Cloud for {topic}")
    
    # Add custom stopwords for the selected topic
    custom_stopwords = STOPWORDS.union(extra_stopwords[topic])
    
    # Generate the word cloud
    wordcloud = WordCloud(
        width=3000, 
        height=2000, 
        random_state=1, 
        background_color='white',  
        colormap='viridis',  
        collocations=False, 
        stopwords=custom_stopwords  # Adding extra stopwords here
    ).generate(text_topic)
    
    # Display the word cloud
    plot_cloud(wordcloud)
    
    # Analyze and plot n-grams (from 2-grams to 4-grams)
    for i in range(2, 5):
        st.subheader(f"{i}-grams for {topic}")
        analyze_ngrams(text_topic, i, top_n=20)

    # Tokenize the text and perform POS tagging
    tokens = nltk.word_tokenize(text_topic)
    pos_tags = nltk.pos_tag(tokens)

    # Filter adjectives, nouns, and verbs, removing stopwords
    stop_words = set(stopwords.words('english'))
    adjectives = [word.lower() for word, pos in pos_tags if pos.startswith('JJ') and word.lower() not in stop_words]
    nouns = [word.lower() for word, pos in pos_tags if pos.startswith('NN') and word.lower() not in stop_words]
    verbs = [word.lower() for word, pos in pos_tags if pos.startswith('VB') and word.lower() not in stop_words]

    # Plot the most common adjectives, nouns, and verbs
    st.subheader("Most common adjectives")
    plot_ngrams(adjectives, 1, top_n=20)

    st.subheader("Most common nouns")
    plot_ngrams(nouns, 1, top_n=20)

    st.subheader("Most common verbs")
    plot_ngrams(verbs, 1, top_n=20)
else:
    st.warning(f"No text found for topic: {topic}")