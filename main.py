import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from plot_wordcloud import plot_cloud
from plot_ngrams import analyze_ngrams
from plot_ngrams import plot_ngrams
import nltk
from nltk.corpus import stopwords

if __name__ == '__main__':
    # Load the data
    chatgpt_df = pd.read_csv('chatgpt_text.csv')

    # combine all the text in the 'text' column with topic 
    topic = 'sports'
    text_topic = ' '.join([str(text) for text in chatgpt_df[chatgpt_df['topic'] == topic]['text']])
    
    # Generate word cloud
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='salmon', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(text_topic)
    plot_cloud(wordcloud)

    # Analyze n-grams
    for i in range(2, 5):
        print("Plotting ", i , "-grams:")
        analyze_ngrams(text_topic, i, top_n=20)

    # now we find the adjectives, nouns and verbs overused by ChatGPT in the text
    # Tokenize the text
    tokens = nltk.word_tokenize(text_topic)

    #nltk.download('averaged_perceptron_tagger_eng')

    # Get the part of speech for each token
    pos_tags = nltk.pos_tag(tokens)

    # Separate the tokens into adjectives, nouns, and verbs and remove stopwords
    stop_words = set(stopwords.words('english'))
    adjectives = [word for word, pos in pos_tags if pos.startswith('JJ') and word not in stop_words]
    nouns = [word for word, pos in pos_tags if pos.startswith('NN') and word not in stop_words]
    verbs = [word for word, pos in pos_tags if pos.startswith('VB') and word not in stop_words]

    # Plot the top 10 most common adjectives, nouns, and verbs
    plot_ngrams(adjectives, 1, top_n=20)
    plot_ngrams(nouns, 1, top_n=20)
    plot_ngrams(verbs, 1, top_n=20)