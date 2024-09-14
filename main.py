import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from plot_wordcloud import plot_cloud
from plot_ngrams import analyze_ngrams


if __name__ == '__main__':
    # Load the data
    chatgpt_df = pd.read_csv('chatgpt_text.csv')

    # combine all the text in the 'text' column with topic 
    topic = 'sports'
    text_topic = ' '.join([str(text) for text in chatgpt_df[chatgpt_df['topic'] == topic]['text']])
    # Generate word cloud
    wordcloud = WordCloud(width = 3000, height = 2000, random_state=1, background_color='salmon', colormap='Set2', collocations=False, stopwords = STOPWORDS).generate(text_topic)
    # Plot
    plot_cloud(wordcloud)

    for i in range(2, 5):
        print("Plotting ", i , "-grams:")
        analyze_ngrams(text_topic, i, top_n=20)
