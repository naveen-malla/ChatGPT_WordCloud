import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()




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

