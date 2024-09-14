
import matplotlib.pyplot as plt
import streamlit as st

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.imshow(wordcloud)
    ax.axis("off")
    plt.show()
    st.pyplot(fig)


