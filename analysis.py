# this script is used to Build a word cloud of most used words and phrases by ChatGPT along with some cool visualizations
import pandas as pd

def convert_xlsx_to_csv():
    df = pd.read_excel('data/ieee-chatgpt-generation.xlsx')
    df.to_csv('data.csv', index=False)




if __name__ == '__main__':
    convert_xlsx_to_csv()
