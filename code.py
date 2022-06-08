
### Import packages 
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import pep8

import spacy
nlp=spacy.load('en_core_web_sm')

from nltk.sentiment import SentimentIntensityAnalyzer
import operator
import nltk
nltk.download('vader_lexicon')

from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
"""### Parameter """

### parameter 
### URL from which data need to be pulled 
url = "https://www.aljazeera.com/where/mozambique/"


"""### Methods"""
def get_url_data(url): 
  data = requests.request("GET", url) ### get request to fetch the data for the URL 
  data_soup = BeautifulSoup(data.content, 'html.parser') ### coverting the data into BeautifulSoup format for easy access of data

  return data_soup
  
def convert_to_lower_case(dataset):
    def lower(input_text):
        return input_text.lower()
    dataset['headline']=dataset['headline'].apply(lower)
    dataset['article']=dataset['article'].apply(lower)
    
def remove_punctuation(dataset):
    def remove_punctuation_from_text(input_text):
        output_list=[word for word in input_text.split() if word.isalpha()]
        return ' '.join(output_list)    
    dataset['headline']=dataset['headline'].apply(remove_punctuation_from_text)
    dataset['article']=dataset['article'].apply(remove_punctuation_from_text)
    
def correct_words(dataset):
    def correct_text(input_text):
        list_1=[str(TextBlob(word).correct()) for word in input_text.split()]
        output_text= ' '.join(list_1)
        return output_text
    dataset['headline']=dataset['headline'].apply(correct_text)
    dataset['article']=dataset['article'].apply(correct_text)
    
def lemmatize(dataset):
    def lematize_text(input_text):
        doc=nlp(input_text)
        lemmas=[token.lemma_ for token in doc]
        output_text=' '.join(lemmas)
        return output_text
    dataset['headline']=dataset['headline'].apply(lematize_text)
    dataset['article']=dataset['article'].apply(lematize_text)
    
def remove_stopwords(dataset):
    def remove_stopwords_from_text(input_text):
        stopwords=spacy.lang.en.stop_words.STOP_WORDS
        output_list=[word for word in input_text.split() if word not in stopwords and not(word=='-PRON-') ]
        return ' '.join(output_list)
    dataset['headline']=dataset['headline'].apply(remove_stopwords_from_text)
    dataset['article']=dataset['article'].apply(remove_stopwords_from_text)


"""### Code Flow"""

### Call get_url_data 
main_page_soup = get_url_data(url)

### All News Links 
News_links = main_page_soup.find_all('a', class_='u-clickable-card__link')

### Modify the links, add https://www.aljazeera.com in prefix of links
top_ten_news = []
for link in News_links: 
  top_ten_news.append("https://www.aljazeera.com" + link["href"])


### Get data from top ten NEWS link
article = []
headline = []
count = 0

for link in top_ten_news:
  news_data = get_url_data(link)
  info = news_data.find_all('div', class_='wysiwyg wysiwyg--all-content css-1ck9wyi')
  res = news_data.find('script')
  
  if len(info) != 0:
    article.append(info[0].get_text())
    json_object = json.loads(res.contents[0])
    headline.append(json_object['headline'])

    count += 1
    if count == 10: ### Filtring the top 10 working articles 
      break

df = pd.DataFrame({"headline": headline, "article": article })

"""### Data cleaning"""

convert_to_lower_case(df)
remove_punctuation(df)
lemmatize(df)
remove_stopwords(df)

sia = SentimentIntensityAnalyzer()
df["sentiment_score"] = df["headline"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],['neg', 'neu', 'pos'])



df["sentiment_score"] = df["headline"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

sia = TextClassifier.load('en-sentiment')
def flair_prediction(x):
    sentence = Sentence(x)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return "pos"
    elif "NEGATIVE" in str(score):
        return "neg"
    else:
        return "neu"
df["sentiment"] = df["headline"].apply(flair_prediction)

"""### Visualization"""

x = df['sentiment'].value_counts()

fig = px.bar(x)
fig.show()

