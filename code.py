### Import packages 
import json
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import plotly.express as px
import spacy
nlp=spacy.load('en_core_web_sm')
from nltk.sentiment import SentimentIntensityAnalyzer
import operator
import nltk
nltk.download('vader_lexicon')
from flair.models import TextClassifier
from flair.data import Sentence
from textblob import TextBlob
from tqdm import tqdm
import time

start = time.time()
"""### Parameter """
### parameter 
### URL from which data need to be pulled 
url = "https://www.aljazeera.com/where/mozambique/"
"""### Methods"""

### Get the data from the URL 
def get_url_data(url): 
  data = requests.request("GET", url) ### get request to fetch the data for the URL 
  data_soup = BeautifulSoup(data.content, 'html.parser') ### coverting the data into BeautifulSoup format for easy access of data

  return data_soup

### Conver the text data into lower case
def convert_to_lower_case(dataset):
    def lower(input_text):
        return input_text.lower()
    dataset['headline']=dataset['headline'].apply(lower)
    dataset['article']=dataset['article'].apply(lower)
  
### Remove punctuation
def remove_punctuation(dataset):
    def remove_punctuation_from_text(input_text):
        output_list=[word for word in input_text.split() if word.isalpha()]
        return ' '.join(output_list)    
    dataset['headline']=dataset['headline'].apply(remove_punctuation_from_text)
    dataset['article']=dataset['article'].apply(remove_punctuation_from_text)

### Correct the words 
def correct_words(dataset):
    def correct_text(input_text):
        list_1=[str(TextBlob(word).correct()) for word in input_text.split()]
        output_text= ' '.join(list_1)
        return output_text
    dataset['headline']=dataset['headline'].apply(correct_text)
    dataset['article']=dataset['article'].apply(correct_text)

### Going to the root word   
def lemmatize(dataset):
    def lematize_text(input_text):
        doc=nlp(input_text)
        lemmas=[token.lemma_ for token in doc]
        output_text=' '.join(lemmas)
        return output_text
    dataset['headline']=dataset['headline'].apply(lematize_text)
    dataset['article']=dataset['article'].apply(lematize_text)

### removing stop words 
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

for link in tqdm(top_ten_news):
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

### Making a dataFrame
df = pd.DataFrame({"headline": headline, "article": article })

### Saving the article in a json file 
df[['headline', 'article']].to_json("News.json")

"""### Data cleaning"""

convert_to_lower_case(df)
remove_punctuation(df)
lemmatize(df)
remove_stopwords(df)

"""### Sentiment Analysis using SentimentIntensityAnalyzer """

df_copy1 = df.copy()
df_copy2 = df.copy()
df_copy3 = df.copy()

sia = SentimentIntensityAnalyzer()
df_copy1["sentiment_score"] = df_copy1["headline"].apply(lambda x: sia.polarity_scores(x)["compound"])
df_copy1["sentiment"] = np.select([df_copy1["sentiment_score"] < 0, df_copy1["sentiment_score"] == 0, df_copy1["sentiment_score"] > 0],['neg', 'neu', 'pos'])

print("SentimentIntensityAnalyzer")
print(df_copy1)

"""### Sentiment Analysis using TextBlob """

df_copy2["sentiment_score"] = df_copy2["headline"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df_copy2["sentiment"] = np.select([df_copy2["sentiment_score"] < 0, df_copy2["sentiment_score"] == 0, df_copy2["sentiment_score"] > 0],
                           ['neg', 'neu', 'pos'])

print("TextBlob")
print(df_copy2)

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
df_copy3["sentiment"] = df_copy3["headline"].apply(flair_prediction)

print("flair_prediction")
print(df_copy3)

"""### Visualization"""

x = df_copy1['sentiment'].value_counts()
fig = px.bar(x)
fig.show()

x = df_copy2['sentiment'].value_counts()
fig = px.bar(x)
fig.show()

x = df_copy3['sentiment'].value_counts()
fig = px.bar(x)
fig.show()

end = time.time()
total_time = end - start
print("\n"+ str(total_time))