# NEWS_Article_sentiment_Analysis

The purpose of the document is to explain the details of the project:

## Code Flow:

### Step 1: Get NEWS link from given URL 
a. Pull all the NEWS articles URL from the page url = "https://www.aljazeera.com/where/mozambique/" using python package request and BeautifulSoup.

b. I did not pull only top 10 links at the start as few NEWS articles were missing description.

c. On close examination of the NEWS link it was observe that it contains Date in the URL and were arranged from latest to older 
example: https://www.aljazeera.com/news/2022/5/23/floods-hit-south-africas-kwazulu-natal-province-again',

d. So not reordering of NEWS url was required 

## step 2: Get NEWS description and headlines from the extracted link 
a. Pull the Headline and description from the NEWS links 

b. Filter top 10 valid links and break

c. Store the data into Pandas Data frame 

## Step 3: Data Cleaning 
a. Apply the below data cleaning for both headline and description in the dataframe
    convert to lowercase 
    remove punctuation
    lemmatize
    remove stop words

## Step 4: Applied 3 an off-the-shelf library for sentiment analysis (NLTK (VADER), TextBlob)

Three combination of text was used for the Sentiment analysis:
1. Using Headline
2. using Article 
3. Using combination of Article and Headline ("News_data")

### *It was observed, there was not much improvement in the prediction with option 2(Article) or option 3(Article + Headline) as compared to option 1 (prediction in Headline). So the final code was trained only on Headline.*

#### *Best result was achived through the Flair*

### NLTK (VADER) and TextBlob:
Both packages rely on a rules-based sentiment analyzer. It, therefore, attaches a positive or negative rating to certain words (ex. horrible has a negative association), pays attention to negation if it exists, and returns values based on these words. This tends to work fine, and has the advantage of being simple and extremely fast, but has some weaknesses.

As sentences get longer, more neutral words exist, and therefore, the overall score tends to normalize more towards neutral as well (or does it)

### Flair:
Flair is a pre-trained embedding-based model. This means that each word is represented inside a vector space. Words with vector representations most similar to another word are often used in the same context. This allows us, to, therefore, determine the sentiment of any given vector, and therefore, any given sentence. The embeddings are based on this paper if you are curious about the more technical aspects.

Flair tends to be much slower than its rule-based counterparts but comes at the advantage of being a trained NLP model instead of a rule-based model, which, if done well comes with added performance. To put in perspective how much slower, in running 1200 sentences, NLTK took 0.78 seconds, Textblob took an impressive 0.55 seconds, and Flair took 49 seconds (50–100x longer), which begs the question of whether the added accuracy is truly worth the increased runtime.

## Step 5: visualization
a. For each Sentiment analysis technique result visualization was done 
b. Visualization picture is stored in visualization folder

## Total Time: 15 Sec

# Data 
## *Please refer to data.json file in the repository for the Headline and Article data fetched from Url*

# How to Run the code 

1. Install python and pip3
3. Open the code using any IDE like Visual studio code/PyCharm etc. 
4. Install all the packages from requiremnet.txt
5. Run the code
6. Check the printed data frames, data.json file and charts for results
